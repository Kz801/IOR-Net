import argparse
import glob
import json
import logging
import os
import random
import re
import sys
import time
import warnings

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
for _p in (CURRENT_DIR, PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from datasets.dfew import DFEWDataset
from datasets.ferv39k import FERV39K_LABEL_MAP, FERV39KDataset
from datasets.mafw import MAFWDataset
from models.ior_net import ior_net_small_patch14_224

try:
    from models.baseline import InternVideo2 as BaselineModel
except ImportError:
    try:
        from internvideo2_org import InternVideo2 as BaselineModel
    except ImportError:
        BaselineModel = None

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    FlopCountAnalysis = None

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")

DATASET_SPECS = {
    "dfew": {"num_classes": 7, "folds": [1, 2, 3, 4, 5], "regex": r"^Search_K(\d+)_L(\d+)_Fold(\d+)\.pth$"},
    "ferv39k": {"num_classes": 7, "folds": [1], "regex": r"^Search_FERV_K(\d+)_L(\d+)\.pth$"},
    "mafw11": {"num_classes": 11, "folds": [1, 2, 3, 4, 5], "regex": r"^Search_MAFW_K(\d+)_L(\d+)_Fold(\d+)\.pth$"},
}


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logger(save_dir: str, name: str):
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.INFO)
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, f"{name}.log")
    fmt = logging.Formatter("%(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8", mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def collate_fn_safe(batch):
    batch = [x for x in batch if x is not None]
    if not batch:
        return torch.empty(0), torch.empty(0)
    return torch.utils.data.dataloader.default_collate(batch)


def mean_std_sample(values):
    arr = np.asarray(values, dtype=np.float64)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return mean_val, std_val


def load_clean_state_dict(ckpt_path: str):
    obj = torch.load(ckpt_path, map_location="cpu")
    sd = obj
    if isinstance(obj, dict):
        if "model" in obj and isinstance(obj["model"], dict):
            sd = obj["model"]
        elif "state_dict" in obj and isinstance(obj["state_dict"], dict):
            sd = obj["state_dict"]
    return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}


def discover_search_checkpoints(dataset_name: str, ckpt_dir: str, ckpt_pattern: str, folds):
    discovered = {}
    if ckpt_pattern:
        for fn in sorted(os.listdir(ckpt_dir)):
            if not fn.endswith(".pth"):
                continue
            m = re.search(r"K(\d+)_L(\d+)", fn)
            if not m:
                continue
            k_val, l_val = int(m.group(1)), int(m.group(2))
            if len(folds) == 1:
                expected = ckpt_pattern.format(K=k_val, L=l_val, fold=folds[0])
                if fn == expected:
                    discovered[(k_val, l_val)] = {folds[0]: os.path.join(ckpt_dir, fn)}
            else:
                for f in folds:
                    if fn == ckpt_pattern.format(K=k_val, L=l_val, fold=f):
                        discovered.setdefault((k_val, l_val), {})[f] = os.path.join(ckpt_dir, fn)
        return discovered

    regex = re.compile(DATASET_SPECS[dataset_name]["regex"])
    for fn in sorted(os.listdir(ckpt_dir)):
        m = regex.match(fn)
        if not m:
            continue
        k_val, l_val = int(m.group(1)), int(m.group(2))
        if len(folds) == 1:
            discovered[(k_val, l_val)] = {folds[0]: os.path.join(ckpt_dir, fn)}
        else:
            f_val = int(m.group(3))
            if f_val in folds:
                discovered.setdefault((k_val, l_val), {})[f_val] = os.path.join(ckpt_dir, fn)
    return discovered


def build_inner_val_and_test_loaders(
    dataset_name: str,
    data_root: str,
    split_dir: str,
    fold: int,
    inner_val_ratio: float,
    seed: int,
    num_frames: int,
    batch_size: int,
    num_workers: int,
):
    if dataset_name == "dfew":
        val_base = DFEWDataset(split_dir, data_root, fold, "train", is_train=False, num_frames=num_frames)
        targets = (val_base.data_info["label"].astype(int) - 1).tolist()
        test_ds = DFEWDataset(split_dir, data_root, fold, "test", is_train=False, num_frames=num_frames)
        split_seed = seed + fold
    elif dataset_name == "ferv39k":
        val_base = FERV39KDataset(split_dir, data_root, split_type="train", is_train=False, num_frames=num_frames)
        targets = [
            FERV39K_LABEL_MAP.get(str(x).strip(), 0)
            for x in val_base.data_info["label"].tolist()
        ]
        test_ds = FERV39KDataset(split_dir, data_root, split_type="test", is_train=False, num_frames=num_frames)
        split_seed = seed
    elif dataset_name == "mafw11":
        val_base = MAFWDataset(data_root, split_dir, fold=fold, mode="train", num_frames=num_frames, is_train=False)
        targets = [lbl for _, lbl in val_base.data_list]
        test_ds = MAFWDataset(data_root, split_dir, fold=fold, mode="test", num_frames=num_frames, is_train=False)
        split_seed = seed + fold
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    indices = list(range(len(val_base)))
    _, val_idx = train_test_split(
        indices,
        test_size=inner_val_ratio,
        random_state=split_seed,
        shuffle=True,
        stratify=targets,
    )
    val_loader = DataLoader(
        Subset(val_base, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_safe,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_safe,
        pin_memory=True,
    )
    return val_loader, test_loader


@torch.no_grad()
def evaluate_loader(model, loader, num_classes: int, device: torch.device, desc: str):
    model.eval()
    all_t, all_p = [], []
    use_amp = device.type == "cuda"
    for videos, labels in tqdm(loader, desc=desc, leave=False):
        if videos is None or videos.nelement() == 0:
            continue
        videos = videos.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(videos)
        all_p.append(torch.argmax(logits, dim=-1).cpu().numpy())
        all_t.append(labels.numpy())
    yt = np.concatenate(all_t)
    yp = np.concatenate(all_p)
    cls_labels = list(range(num_classes))
    uar = float(recall_score(yt, yp, labels=cls_labels, average="macro", zero_division=0) * 100.0)
    war = float(accuracy_score(yt, yp) * 100.0)
    return {
        "uar": uar,
        "war": war,
        "score": float(0.7 * uar + 0.3 * war),
        "per_class": (recall_score(yt, yp, labels=cls_labels, average=None, zero_division=0) * 100.0).tolist(),
    }


def compute_params_flops(model, device: torch.device, num_frames: int):
    params_m = float(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6)
    flops_g = float("nan")
    if FlopCountAnalysis is not None and device.type == "cuda":
        m_fp16 = model.half().eval()
        dummy = torch.randn(1, 3, num_frames, 224, 224, device=device).half()
        fc = FlopCountAnalysis(m_fp16, dummy)
        fc.unsupported_ops_warnings(False)
        fc.uncalled_modules_warnings(False)
        flops_g = float(fc.total() / 1e9)
        model.float()
        del dummy
        torch.cuda.empty_cache()
    return params_m, flops_g


def resolve_baseline_ckpt(root: str, pattern: str, fold: int):
    if pattern:
        return os.path.join(root, pattern.format(fold=fold))
    if os.path.isfile(root):
        return root
    fold_dir = os.path.join(root, f"fold_{fold}")
    pths = sorted(glob.glob(os.path.join(fold_dir, "*.pth")))
    if len(pths) == 1:
        return pths[0]
    target = os.path.join(fold_dir, "target_reached_model.pth")
    return target if os.path.exists(target) else pths[0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["dfew", "ferv39k", "mafw11"])
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--split_dir", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--ckpt_pattern", type=str, default="")
    parser.add_argument("--base_ckpt_dir", type=str, default="")
    parser.add_argument("--base_ckpt_pattern", type=str, default="")
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--inner_val_ratio", type=float, default=0.10)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile_flops", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./outputs/arch_search")
    return parser.parse_args()


def main():
    t0 = time.time()
    args = parse_args()
    set_seed(args.seed)
    logger = setup_logger(args.output_dir, name=f"arch_search_{args.dataset}")

    spec = DATASET_SPECS[args.dataset]
    num_classes = spec["num_classes"]
    folds = args.folds if args.folds else spec["folds"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    discovered = discover_search_checkpoints(args.dataset, args.ckpt_dir, args.ckpt_pattern, folds)
    if not discovered:
        raise FileNotFoundError(f"No search checkpoints discovered in {args.ckpt_dir}")

    val_loaders = {}
    test_loaders = {}
    for fold in folds:
        val_loaders[fold], test_loaders[fold] = build_inner_val_and_test_loaders(
            args.dataset,
            args.data_root,
            args.split_dir,
            fold,
            args.inner_val_ratio,
            args.seed,
            args.num_frames,
            args.batch_size,
            args.num_workers,
        )

    base_uars = None
    if args.base_ckpt_dir and BaselineModel is not None:
        base_uars = []
        for fold in folds:
            b_ckpt = resolve_baseline_ckpt(args.base_ckpt_dir, args.base_ckpt_pattern, fold)
            b_model = BaselineModel(
                img_size=224,
                patch_size=14,
                embed_dim=384,
                depth=12,
                num_heads=6,
                mlp_ratio=4,
                attn_pool_num_heads=16,
                clip_embed_dim=768,
                num_classes=num_classes,
                num_frames=args.num_frames,
                use_flash_attn=True,
                use_fused_mlp=True,
                use_fused_rmsnorm=True,
            ).to(device)
            b_model.load_state_dict(load_clean_state_dict(b_ckpt), strict=True)
            res = evaluate_loader(b_model, test_loaders[fold], num_classes, device, desc=f"Base F{fold}")
            base_uars.append(res["uar"])
            del b_model
            torch.cuda.empty_cache()

    rows = []
    for k_val, l_val in sorted(discovered.keys()):
        fold_map = discovered[(k_val, l_val)]
        val_scores, val_uars, val_wars = [], [], []
        test_uars, test_wars = [], []
        params_m, flops_g = float("nan"), float("nan")

        for idx_f, fold in enumerate(sorted(fold_map.keys())):
            model = ior_net_small_patch14_224(
                num_frames=args.num_frames,
                num_classes=num_classes,
                max_iterations=k_val,
                isr_mem_refiner_layers=l_val,
                isr_global_decoder_layers=1,
                isr_local_decoder_layers=1,
                isr_query_gen_layers=1,
                isr_query_synth_layers=1,
                isr_global_fuser_layers=1,
            ).to(device)
            if idx_f == 0:
                if args.profile_flops:
                    params_m, flops_g = compute_params_flops(model, device, args.num_frames)
                else:
                    params_m = float(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6)
            model.load_state_dict(load_clean_state_dict(fold_map[fold]), strict=True)
            vr = evaluate_loader(model, val_loaders[fold], num_classes, device, desc=f"Val K{k_val}L{l_val} F{fold}")
            tr = evaluate_loader(model, test_loaders[fold], num_classes, device, desc=f"Test K{k_val}L{l_val} F{fold}")
            val_scores.append(vr["score"])
            val_uars.append(vr["uar"])
            val_wars.append(vr["war"])
            test_uars.append(tr["uar"])
            test_wars.append(tr["war"])
            del model
            torch.cuda.empty_cache()

        vs_m, vs_s = mean_std_sample(val_scores)
        vu_m, vu_s = mean_std_sample(val_uars)
        vw_m, vw_s = mean_std_sample(val_wars)
        tu_m, tu_s = mean_std_sample(test_uars)
        tw_m, tw_s = mean_std_sample(test_wars)
        p_base = float("nan")
        if base_uars is not None and len(base_uars) == len(test_uars) and len(test_uars) > 1:
            _, p_t = stats.ttest_rel(np.asarray(test_uars), np.asarray(base_uars))
            p_base = float(p_t)

        row = {
            "K": k_val,
            "L": l_val,
            "params_m": params_m,
            "flops_g": flops_g,
            "val_score_mean": vs_m,
            "val_score_std": vs_s,
            "val_uar_mean": vu_m,
            "val_uar_std": vu_s,
            "val_war_mean": vw_m,
            "val_war_std": vw_s,
            "test_uar_mean": tu_m,
            "test_uar_std": tu_s,
            "test_war_mean": tw_m,
            "test_war_std": tw_s,
            "p_vs_base": p_base,
        }
        rows.append(row)
        logger.info(
            "K=%d, L=%d | Params=%.2fM | FLOPs=%.2fG | ValScore=%.2f±%.2f | ValUAR=%.2f±%.2f | TestUAR=%.2f±%.2f | TestWAR=%.2f±%.2f | p=%.2e",
            k_val,
            l_val,
            params_m,
            flops_g,
            vs_m,
            vs_s,
            vu_m,
            vu_s,
            tu_m,
            tu_s,
            tw_m,
            tw_s,
            p_base,
        )

    best_row = max(rows, key=lambda r: r["val_score_mean"])
    logger.info(
        "Best Inner-Val Config on %s: K=%d, L=%d (ValScore=%.2f±%.2f, TestUAR=%.2f±%.2f)",
        args.dataset.upper(),
        best_row["K"],
        best_row["L"],
        best_row["val_score_mean"],
        best_row["val_score_std"],
        best_row["test_uar_mean"],
        best_row["test_uar_std"],
    )

    json_path = os.path.join(args.output_dir, f"arch_search_{args.dataset}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"dataset": args.dataset, "best": best_row, "results": rows}, f, indent=2)
    logger.info("Saved JSON summary to %s (elapsed %.2f mins).", json_path, (time.time() - t0) / 60.0)


if __name__ == "__main__":
    main()
