import argparse
import json
import logging
import os
import random
import sys
import time
import warnings

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import accuracy_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
for _p in (CURRENT_DIR, PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from datasets.dfew import DFEWDataset
from datasets.ferv39k import FERV39KDataset
from datasets.mafw import MAFWDataset
from models.ior_net import ior_net_small_patch14_224

try:
    from models.baseline import InternVideo2 as BaselineModel
except ImportError:
    try:
        from internvideo2_org import InternVideo2 as BaselineModel
    except ImportError:
        BaselineModel = None

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")

DATASET_SPECS = {
    "dfew": {"K": 5, "L": 1, "num_classes": 7, "folds": [1, 2, 3, 4, 5], "pattern": "DFEW_Seed{seed}_K5_L1_Fold{fold}.pth"},
    "ferv39k": {"K": 6, "L": 1, "num_classes": 7, "folds": [1], "pattern": "FERV39K_IOR_K6_L1_Seed{seed}.pth"},
    "mafw11": {"K": 4, "L": 1, "num_classes": 11, "folds": [1, 2, 3, 4, 5], "pattern": "MAFW_Seed{seed}_K4_L1_Fold{fold}.pth"},
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


def compute_mcnemar(y_true, pred_a, pred_b):
    ca = pred_a == y_true
    cb = pred_b == y_true
    b_cnt = int(np.sum(ca & (~cb)))
    c_cnt = int(np.sum((~ca) & cb))
    n_disc = b_cnt + c_cnt
    if n_disc == 0:
        return 1.0
    if hasattr(stats, "binomtest"):
        return float(
            stats.binomtest(b_cnt, n_disc, p=0.5, alternative="two-sided").pvalue
        )
    chi2 = float((abs(b_cnt - c_cnt) - 1.0) ** 2 / n_disc)
    return float(stats.chi2.sf(chi2, df=1))


def load_clean_state_dict(ckpt_path: str):
    obj = torch.load(ckpt_path, map_location="cpu")
    sd = obj
    if isinstance(obj, dict):
        if "model" in obj and isinstance(obj["model"], dict):
            sd = obj["model"]
        elif "state_dict" in obj and isinstance(obj["state_dict"], dict):
            sd = obj["state_dict"]
    return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}


def build_test_dataset(dataset_name: str, data_root: str, split_dir: str, fold: int, num_frames: int):
    if dataset_name == "dfew":
        return DFEWDataset(split_dir, data_root, fold, "test", is_train=False, num_frames=num_frames)
    if dataset_name == "ferv39k":
        return FERV39KDataset(split_dir, data_root, split_type="test", is_train=False, num_frames=num_frames)
    if dataset_name == "mafw11":
        return MAFWDataset(data_root, split_dir, fold=fold, mode="test", num_frames=num_frames, is_train=False)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


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
    return {
        "uar": float(recall_score(yt, yp, labels=cls_labels, average="macro", zero_division=0) * 100.0),
        "war": float(accuracy_score(yt, yp) * 100.0),
        "targets": yt,
        "preds": yp,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["dfew", "ferv39k", "mafw11"])
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--split_dir", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 2026])
    parser.add_argument("--ior_ckpt_dir", type=str, required=True)
    parser.add_argument("--ior_ckpt_pattern", type=str, default="")
    parser.add_argument("--seed42_ior_dir", type=str, default="")
    parser.add_argument("--seed42_ior_pattern", type=str, default="")
    parser.add_argument("--base_ckpt_dir", type=str, default="")
    parser.add_argument("--base_ckpt_pattern", type=str, default="FERV39K_Base_Seed{seed}.pth")
    parser.add_argument("--seed42_base_ckpt", type=str, default="")
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--mem_refiner_layers", type=int, default=None)
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="./outputs/multiseed")
    return parser.parse_args()


def main():
    t0 = time.time()
    args = parse_args()
    set_seed(42)
    logger = setup_logger(args.output_dir, name=f"multiseed_{args.dataset}")

    spec = DATASET_SPECS[args.dataset]
    max_iter = args.max_iterations if args.max_iterations is not None else spec["K"]
    mem_layers = args.mem_refiner_layers if args.mem_refiner_layers is not None else spec["L"]
    num_classes = spec["num_classes"]
    folds = args.folds if args.folds else spec["folds"]
    pattern = args.ior_ckpt_pattern if args.ior_ckpt_pattern else spec["pattern"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = {
        fold: DataLoader(
            build_test_dataset(args.dataset, args.data_root, args.split_dir, fold, args.num_frames),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn_safe,
            pin_memory=True,
        )
        for fold in folds
    }

    seed_rows = []
    pooled_uars, pooled_wars = [], []
    base_seed_uars, base_seed_wars = [], []

    for seed in args.seeds:
        fold_uars, fold_wars = [], []
        last_ior_res = None
        for fold in folds:
            if seed == 42 and args.seed42_ior_dir:
                if args.seed42_ior_pattern:
                    ckpt_path = os.path.join(args.seed42_ior_dir, args.seed42_ior_pattern.format(fold=fold, seed=seed))
                elif os.path.isfile(args.seed42_ior_dir):
                    ckpt_path = args.seed42_ior_dir
                else:
                    ckpt_path = os.path.join(args.ior_ckpt_dir, pattern.format(seed=seed, fold=fold))
            else:
                ckpt_path = os.path.join(args.ior_ckpt_dir, pattern.format(seed=seed, fold=fold))
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

            model = ior_net_small_patch14_224(
                num_frames=args.num_frames,
                num_classes=num_classes,
                max_iterations=max_iter,
                isr_mem_refiner_layers=mem_layers,
                isr_global_decoder_layers=1,
                isr_local_decoder_layers=1,
                isr_query_gen_layers=1,
                isr_query_synth_layers=1,
                isr_global_fuser_layers=1,
            ).to(device)
            model.load_state_dict(load_clean_state_dict(ckpt_path), strict=True)
            res = evaluate_loader(model, loaders[fold], num_classes, device, desc=f"{args.dataset.upper()} S{seed} F{fold}")
            fold_uars.append(res["uar"])
            fold_wars.append(res["war"])
            last_ior_res = res
            del model
            torch.cuda.empty_cache()

        pooled_uars.extend(fold_uars)
        pooled_wars.extend(fold_wars)
        um, us = mean_std_sample(fold_uars)
        wm, ws = mean_std_sample(fold_wars)
        row = {
            "seed": seed,
            "fold_uars": fold_uars,
            "fold_wars": fold_wars,
            "uar_mean": um,
            "uar_std": us,
            "war_mean": wm,
            "war_std": ws,
        }

        if len(folds) == 1 and (args.base_ckpt_dir or (seed == 42 and args.seed42_base_ckpt)) and BaselineModel is not None:
            b_path = (
                args.seed42_base_ckpt
                if (seed == 42 and args.seed42_base_ckpt)
                else os.path.join(args.base_ckpt_dir, args.base_ckpt_pattern.format(seed=seed))
            )
            if os.path.exists(b_path):
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
                b_model.load_state_dict(load_clean_state_dict(b_path), strict=True)
                b_res = evaluate_loader(b_model, loaders[folds[0]], num_classes, device, desc=f"Base S{seed}")
                p_mc = compute_mcnemar(last_ior_res["targets"], last_ior_res["preds"], b_res["preds"])
                row.update({"base_uar": b_res["uar"], "base_war": b_res["war"], "p_mcnemar": p_mc})
                base_seed_uars.append(b_res["uar"])
                base_seed_wars.append(b_res["war"])
                del b_model
                torch.cuda.empty_cache()

        seed_rows.append(row)
        logger.info("Seed %-4d | UAR=%.2f±%.2f%% | WAR=%.2f±%.2f%%", seed, um, us, wm, ws)

    across_uar_m, across_uar_s = mean_std_sample([r["uar_mean"] for r in seed_rows])
    across_war_m, across_war_s = mean_std_sample([r["war_mean"] for r in seed_rows])
    pooled_uar_m, pooled_uar_s = mean_std_sample(pooled_uars)
    pooled_war_m, pooled_war_s = mean_std_sample(pooled_wars)

    summary = {
        "dataset": args.dataset,
        "K": max_iter,
        "L": mem_layers,
        "seeds": seed_rows,
        "across_seeds_uar_mean": across_uar_m,
        "across_seeds_uar_std": across_uar_s,
        "across_seeds_war_mean": across_war_m,
        "across_seeds_war_std": across_war_s,
        "pooled_runs": len(pooled_uars),
        "pooled_uar_mean": pooled_uar_m,
        "pooled_uar_std": pooled_uar_s,
        "pooled_war_mean": pooled_war_m,
        "pooled_war_std": pooled_war_s,
    }

    if len(base_seed_uars) == len(seed_rows) and len(seed_rows) > 1:
        b_um, b_us = mean_std_sample(base_seed_uars)
        b_wm, b_ws = mean_std_sample(base_seed_wars)
        _, p_t = stats.ttest_rel(
            np.asarray([r["uar_mean"] for r in seed_rows]),
            np.asarray(base_seed_uars),
        )
        summary.update(
            {
                "base_uar_mean": b_um,
                "base_uar_std": b_us,
                "base_war_mean": b_wm,
                "base_war_std": b_ws,
                "p_uar_ttest": float(p_t),
            }
        )

    logger.info(
        "%s Across-%d-Seed: UAR=%.2f±%.2f%% | WAR=%.2f±%.2f%% | Pooled-%d: UAR=%.2f±%.2f%% | WAR=%.2f±%.2f%%",
        args.dataset.upper(),
        len(seed_rows),
        across_uar_m,
        across_uar_s,
        across_war_m,
        across_war_s,
        len(pooled_uars),
        pooled_uar_m,
        pooled_uar_s,
        pooled_war_m,
        pooled_war_s,
    )

    json_path = os.path.join(args.output_dir, f"multiseed_{args.dataset}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved JSON summary to %s (elapsed %.2f mins).", json_path, (time.time() - t0) / 60.0)


if __name__ == "__main__":
    main()
