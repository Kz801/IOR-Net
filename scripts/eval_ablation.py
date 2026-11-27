import argparse
import glob
import json
import logging
import os
import random
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
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

try:
    from models.ior_net_ablation import internvideo2_ablation_sota_small as build_ablation_net
except ImportError:
    try:
        from internvideo2_Ablation import internvideo2_ablation_sota_small as build_ablation_net
    except ImportError:
        build_ablation_net = None

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")

DATASET_SPECS = {
    "dfew": {"K": 5, "L": 1, "num_classes": 7, "folds": [1, 2, 3, 4, 5], "pattern": "Weights_{variant}_Fold{fold}.pth"},
    "ferv39k": {"K": 6, "L": 1, "num_classes": 7, "folds": [1], "pattern": "Weights_FERV_{variant}.pth"},
    "mafw11": {"K": 4, "L": 1, "num_classes": 11, "folds": [1, 2, 3, 4, 5], "pattern": "Weights_MAFW_{variant}_Fold{fold}.pth"},
}

ABLATION_SPECS = [
    ("1_IFM_Only", False, False, "etf_learnable", "none"),
    ("2_IFM_w_ODUM", True, False, "etf_learnable", "none"),
    ("3_IFM_w_PGSM", False, True, "etf_learnable", "max_proto"),
    ("4_Full_IOR_Net", True, True, "etf_learnable", "max_proto"),
    ("5_Proto_Random_Init", True, True, "random_init", "max_proto"),
    ("6_Proto_Fixed_ETF", True, True, "fixed_etf", "max_proto"),
    ("7_Saliency_L2Norm", True, True, "etf_learnable", "l2_norm"),
    ("8_Saliency_MeanProto", True, True, "etf_learnable", "mean_proto"),
]


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


def compute_paired_ttest(a, b):
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if len(a_arr) != len(b_arr) or len(a_arr) < 2:
        return float("nan")
    _, p_val = stats.ttest_rel(a_arr, b_arr)
    return float(p_val)


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


def build_variant_model(
    variant_name: str,
    use_odu: bool,
    use_pgs: bool,
    proto_mode: str,
    saliency_mode: str,
    max_iterations: int,
    mem_refiner_layers: int,
    num_classes: int,
    num_frames: int,
):
    if variant_name == "4_Full_IOR_Net" or build_ablation_net is None:
        model = ior_net_small_patch14_224(
            num_frames=num_frames,
            num_classes=num_classes,
            max_iterations=max_iterations,
            isr_mem_refiner_layers=mem_refiner_layers,
            isr_global_decoder_layers=1,
            isr_local_decoder_layers=1,
            isr_query_gen_layers=1,
            isr_query_synth_layers=1,
            isr_global_fuser_layers=1,
        )
    else:
        kwargs = {
            "num_frames": num_frames,
            "num_classes": num_classes,
            "max_iterations": max_iterations,
            "use_odu": use_odu,
            "use_pgs": use_pgs,
            "isr_mem_refiner_layers": mem_refiner_layers,
            "isr_global_decoder_layers": 1,
            "isr_local_decoder_layers": 1,
            "isr_query_gen_layers": 1,
            "isr_query_synth_layers": 1,
            "isr_global_fuser_layers": 1,
        }
        try:
            model = build_ablation_net(
                proto_mode=proto_mode, saliency_mode=saliency_mode, **kwargs
            )
        except TypeError:
            model = build_ablation_net(**kwargs)
    if proto_mode == "fixed_etf" and hasattr(model, "prototypes"):
        model.prototypes.requires_grad = False
    return model


@torch.no_grad()
def evaluate_end_to_end(model, loader, num_classes: int, device: torch.device, desc: str):
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
        "per_class": (recall_score(yt, yp, labels=cls_labels, average=None, zero_division=0) * 100.0).tolist(),
        "targets": yt,
        "preds": yp,
    }


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
    parser.add_argument("--abl_ckpt_dir", type=str, required=True)
    parser.add_argument("--abl_ckpt_pattern", type=str, default="")
    parser.add_argument("--full_ckpt_dir", type=str, default="")
    parser.add_argument("--full_ckpt_pattern", type=str, default="")
    parser.add_argument("--base_ckpt_dir", type=str, default="")
    parser.add_argument("--base_ckpt_pattern", type=str, default="")
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--mem_refiner_layers", type=int, default=None)
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./outputs/ablation")
    return parser.parse_args()


def main():
    t0 = time.time()
    args = parse_args()
    set_seed(args.seed)
    logger = setup_logger(args.output_dir, name=f"ablation_{args.dataset}")

    spec = DATASET_SPECS[args.dataset]
    max_iter = args.max_iterations if args.max_iterations is not None else spec["K"]
    mem_layers = args.mem_refiner_layers if args.mem_refiner_layers is not None else spec["L"]
    num_classes = spec["num_classes"]
    folds = args.folds if args.folds else spec["folds"]
    abl_pattern = args.abl_ckpt_pattern if args.abl_ckpt_pattern else spec["pattern"]
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

    raw_results = {}

    if args.base_ckpt_dir and BaselineModel is not None:
        uars, wars, per_cls, yt_all, yp_all = [], [], [], [], []
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
            res = evaluate_end_to_end(b_model, loaders[fold], num_classes, device, desc=f"Base F{fold}")
            uars.append(res["uar"])
            wars.append(res["war"])
            per_cls.append(res["per_class"])
            yt_all.append(res["targets"])
            yp_all.append(res["preds"])
            del b_model
            torch.cuda.empty_cache()

        um, us = mean_std_sample(uars)
        wm, ws = mean_std_sample(wars)
        raw_results["0_Static_Base"] = {
            "use_odu": False,
            "use_pgs": False,
            "fold_uars": uars,
            "fold_wars": wars,
            "uar_mean": um,
            "uar_std": us,
            "war_mean": wm,
            "war_std": ws,
            "per_class_mean": np.mean(np.array(per_cls), axis=0).tolist(),
            "y_true": np.concatenate(yt_all),
            "y_pred": np.concatenate(yp_all),
        }

    for variant_name, use_odu, use_pgs, proto_mode, saliency_mode in ABLATION_SPECS:
        ckpt_paths = {}
        for fold in folds:
            if variant_name == "4_Full_IOR_Net" and args.full_ckpt_dir:
                if args.full_ckpt_pattern:
                    p = os.path.join(args.full_ckpt_dir, args.full_ckpt_pattern.format(fold=fold))
                elif os.path.isfile(args.full_ckpt_dir):
                    p = args.full_ckpt_dir
                else:
                    p = os.path.join(args.abl_ckpt_dir, abl_pattern.format(variant=variant_name, fold=fold))
            else:
                p = os.path.join(args.abl_ckpt_dir, abl_pattern.format(variant=variant_name, fold=fold))
            if os.path.exists(p):
                ckpt_paths[fold] = p

        if len(ckpt_paths) != len(folds):
            continue

        uars, wars, per_cls, yt_all, yp_all = [], [], [], [], []
        for fold in folds:
            model = build_variant_model(
                variant_name=variant_name,
                use_odu=use_odu,
                use_pgs=use_pgs,
                proto_mode=proto_mode,
                saliency_mode=saliency_mode,
                max_iterations=max_iter,
                mem_refiner_layers=mem_layers,
                num_classes=num_classes,
                num_frames=args.num_frames,
            ).to(device)
            sd = load_clean_state_dict(ckpt_paths[fold])
            model.load_state_dict(sd, strict=True)
            res = evaluate_end_to_end(model, loaders[fold], num_classes, device, desc=f"{variant_name} F{fold}")
            uars.append(res["uar"])
            wars.append(res["war"])
            per_cls.append(res["per_class"])
            yt_all.append(res["targets"])
            yp_all.append(res["preds"])
            del model
            torch.cuda.empty_cache()

        um, us = mean_std_sample(uars)
        wm, ws = mean_std_sample(wars)
        raw_results[variant_name] = {
            "use_odu": use_odu,
            "use_pgs": use_pgs,
            "proto_mode": proto_mode,
            "saliency_mode": saliency_mode,
            "fold_uars": uars,
            "fold_wars": wars,
            "uar_mean": um,
            "uar_std": us,
            "war_mean": wm,
            "war_std": ws,
            "per_class_mean": np.mean(np.array(per_cls), axis=0).tolist(),
            "y_true": np.concatenate(yt_all),
            "y_pred": np.concatenate(yp_all),
        }

    full_entry = raw_results.get("4_Full_IOR_Net")
    export_summary = {}
    for key, rec in raw_results.items():
        if full_entry is not None and key != "4_Full_IOR_Net":
            if len(folds) > 1:
                p_val = compute_paired_ttest(full_entry["fold_uars"], rec["fold_uars"])
            else:
                p_val = compute_mcnemar(full_entry["y_true"], full_entry["y_pred"], rec["y_pred"])
        else:
            p_val = 1.0 if key == "4_Full_IOR_Net" else float("nan")

        cm_norm = (
            confusion_matrix(
                rec["y_true"], rec["y_pred"], labels=list(range(num_classes)), normalize="true"
            )
            * 100.0
        )
        export_summary[key] = {
            "use_odu": rec["use_odu"],
            "use_pgs": rec["use_pgs"],
            "uar_mean": rec["uar_mean"],
            "uar_std": rec["uar_std"],
            "war_mean": rec["war_mean"],
            "war_std": rec["war_std"],
            "p_vs_full": p_val,
            "per_class_mean": rec["per_class_mean"],
            "confusion_matrix_pct": cm_norm.tolist(),
        }
        logger.info(
            "%-20s | UAR=%5.2f±%4.2f%% | WAR=%5.2f±%4.2f%% | p_vs_Full=%.2e",
            key,
            rec["uar_mean"],
            rec["uar_std"],
            rec["war_mean"],
            rec["war_std"],
            p_val,
        )

    json_path = os.path.join(args.output_dir, f"ablation_{args.dataset}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export_summary, f, indent=2)
    logger.info("Saved JSON summary to %s (elapsed %.2f mins).", json_path, (time.time() - t0) / 60.0)


if __name__ == "__main__":
    main()
