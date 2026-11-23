import argparse
import glob
import json
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import accuracy_score, recall_score
from tqdm import tqdm

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    FlopCountAnalysis = None

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
for _p in (CURRENT_DIR, PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.train import (
    DATASET_SPECS,
    build_cosine_warmup_scheduler,
    build_dataloaders,
    build_model,
    build_optimizer,
    evaluate_loader,
    load_pretrained_backbone,
    mean_std_sample,
    set_seed,
    setup_logger,
)

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")

ABLATION_VARIANTS = [
    {
        "name": "0_Static_Base",
        "model_type": "baseline",
        "use_odu": False,
        "use_pgs": False,
        "proto_mode": "none",
        "saliency_mode": "none",
    },
    {
        "name": "1_IFM_Only",
        "model_type": "ablation",
        "use_odu": False,
        "use_pgs": False,
        "proto_mode": "etf_learnable",
        "saliency_mode": "none",
    },
    {
        "name": "2_IFM_w_ODUM",
        "model_type": "ablation",
        "use_odu": True,
        "use_pgs": False,
        "proto_mode": "etf_learnable",
        "saliency_mode": "none",
    },
    {
        "name": "3_IFM_w_PGSM",
        "model_type": "ablation",
        "use_odu": False,
        "use_pgs": True,
        "proto_mode": "etf_learnable",
        "saliency_mode": "max_proto",
    },
    {
        "name": "4_Full_IOR_Net",
        "model_type": "ior_net",
        "use_odu": True,
        "use_pgs": True,
        "proto_mode": "etf_learnable",
        "saliency_mode": "max_proto",
    },
    {
        "name": "5_Proto_Random_Init",
        "model_type": "ablation",
        "use_odu": True,
        "use_pgs": True,
        "proto_mode": "random_init",
        "saliency_mode": "max_proto",
    },
    {
        "name": "6_Proto_Fixed_ETF",
        "model_type": "ablation",
        "use_odu": True,
        "use_pgs": True,
        "proto_mode": "fixed_etf",
        "saliency_mode": "max_proto",
    },
    {
        "name": "7_Saliency_L2Norm",
        "model_type": "ablation",
        "use_odu": True,
        "use_pgs": True,
        "proto_mode": "etf_learnable",
        "saliency_mode": "l2_norm",
    },
    {
        "name": "8_Saliency_MeanProto",
        "model_type": "ablation",
        "use_odu": True,
        "use_pgs": True,
        "proto_mode": "etf_learnable",
        "saliency_mode": "mean_proto",
    },
]

VARIANT_MAP = {v["name"]: v for v in ABLATION_VARIANTS}


def load_clean_state_dict(ckpt_path: str):
    obj = torch.load(ckpt_path, map_location="cpu")
    sd = obj
    if isinstance(obj, dict):
        if "model" in obj and isinstance(obj["model"], dict):
            sd = obj["model"]
        elif "state_dict" in obj and isinstance(obj["state_dict"], dict):
            sd = obj["state_dict"]
    return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}


def apply_ablation_config(model: nn.Module, proto_mode: str, saliency_mode: str) -> nn.Module:
    if hasattr(model, "prototypes"):
        if proto_mode == "random_init":
            with torch.no_grad():
                nn.init.normal_(model.prototypes, mean=0.0, std=0.02)
                model.prototypes.copy_(F.normalize(model.prototypes, p=2, dim=-1))
            model.prototypes.requires_grad = True
        elif proto_mode == "fixed_etf":
            model.prototypes.requires_grad = False
        elif proto_mode == "etf_learnable":
            model.prototypes.requires_grad = True
    model.saliency_mode = saliency_mode
    return model


def compute_paired_ttest(scores_a, scores_b) -> float:
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    _, p_val = stats.ttest_rel(a, b)
    return float(p_val)


def compute_mcnemar_pvalue(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> float:
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


def profile_variant_fp16(
    variant_cfg: dict,
    num_classes: int,
    max_iterations: int,
    mem_refiner_layers: int,
    num_frames: int,
    batch_size: int,
    device: torch.device,
):
    if device.type != "cuda":
        return None

    model = build_model(
        model_type=variant_cfg["model_type"],
        num_classes=num_classes,
        max_iterations=max_iterations,
        mem_refiner_layers=mem_refiner_layers,
        num_frames=num_frames,
        use_odu=variant_cfg["use_odu"],
        use_pgs=variant_cfg["use_pgs"],
        proto_mode=variant_cfg["proto_mode"],
        saliency_mode=variant_cfg["saliency_mode"],
    )
    model = apply_ablation_config(
        model, variant_cfg["proto_mode"], variant_cfg["saliency_mode"]
    ).to(device).half().eval()

    dummy_b1 = torch.randn(1, 3, num_frames, 224, 224, device=device).half()
    dummy_bn = torch.randn(batch_size, 3, num_frames, 224, 224, device=device).half()

    params_m = float(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6)
    flops_g = float("nan")
    if FlopCountAnalysis is not None:
        fc = FlopCountAnalysis(model, dummy_b1)
        fc.unsupported_ops_warnings(False)
        fc.uncalled_modules_warnings(False)
        flops_g = float(fc.total() / 1e9)

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    lat_ms = []
    with torch.no_grad():
        for _ in range(8):
            _ = model(dummy_b1)
        torch.cuda.synchronize()
        for _ in range(25):
            starter.record()
            _ = model(dummy_b1)
            ender.record()
            torch.cuda.synchronize()
            lat_ms.append(starter.elapsed_time(ender))
    latency_ms = float(np.median(lat_ms))

    torch.cuda.reset_peak_memory_stats(device)
    batch_times = []
    with torch.no_grad():
        for _ in range(8):
            _ = model(dummy_bn)
        torch.cuda.synchronize()
        for _ in range(20):
            starter.record()
            _ = model(dummy_bn)
            ender.record()
            torch.cuda.synchronize()
            batch_times.append(starter.elapsed_time(ender) / 1000.0)
    clips_per_sec = float(batch_size / np.median(batch_times))
    fps = float(clips_per_sec * num_frames)
    peak_mem_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))

    del model, dummy_b1, dummy_bn
    torch.cuda.empty_cache()

    return {
        "params_m": params_m,
        "flops_g": flops_g,
        "latency_ms": latency_ms,
        "clips_per_sec": clips_per_sec,
        "fps": fps,
        "peak_mem_mb": peak_mem_mb,
    }


def resolve_variant_ckpt(args, variant_name: str, fold: int, is_single_split: bool) -> str:
    if variant_name == "4_Full_IOR_Net" and args.full_ckpt_dir:
        if args.full_ckpt_pattern:
            candidate = os.path.join(args.full_ckpt_dir, args.full_ckpt_pattern.format(fold=fold))
            if os.path.exists(candidate):
                return candidate
        elif os.path.isfile(args.full_ckpt_dir):
            return args.full_ckpt_dir

    if variant_name == "0_Static_Base" and args.base_ckpt_dir:
        if args.base_ckpt_pattern:
            candidate = os.path.join(args.base_ckpt_dir, args.base_ckpt_pattern.format(fold=fold))
            if os.path.exists(candidate):
                return candidate
        elif os.path.isfile(args.base_ckpt_dir):
            return args.base_ckpt_dir
        else:
            fold_dir = os.path.join(args.base_ckpt_dir, f"fold_{fold}")
            if os.path.isdir(fold_dir):
                pths = sorted(glob.glob(os.path.join(fold_dir, "*.pth")))
                if pths:
                    return pths[0]

    if is_single_split:
        return os.path.join(args.output_dir, f"Weights_{args.dataset}_{variant_name}.pth")
    return os.path.join(args.output_dir, f"Weights_{args.dataset}_{variant_name}_Fold{fold}.pth")


def run_variant_fold(
    args,
    variant_cfg: dict,
    fold: int,
    max_iterations: int,
    mem_refiner_layers: int,
    num_classes: int,
    ckpt_path: str,
    device: torch.device,
    logger,
):
    is_single_split = args.dataset == "ferv39k"
    fold_seed = args.seed if is_single_split else (args.seed + fold)
    set_seed(fold_seed)

    train_loader, _, test_loader = build_dataloaders(
        dataset_name=args.dataset,
        data_root=args.data_root,
        split_dir=args.split_dir,
        fold=fold,
        split_mode="full",
        inner_val_ratio=0.10,
        split_seed=fold_seed,
        num_frames=args.num_frames,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(
        model_type=variant_cfg["model_type"],
        num_classes=num_classes,
        max_iterations=max_iterations,
        mem_refiner_layers=mem_refiner_layers,
        num_frames=args.num_frames,
        use_odu=variant_cfg["use_odu"],
        use_pgs=variant_cfg["use_pgs"],
        proto_mode=variant_cfg["proto_mode"],
        saliency_mode=variant_cfg["saliency_mode"],
    )
    model = apply_ablation_config(
        model, variant_cfg["proto_mode"], variant_cfg["saliency_mode"]
    ).to(device)

    os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)
    variant_name = variant_cfg["name"]

    if args.resume_eval_if_exists and os.path.exists(ckpt_path):
        logger.info("[%s | F%d] Loading checkpoint %s", variant_name, fold, ckpt_path)
        model.load_state_dict(load_clean_state_dict(ckpt_path), strict=True)
        res = evaluate_loader(model, test_loader, num_classes, device, desc=f"{variant_name} F{fold}")
        del model
        torch.cuda.empty_cache()
        return res

    load_pretrained_backbone(model, args.pretrained_path, args.num_frames, logger)
    model = apply_ablation_config(
        model, variant_cfg["proto_mode"], variant_cfg["saliency_mode"]
    )
    optimizer = build_optimizer(
        model,
        base_lr=args.base_lr,
        new_module_lr_mult=args.new_module_lr_mult,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)

    total_steps = len(train_loader) * args.epochs
    scheduler = build_cosine_warmup_scheduler(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", init_scale=2 ** 12, enabled=use_amp)

    best_score = -1.0
    best_epoch = -1

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        steps_in_epoch = 0

        for videos, labels in tqdm(
            train_loader,
            desc=f"{variant_name} F{fold} Ep{epoch + 1}/{args.epochs}",
            leave=False,
        ):
            if videos is None or videos.nelement() == 0:
                continue
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(videos)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += float(loss.item())
            steps_in_epoch += 1

        avg_loss = running_loss / max(1, steps_in_epoch)
        ep_res = evaluate_loader(
            model, test_loader, num_classes, device, desc=f"{variant_name} F{fold} Eval"
        )
        if ep_res["score"] > best_score:
            best_score = ep_res["score"]
            best_epoch = epoch + 1
            torch.save(model.state_dict(), ckpt_path)

        logger.info(
            "[%s | F%d | Ep %02d/%02d] Loss=%.4f | UAR=%.2f%% | WAR=%.2f%% | BestScore=%.2f (Ep %d)",
            variant_name,
            fold,
            epoch + 1,
            args.epochs,
            avg_loss,
            ep_res["uar"],
            ep_res["war"],
            best_score,
            best_epoch,
        )

    model.load_state_dict(load_clean_state_dict(ckpt_path), strict=True)
    final_res = evaluate_loader(
        model, test_loader, num_classes, device, desc=f"{variant_name} F{fold} Final"
    )

    del model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    return final_res


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["dfew", "ferv39k", "mafw11"])
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--split_dir", type=str, required=True)
    parser.add_argument("--pretrained_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs/ablation_train")
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=None,
        choices=[v["name"] for v in ABLATION_VARIANTS],
    )
    parser.add_argument("--full_ckpt_dir", type=str, default="")
    parser.add_argument("--full_ckpt_pattern", type=str, default="")
    parser.add_argument("--base_ckpt_dir", type=str, default="")
    parser.add_argument("--base_ckpt_pattern", type=str, default="")
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--mem_refiner_layers", type=int, default=None)
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--base_lr", type=float, default=1e-5)
    parser.add_argument("--new_module_lr_mult", type=float, default=10.0)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_eval_if_exists", action="store_true")
    parser.add_argument("--profile_efficiency", action="store_true")
    return parser.parse_args()


def main():
    t0 = time.time()
    args = parse_args()
    spec = DATASET_SPECS[args.dataset]
    max_iter = args.max_iterations if args.max_iterations is not None else spec["K"]
    mem_layers = args.mem_refiner_layers if args.mem_refiner_layers is not None else spec["L"]
    num_classes = spec["num_classes"]
    folds = args.folds if args.folds else spec["folds"]
    is_single_split = args.dataset == "ferv39k"

    logger = setup_logger(args.output_dir, name=f"train_ablation_{args.dataset}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selected_names = args.variants if args.variants else [v["name"] for v in ABLATION_VARIANTS]
    selected_variants = [VARIANT_MAP[name] for name in selected_names]

    raw_results = {}
    for variant_cfg in selected_variants:
        v_name = variant_cfg["name"]
        v_t0 = time.time()
        eff = (
            profile_variant_fp16(
                variant_cfg=variant_cfg,
                num_classes=num_classes,
                max_iterations=max_iter,
                mem_refiner_layers=mem_layers,
                num_frames=args.num_frames,
                batch_size=args.batch_size,
                device=device,
            )
            if args.profile_efficiency
            else None
        )

        fold_uars, fold_wars, fold_cls = [], [], []
        yt_list, yp_list = [], []
        for fold in folds:
            ckpt_path = resolve_variant_ckpt(args, v_name, fold, is_single_split)
            res = run_variant_fold(
                args=args,
                variant_cfg=variant_cfg,
                fold=fold,
                max_iterations=max_iter,
                mem_refiner_layers=mem_layers,
                num_classes=num_classes,
                ckpt_path=ckpt_path,
                device=device,
                logger=logger,
            )
            fold_uars.append(res["uar"])
            fold_wars.append(res["war"])
            fold_cls.append(res["per_class"])
            yt_list.append(res["targets"])
            yp_list.append(res["preds"])

        um, us = mean_std_sample(fold_uars)
        wm, ws = mean_std_sample(fold_wars)
        cm, cs = mean_std_sample(fold_cls, axis=0)
        yt_all = np.concatenate(yt_list)
        yp_all = np.concatenate(yp_list)
        cls_labels = list(range(num_classes))

        raw_results[v_name] = {
            "name": v_name,
            "use_odu": variant_cfg["use_odu"],
            "use_pgs": variant_cfg["use_pgs"],
            "proto_mode": variant_cfg["proto_mode"],
            "saliency_mode": variant_cfg["saliency_mode"],
            "fold_uars": fold_uars,
            "fold_wars": fold_wars,
            "uar_mean": float(um),
            "uar_std": float(us),
            "war_mean": float(wm),
            "war_std": float(ws),
            "pooled_uar": float(
                recall_score(yt_all, yp_all, labels=cls_labels, average="macro", zero_division=0)
                * 100.0
            ),
            "pooled_war": float(accuracy_score(yt_all, yp_all) * 100.0),
            "per_class_mean": cm.tolist(),
            "per_class_std": cs.tolist(),
            "efficiency": eff,
            "elapsed_min": float((time.time() - v_t0) / 60.0),
            "y_true": yt_all,
            "y_pred": yp_all,
        }
        logger.info(
            "[%s] UAR=%.2f±%.2f%% | WAR=%.2f±%.2f%% | Elapsed=%.2f mins",
            v_name,
            um,
            us,
            wm,
            ws,
            raw_results[v_name]["elapsed_min"],
        )

    full_entry = raw_results.get("4_Full_IOR_Net")
    base_entry = raw_results.get("0_Static_Base")
    export_summary = {}

    for v_name, rec in raw_results.items():
        if full_entry is not None and v_name != "4_Full_IOR_Net":
            p_vs_full = (
                compute_paired_ttest(full_entry["fold_uars"], rec["fold_uars"])
                if len(folds) > 1
                else compute_mcnemar_pvalue(full_entry["y_true"], full_entry["y_pred"], rec["y_pred"])
            )
        else:
            p_vs_full = 1.0 if v_name == "4_Full_IOR_Net" else float("nan")

        if base_entry is not None and v_name != "0_Static_Base":
            p_vs_base = (
                compute_paired_ttest(rec["fold_uars"], base_entry["fold_uars"])
                if len(folds) > 1
                else compute_mcnemar_pvalue(rec["y_true"], rec["y_pred"], base_entry["y_pred"])
            )
        else:
            p_vs_base = 1.0 if v_name == "0_Static_Base" else float("nan")

        export_summary[v_name] = {
            k: v for k, v in rec.items() if k not in ("y_true", "y_pred")
        }
        export_summary[v_name]["p_vs_full"] = p_vs_full
        export_summary[v_name]["p_vs_base"] = p_vs_base
        logger.info(
            "Summary %-20s | UAR=%5.2f±%4.2f%% | WAR=%5.2f±%4.2f%% | p_vs_Full=%.2e | p_vs_Base=%.2e",
            v_name,
            rec["uar_mean"],
            rec["uar_std"],
            rec["war_mean"],
            rec["war_std"],
            p_vs_full,
            p_vs_base,
        )

    json_path = os.path.join(args.output_dir, f"ablation_suite_{args.dataset}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "K": max_iter,
                "L": mem_layers,
                "epochs": args.epochs,
                "seed": args.seed,
                "elapsed_min": float((time.time() - t0) / 60.0),
                "variants": export_summary,
            },
            f,
            indent=2,
        )
    logger.info("Saved ablation suite summary to %s", json_path)


if __name__ == "__main__":
    main()