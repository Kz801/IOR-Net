import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
for _p in (CURRENT_DIR, PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.train import (
    DATASET_SPECS,
    mean_std_sample,
    setup_logger,
    train_single_fold,
)

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")

ISO_TIERS_BY_DATASET = {
    "dfew": [
        ("Tier-1", [(2, 2), (3, 1)]),
        ("Tier-2", [(2, 3), (4, 1)]),
        ("Tier-3", [(3, 2), (5, 1)]),
    ],
    "ferv39k": [
        ("Tier-1", [(2, 2), (3, 1)]),
        ("Tier-2", [(2, 3), (4, 1)]),
        ("Mid-High", [(3, 2), (5, 1)]),
        ("Tier-3", [(3, 3), (6, 1)]),
    ],
    "mafw11": [
        ("Tier-1", [(2, 2), (3, 1)]),
        ("Tier-2", [(2, 3), (4, 1)]),
        ("Tier-3", [(3, 2), (5, 1)]),
    ],
}


def run_search_config(
    args,
    k_val: int,
    l_val: int,
    phase_name: str,
    folds,
    num_classes: int,
    device: torch.device,
    logger,
):
    t0 = time.time()
    val_scores, val_uars, val_wars = [], [], []
    test_uars, test_wars = [], []

    for fold in folds:
        if len(folds) == 1 and args.dataset == "ferv39k":
            ckpt_name = f"Search_{args.dataset}_K{k_val}_L{l_val}.pth"
        else:
            ckpt_name = f"Search_{args.dataset}_K{k_val}_L{l_val}_Fold{fold}.pth"
        ckpt_path = os.path.join(args.output_dir, ckpt_name)

        rec = train_single_fold(
            dataset_name=args.dataset,
            data_root=args.data_root,
            split_dir=args.split_dir,
            pretrained_path=args.pretrained_path,
            fold=fold,
            max_iterations=k_val,
            mem_refiner_layers=l_val,
            num_classes=num_classes,
            split_mode="inner_val",
            inner_val_ratio=args.inner_val_ratio,
            epochs=args.search_epochs,
            batch_size=args.batch_size,
            base_lr=args.base_lr,
            new_module_lr_mult=args.new_module_lr_mult,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            warmup_ratio=args.warmup_ratio,
            grad_clip=args.grad_clip,
            num_frames=args.num_frames,
            num_workers=args.num_workers,
            seed=args.seed,
            ckpt_path=ckpt_path,
            model_type="ior_net",
            use_odu=True,
            use_pgs=True,
            proto_mode="etf_learnable",
            saliency_mode="max_proto",
            eval_test_after_val=args.eval_blind_test,
            resume_eval_if_exists=True,
            device=device,
            logger=logger,
        )
        val_scores.append(rec["val_res"]["score"])
        val_uars.append(rec["val_res"]["uar"])
        val_wars.append(rec["val_res"]["war"])
        if rec["test_res"] is not None:
            test_uars.append(rec["test_res"]["uar"])
            test_wars.append(rec["test_res"]["war"])

    vs_m, vs_s = mean_std_sample(val_scores)
    vu_m, vu_s = mean_std_sample(val_uars)
    vw_m, vw_s = mean_std_sample(val_wars)

    summary = {
        "K": k_val,
        "L": l_val,
        "phase": phase_name,
        "val_score_mean": float(vs_m),
        "val_score_std": float(vs_s),
        "val_uar_mean": float(vu_m),
        "val_uar_std": float(vu_s),
        "val_war_mean": float(vw_m),
        "val_war_std": float(vw_s),
        "elapsed_min": float((time.time() - t0) / 60.0),
    }
    if test_uars:
        tu_m, tu_s = mean_std_sample(test_uars)
        tw_m, tw_s = mean_std_sample(test_wars)
        summary.update(
            {
                "test_uar_mean": float(tu_m),
                "test_uar_std": float(tu_s),
                "test_war_mean": float(tw_m),
                "test_war_std": float(tw_s),
            }
        )
        logger.info(
            "[Search %s K=%d L=%d | %s] ValScore=%.2f±%.2f (UAR=%.2f±%.2f, WAR=%.2f±%.2f) | BlindTest UAR=%.2f±%.2f WAR=%.2f±%.2f",
            args.dataset.upper(),
            k_val,
            l_val,
            phase_name,
            vs_m,
            vs_s,
            vu_m,
            vu_s,
            vw_m,
            vw_s,
            tu_m,
            tu_s,
            tw_m,
            tw_s,
        )
    else:
        logger.info(
            "[Search %s K=%d L=%d | %s] ValScore=%.2f±%.2f (UAR=%.2f±%.2f, WAR=%.2f±%.2f)",
            args.dataset.upper(),
            k_val,
            l_val,
            phase_name,
            vs_m,
            vs_s,
            vu_m,
            vu_s,
            vw_m,
            vw_s,
        )
    return summary


def run_full_refit(
    args,
    best_k: int,
    best_l: int,
    folds,
    num_classes: int,
    device: torch.device,
    logger,
):
    t0 = time.time()
    test_uars, test_wars, test_per_cls = [], [], []
    for fold in folds:
        if len(folds) == 1 and args.dataset == "ferv39k":
            ckpt_name = f"Final_FullTrain_{args.dataset}_K{best_k}_L{best_l}.pth"
        else:
            ckpt_name = (
                f"Final_FullTrain_{args.dataset}_K{best_k}_L{best_l}_Fold{fold}.pth"
            )
        ckpt_path = os.path.join(args.output_dir, ckpt_name)

        rec = train_single_fold(
            dataset_name=args.dataset,
            data_root=args.data_root,
            split_dir=args.split_dir,
            pretrained_path=args.pretrained_path,
            fold=fold,
            max_iterations=best_k,
            mem_refiner_layers=best_l,
            num_classes=num_classes,
            split_mode="full",
            inner_val_ratio=args.inner_val_ratio,
            epochs=args.refit_epochs,
            batch_size=args.batch_size,
            base_lr=args.base_lr,
            new_module_lr_mult=args.new_module_lr_mult,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            warmup_ratio=args.warmup_ratio,
            grad_clip=args.grad_clip,
            num_frames=args.num_frames,
            num_workers=args.num_workers,
            seed=args.seed,
            ckpt_path=ckpt_path,
            model_type="ior_net",
            use_odu=True,
            use_pgs=True,
            proto_mode="etf_learnable",
            saliency_mode="max_proto",
            eval_test_after_val=False,
            resume_eval_if_exists=True,
            device=device,
            logger=logger,
        )
        test_uars.append(rec["test_res"]["uar"])
        test_wars.append(rec["test_res"]["war"])
        test_per_cls.append(rec["test_res"]["per_class"])

    tu_m, tu_s = mean_std_sample(test_uars)
    tw_m, tw_s = mean_std_sample(test_wars)
    cls_m, cls_s = mean_std_sample(test_per_cls, axis=0)
    summary = {
        "K": best_k,
        "L": best_l,
        "test_uar_mean": float(tu_m),
        "test_uar_std": float(tu_s),
        "test_war_mean": float(tw_m),
        "test_war_std": float(tw_s),
        "per_class_mean": cls_m.tolist(),
        "per_class_std": cls_s.tolist(),
        "elapsed_min": float((time.time() - t0) / 60.0),
    }
    logger.info(
        "[Full Refit %s K=%d L=%d] Test UAR=%.2f±%.2f%% | Test WAR=%.2f±%.2f%%",
        args.dataset.upper(),
        best_k,
        best_l,
        tu_m,
        tu_s,
        tw_m,
        tw_s,
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, choices=["dfew", "ferv39k", "mafw11"]
    )
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--split_dir", type=str, required=True)
    parser.add_argument("--pretrained_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs/arch_search")
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--inner_val_ratio", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min_k", type=int, default=2)
    parser.add_argument("--max_k", type=int, default=10)
    parser.add_argument("--max_l", type=int, default=6)
    parser.add_argument("--search_epochs", type=int, default=50)
    parser.add_argument("--refit_epochs", type=int, default=50)
    parser.add_argument("--run_iso_tiers", action="store_true")
    parser.add_argument("--run_full_refit", action="store_true")
    parser.add_argument("--eval_blind_test", action="store_true")
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
    return parser.parse_args()


def main():
    args = parse_args()
    spec = DATASET_SPECS[args.dataset]
    num_classes = spec["num_classes"]
    folds = args.folds if args.folds else spec["folds"]
    logger = setup_logger(args.output_dir, name=f"arch_search_{args.dataset}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    history = {}
    ordered_summaries = []

    def eval_cached(k_val: int, l_val: int, phase_name: str):
        key = (k_val, l_val)
        if key in history:
            return history[key]
        res = run_search_config(
            args, k_val, l_val, phase_name, folds, num_classes, device, logger
        )
        history[key] = res
        ordered_summaries.append(res)
        return res

    best_k = args.min_k
    best_k_res = eval_cached(best_k, 1, "Stage-1: Search K")
    best_k_score = best_k_res["val_score_mean"]
    drops_k = 0

    for curr_k in range(args.min_k + 1, args.max_k + 1):
        res = eval_cached(curr_k, 1, "Stage-1: Search K")
        if res["val_score_mean"] > best_k_score:
            best_k = curr_k
            best_k_score = res["val_score_mean"]
            drops_k = 0
        else:
            drops_k += 1
            if drops_k >= args.patience:
                break

    best_l = 1
    best_l_score = history[(best_k, 1)]["val_score_mean"]
    drops_l = 0

    for curr_l in range(2, args.max_l + 1):
        res = eval_cached(best_k, curr_l, "Stage-2: Search L")
        if res["val_score_mean"] > best_l_score:
            best_l = curr_l
            best_l_score = res["val_score_mean"]
            drops_l = 0
        else:
            drops_l += 1
            if drops_l >= args.patience:
                break

    if args.run_iso_tiers:
        for tier_name, pairs in ISO_TIERS_BY_DATASET[args.dataset]:
            for cand_k, cand_l in pairs:
                eval_cached(cand_k, cand_l, f"Stage-3: Iso-FLOPs {tier_name}")

    refit_summary = None
    if args.run_full_refit:
        refit_summary = run_full_refit(
            args, best_k, best_l, folds, num_classes, device, logger
        )

    out_payload = {
        "dataset": args.dataset,
        "selected_K": best_k,
        "selected_L": best_l,
        "search_summaries": ordered_summaries,
        "full_refit": refit_summary,
    }
    json_path = os.path.join(args.output_dir, f"arch_search_{args.dataset}_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    logger.info(
        "Architecture search finished on %s: K*=%d, L*=%d. Saved summary to %s",
        args.dataset.upper(),
        best_k,
        best_l,
        json_path,
    )


if __name__ == "__main__":
    main()
