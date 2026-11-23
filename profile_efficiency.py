import argparse
import json
import logging
import os
import sys
import warnings

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
for _p in (CURRENT_DIR, PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

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

CONFIGS_7C = [
    (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1),
    (5, 2), (5, 3), (6, 2), (6, 3),
    (2, 2), (2, 3), (3, 2), (3, 3),
]

CONFIGS_11C = [
    (2, 1), (3, 1), (4, 1), (5, 1), (6, 1),
    (4, 2), (4, 3),
    (2, 2), (2, 3), (3, 2),
]

ISO_TIERS_7C = [
    ("Tier-1", (2, 2), (3, 1)),
    ("Tier-2", (2, 3), (4, 1)),
    ("Mid-High", (3, 2), (5, 1)),
    ("Tier-3", (3, 3), (6, 1)),
]

ISO_TIERS_11C = [
    ("Tier-1", (2, 2), (3, 1)),
    ("Tier-2", (2, 3), (4, 1)),
    ("Tier-3", (3, 2), (5, 1)),
]


def setup_logger(save_dir: str):
    logger = logging.getLogger("profile_efficiency")
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.INFO)
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "efficiency_profile.log")
    fmt = logging.Formatter("%(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8", mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def build_baseline(num_classes: int, num_frames: int, img_size: int):
    if BaselineModel is None:
        return None
    return BaselineModel(
        img_size=img_size,
        patch_size=14,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        attn_pool_num_heads=16,
        clip_embed_dim=768,
        num_classes=num_classes,
        num_frames=num_frames,
        use_flash_attn=True,
        use_fused_mlp=True,
        use_fused_rmsnorm=True,
    )


def build_ior_net(max_iterations: int, mem_refiner_layers: int, num_classes: int, num_frames: int):
    return ior_net_small_patch14_224(
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


def profile_model_fp16(
    model,
    device: torch.device,
    batch_size: int = 8,
    num_frames: int = 16,
    img_size: int = 224,
    warmup: int = 8,
    lat_repeats: int = 25,
    thr_repeats: int = 20,
):
    model = model.to(device).half().eval()
    dummy_b1 = torch.randn(1, 3, num_frames, img_size, img_size, device=device).half()
    dummy_bn = torch.randn(batch_size, 3, num_frames, img_size, img_size, device=device).half()

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
        for _ in range(warmup):
            _ = model(dummy_b1)
        torch.cuda.synchronize()
        for _ in range(lat_repeats):
            starter.record()
            _ = model(dummy_b1)
            ender.record()
            torch.cuda.synchronize()
            lat_ms.append(starter.elapsed_time(ender))
    latency_ms = float(np.median(lat_ms))

    torch.cuda.reset_peak_memory_stats(device)
    batch_times = []
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_bn)
        torch.cuda.synchronize()
        for _ in range(thr_repeats):
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=str, default="all", choices=["all", "7c", "11c", "43c"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--lat_repeats", type=int, default=25)
    parser.add_argument("--thr_repeats", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="./outputs/efficiency")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger(args.output_dir)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for FP16 efficiency profiling.")

    device = torch.device("cuda")
    specs = []

    if args.suite in ("all", "7c"):
        if BaselineModel is not None:
            specs.append(
                (
                    "Base_7C",
                    "Static Baseline (K=0, 7C)",
                    lambda: build_baseline(7, args.num_frames, args.img_size),
                )
            )
        for k, l in CONFIGS_7C:
            specs.append(
                (
                    f"IOR_K{k}_L{l}_7C",
                    f"IOR-Net (K={k}, L={l}, 7C)",
                    lambda kk=k, ll=l: build_ior_net(kk, ll, 7, args.num_frames),
                )
            )

    if args.suite in ("all", "11c"):
        if BaselineModel is not None:
            specs.append(
                (
                    "Base_11C",
                    "Static Baseline (K=0, 11C)",
                    lambda: build_baseline(11, args.num_frames, args.img_size),
                )
            )
        for k, l in CONFIGS_11C:
            specs.append(
                (
                    f"IOR_K{k}_L{l}_11C",
                    f"IOR-Net (K={k}, L={l}, 11C)",
                    lambda kk=k, ll=l: build_ior_net(kk, ll, 11, args.num_frames),
                )
            )

    if args.suite in ("all", "43c"):
        specs.append(
            (
                "IOR_K4_L1_43C",
                "IOR-Net MAFW-43 (K=4, L=1, 43C)",
                lambda: build_ior_net(4, 1, 43, args.num_frames),
            )
        )

    logger.info(
        "%-34s | %-10s | %-10s | %-10s | %-13s | %-11s | %s",
        "Configuration",
        "Params(M)",
        "FLOPs(G)",
        "Lat(B=1)",
        f"Speed(B={args.batch_size})",
        "FPS",
        "Mem(MB)",
    )
    logger.info("-" * 116)

    results = {}
    for key, label, builder in specs:
        model = builder()
        if model is None:
            continue
        metrics = profile_model_fp16(
            model,
            device=device,
            batch_size=args.batch_size,
            num_frames=args.num_frames,
            img_size=args.img_size,
            warmup=args.warmup,
            lat_repeats=args.lat_repeats,
            thr_repeats=args.thr_repeats,
        )
        results[key] = metrics
        logger.info(
            "%-34s | %-10.2f | %-10.2f | %6.2f ms | %7.1f cl/s | %7.1f FPS | %6.1f",
            label,
            metrics["params_m"],
            metrics["flops_g"],
            metrics["latency_ms"],
            metrics["clips_per_sec"],
            metrics["fps"],
            metrics["peak_mem_mb"],
        )

    iso_summary = {}
    for suite_lbl, suffix, tiers in (
        ("7C", "7C", ISO_TIERS_7C),
        ("11C", "11C", ISO_TIERS_11C),
    ):
        tier_rows = []
        for tier_name, (k1, l1), (k2, l2) in tiers:
            key1 = f"IOR_K{k1}_L{l1}_{suffix}"
            key2 = f"IOR_K{k2}_L{l2}_{suffix}"
            if key1 not in results or key2 not in results:
                continue
            f1 = results[key1]["flops_g"]
            f2 = results[key2]["flops_g"]
            rel_diff_pct = float(abs(f1 - f2) / max(f1, f2) * 100.0)
            tier_rows.append(
                {
                    "tier": tier_name,
                    "config_a": f"({k1},{l1})",
                    "flops_a": f1,
                    "config_b": f"({k2},{l2})",
                    "flops_b": f2,
                    "rel_diff_pct": rel_diff_pct,
                }
            )
            logger.info(
                "Iso-FLOPs %s %-8s: (%d,%d)=%.2fG vs (%d,%d)=%.2fG -> RelDiff=%.2f%%",
                suite_lbl,
                tier_name,
                k1,
                l1,
                f1,
                k2,
                l2,
                f2,
                rel_diff_pct,
            )
        if tier_rows:
            iso_summary[suite_lbl] = tier_rows

    json_path = os.path.join(args.output_dir, "efficiency_profile.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"profiles": results, "iso_flops": iso_summary}, f, indent=2)
    logger.info("Saved efficiency profile to %s", json_path)


if __name__ == "__main__":
    main()
