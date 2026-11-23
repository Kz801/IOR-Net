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
from sklearn.metrics import accuracy_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
for _p in (CURRENT_DIR, PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from datasets.corruptions import (
    ALL_12_CONDITIONS,
    DLIB_CONDITIONS,
    DlibLandmarkMaskWrapper,
    build_corruptions_batch,
    collate_fn_dlib,
)
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

DATASET_DEFAULTS = {
    "dfew": {"K": 5, "L": 1, "num_classes": 7, "folds": [1, 2, 3, 4, 5]},
    "ferv39k": {"K": 6, "L": 1, "num_classes": 7, "folds": [1]},
    "mafw11": {"K": 4, "L": 1, "num_classes": 11, "folds": [1, 2, 3, 4, 5]},
}


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logger(save_dir: str, name: str = "eval_robustness"):
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


def mean_std_sample(values):
    arr = np.asarray(values, dtype=np.float64)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return mean_val, std_val


def compute_paired_ttest(a, b):
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if len(a_arr) != len(b_arr) or len(a_arr) < 2:
        return {"t": float("nan"), "p_t": float("nan")}
    t_val, p_t = stats.ttest_rel(a_arr, b_arr)
    return {"t": float(t_val), "p_t": float(p_t)}


def compute_mcnemar(y_true, pred_a, pred_b):
    ca = pred_a == y_true
    cb = pred_b == y_true
    b_cnt = int(np.sum(ca & (~cb)))
    c_cnt = int(np.sum((~ca) & cb))
    n_disc = b_cnt + c_cnt
    if n_disc == 0:
        return {"b": 0, "c": 0, "p": 1.0}
    if hasattr(stats, "binomtest"):
        p_val = float(
            stats.binomtest(b_cnt, n_disc, p=0.5, alternative="two-sided").pvalue
        )
    else:
        chi2 = float((abs(b_cnt - c_cnt) - 1.0) ** 2 / n_disc)
        p_val = float(stats.chi2.sf(chi2, df=1))
    return {"b": b_cnt, "c": c_cnt, "p": p_val}


def load_clean_state_dict(ckpt_path: str):
    obj = torch.load(ckpt_path, map_location="cpu")
    sd = obj
    if isinstance(obj, dict):
        if "model" in obj and isinstance(obj["model"], dict):
            sd = obj["model"]
        elif "state_dict" in obj and isinstance(obj["state_dict"], dict):
            sd = obj["state_dict"]
    return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}


def resolve_ckpt_path(ckpt_dir: str, pattern: str, fold: int):
    if pattern:
        candidate = os.path.join(ckpt_dir, pattern.format(fold=fold))
        if os.path.exists(candidate):
            return candidate
        raise FileNotFoundError(f"Checkpoint not found: {candidate}")

    if os.path.isfile(ckpt_dir):
        return ckpt_dir

    fold_dir = os.path.join(ckpt_dir, f"fold_{fold}")
    if os.path.isdir(fold_dir):
        pths = sorted(glob.glob(os.path.join(fold_dir, "*.pth")))
        if len(pths) == 1:
            return pths[0]
        target = os.path.join(fold_dir, "target_reached_model.pth")
        if os.path.exists(target):
            return target
        if pths:
            return pths[0]

    raise FileNotFoundError(f"Unable to resolve checkpoint in {ckpt_dir} for fold {fold}")


def build_baseline_model(num_classes: int, num_frames: int = 16, img_size: int = 224):
    if BaselineModel is None:
        raise ImportError("Baseline model definition could not be imported.")
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


def build_ior_model(num_classes: int, max_iterations: int, mem_refiner_layers: int, num_frames: int = 16):
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


@torch.no_grad()
def forward_ior_dynamics(model, x):
    x_cls_normed, patches_normed = model._encode_backbone(x)
    batch_size = x_cls_normed.shape[0]

    tgt_init = model.norm_q_gen(model.initial_queries.expand(batch_size, -1, -1))
    mem_init = model.norm_kv_gen(patches_normed)
    h_init = tgt_init
    for layer in model.query_generator_layers:
        h_init = layer(tgt=h_init, memory=mem_init)
    h_init = model.query_generator_norm(h_init)

    target_norm_fp32 = F.normalize(model.prototypes.float(), p=2, dim=-1)
    target_norm = F.normalize(model.prototypes, p=2, dim=-1)

    h_glo, h_loc = h_init, h_init
    m_glo, m_loc = x_cls_normed.unsqueeze(1), patches_normed
    v_m_loc = torch.zeros_like(m_loc)
    v_m_glo = torch.zeros_like(m_glo)
    v_h_glo = torch.zeros_like(h_glo)
    v_h_loc = torch.zeros_like(h_loc)

    step_logits, ortho_errors = [], []
    for k in range(model.max_iterations):
        p_glo, r_glo = model._orthogonal_decompose(h_glo, target_norm)
        p_loc, r_loc = model._orthogonal_decompose(h_loc, target_norm)

        h_loc_fp32 = h_loc.float()
        w_fp32 = target_norm_fp32.unsqueeze(0)
        p_fp32 = torch.sum(h_loc_fp32 * w_fp32, dim=-1, keepdim=True) * w_fp32
        r_fp32 = h_loc_fp32 - p_fp32
        cos_ortho = torch.abs(
            torch.sum(p_fp32 * r_fp32, dim=-1)
            / (
                torch.norm(p_fp32, p=2, dim=-1) * torch.norm(r_fp32, p=2, dim=-1)
                + 1e-12
            )
        ).mean().item()
        ortho_errors.append(cos_ortho)

        q_glo = p_glo + model.global_decoder(
            tgt=model.norm_q_L1_global(r_glo),
            memory=model.norm_kv_L1_global(m_glo),
        )
        q_loc = p_loc + model.local_decoder(
            tgt=model.norm_q_L2_local(r_loc),
            memory=model.norm_kv_L2_local(m_loc),
        )

        scale_factor = model.logit_scale.exp()
        logits_glo = (
            torch.sum(F.normalize(q_glo, p=2, dim=-1) * target_norm.unsqueeze(0), dim=-1)
            * scale_factor
        )
        logits_loc = (
            torch.sum(F.normalize(q_loc, p=2, dim=-1) * target_norm.unsqueeze(0), dim=-1)
            * scale_factor
        )

        gate_input = torch.cat(
            [
                x_cls_normed,
                q_glo.mean(dim=1),
                q_loc.mean(dim=1),
                torch.abs(q_glo - q_loc).mean(dim=1),
            ],
            dim=-1,
        )
        gate_glo, gate_loc = torch.chunk(model.gating_network(gate_input), 2, dim=-1)
        cur_logits = (
            torch.sigmoid(gate_glo) * logits_glo
            + torch.sigmoid(gate_loc) * logits_loc
        )
        step_logits.append(cur_logits.float().detach().cpu())

        if k < model.max_iterations - 1:
            delta_m_loc = model.norm_q_mem_ref(m_loc)
            feedback_loc = model.norm_kv_mem_ref(q_loc)
            for layer in model.memory_refiner_local_layers:
                delta_m_loc = layer(
                    tgt=delta_m_loc, memory=feedback_loc, prototypes=model.prototypes
                )
            v_m_loc = (
                model.momentum_factor * v_m_loc
                + (1.0 - model.momentum_factor)
                * model.memory_refiner_local_norm(delta_m_loc)
            )
            m_loc = m_loc + v_m_loc

            pooled_loc = model.norm_for_global_calc(m_loc).mean(dim=1, keepdim=True)
            delta_m_glo = model.norm_q_fuser(m_glo)
            mem_fuser = model.norm_kv_fuser(pooled_loc)
            for layer in model.global_memory_fuser_layers:
                delta_m_glo = layer(tgt=delta_m_glo, memory=mem_fuser)
            v_m_glo = (
                model.momentum_factor * v_m_glo
                + (1.0 - model.momentum_factor)
                * model.global_memory_fuser_norm(delta_m_glo)
            )
            m_glo = m_glo + v_m_glo

            delta_h_glo = model.global_query_synthesizer(
                tgt=model.norm_q_synth_g(q_glo), memory=model.norm_kv_synth_g(q_loc)
            )
            delta_h_loc = model.local_query_synthesizer(
                tgt=model.norm_q_synth_l(q_loc), memory=model.norm_kv_synth_l(q_glo)
            )
            v_h_glo = (
                model.momentum_factor * v_h_glo
                + (1.0 - model.momentum_factor) * delta_h_glo
            )
            v_h_loc = (
                model.momentum_factor * v_h_loc
                + (1.0 - model.momentum_factor) * delta_h_loc
            )
            h_glo = h_glo + v_h_glo
            h_loc = h_loc + v_h_loc

    return step_logits, ortho_errors


def build_test_dataset(dataset_name: str, data_root: str, split_dir: str, fold: int, num_frames: int):
    if dataset_name == "dfew":
        return DFEWDataset(
            split_csv_dir=split_dir,
            clip_root=data_root,
            fold_num=fold,
            split_type="test",
            is_train=False,
            num_frames=num_frames,
        )
    if dataset_name == "ferv39k":
        return FERV39KDataset(
            csv_root_dir=split_dir,
            clip_root=data_root,
            split_type="test",
            is_train=False,
            num_frames=num_frames,
        )
    if dataset_name == "mafw11":
        return MAFWDataset(
            frames_root=data_root,
            txt_root_base=split_dir,
            fold=fold,
            mode="test",
            num_frames=num_frames,
            is_train=False,
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def evaluate_robustness(args, logger):
    cfg = DATASET_DEFAULTS[args.dataset]
    max_iter = args.max_iterations if args.max_iterations is not None else cfg["K"]
    mem_layers = args.mem_refiner_layers if args.mem_refiner_layers is not None else cfg["L"]
    num_classes = cfg["num_classes"]
    folds = args.folds if args.folds else cfg["folds"]

    include_all_12 = args.conditions == "all"
    cond_keys = ALL_12_CONDITIONS if include_all_12 else DLIB_CONDITIONS
    eval_baseline = bool(args.base_ckpt_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    cls_labels = list(range(num_classes))

    store = {ck: {"base": [], "ior": []} for ck in cond_keys}
    tot_frames_all, tot_det_frames, tot_clips_all, tot_clip_hits = 0, 0, 0, 0
    area_bg_sum, area_eyes_sum, area_mouth_sum = 0.0, 0.0, 0.0

    for fold in folds:
        ior_ckpt = resolve_ckpt_path(args.ior_ckpt_dir, args.ior_ckpt_pattern, fold)
        ior_model = build_ior_model(num_classes, max_iter, mem_layers, args.num_frames).to(device)
        ior_model.load_state_dict(load_clean_state_dict(ior_ckpt), strict=True)
        ior_model.eval()

        base_model = None
        if eval_baseline:
            base_ckpt = resolve_ckpt_path(args.base_ckpt_dir, args.base_ckpt_pattern, fold)
            base_model = build_baseline_model(num_classes, args.num_frames).to(device)
            base_model.load_state_dict(load_clean_state_dict(base_ckpt), strict=True)
            base_model.eval()

        base_ds = build_test_dataset(
            args.dataset, args.data_root, args.split_dir, fold, args.num_frames
        )
        wrapped_ds = DlibLandmarkMaskWrapper(base_ds, args.dlib_predictor)
        loader = DataLoader(
            wrapped_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn_dlib,
            pin_memory=True,
        )

        buf = {
            ck: {
                "yt": [],
                "bp": [],
                "ilg": [[] for _ in range(max_iter)],
                "ortho": [],
            }
            for ck in cond_keys
        }
        seed_base = args.seed if len(folds) == 1 else (args.seed + fold * 100000)
        sample_offset = 0

        for batch in tqdm(loader, desc=f"{args.dataset.upper()} F{fold}", leave=False):
            if batch is None:
                continue
            videos, labels, m_bg, m_eyes, m_mouth, n_det, clip_hit = batch
            batch_size = videos.shape[0]

            tot_clips_all += batch_size
            tot_frames_all += batch_size * args.num_frames
            tot_det_frames += int(n_det.sum().item())
            tot_clip_hits += int(clip_hit.sum().item())
            area_bg_sum += float(m_bg.float().mean(dim=(1, 2, 3)).sum().item())
            area_eyes_sum += float(m_eyes.float().mean(dim=(1, 2, 3)).sum().item())
            area_mouth_sum += float(m_mouth.float().mean(dim=(1, 2, 3)).sum().item())

            videos = videos.to(device, non_blocking=True)
            m_bg = m_bg.to(device, non_blocking=True)
            m_eyes = m_eyes.to(device, non_blocking=True)
            m_mouth = m_mouth.to(device, non_blocking=True)

            corrupted_dict = build_corruptions_batch(
                videos,
                m_bg,
                m_eyes,
                m_mouth,
                seed_base=seed_base,
                batch_start_idx=sample_offset,
                device=device,
                include_all_12=include_all_12,
            )
            sample_offset += batch_size
            labels_np = labels.numpy()

            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                for ck in cond_keys:
                    x_c = corrupted_dict[ck]
                    if base_model is not None:
                        buf[ck]["bp"].append(
                            torch.argmax(base_model(x_c), dim=-1).cpu().numpy()
                        )
                    step_lgs, ortho_errs = forward_ior_dynamics(ior_model, x_c)
                    for k in range(max_iter):
                        buf[ck]["ilg"][k].append(step_lgs[k].numpy())
                    buf[ck]["ortho"].append(ortho_errs)
                    buf[ck]["yt"].append(labels_np)

        for ck in cond_keys:
            yt = np.concatenate(buf[ck]["yt"])
            sp = [np.argmax(np.concatenate(buf[ck]["ilg"][k]), axis=-1) for k in range(max_iter)]
            store[ck]["ior"].append(
                {
                    "uar": float(
                        recall_score(yt, sp[-1], labels=cls_labels, average="macro", zero_division=0)
                        * 100.0
                    ),
                    "war": float(accuracy_score(yt, sp[-1]) * 100.0),
                    "yt": yt,
                    "step_preds": sp,
                    "ortho_mean": np.mean(np.array(buf[ck]["ortho"]), axis=0).tolist(),
                }
            )
            if base_model is not None:
                bp = np.concatenate(buf[ck]["bp"])
                store[ck]["base"].append(
                    {
                        "uar": float(
                            recall_score(yt, bp, labels=cls_labels, average="macro", zero_division=0)
                            * 100.0
                        ),
                        "war": float(accuracy_score(yt, bp) * 100.0),
                        "yt": yt,
                        "preds": bp,
                    }
                )

        del ior_model
        if base_model is not None:
            del base_model
        torch.cuda.empty_cache()

    frame_det_rate = tot_det_frames / max(tot_frames_all, 1) * 100.0
    clip_det_rate = tot_clip_hits / max(tot_clips_all, 1) * 100.0
    avg_bg_pct = area_bg_sum / max(tot_clips_all, 1) * 100.0
    avg_eyes_pct = area_eyes_sum / max(tot_clips_all, 1) * 100.0
    avg_mouth_pct = area_mouth_sum / max(tot_clips_all, 1) * 100.0

    logger.info(
        "Dataset: %s | K=%d, L=%d | Clips=%d | Folds=%d",
        args.dataset.upper(),
        max_iter,
        mem_layers,
        tot_clips_all,
        len(folds),
    )
    logger.info(
        "dlib Coverage: Frame=%.2f%% | Clip(>=1f)=%.2f%% | Avg Area (BG/Eyes/Mouth)=%.1f%%/%.1f%%/%.1f%%",
        frame_det_rate,
        clip_det_rate,
        avg_bg_pct,
        avg_eyes_pct,
        avg_mouth_pct,
    )

    clean_ior_uar = float(np.mean([r["uar"] for r in store["Clean"]["ior"]]))
    summary_rows = []
    for ck in cond_keys:
        iu_m, iu_s = mean_std_sample([r["uar"] for r in store[ck]["ior"]])
        iw_m, iw_s = mean_std_sample([r["war"] for r in store[ck]["ior"]])
        ret = iu_m / clean_ior_uar * 100.0

        yt_all = np.concatenate([r["yt"] for r in store[ck]["ior"]])
        p1_all = np.concatenate([r["step_preds"][0] for r in store[ck]["ior"]])
        pk_all = np.concatenate([r["step_preds"][-1] for r in store[ck]["ior"]])
        c1k = int(np.sum((p1_all != yt_all) & (pk_all == yt_all)))
        d1k = int(np.sum((p1_all == yt_all) & (pk_all != yt_all)))
        ratio = float(c1k / max(d1k, 1))
        omax = float(
            np.max(np.mean(np.array([r["ortho_mean"] for r in store[ck]["ior"]]), axis=0))
        )

        row = {
            "condition": ck,
            "ior_uar_mean": iu_m,
            "ior_uar_std": iu_s,
            "ior_ret_pct": ret,
            "ior_war_mean": iw_m,
            "ior_war_std": iw_s,
            "corrected_1k": c1k,
            "erroneous_1k": d1k,
            "ratio_1k": ratio,
            "max_ortho_err": omax,
        }

        if eval_baseline:
            bu_m, bu_s = mean_std_sample([r["uar"] for r in store[ck]["base"]])
            bw_m, bw_s = mean_std_sample([r["war"] for r in store[ck]["base"]])
            bp_all = np.concatenate([r["preds"] for r in store[ck]["base"]])
            pt = (
                compute_paired_ttest(
                    [r["uar"] for r in store[ck]["ior"]],
                    [r["uar"] for r in store[ck]["base"]],
                )["p_t"]
                if len(folds) > 1
                else float("nan")
            )
            pmc = compute_mcnemar(yt_all, pk_all, bp_all)["p"]
            row.update(
                {
                    "base_uar_mean": bu_m,
                    "base_uar_std": bu_s,
                    "base_war_mean": bw_m,
                    "base_war_std": bw_s,
                    "delta_uar": iu_m - bu_m,
                    "delta_war": iw_m - bw_m,
                    "p_ttest": pt,
                    "p_mcnemar": pmc,
                }
            )
            logger.info(
                "%-16s | Base: %5.2f±%4.2f / %5.2f | IOR: %5.2f±%4.2f (Ret %5.1f%%) / %5.2f | dUAR=%+6.2f%% (p_t=%.2e, p_mc=%.2e) | +%d/-%d (%.2f:1) | |<P,R>|=%.2e",
                ck,
                bu_m,
                bu_s,
                bw_m,
                iu_m,
                iu_s,
                ret,
                iw_m,
                iu_m - bu_m,
                pt,
                pmc,
                c1k,
                d1k,
                ratio,
                omax,
            )
        else:
            logger.info(
                "%-16s | IOR: %5.2f±%4.2f (Ret %5.1f%%) / %5.2f±%4.2f | +%d/-%d (%.2f:1) | |<P,R>|=%.2e",
                ck,
                iu_m,
                iu_s,
                ret,
                iw_m,
                iw_s,
                c1k,
                d1k,
                ratio,
                omax,
            )
        summary_rows.append(row)

    result = {
        "dataset": args.dataset,
        "K": max_iter,
        "L": mem_layers,
        "dlib_stats": {
            "total_clips": tot_clips_all,
            "frame_det_rate": frame_det_rate,
            "clip_det_rate": clip_det_rate,
            "avg_bg_area_pct": avg_bg_pct,
            "avg_eyes_area_pct": avg_eyes_pct,
            "avg_mouth_area_pct": avg_mouth_pct,
        },
        "conditions": summary_rows,
    }
    json_path = os.path.join(args.output_dir, f"robustness_{args.dataset}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved JSON summary to %s", json_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["dfew", "ferv39k", "mafw11"])
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--split_dir", type=str, required=True)
    parser.add_argument("--ior_ckpt_dir", type=str, required=True)
    parser.add_argument("--ior_ckpt_pattern", type=str, default="")
    parser.add_argument("--base_ckpt_dir", type=str, default="")
    parser.add_argument("--base_ckpt_pattern", type=str, default="")
    parser.add_argument("--dlib_predictor", type=str, required=True)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--mem_refiner_layers", type=int, default=None)
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--conditions", type=str, default="all", choices=["all", "dlib_only"])
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./outputs/robustness")
    return parser.parse_args()


def main():
    t0 = time.time()
    args = parse_args()
    set_seed(args.seed)
    logger = setup_logger(args.output_dir, name=f"robustness_{args.dataset}")
    evaluate_robustness(args, logger)
    logger.info("Finished in %.2f mins.", (time.time() - t0) / 60.0)


if __name__ == "__main__":
    main()
