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

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
for _p in (CURRENT_DIR, PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from datasets.dfew import DFEW_CLASS_NAMES, DFEWDataset
from datasets.ferv39k import FERV39KDataset
from datasets.mafw import MAFWDataset
from datasets.mafw_43 import MAFW_43_LABELS, MAFWDataset43
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

FERV_CLASS_NAMES = ["Happy", "Sad", "Neutral", "Angry", "Surprise", "Disgust", "Fear"]
MAFW11_CLASS_NAMES = [
    "Anger",
    "Anxiety",
    "Contempt",
    "Disappointment",
    "Disgust",
    "Fear",
    "Happiness",
    "Helplessness",
    "Neutral",
    "Sadness",
    "Surprise",
]

DATASET_SPECS = {
    "dfew": {
        "K": 5,
        "L": 1,
        "num_classes": 7,
        "folds": [1, 2, 3, 4, 5],
        "class_names": DFEW_CLASS_NAMES,
        "plot_indices": [5, 6, 1, 0],
    },
    "ferv39k": {
        "K": 6,
        "L": 1,
        "num_classes": 7,
        "folds": [1],
        "class_names": FERV_CLASS_NAMES,
        "plot_indices": [4, 5, 1, 2],
    },
    "mafw11": {
        "K": 4,
        "L": 1,
        "num_classes": 11,
        "folds": [1, 2, 3, 4, 5],
        "class_names": MAFW11_CLASS_NAMES,
        "plot_indices": [4, 5, 7, 3],
    },
    "mafw43": {
        "K": 4,
        "L": 1,
        "num_classes": 43,
        "folds": [1, 2, 3, 4, 5],
        "class_names": MAFW_43_LABELS,
        "plot_indices": [],
    },
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


def mean_std_sample(values, axis=0):
    arr = np.asarray(values, dtype=np.float64)
    n = arr.shape[axis] if arr.ndim > 0 else 1
    ddof = 1 if n > 1 else 0
    return np.mean(arr, axis=axis), np.std(arr, axis=axis, ddof=ddof)


def compute_paired_stats(a, b):
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if len(a_arr) != len(b_arr) or len(a_arr) < 2:
        return {"t": float("nan"), "p_t": float("nan"), "w": float("nan"), "p_w": float("nan")}
    t_val, p_t = stats.ttest_rel(a_arr, b_arr)
    try:
        w_val, p_w = stats.wilcoxon(a_arr - b_arr)
    except Exception:
        w_val, p_w = float("nan"), float("nan")
    return {"t": float(t_val), "p_t": float(p_t), "w": float(w_val), "p_w": float(p_w)}


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


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_l = np.exp(shifted)
    return exp_l / (np.sum(exp_l, axis=-1, keepdims=True) + 1e-12)


def shannon_entropy_np(probs: np.ndarray) -> np.ndarray:
    return -np.sum(probs * np.log(probs + 1e-12), axis=-1)


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
    if dataset_name == "mafw43":
        return MAFWDataset43(
            frames_root=data_root,
            txt_root_base=split_dir,
            fold=fold,
            mode="test",
            num_frames=num_frames,
            is_train=False,
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_baseline_model(num_classes: int, num_frames: int = 16):
    if BaselineModel is None:
        raise ImportError("Baseline model definition could not be imported.")
    return BaselineModel(
        img_size=224,
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


def inspect_prototype_geometry(ckpt_paths, num_classes: int):
    signed_means, abs_means, abs_maxes, angles, scales = [], [], [], [], []
    for path in ckpt_paths:
        sd = load_clean_state_dict(path)
        if "prototypes" not in sd:
            continue
        pn = F.normalize(sd["prototypes"].float(), p=2, dim=-1)
        gram = torch.matmul(pn, pn.t())
        mask = ~torch.eye(gram.shape[0], dtype=torch.bool)
        off = gram[mask].numpy()
        signed_means.append(float(np.mean(off)))
        abs_means.append(float(np.mean(np.abs(off))))
        abs_maxes.append(float(np.max(np.abs(off))))
        angles.append(float(np.mean(np.degrees(np.arccos(np.clip(off, -1.0, 1.0))))))
        if "logit_scale" in sd:
            scales.append(float(torch.exp(sd["logit_scale"].float()).item()))
    if not abs_means:
        return None
    abs_m, abs_s = mean_std_sample(abs_means)
    ang_m, ang_s = mean_std_sample(angles)
    return {
        "mean_signed_cos": float(np.mean(signed_means)),
        "mean_abs_cos": float(abs_m),
        "std_abs_cos": float(abs_s),
        "max_abs_cos": float(np.max(abs_maxes)),
        "mean_angle_deg": float(ang_m),
        "std_angle_deg": float(ang_s),
        "mean_logit_scale": float(np.mean(scales)) if scales else float("nan"),
        "num_classes": num_classes,
    }


@torch.no_grad()
def evaluate_baseline_fold(model, loader, num_classes: int, device: torch.device, desc: str):
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


@torch.no_grad()
def forward_ior_trace(model, x):
    x_cls_normed, patches_normed = model._encode_backbone(x)
    batch_size = x_cls_normed.shape[0]

    tgt_init = model.norm_q_gen(model.initial_queries.expand(batch_size, -1, -1))
    mem_init = model.norm_kv_gen(patches_normed)
    h_init = tgt_init
    for layer in model.query_generator_layers:
        h_init = layer(tgt=h_init, memory=mem_init)
    h_init = model.query_generator_norm(h_init)

    target_norm_fp32 = F.normalize(model.prototypes.float(), p=2, dim=-1)
    scale_fp32 = model.logit_scale.exp().float()
    h0_logits = (
        torch.sum(
            F.normalize(h_init.float(), p=2, dim=-1) * target_norm_fp32.unsqueeze(0),
            dim=-1,
        )
        * scale_fp32
    )

    target_norm = F.normalize(model.prototypes, p=2, dim=-1)
    h_glo, h_loc = h_init, h_init
    m_glo, m_loc = x_cls_normed.unsqueeze(1), patches_normed
    v_m_loc = torch.zeros_like(m_loc)
    v_m_glo = torch.zeros_like(m_glo)
    v_h_glo = torch.zeros_like(h_glo)
    v_h_loc = torch.zeros_like(h_loc)

    step_logits, active_feats, ortho_local, ortho_global = [], [], [], []
    for k in range(model.max_iterations):
        p_glo, r_glo = model._orthogonal_decompose(h_glo, target_norm)
        p_loc, r_loc = model._orthogonal_decompose(h_loc, target_norm)

        w32 = target_norm_fp32.unsqueeze(0)
        for state_fp32, collector in ((h_loc.float(), ortho_local), (h_glo.float(), ortho_global)):
            p32 = torch.sum(state_fp32 * w32, dim=-1, keepdim=True) * w32
            r32 = state_fp32 - p32
            cos_o = torch.abs(
                torch.sum(p32 * r32, dim=-1)
                / (
                    torch.norm(p32, p=2, dim=-1) * torch.norm(r32, p=2, dim=-1)
                    + 1e-12
                )
            ).mean().item()
            collector.append(cos_o)

        q_glo = p_glo + model.global_decoder(
            tgt=model.norm_q_L1_global(r_glo),
            memory=model.norm_kv_L1_global(m_glo),
        )
        q_loc = p_loc + model.local_decoder(
            tgt=model.norm_q_L2_local(r_loc),
            memory=model.norm_kv_L2_local(m_loc),
        )

        q_comb = 0.5 * (
            F.normalize(q_glo.float(), p=2, dim=-1)
            + F.normalize(q_loc.float(), p=2, dim=-1)
        )
        active_feats.append(q_comb.detach().cpu())

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

    return {
        "h0_logits": h0_logits.detach().cpu(),
        "step_logits": step_logits,
        "active_feats": active_feats,
        "ortho_local": ortho_local,
        "ortho_global": ortho_global,
        "prototypes_norm": target_norm_fp32.detach().cpu(),
    }


def evaluate_ior_fold(model, loader, max_iterations: int, fold: int, device: torch.device, desc: str):
    model.eval()
    f_targets, f_h0, f_ol, f_og = [], [], [], []
    f_step_logits = [[] for _ in range(max_iterations)]
    f_feat_drift = [[] for _ in range(max(0, max_iterations - 1))]
    f_gt_align = [[] for _ in range(max_iterations)]
    use_amp = device.type == "cuda"

    for videos, labels in tqdm(loader, desc=desc, leave=False):
        if videos is None or videos.nelement() == 0:
            continue
        videos = videos.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = forward_ior_trace(model, videos)

        batch_size = labels.shape[0]
        f_targets.append(labels.numpy())
        f_h0.append(out["h0_logits"].numpy())
        f_ol.append(out["ortho_local"])
        f_og.append(out["ortho_global"])
        protos = out["prototypes_norm"]

        for k in range(max_iterations):
            f_step_logits[k].append(out["step_logits"][k].numpy())
            qk = F.normalize(out["active_feats"][k], p=2, dim=-1)
            for b_idx in range(batch_size):
                y_c = int(labels[b_idx].item())
                f_gt_align[k].append(torch.sum(qk[b_idx, y_c] * protos[y_c]).item())
            if k > 0:
                q_prev = out["active_feats"][k - 1]
                q_curr = out["active_feats"][k]
                drift = (
                    torch.norm(q_curr - q_prev, p=2, dim=-1)
                    / (torch.norm(q_prev, p=2, dim=-1) + 1e-8)
                ).mean(dim=-1).numpy()
                f_feat_drift[k - 1].append(drift)

    return {
        "fold": fold,
        "targets": np.concatenate(f_targets),
        "h0_logits": np.concatenate(f_h0),
        "step_logits": [np.concatenate(f_step_logits[k]) for k in range(max_iterations)],
        "feat_drift": [float(np.mean(np.concatenate(f_feat_drift[k]))) for k in range(max(0, max_iterations - 1))],
        "gt_align": [float(np.mean(f_gt_align[k])) for k in range(max_iterations)],
        "ortho_local": np.mean(np.array(f_ol, dtype=np.float64), axis=0).tolist(),
        "ortho_global": np.mean(np.array(f_og, dtype=np.float64), axis=0).tolist(),
    }


def save_dynamics_figure(summary: dict, dataset_name: str, save_path: str):
    if not HAS_MPL:
        return
    spec = DATASET_SPECS[dataset_name]
    max_iter = summary["K"]
    steps = np.arange(1, max_iter + 1)

    um = np.array(summary["step_uar_mean"])
    us = np.array(summary["step_uar_std"])
    wm = np.array(summary["step_war_mean"])
    em = np.array(summary["step_entropy_mean"])
    es = np.array(summary["step_entropy_std"])
    flips = np.array(summary["flip_rate_mean"])
    cls_mat = np.array(summary["step_per_class_mean"])

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 10.0
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.2, 4.4), dpi=300)

    ax1.plot(steps, um, marker="s", markersize=6.5, lw=2.2, color="#d62728")
    ax1.fill_between(steps, um - us, um + us, color="#d62728", alpha=0.15)
    ax1_r = ax1.twinx()
    ax1_r.plot(steps, wm, marker="o", markersize=6.0, lw=2.0, ls="--", color="#1f77b4")
    ax1.set_xlabel("Iteration Step ($k$)")
    ax1.set_ylabel("UAR (%)", color="#d62728")
    ax1_r.set_ylabel("WAR (%)", color="#1f77b4")
    ax1.set_xticks(steps)
    ax1.set_xticklabels([f"$y_{k}$" for k in steps])
    ax1.grid(True, ls="--", alpha=0.35)
    ax1.set_title("(a) Step-wise UAR and WAR Progression")

    colors = ["#e377c2", "#9467bd", "#2ca02c", "#17becf"]
    markers = ["*", "D", "^", "o"]
    for idx_i, cls_id in enumerate(spec["plot_indices"]):
        gains = cls_mat[:, cls_id] - cls_mat[0, cls_id]
        ax2.plot(
            steps,
            gains,
            marker=markers[idx_i],
            markersize=6.5,
            lw=2.0,
            color=colors[idx_i],
            label=f"{spec['class_names'][cls_id]} ({gains[-1]:+.2f}%)",
        )
    ax2.axhline(0.0, color="gray", ls=":", lw=1.0)
    ax2.set_xlabel("Iteration Step ($k$)")
    ax2.set_ylabel("Recall Gain over $y_1$ (%)")
    ax2.set_xticks(steps)
    ax2.set_xticklabels([f"$y_{k}$" for k in steps])
    if spec["plot_indices"]:
        ax2.legend(loc="best", fontsize=8.5)
    ax2.grid(True, ls="--", alpha=0.35)
    ax2.set_title("(b) Minority and Subtle Emotion Recall Gains")

    ax3.plot(steps, em, marker="^", markersize=6.5, lw=2.2, color="#ff7f0e")
    ax3.fill_between(steps, em - es, em + es, color="#ff7f0e", alpha=0.15)
    ax3_r = ax3.twinx()
    ax3_r.plot(steps[1:], flips, marker="D", markersize=6.0, lw=2.0, ls="--", color="#2ca02c")
    ax3.set_xlabel("Iteration Step ($k$)")
    ax3.set_ylabel("Shannon Entropy $H(y_k)$", color="#ff7f0e")
    ax3_r.set_ylabel("Prediction Flip Rate (%)", color="#2ca02c")
    ax3.set_xticks(steps)
    ax3.set_xticklabels([f"$y_{k}$" for k in steps])
    ax3.grid(True, ls="--", alpha=0.35)
    ax3.set_title("(c) Predictive Entropy and Decision Stability")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["dfew", "ferv39k", "mafw11", "mafw43"])
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--split_dir", type=str, required=True)
    parser.add_argument("--ior_ckpt_dir", type=str, required=True)
    parser.add_argument("--ior_ckpt_pattern", type=str, default="")
    parser.add_argument("--base_ckpt_dir", type=str, default="")
    parser.add_argument("--base_ckpt_pattern", type=str, default="")
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--mem_refiner_layers", type=int, default=None)
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_fig7", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./outputs/benchmark")
    return parser.parse_args()


def main():
    t0 = time.time()
    args = parse_args()
    set_seed(args.seed)
    logger = setup_logger(args.output_dir, name=f"benchmark_{args.dataset}")

    spec = DATASET_SPECS[args.dataset]
    max_iter = args.max_iterations if args.max_iterations is not None else spec["K"]
    mem_layers = args.mem_refiner_layers if args.mem_refiner_layers is not None else spec["L"]
    num_classes = spec["num_classes"]
    folds = args.folds if args.folds else spec["folds"]
    class_names = spec["class_names"]
    cls_labels = list(range(num_classes))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ior_ckpt_paths = []
    ior_recs = []
    base_recs = []

    for fold in folds:
        ds = build_test_dataset(args.dataset, args.data_root, args.split_dir, fold, args.num_frames)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn_safe,
            pin_memory=True,
        )

        if args.base_ckpt_dir:
            base_ckpt = resolve_ckpt_path(args.base_ckpt_dir, args.base_ckpt_pattern, fold)
            base_model = build_baseline_model(num_classes, args.num_frames).to(device)
            base_model.load_state_dict(load_clean_state_dict(base_ckpt), strict=True)
            base_recs.append(
                evaluate_baseline_fold(base_model, loader, num_classes, device, desc=f"{args.dataset.upper()} Base F{fold}")
            )
            del base_model
            torch.cuda.empty_cache()

        ior_ckpt = resolve_ckpt_path(args.ior_ckpt_dir, args.ior_ckpt_pattern, fold)
        ior_ckpt_paths.append(ior_ckpt)
        ior_model = build_ior_model(num_classes, max_iter, mem_layers, args.num_frames).to(device)
        ior_model.load_state_dict(load_clean_state_dict(ior_ckpt), strict=True)
        ior_recs.append(
            evaluate_ior_fold(ior_model, loader, max_iter, fold, device, desc=f"{args.dataset.upper()} IOR F{fold}")
        )
        del ior_model
        torch.cuda.empty_cache()

    n_f = len(ior_recs)
    h0_uar = np.zeros(n_f)
    h0_war = np.zeros(n_f)
    h0_ent = np.zeros(n_f)
    step_uar = np.zeros((n_f, max_iter))
    step_war = np.zeros((n_f, max_iter))
    step_ent = np.zeros((n_f, max_iter))
    step_cls = np.zeros((n_f, max_iter, num_classes))

    prob_l2 = np.zeros((n_f, max(0, max_iter - 1)))
    prob_tv = np.zeros((n_f, max(0, max_iter - 1)))
    prob_kl = np.zeros((n_f, max(0, max_iter - 1)))
    prob_jsd = np.zeros((n_f, max(0, max_iter - 1)))
    flip_rate = np.zeros((n_f, max(0, max_iter - 1)))
    logit_rel = np.zeros((n_f, max(0, max_iter - 1)))

    pooled_y = []
    pooled_steps = [[] for _ in range(max_iter)]

    for fi, rec in enumerate(ior_recs):
        yt = rec["targets"]
        pooled_y.append(yt)
        h0_probs = softmax_np(rec["h0_logits"])
        h0_preds = np.argmax(rec["h0_logits"], axis=-1)
        h0_uar[fi] = recall_score(yt, h0_preds, labels=cls_labels, average="macro", zero_division=0) * 100.0
        h0_war[fi] = accuracy_score(yt, h0_preds) * 100.0
        h0_ent[fi] = float(np.mean(shannon_entropy_np(h0_probs)))

        probs_list = [softmax_np(rec["step_logits"][k]) for k in range(max_iter)]
        preds_list = [np.argmax(rec["step_logits"][k], axis=-1) for k in range(max_iter)]

        for k in range(max_iter):
            pooled_steps[k].append(preds_list[k])
            step_uar[fi, k] = recall_score(yt, preds_list[k], labels=cls_labels, average="macro", zero_division=0) * 100.0
            step_war[fi, k] = accuracy_score(yt, preds_list[k]) * 100.0
            step_ent[fi, k] = float(np.mean(shannon_entropy_np(probs_list[k])))
            step_cls[fi, k] = recall_score(yt, preds_list[k], labels=cls_labels, average=None, zero_division=0) * 100.0
            if k > 0:
                p_cur = probs_list[k]
                p_prev = probs_list[k - 1]
                prob_l2[fi, k - 1] = float(np.mean(np.linalg.norm(p_cur - p_prev, axis=-1)))
                prob_tv[fi, k - 1] = float(0.5 * np.mean(np.sum(np.abs(p_cur - p_prev), axis=-1)))
                prob_kl[fi, k - 1] = float(np.mean(np.sum(p_cur * np.log((p_cur + 1e-12) / (p_prev + 1e-12)), axis=-1)))
                m_prob = 0.5 * (p_cur + p_prev)
                jsd = 0.5 * np.sum(p_cur * np.log((p_cur + 1e-12) / (m_prob + 1e-12)), axis=-1) + 0.5 * np.sum(
                    p_prev * np.log((p_prev + 1e-12) / (m_prob + 1e-12)), axis=-1
                )
                prob_jsd[fi, k - 1] = float(np.mean(jsd))
                flip_rate[fi, k - 1] = float(np.mean(preds_list[k] != preds_list[k - 1]) * 100.0)
                lg_cur = rec["step_logits"][k]
                lg_prev = rec["step_logits"][k - 1]
                logit_rel[fi, k - 1] = float(
                    np.mean(
                        np.linalg.norm(lg_cur - lg_prev, axis=-1)
                        / (np.linalg.norm(lg_prev, axis=-1) + 1e-8)
                    )
                )

    y_all = np.concatenate(pooled_y)
    p_steps = [np.concatenate(pooled_steps[k]) for k in range(max_iter)]
    p1_all, pk_all = p_steps[0], p_steps[-1]

    c_1k = int(np.sum((p1_all != y_all) & (pk_all == y_all)))
    d_1k = int(np.sum((p1_all == y_all) & (pk_all != y_all)))
    ratio_1k = float(c_1k / max(d_1k, 1))
    mcnemar_1k = compute_mcnemar(y_all, pk_all, p1_all)

    step_uar_m, step_uar_s = mean_std_sample(step_uar, axis=0)
    step_war_m, step_war_s = mean_std_sample(step_war, axis=0)
    step_ent_m, step_ent_s = mean_std_sample(step_ent, axis=0)
    step_cls_m, step_cls_s = mean_std_sample(step_cls, axis=0)
    h0_uar_m, h0_uar_s = mean_std_sample(h0_uar, axis=0)
    h0_war_m, h0_war_s = mean_std_sample(h0_war, axis=0)
    h0_ent_m, h0_ent_s = mean_std_sample(h0_ent, axis=0)

    flip_m, flip_s = mean_std_sample(flip_rate, axis=0)
    pl2_m, _ = mean_std_sample(prob_l2, axis=0)
    ptv_m, _ = mean_std_sample(prob_tv, axis=0)
    kl_m, _ = mean_std_sample(prob_kl, axis=0)
    jsd_m, _ = mean_std_sample(prob_jsd, axis=0)
    lgl2_m, _ = mean_std_sample(logit_rel, axis=0)

    ortho_l_m = np.mean(np.array([r["ortho_local"] for r in ior_recs], dtype=np.float64), axis=0)
    ortho_g_m = np.mean(np.array([r["ortho_global"] for r in ior_recs], dtype=np.float64), axis=0)
    feat_drift_m = np.mean(np.array([r["feat_drift"] for r in ior_recs], dtype=np.float64), axis=0)
    gt_align_m = np.mean(np.array([r["gt_align"] for r in ior_recs], dtype=np.float64), axis=0)
    proto_geom = inspect_prototype_geometry(ior_ckpt_paths, num_classes)

    base_summary = None
    if base_recs:
        b_u = [r["uar"] for r in base_recs]
        b_w = [r["war"] for r in base_recs]
        b_um, b_us = mean_std_sample(b_u)
        b_wm, b_ws = mean_std_sample(b_w)
        b_preds_all = np.concatenate([r["preds"] for r in base_recs])
        base_summary = {
            "uar_mean": float(b_um),
            "uar_std": float(b_us),
            "war_mean": float(b_wm),
            "war_std": float(b_ws),
            "paired_uar": compute_paired_stats(step_uar[:, -1], b_u),
            "paired_war": compute_paired_stats(step_war[:, -1], b_w),
            "mcnemar": compute_mcnemar(y_all, pk_all, b_preds_all),
        }

    summary = {
        "dataset": args.dataset,
        "K": max_iter,
        "L": mem_layers,
        "num_classes": num_classes,
        "num_folds": n_f,
        "total_samples": int(len(y_all)),
        "ior_uar_mean": float(step_uar_m[-1]),
        "ior_uar_std": float(step_uar_s[-1]),
        "ior_war_mean": float(step_war_m[-1]),
        "ior_war_std": float(step_war_s[-1]),
        "ior_pooled_uar": float(recall_score(y_all, pk_all, labels=cls_labels, average="macro", zero_division=0) * 100.0),
        "ior_pooled_war": float(accuracy_score(y_all, pk_all) * 100.0),
        "h0_uar_mean": float(h0_uar_m),
        "h0_uar_std": float(h0_uar_s),
        "h0_war_mean": float(h0_war_m),
        "h0_war_std": float(h0_war_s),
        "h0_entropy_mean": float(h0_ent_m),
        "h0_entropy_std": float(h0_ent_s),
        "step_uar_mean": step_uar_m.tolist(),
        "step_uar_std": step_uar_s.tolist(),
        "step_war_mean": step_war_m.tolist(),
        "step_war_std": step_war_s.tolist(),
        "step_entropy_mean": step_ent_m.tolist(),
        "step_entropy_std": step_ent_s.tolist(),
        "step_per_class_mean": step_cls_m.tolist(),
        "step_per_class_std": step_cls_s.tolist(),
        "flip_rate_mean": flip_m.tolist(),
        "flip_rate_std": flip_s.tolist(),
        "prob_l2_mean": pl2_m.tolist(),
        "prob_tv_mean": ptv_m.tolist(),
        "prob_kl_mean": kl_m.tolist(),
        "prob_jsd_mean": jsd_m.tolist(),
        "logit_rel_shift_mean": lgl2_m.tolist(),
        "feat_drift_mean": feat_drift_m.tolist(),
        "gt_align_mean": gt_align_m.tolist(),
        "ortho_local_mean": ortho_l_m.tolist(),
        "ortho_global_mean": ortho_g_m.tolist(),
        "corrected_1k": c_1k,
        "erroneous_1k": d_1k,
        "ratio_1k": ratio_1k,
        "mcnemar_1k": mcnemar_1k,
        "paired_1k_uar": compute_paired_stats(step_uar[:, -1], step_uar[:, 0]),
        "paired_1k_war": compute_paired_stats(step_war[:, -1], step_war[:, 0]),
        "prototype_geometry": proto_geom,
        "baseline": base_summary,
    }

    logger.info(
        "%s (K=%d, L=%d, N=%d, Folds=%d) | Final UAR=%.2f±%.2f%% | Final WAR=%.2f±%.2f%%",
        args.dataset.upper(),
        max_iter,
        mem_layers,
        len(y_all),
        n_f,
        summary["ior_uar_mean"],
        summary["ior_uar_std"],
        summary["ior_war_mean"],
        summary["ior_war_std"],
    )
    if base_summary:
        logger.info(
            "Baseline: UAR=%.2f±%.2f%% | WAR=%.2f±%.2f%% | dUAR=%+.2f%% (p_t=%.2e, McNemar p=%.2e)",
            base_summary["uar_mean"],
            base_summary["uar_std"],
            base_summary["war_mean"],
            base_summary["war_std"],
            summary["ior_uar_mean"] - base_summary["uar_mean"],
            base_summary["paired_uar"]["p_t"],
            base_summary["mcnemar"]["p"],
        )
    logger.info(
        "y0: UAR=%.2f±%.2f%% | WAR=%.2f±%.2f%% | Entropy=%.4f",
        summary["h0_uar_mean"],
        summary["h0_uar_std"],
        summary["h0_war_mean"],
        summary["h0_war_std"],
        summary["h0_entropy_mean"],
    )
    for k in range(max_iter):
        logger.info(
            "y%d: UAR=%.2f±%.2f%% | WAR=%.2f±%.2f%% | Ent=%.4f | |<P,R>|_loc=%.2e | |<P,R>|_glo=%.2e",
            k + 1,
            step_uar_m[k],
            step_uar_s[k],
            step_war_m[k],
            step_war_s[k],
            step_ent_m[k],
            ortho_l_m[k],
            ortho_g_m[k],
        )
    logger.info(
        "y1->y%d: +%d / -%d (Ratio=%.2f:1, McNemar p=%.2e) | FlipRate=%s | Prob-L2=%s | JSD=%s",
        max_iter,
        c_1k,
        d_1k,
        ratio_1k,
        mcnemar_1k["p"],
        [round(x, 3) for x in flip_m.tolist()],
        [round(x, 5) for x in pl2_m.tolist()],
        [round(x, 5) for x in jsd_m.tolist()],
    )
    if proto_geom:
        logger.info(
            "O-ETF Prototypes: mean|cos|=%.4f±%.4f | max|cos|=%.4f | angle=%.2f±%.2f deg | exp(tau)=%.2f",
            proto_geom["mean_abs_cos"],
            proto_geom["std_abs_cos"],
            proto_geom["max_abs_cos"],
            proto_geom["mean_angle_deg"],
            proto_geom["std_angle_deg"],
            proto_geom["mean_logit_scale"],
        )
    if num_classes <= 11:
        for ci, cname in enumerate(class_names):
            logger.info("  %-14s: %.2f±%.2f%%", cname, step_cls_m[-1, ci], step_cls_s[-1, ci])

    json_path = os.path.join(args.output_dir, f"benchmark_{args.dataset}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved JSON summary to %s (elapsed %.2f mins).", json_path, (time.time() - t0) / 60.0)

    if args.save_fig7 and num_classes <= 11:
        fig_path = os.path.join(args.output_dir, f"fig7_dynamics_{args.dataset}.pdf")
        save_dynamics_figure(summary, args.dataset, fig_path)
        logger.info("Saved Fig. 7 plot to %s", fig_path)


if __name__ == "__main__":
    main()
