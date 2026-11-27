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
from torch.utils.data import DataLoader, Dataset
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

CANONICAL_7_NAMES = ["Happy", "Sad", "Neutral", "Anger", "Surprise", "Disgust", "Fear"]
CANONICAL_TO_MAFW_IDX = [6, 9, 8, 0, 10, 4, 5]
MAFW_TO_CANONICAL_MAP = {
    mafw_id: canon_id for canon_id, mafw_id in enumerate(CANONICAL_TO_MAFW_IDX)
}


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logger(save_dir: str):
    logger = logging.getLogger("eval_cross_dataset")
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.INFO)
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "cross_dataset_zeroshot.log")
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


def paired_ttest(a, b):
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if len(a_arr) != len(b_arr) or len(a_arr) < 2:
        return {"t": float("nan"), "p_t": float("nan")}
    t_val, p_t = stats.ttest_rel(a_arr, b_arr)
    return {"t": float(t_val), "p_t": float(p_t)}


class MAFW7FilteredDataset(Dataset):
    def __init__(self, frames_root: str, txt_root_base: str, fold: int, num_frames: int = 16):
        self.base_ds = MAFWDataset(
            frames_root=frames_root,
            txt_root_base=txt_root_base,
            fold=fold,
            mode="test",
            num_frames=num_frames,
            is_train=False,
        )
        self.filtered_indices = [
            (idx, MAFW_TO_CANONICAL_MAP[raw_lbl])
            for idx, (_, raw_lbl) in enumerate(self.base_ds.data_list)
            if raw_lbl in MAFW_TO_CANONICAL_MAP
        ]

    def __len__(self) -> int:
        return len(self.filtered_indices)

    def __getitem__(self, idx: int):
        raw_idx, canon_label = self.filtered_indices[idx]
        video, _ = self.base_ds[raw_idx]
        return video, canon_label


def load_clean_state_dict(ckpt_path: str):
    obj = torch.load(ckpt_path, map_location="cpu")
    sd = obj
    if isinstance(obj, dict):
        if "model" in obj and isinstance(obj["model"], dict):
            sd = obj["model"]
        elif "state_dict" in obj and isinstance(obj["state_dict"], dict):
            sd = obj["state_dict"]
    return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}


def resolve_ckpt(root: str, pattern: str, fold: int):
    if pattern:
        path = os.path.join(root, pattern.format(fold=fold))
        if os.path.exists(path):
            return path
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if os.path.isfile(root):
        return root
    fold_dir = os.path.join(root, f"fold_{fold}")
    if os.path.isdir(fold_dir):
        pths = sorted(glob.glob(os.path.join(fold_dir, "*.pth")))
        if len(pths) == 1:
            return pths[0]
        target = os.path.join(fold_dir, "target_reached_model.pth")
        if os.path.exists(target):
            return target
        if pths:
            return pths[0]
    raise FileNotFoundError(f"Unable to resolve checkpoint in {root} for fold {fold}")


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


def build_ior_model(num_classes: int, max_iterations: int, mem_refiner_layers: int = 1, num_frames: int = 16):
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
def evaluate_baseline_loader(model, loader, device, slice_indices=None, desc="Baseline"):
    model.eval()
    all_targets, all_preds = [], []
    use_amp = device.type == "cuda"
    for videos, labels in tqdm(loader, desc=desc, leave=False):
        if videos.nelement() == 0:
            continue
        videos = videos.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(videos)
        if slice_indices is not None:
            logits = logits[:, slice_indices]
        all_preds.append(torch.argmax(logits, dim=-1).cpu().numpy())
        all_targets.append(labels.numpy())
    yt = np.concatenate(all_targets)
    yp = np.concatenate(all_preds)
    uar = float(recall_score(yt, yp, labels=list(range(7)), average="macro", zero_division=0) * 100.0)
    war = float(accuracy_score(yt, yp) * 100.0)
    return {"targets": yt, "preds": yp, "uar": uar, "war": war}


@torch.no_grad()
def forward_ior_cross(model, x, slice_indices=None):
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
        if slice_indices is not None:
            cur_logits = cur_logits[:, slice_indices]
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


@torch.no_grad()
def evaluate_ior_loader(model, loader, max_iterations, device, slice_indices=None, desc="IOR-Net"):
    model.eval()
    all_targets, all_ortho = [], []
    step_lgs = [[] for _ in range(max_iterations)]
    use_amp = device.type == "cuda"
    for videos, labels in tqdm(loader, desc=desc, leave=False):
        if videos.nelement() == 0:
            continue
        videos = videos.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            lgs, ortho_errs = forward_ior_cross(model, videos, slice_indices=slice_indices)
        all_targets.append(labels.numpy())
        all_ortho.append(ortho_errs)
        for k in range(max_iterations):
            step_lgs[k].append(lgs[k].numpy())

    yt = np.concatenate(all_targets)
    step_preds = [np.argmax(np.concatenate(step_lgs[k]), axis=-1) for k in range(max_iterations)]
    step_uars = [
        float(recall_score(yt, step_preds[k], labels=list(range(7)), average="macro", zero_division=0) * 100.0)
        for k in range(max_iterations)
    ]
    step_wars = [float(accuracy_score(yt, step_preds[k]) * 100.0) for k in range(max_iterations)]
    return {
        "targets": yt,
        "step_preds": step_preds,
        "step_uars": step_uars,
        "step_wars": step_wars,
        "uar": step_uars[-1],
        "war": step_wars[-1],
        "ortho_mean": np.mean(np.array(all_ortho), axis=0).tolist(),
    }


def summarize_transfer(pair_name, max_iterations, base_recs, ior_recs, logger):
    iu_list = [r["uar"] for r in ior_recs]
    iw_list = [r["war"] for r in ior_recs]
    iu_m, iu_s = mean_std_sample(iu_list)
    iw_m, iw_s = mean_std_sample(iw_list)

    y_all = np.concatenate([r["targets"] for r in ior_recs])
    p1_all = np.concatenate([r["step_preds"][0] for r in ior_recs])
    pk_all = np.concatenate([r["step_preds"][-1] for r in ior_recs])
    c1k = int(np.sum((p1_all != y_all) & (pk_all == y_all)))
    d1k = int(np.sum((p1_all == y_all) & (pk_all != y_all)))
    ratio = float(c1k / max(d1k, 1))
    max_ortho = float(
        np.max(np.mean(np.array([r["ortho_mean"] for r in ior_recs]), axis=0))
    )

    summary = {
        "pair": pair_name,
        "K": max_iterations,
        "ior_uar_mean": iu_m,
        "ior_uar_std": iu_s,
        "ior_war_mean": iw_m,
        "ior_war_std": iw_s,
        "corrected_1k": c1k,
        "erroneous_1k": d1k,
        "ratio_1k": ratio,
        "max_ortho_err": max_ortho,
    }

    if base_recs:
        bu_list = [r["uar"] for r in base_recs]
        bw_list = [r["war"] for r in base_recs]
        bu_m, bu_s = mean_std_sample(bu_list)
        bw_m, bw_s = mean_std_sample(bw_list)
        p_uar = paired_ttest(iu_list, bu_list)["p_t"]
        p_war = paired_ttest(iw_list, bw_list)["p_t"]
        summary.update(
            {
                "base_uar_mean": bu_m,
                "base_uar_std": bu_s,
                "base_war_mean": bw_m,
                "base_war_std": bw_s,
                "delta_uar": iu_m - bu_m,
                "delta_war": iw_m - bw_m,
                "p_uar": p_uar,
                "p_war": p_war,
            }
        )
        logger.info(
            "%-18s (K=%d) | Base: %5.2f±%4.2f / %5.2f±%4.2f | IOR: %5.2f±%4.2f / %5.2f±%4.2f | dUAR=%+6.2f%% (p=%.2e) | dWAR=%+6.2f%% (p=%.2e) | +%d/-%d (%.2f:1) | |<P,R>|=%.2e",
            pair_name,
            max_iterations,
            bu_m,
            bu_s,
            bw_m,
            bw_s,
            iu_m,
            iu_s,
            iw_m,
            iw_s,
            iu_m - bu_m,
            p_uar,
            iw_m - bw_m,
            p_war,
            c1k,
            d1k,
            ratio,
            max_ortho,
        )
    else:
        logger.info(
            "%-18s (K=%d) | IOR: %5.2f±%4.2f / %5.2f±%4.2f | +%d/-%d (%.2f:1) | |<P,R>|=%.2e",
            pair_name,
            max_iterations,
            iu_m,
            iu_s,
            iw_m,
            iw_s,
            c1k,
            d1k,
            ratio,
            max_ortho,
        )
    return summary


def make_loader(dataset, batch_size: int, num_workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_safe,
        pin_memory=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dfew_data_root", type=str, required=True)
    parser.add_argument("--dfew_split_dir", type=str, required=True)
    parser.add_argument("--ferv_data_root", type=str, required=True)
    parser.add_argument("--ferv_split_dir", type=str, required=True)
    parser.add_argument("--mafw_data_root", type=str, required=True)
    parser.add_argument("--mafw_split_dir", type=str, required=True)

    parser.add_argument("--dfew_ior_dir", type=str, required=True)
    parser.add_argument("--dfew_ior_pattern", type=str, default="Final_FullTrain_K5_L1_Fold{fold}.pth")
    parser.add_argument("--ferv_ior_ckpt", type=str, required=True)
    parser.add_argument("--mafw_ior_dir", type=str, required=True)
    parser.add_argument("--mafw_ior_pattern", type=str, default="Final_FullTrain_MAFW_K4_L1_Fold{fold}.pth")

    parser.add_argument("--dfew_base_dir", type=str, default="")
    parser.add_argument("--dfew_base_pattern", type=str, default="")
    parser.add_argument("--ferv_base_ckpt", type=str, default="")
    parser.add_argument("--mafw_base_dir", type=str, default="")
    parser.add_argument("--mafw_base_pattern", type=str, default="")

    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./outputs/cross_dataset")
    return parser.parse_args()


def main():
    t0 = time.time()
    args = parse_args()
    set_seed(args.seed)
    logger = setup_logger(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = [1, 2, 3, 4, 5]

    dfew_loaders = {
        f: make_loader(
            DFEWDataset(args.dfew_split_dir, args.dfew_data_root, f, "test", is_train=False, num_frames=args.num_frames),
            args.batch_size,
            args.num_workers,
        )
        for f in folds
    }
    mafw7_loaders = {
        f: make_loader(
            MAFW7FilteredDataset(args.mafw_data_root, args.mafw_split_dir, f, num_frames=args.num_frames),
            args.batch_size,
            args.num_workers,
        )
        for f in folds
    }
    ferv_loader = make_loader(
        FERV39KDataset(args.ferv_split_dir, args.ferv_data_root, split_type="test", is_train=False, num_frames=args.num_frames),
        args.batch_size,
        args.num_workers,
    )

    dfew_b_ferv, dfew_b_mafw = [], []
    dfew_i_ferv, dfew_i_mafw = [], []
    for f in folds:
        if args.dfew_base_dir:
            b_ckpt = resolve_ckpt(args.dfew_base_dir, args.dfew_base_pattern, f)
            b_model = build_baseline_model(7, args.num_frames).to(device)
            b_model.load_state_dict(load_clean_state_dict(b_ckpt), strict=True)
            dfew_b_ferv.append(evaluate_baseline_loader(b_model, ferv_loader, device, desc=f"Base DFEW->FERV F{f}"))
            dfew_b_mafw.append(evaluate_baseline_loader(b_model, mafw7_loaders[f], device, desc=f"Base DFEW->MAFW7 F{f}"))
            del b_model
            torch.cuda.empty_cache()

        i_ckpt = resolve_ckpt(args.dfew_ior_dir, args.dfew_ior_pattern, f)
        i_model = build_ior_model(7, max_iterations=5, mem_refiner_layers=1, num_frames=args.num_frames).to(device)
        i_model.load_state_dict(load_clean_state_dict(i_ckpt), strict=True)
        dfew_i_ferv.append(evaluate_ior_loader(i_model, ferv_loader, 5, device, desc=f"IOR DFEW->FERV F{f}"))
        dfew_i_mafw.append(evaluate_ior_loader(i_model, mafw7_loaders[f], 5, device, desc=f"IOR DFEW->MAFW7 F{f}"))
        del i_model
        torch.cuda.empty_cache()

    ferv_b_dfew, ferv_b_mafw = [], []
    if args.ferv_base_ckpt:
        b_model = build_baseline_model(7, args.num_frames).to(device)
        b_model.load_state_dict(load_clean_state_dict(args.ferv_base_ckpt), strict=True)
        ferv_b_dfew = [evaluate_baseline_loader(b_model, dfew_loaders[f], device, desc=f"Base FERV->DFEW F{f}") for f in folds]
        ferv_b_mafw = [evaluate_baseline_loader(b_model, mafw7_loaders[f], device, desc=f"Base FERV->MAFW7 F{f}") for f in folds]
        del b_model
        torch.cuda.empty_cache()

    i_model = build_ior_model(7, max_iterations=6, mem_refiner_layers=1, num_frames=args.num_frames).to(device)
    i_model.load_state_dict(load_clean_state_dict(args.ferv_ior_ckpt), strict=True)
    ferv_i_dfew = [evaluate_ior_loader(i_model, dfew_loaders[f], 6, device, desc=f"IOR FERV->DFEW F{f}") for f in folds]
    ferv_i_mafw = [evaluate_ior_loader(i_model, mafw7_loaders[f], 6, device, desc=f"IOR FERV->MAFW7 F{f}") for f in folds]
    del i_model
    torch.cuda.empty_cache()

    mafw_b_dfew, mafw_b_ferv = [], []
    mafw_i_dfew, mafw_i_ferv = [], []
    for f in folds:
        if args.mafw_base_dir:
            b_ckpt = resolve_ckpt(args.mafw_base_dir, args.mafw_base_pattern, f)
            b_model = build_baseline_model(11, args.num_frames).to(device)
            b_model.load_state_dict(load_clean_state_dict(b_ckpt), strict=True)
            mafw_b_dfew.append(evaluate_baseline_loader(b_model, dfew_loaders[f], device, slice_indices=CANONICAL_TO_MAFW_IDX, desc=f"Base MAFW->DFEW F{f}"))
            mafw_b_ferv.append(evaluate_baseline_loader(b_model, ferv_loader, device, slice_indices=CANONICAL_TO_MAFW_IDX, desc=f"Base MAFW->FERV F{f}"))
            del b_model
            torch.cuda.empty_cache()

        i_ckpt = resolve_ckpt(args.mafw_ior_dir, args.mafw_ior_pattern, f)
        i_model = build_ior_model(11, max_iterations=4, mem_refiner_layers=1, num_frames=args.num_frames).to(device)
        i_model.load_state_dict(load_clean_state_dict(i_ckpt), strict=True)
        mafw_i_dfew.append(evaluate_ior_loader(i_model, dfew_loaders[f], 4, device, slice_indices=CANONICAL_TO_MAFW_IDX, desc=f"IOR MAFW->DFEW F{f}"))
        mafw_i_ferv.append(evaluate_ior_loader(i_model, ferv_loader, 4, device, slice_indices=CANONICAL_TO_MAFW_IDX, desc=f"IOR MAFW->FERV F{f}"))
        del i_model
        torch.cuda.empty_cache()

    results = [
        summarize_transfer("DFEW -> FERV39k", 5, dfew_b_ferv, dfew_i_ferv, logger),
        summarize_transfer("DFEW -> MAFW-7", 5, dfew_b_mafw, dfew_i_mafw, logger),
        summarize_transfer("FERV39k -> DFEW", 6, ferv_b_dfew, ferv_i_dfew, logger),
        summarize_transfer("FERV39k -> MAFW-7", 6, ferv_b_mafw, ferv_i_mafw, logger),
        summarize_transfer("MAFW -> DFEW", 4, mafw_b_dfew, mafw_i_dfew, logger),
        summarize_transfer("MAFW -> FERV39k", 4, mafw_b_ferv, mafw_i_ferv, logger),
    ]

    json_path = os.path.join(args.output_dir, "cross_dataset_zeroshot.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved JSON summary to %s (elapsed %.2f mins).", json_path, (time.time() - t0) / 60.0)


if __name__ == "__main__":
    main()
