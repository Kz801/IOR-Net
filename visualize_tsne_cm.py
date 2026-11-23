import argparse
import glob
import os
import random
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
for _p in (CURRENT_DIR, PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from datasets.dfew import DFEWDataset
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

STAGES = [
    ("0_Static_Base", "Baseline", False, False),
    ("1_IFM_Only", "+IFM", False, False),
    ("2_IFM_w_ODUM", "+IFM+ODUM", True, False),
    ("3_IFM_w_PGSM", "+IFM+PGSM", False, True),
    ("4_Full_IOR_Net", "FULL", True, True),
]

CLASS_SHORT = ["Hap", "Sad", "Neu", "Ang", "Sur", "Dis", "Fea"]
CLASS_FULL = ["Happy", "Sad", "Neutral", "Anger", "Surprise", "Disgust", "Fear"]

PALETTE = [
    "#E64B35",
    "#3C5488",
    "#8FA0B5",
    "#E69F00",
    "#00A087",
    "#8E44AD",
    "#E7298A",
]

TIP_BLUES = LinearSegmentedColormap.from_list(
    "TIP_Blues",
    ["#F8FAFD", "#DEEBF7", "#9ECAE1", "#4292C6", "#2171B5", "#084594"],
    N=256,
)


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn_safe(batch):
    batch = [x for x in batch if x is not None]
    if not batch:
        return torch.empty(0), torch.empty(0)
    return torch.utils.data.dataloader.default_collate(batch)


def load_clean_state_dict(ckpt_path: str):
    obj = torch.load(ckpt_path, map_location="cpu")
    sd = obj
    if isinstance(obj, dict):
        if "model" in obj and isinstance(obj["model"], dict):
            sd = obj["model"]
        elif "state_dict" in obj and isinstance(obj["state_dict"], dict):
            sd = obj["state_dict"]
    return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}


def resolve_baseline_ckpt(root: str, pattern: str, fold: int):
    if pattern:
        return os.path.join(root, pattern.format(fold=fold))
    fold_dir = os.path.join(root, f"fold_{fold}")
    pths = sorted(glob.glob(os.path.join(fold_dir, "*.pth")))
    if len(pths) == 1:
        return pths[0]
    target = os.path.join(fold_dir, "target_reached_model.pth")
    return target if os.path.exists(target) else pths[0]


def build_stage_model(key: str, use_odu: bool, use_pgs: bool, max_iterations: int, mem_refiner_layers: int, num_frames: int):
    if key == "0_Static_Base":
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
            num_classes=7,
            num_frames=num_frames,
            use_flash_attn=True,
            use_fused_mlp=True,
            use_fused_rmsnorm=True,
        )
    if key == "4_Full_IOR_Net" or build_ablation_net is None:
        return ior_net_small_patch14_224(
            num_frames=num_frames,
            num_classes=7,
            max_iterations=max_iterations,
            isr_mem_refiner_layers=mem_refiner_layers,
            isr_global_decoder_layers=1,
            isr_local_decoder_layers=1,
            isr_query_gen_layers=1,
            isr_query_synth_layers=1,
            isr_global_fuser_layers=1,
        )
    return build_ablation_net(
        num_frames=num_frames,
        num_classes=7,
        max_iterations=max_iterations,
        use_odu=use_odu,
        use_pgs=use_pgs,
        isr_mem_refiner_layers=mem_refiner_layers,
        isr_global_decoder_layers=1,
        isr_local_decoder_layers=1,
        isr_query_gen_layers=1,
        isr_query_synth_layers=1,
        isr_global_fuser_layers=1,
    )


@torch.no_grad()
def forward_ior_prob_q_and_logits(model, x, use_odu: bool, temperature: float):
    x_cls_normed, patches_normed = model._encode_backbone(x)
    batch_size = x_cls_normed.shape[0]

    tgt_init = model.norm_q_gen(model.initial_queries.expand(batch_size, -1, -1))
    mem_init = model.norm_kv_gen(patches_normed)
    h_init = tgt_init
    for layer in model.query_generator_layers:
        h_init = layer(tgt=h_init, memory=mem_init)
    h_init = model.query_generator_norm(h_init)

    target_norm = F.normalize(model.prototypes, p=2, dim=-1)
    h_glo, h_loc = h_init, h_init
    m_glo, m_loc = x_cls_normed.unsqueeze(1), patches_normed
    v_m_loc = torch.zeros_like(m_loc)
    v_m_glo = torch.zeros_like(m_glo)
    v_h_glo = torch.zeros_like(h_glo)
    v_h_loc = torch.zeros_like(h_loc)

    final_logits = None
    final_prob_q = None
    for k in range(model.max_iterations):
        if use_odu:
            p_glo, r_glo = model._orthogonal_decompose(h_glo, target_norm)
            p_loc, r_loc = model._orthogonal_decompose(h_loc, target_norm)
            q_glo = p_glo + model.global_decoder(
                tgt=model.norm_q_L1_global(r_glo),
                memory=model.norm_kv_L1_global(m_glo),
            )
            q_loc = p_loc + model.local_decoder(
                tgt=model.norm_q_L2_local(r_loc),
                memory=model.norm_kv_L2_local(m_loc),
            )
        else:
            q_glo = h_glo + model.global_decoder(
                tgt=model.norm_q_L1_global(h_glo),
                memory=model.norm_kv_L1_global(m_glo),
            )
            q_loc = h_loc + model.local_decoder(
                tgt=model.norm_q_L2_local(h_loc),
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
        logits_k = (
            torch.sigmoid(gate_glo) * logits_glo
            + torch.sigmoid(gate_loc) * logits_loc
        )

        if k == model.max_iterations - 1:
            final_logits = logits_k
            probs = F.softmax(logits_k.float() / temperature, dim=-1)
            q_fused = F.normalize(0.5 * (q_glo.float() + q_loc.float()), p=2, dim=-1)
            final_prob_q = torch.sum(probs.unsqueeze(-1) * q_fused, dim=1)
        else:
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

    return final_logits, final_prob_q


@torch.no_grad()
def evaluate_stage_fold(
    model,
    loader,
    stage_key: str,
    use_odu: bool,
    device: torch.device,
    extract_features: bool,
    temperature: float,
):
    model.eval()
    use_amp = device.type == "cuda"
    all_y, all_p, all_f = [], [], []

    hook_buf = []
    hook_handle = None
    if extract_features and stage_key == "0_Static_Base":
        head_module = getattr(model, "head", getattr(model, "fc", None))
        if head_module is not None:
            def _capture_pre_head(_mod, inputs):
                hook_buf.append(inputs[0].float().detach().cpu())

            hook_handle = head_module.register_forward_pre_hook(_capture_pre_head)

    try:
        for videos, labels in tqdm(loader, desc=stage_key, leave=False):
            if videos is None or videos.nelement() == 0:
                continue
            videos = videos.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                if stage_key == "0_Static_Base":
                    logits = model(videos)
                    if extract_features:
                        if hook_buf:
                            all_f.append(hook_buf.pop().numpy())
                        else:
                            all_f.append(logits.float().detach().cpu().numpy())
                else:
                    logits, prob_q = forward_ior_prob_q_and_logits(
                        model, videos, use_odu=use_odu, temperature=temperature
                    )
                    if extract_features:
                        all_f.append(prob_q.detach().cpu().numpy())
            all_p.append(torch.argmax(logits, dim=-1).cpu().numpy())
            all_y.append(labels.numpy())
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    yt = np.concatenate(all_y)
    yp = np.concatenate(all_p)
    cm = confusion_matrix(yt, yp, labels=list(range(7)), normalize="true")
    feats = np.concatenate(all_f, axis=0) if (extract_features and all_f) else None
    return cm, feats, yt


def run_tsne_2d(feat: np.ndarray, perplexity: float = 35.0, seed: int = 42) -> np.ndarray:
    feat = np.asarray(feat, dtype=np.float32)
    norms = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(feat / norms)


def render_vis2x5(tsne_dict: dict, cms: dict, out_pdf: str, out_png: str):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(13.0, 5.40), facecolor="white")
    gs = GridSpec(
        2,
        5,
        figure=fig,
        height_ratios=[1.0, 1.0],
        left=0.056,
        right=0.940,
        top=0.935,
        bottom=0.172,
        hspace=0.085,
        wspace=0.11,
    )

    top_axes, bot_axes = [], []
    draw_order = [2, 0, 3, 1, 4, 6, 5]

    for col, (key, title, _, _) in enumerate(STAGES):
        ax = fig.add_subplot(gs[0, col])
        top_axes.append(ax)
        xy, labels = tsne_dict[key]

        for cls_id in draw_order:
            mask = labels == cls_id
            if not np.any(mask):
                continue
            is_tail = cls_id in (5, 6)
            ax.scatter(
                xy[mask, 0],
                xy[mask, 1],
                s=7.8 if is_tail else 5.6,
                c=PALETTE[cls_id],
                alpha=0.88 if is_tail else 0.76,
                edgecolors="none",
                rasterized=True,
            )

        ax.set_box_aspect(1)
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=5.5, color="#111111")
        ax.margins(0.03)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

        if col == 0:
            ax.set_ylabel(
                "Feature\nDistribution",
                fontsize=10.2,
                fontweight="bold",
                color="#111111",
                labelpad=14.0,
            )

    im_last = None
    for col, (key, _, _, _) in enumerate(STAGES):
        ax = fig.add_subplot(gs[1, col])
        bot_axes.append(ax)
        cm = cms[key]

        im = ax.imshow(
            cm, cmap=TIP_BLUES, vmin=0.0, vmax=1.0, aspect="equal", interpolation="nearest"
        )
        ax.set_box_aspect(1)
        im_last = im

        ax.set_xticks(np.arange(-0.5, 7, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 7, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.45)
        ax.tick_params(which="minor", bottom=False, left=False)

        for i in range(7):
            for j in range(7):
                val = cm[i, j]
                if val >= 0.48:
                    txt_color = "#FFFFFF"
                elif val < 0.005:
                    txt_color = "#94A3B8"
                else:
                    txt_color = "#1A1A1A"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.8 if i == j else 6.4,
                    fontweight="bold" if i == j else "normal",
                    color=txt_color,
                )

        ax.set_xticks(range(7))
        ax.set_xticklabels(CLASS_SHORT, rotation=40, ha="right", fontsize=8.0, color="#222222")
        ax.tick_params(axis="x", pad=1.6, length=2.0, width=0.5)
        ax.set_xlabel("Predicted", fontsize=9.2, fontweight="bold", labelpad=2.2, color="#111111")

        if col == 0:
            ax.set_yticks(range(7))
            ax.set_yticklabels(CLASS_SHORT, fontsize=8.0, color="#222222")
            ax.tick_params(axis="y", pad=2.0, length=2.0, width=0.5)
            ax.set_ylabel("True Label", fontsize=9.8, fontweight="bold", labelpad=4.0, color="#111111")
        else:
            ax.set_yticks([])

        for sp in ax.spines.values():
            sp.set_color("#888888")
            sp.set_linewidth(0.6)

    fig.canvas.draw()
    fig.align_ylabels([top_axes[0], bot_axes[0]])

    pos_bot = bot_axes[-1].get_position()
    cax = fig.add_axes([0.950, pos_bot.y0, 0.010, pos_bot.height])
    cb = fig.colorbar(im_last, cax=cax)
    cb.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb.ax.tick_params(labelsize=7.6, length=2.0, width=0.5)
    cb.outline.set_linewidth(0.55)
    cb.outline.set_edgecolor("#888888")
    cb.set_label("Normalized Recall", fontsize=8.8, fontweight="bold", labelpad=5.0, color="#111111")

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=PALETTE[i],
            markeredgecolor="none",
            markersize=6.8,
            label=CLASS_FULL[i],
        )
        for i in range(7)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.498, 0.012),
        ncol=7,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.6,
        fontsize=9.2,
    )

    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_pdf, format="pdf", dpi=300, facecolor="white")
    fig.savefig(out_png, format="png", dpi=300, facecolor="white")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--split_dir", type=str, required=True)
    parser.add_argument("--base_ckpt_dir", type=str, required=True)
    parser.add_argument("--base_ckpt_pattern", type=str, default="")
    parser.add_argument("--abl_ckpt_dir", type=str, required=True)
    parser.add_argument("--abl_ckpt_pattern", type=str, default="Weights_{variant}_Fold{fold}.pth")
    parser.add_argument("--full_ckpt_dir", type=str, default="")
    parser.add_argument("--full_ckpt_pattern", type=str, default="Final_FullTrain_K5_L1_Fold{fold}.pth")
    parser.add_argument("--max_iterations", type=int, default=5)
    parser.add_argument("--mem_refiner_layers", type=int, default=1)
    parser.add_argument("--tsne_fold", type=int, default=1, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--temperature", type=float, default=1.6)
    parser.add_argument("--perplexity", type=float, default=35.0)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_path", type=str, default="")
    parser.add_argument("--force_recompute", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./outputs/visualizations")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    cache_path = args.cache_path or os.path.join(
        args.output_dir,
        f"dfew_vis2x5_cache_f{args.tsne_fold}_T{args.temperature:.1f}.npz",
    )

    if (not args.force_recompute) and os.path.isfile(cache_path):
        cached = np.load(cache_path)
        cms = {k: cached[f"cm_{k}"] for k, _, _, _ in STAGES}
        tsne_dict = {k: (cached[f"xy_{k}"], cached[f"y_{k}"]) for k, _, _, _ in STAGES}
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        folds = [1, 2, 3, 4, 5]
        loaders = {
            fold: DataLoader(
                DFEWDataset(args.split_dir, args.data_root, fold, "test", is_train=False, num_frames=args.num_frames),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_fn_safe,
                pin_memory=True,
            )
            for fold in folds
        }

        cms = {}
        tsne_dict = {}
        for key, _, use_odu, use_pgs in STAGES:
            fold_cms = []
            for fold in folds:
                if key == "0_Static_Base":
                    ckpt_path = resolve_baseline_ckpt(args.base_ckpt_dir, args.base_ckpt_pattern, fold)
                elif key == "4_Full_IOR_Net" and args.full_ckpt_dir:
                    ckpt_path = os.path.join(args.full_ckpt_dir, args.full_ckpt_pattern.format(fold=fold))
                else:
                    ckpt_path = os.path.join(args.abl_ckpt_dir, args.abl_ckpt_pattern.format(variant=key, fold=fold))
                if not os.path.exists(ckpt_path):
                    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

                model = build_stage_model(
                    key,
                    use_odu=use_odu,
                    use_pgs=use_pgs,
                    max_iterations=args.max_iterations,
                    mem_refiner_layers=args.mem_refiner_layers,
                    num_frames=args.num_frames,
                ).to(device)
                model.load_state_dict(load_clean_state_dict(ckpt_path), strict=True)

                cm_fold, feats_fold, y_fold = evaluate_stage_fold(
                    model,
                    loaders[fold],
                    stage_key=key,
                    use_odu=use_odu,
                    device=device,
                    extract_features=(fold == args.tsne_fold),
                    temperature=args.temperature,
                )
                fold_cms.append(cm_fold)
                if fold == args.tsne_fold:
                    xy = run_tsne_2d(feats_fold, perplexity=args.perplexity, seed=args.seed)
                    tsne_dict[key] = (xy, y_fold)

                del model
                torch.cuda.empty_cache()

            cms[key] = np.mean(fold_cms, axis=0)

        save_payload = {}
        for key, _, _, _ in STAGES:
            save_payload[f"cm_{key}"] = cms[key]
            save_payload[f"xy_{key}"] = tsne_dict[key][0]
            save_payload[f"y_{key}"] = tsne_dict[key][1]
        np.savez(cache_path, **save_payload)

    render_vis2x5(
        tsne_dict=tsne_dict,
        cms=cms,
        out_pdf=os.path.join(args.output_dir, "fig9_tsne_cm_2x5.pdf"),
        out_png=os.path.join(args.output_dir, "fig9_tsne_cm_2x5.png"),
    )


if __name__ == "__main__":
    main()
