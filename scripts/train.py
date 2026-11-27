import argparse
import json
import logging
import math
import os
import random
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

try:
    import yaml
except ImportError:
    yaml = None

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
for _p in (CURRENT_DIR, PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from datasets.dfew import DFEW_CLASS_NAMES, DFEWDataset
from datasets.ferv39k import FERV39K_LABEL_MAP, FERV39KDataset
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

try:
    from models.ior_net_ablation import internvideo2_ablation_sota_small as build_ablation_net
except ImportError:
    try:
        from internvideo2_Ablation import internvideo2_ablation_sota_small as build_ablation_net
    except ImportError:
        build_ablation_net = None

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
    },
    "ferv39k": {
        "K": 6,
        "L": 1,
        "num_classes": 7,
        "folds": [1],
        "class_names": FERV_CLASS_NAMES,
    },
    "mafw11": {
        "K": 4,
        "L": 1,
        "num_classes": 11,
        "folds": [1, 2, 3, 4, 5],
        "class_names": MAFW11_CLASS_NAMES,
    },
    "mafw43": {
        "K": 4,
        "L": 1,
        "num_classes": 43,
        "folds": [1, 2, 3, 4, 5],
        "class_names": MAFW_43_LABELS,
    },
}

NEW_MODULE_KEYWORDS = (
    "isr_",
    "decoder",
    "refiner",
    "query_",
    "gating",
    "head",
    "fuser",
    "synthesizer",
    "logit_scale",
    "prototypes",
)


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def setup_logger(save_dir: str, name: str):
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.INFO)
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, f"{name}.log")
    fmt = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8", mode="a")
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


def build_cosine_warmup_scheduler(
    optimizer: optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def load_pretrained_backbone(model: nn.Module, path: str, num_frames: int, logger):
    if not path:
        return
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pretrained backbone weights not found: {path}")

    obj = torch.load(path, map_location="cpu")
    state_dict = obj
    if isinstance(obj, dict):
        if "model" in obj and isinstance(obj["model"], dict):
            state_dict = obj["model"]
        elif "state_dict" in obj and isinstance(obj["state_dict"], dict):
            state_dict = obj["state_dict"]
    state_dict = {
        k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()
    }

    if "pos_embed" in state_dict and hasattr(model, "pos_embed"):
        pos_embed_ckpt = state_dict["pos_embed"]
        pos_embed_model = model.pos_embed.data
        if pos_embed_ckpt.shape != pos_embed_model.shape:
            cls_token_ckpt = pos_embed_ckpt[:, 0:1, :]
            patch_embed_ckpt = pos_embed_ckpt[:, 1:, :]
            embed_dim = patch_embed_ckpt.shape[-1]
            num_patches = patch_embed_ckpt.shape[1]
            target_grid_size = model.patch_embed.grid_size[1]
            ckpt_t = 8
            if num_patches % ckpt_t != 0 and (
                num_patches % (target_grid_size * target_grid_size) == 0
            ):
                ckpt_t = num_patches // (target_grid_size * target_grid_size)
            spatial_patches = num_patches // ckpt_t
            grid_size = int(round(spatial_patches ** 0.5))
            if grid_size * grid_size * ckpt_t == num_patches:
                patch_embed_ckpt = (
                    patch_embed_ckpt.view(1, ckpt_t, grid_size, grid_size, embed_dim)
                    .permute(0, 4, 1, 2, 3)
                    .float()
                )
                patch_embed_new = F.interpolate(
                    patch_embed_ckpt,
                    size=(num_frames, target_grid_size, target_grid_size),
                    mode="trilinear",
                    align_corners=False,
                )
                patch_embed_new = patch_embed_new.permute(0, 2, 3, 4, 1).view(
                    1, -1, embed_dim
                )
                state_dict["pos_embed"] = torch.cat(
                    (cls_token_ckpt, patch_embed_new), dim=1
                )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info(
        "Loaded pretrained backbone from %s (missing=%d, unexpected=%d)",
        path,
        len(missing),
        len(unexpected),
    )


def build_model(
    model_type: str,
    num_classes: int,
    max_iterations: int,
    mem_refiner_layers: int,
    num_frames: int = 16,
    use_odu: bool = True,
    use_pgs: bool = True,
    proto_mode: str = "etf_learnable",
    saliency_mode: str = "max_proto",
) -> nn.Module:
    if model_type == "baseline":
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

    if model_type == "ablation":
        if build_ablation_net is None:
            raise ImportError("Ablation model definition could not be imported.")
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


def build_datasets(
    dataset_name: str,
    data_root: str,
    split_dir: str,
    fold: int,
    num_frames: int,
):
    if dataset_name == "dfew":
        train_ds = DFEWDataset(
            split_csv_dir=split_dir,
            clip_root=data_root,
            fold_num=fold,
            split_type="train",
            is_train=True,
            num_frames=num_frames,
        )
        val_ds = DFEWDataset(
            split_csv_dir=split_dir,
            clip_root=data_root,
            fold_num=fold,
            split_type="train",
            is_train=False,
            num_frames=num_frames,
        )
        test_ds = DFEWDataset(
            split_csv_dir=split_dir,
            clip_root=data_root,
            fold_num=fold,
            split_type="test",
            is_train=False,
            num_frames=num_frames,
        )
        targets = (train_ds.data_info["label"].astype(int) - 1).tolist()
        return train_ds, val_ds, test_ds, targets

    if dataset_name == "ferv39k":
        train_ds = FERV39KDataset(
            csv_root_dir=split_dir,
            clip_root=data_root,
            split_type="train",
            is_train=True,
            num_frames=num_frames,
        )
        val_ds = FERV39KDataset(
            csv_root_dir=split_dir,
            clip_root=data_root,
            split_type="train",
            is_train=False,
            num_frames=num_frames,
        )
        test_ds = FERV39KDataset(
            csv_root_dir=split_dir,
            clip_root=data_root,
            split_type="test",
            is_train=False,
            num_frames=num_frames,
        )
        targets = [
            FERV39K_LABEL_MAP.get(str(x).strip(), 0)
            for x in train_ds.data_info["label"].tolist()
        ]
        return train_ds, val_ds, test_ds, targets

    if dataset_name == "mafw11":
        train_ds = MAFWDataset(
            frames_root=data_root,
            txt_root_base=split_dir,
            fold=fold,
            mode="train",
            num_frames=num_frames,
            is_train=True,
        )
        val_ds = MAFWDataset(
            frames_root=data_root,
            txt_root_base=split_dir,
            fold=fold,
            mode="train",
            num_frames=num_frames,
            is_train=False,
        )
        test_ds = MAFWDataset(
            frames_root=data_root,
            txt_root_base=split_dir,
            fold=fold,
            mode="test",
            num_frames=num_frames,
            is_train=False,
        )
        targets = [lbl for _, lbl in train_ds.data_list]
        return train_ds, val_ds, test_ds, targets

    if dataset_name == "mafw43":
        train_ds = MAFWDataset43(
            frames_root=data_root,
            txt_root_base=split_dir,
            fold=fold,
            mode="train",
            num_frames=num_frames,
            is_train=True,
        )
        val_ds = MAFWDataset43(
            frames_root=data_root,
            txt_root_base=split_dir,
            fold=fold,
            mode="train",
            num_frames=num_frames,
            is_train=False,
        )
        test_ds = MAFWDataset43(
            frames_root=data_root,
            txt_root_base=split_dir,
            fold=fold,
            mode="test",
            num_frames=num_frames,
            is_train=False,
        )
        targets = [lbl for _, lbl in train_ds.data_list]
        return train_ds, val_ds, test_ds, targets

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_dataloaders(
    dataset_name: str,
    data_root: str,
    split_dir: str,
    fold: int,
    split_mode: str,
    inner_val_ratio: float,
    split_seed: int,
    num_frames: int,
    batch_size: int,
    num_workers: int,
):
    train_ds, val_ds, test_ds, targets = build_datasets(
        dataset_name, data_root, split_dir, fold, num_frames
    )

    if split_mode == "inner_val":
        indices = list(range(len(train_ds)))
        try:
            train_idx, val_idx = train_test_split(
                indices,
                test_size=inner_val_ratio,
                random_state=split_seed,
                shuffle=True,
                stratify=targets,
            )
        except ValueError:
            train_idx, val_idx = train_test_split(
                indices,
                test_size=inner_val_ratio,
                random_state=split_seed,
                shuffle=True,
            )
        train_subset = Subset(train_ds, train_idx)
        val_subset = Subset(val_ds, val_idx)
    else:
        train_subset = train_ds
        val_subset = None

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_safe,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = (
        DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn_safe,
            pin_memory=True,
        )
        if val_subset is not None
        else None
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_safe,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def build_optimizer(
    model: nn.Module,
    base_lr: float,
    new_module_lr_mult: float,
    weight_decay: float,
) -> optim.AdamW:
    backbone_params = []
    new_module_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(kw in name for kw in NEW_MODULE_KEYWORDS):
            new_module_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": base_lr})
    if new_module_params:
        param_groups.append(
            {"params": new_module_params, "lr": base_lr * new_module_lr_mult}
        )
    return optim.AdamW(param_groups, weight_decay=weight_decay)


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
    desc: str = "Eval",
):
    model.eval()
    all_targets = []
    all_preds = []
    use_amp = device.type == "cuda"

    for videos, labels in tqdm(loader, desc=desc, leave=False):
        if videos is None or videos.nelement() == 0:
            continue
        videos = videos.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(videos)
        all_preds.append(torch.argmax(logits, dim=-1).cpu().numpy())
        all_targets.append(labels.numpy())

    yt = np.concatenate(all_targets)
    yp = np.concatenate(all_preds)
    cls_labels = list(range(num_classes))
    uar = float(
        recall_score(yt, yp, labels=cls_labels, average="macro", zero_division=0)
        * 100.0
    )
    war = float(accuracy_score(yt, yp) * 100.0)
    per_class = (
        recall_score(yt, yp, labels=cls_labels, average=None, zero_division=0) * 100.0
    ).tolist()
    return {
        "uar": uar,
        "war": war,
        "score": float(0.7 * uar + 0.3 * war),
        "per_class": per_class,
        "targets": yt,
        "preds": yp,
    }


def train_single_fold(
    dataset_name: str,
    data_root: str,
    split_dir: str,
    pretrained_path: str,
    fold: int,
    max_iterations: int,
    mem_refiner_layers: int,
    num_classes: int,
    split_mode: str,
    inner_val_ratio: float,
    epochs: int,
    batch_size: int,
    base_lr: float,
    new_module_lr_mult: float,
    weight_decay: float,
    label_smoothing: float,
    warmup_ratio: float,
    grad_clip: float,
    num_frames: int,
    num_workers: int,
    seed: int,
    ckpt_path: str,
    model_type: str,
    use_odu: bool,
    use_pgs: bool,
    proto_mode: str,
    saliency_mode: str,
    eval_test_after_val: bool,
    resume_eval_if_exists: bool,
    device: torch.device,
    logger,
):
    is_single_split = dataset_name == "ferv39k"
    fold_seed = seed if is_single_split else (seed + fold)
    set_seed(fold_seed)

    train_loader, val_loader, test_loader = build_dataloaders(
        dataset_name=dataset_name,
        data_root=data_root,
        split_dir=split_dir,
        fold=fold,
        split_mode=split_mode,
        inner_val_ratio=inner_val_ratio,
        split_seed=fold_seed,
        num_frames=num_frames,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = build_model(
        model_type=model_type,
        num_classes=num_classes,
        max_iterations=max_iterations,
        mem_refiner_layers=mem_refiner_layers,
        num_frames=num_frames,
        use_odu=use_odu,
        use_pgs=use_pgs,
        proto_mode=proto_mode,
        saliency_mode=saliency_mode,
    ).to(device)

    os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)
    eval_loader = val_loader if split_mode == "inner_val" else test_loader

    if resume_eval_if_exists and os.path.exists(ckpt_path):
        obj = torch.load(ckpt_path, map_location="cpu")
        sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        primary_res = evaluate_loader(
            model, eval_loader, num_classes, device, desc=f"Eval F{fold}"
        )
        test_res = (
            evaluate_loader(
                model, test_loader, num_classes, device, desc=f"Test F{fold}"
            )
            if (split_mode == "inner_val" and eval_test_after_val)
            else (primary_res if split_mode == "full" else None)
        )
        del model
        torch.cuda.empty_cache()
        return {
            "fold": fold,
            "best_epoch": -1,
            "val_res": primary_res if split_mode == "inner_val" else None,
            "test_res": test_res,
            "ckpt_path": ckpt_path,
        }

    load_pretrained_backbone(model, pretrained_path, num_frames, logger)
    optimizer = build_optimizer(model, base_lr, new_module_lr_mult, weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)

    total_steps = len(train_loader) * epochs
    scheduler = build_cosine_warmup_scheduler(
        optimizer,
        num_warmup_steps=int(warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", init_scale=2 ** 12, enabled=use_amp)

    best_score = -1.0
    best_epoch = -1
    best_primary_res = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        steps_in_epoch = 0

        for videos, labels in tqdm(
            train_loader,
            desc=f"{dataset_name.upper()} F{fold} Ep{epoch + 1}/{epochs}",
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += float(loss.item())
            steps_in_epoch += 1

        avg_loss = running_loss / max(1, steps_in_epoch)
        ep_res = evaluate_loader(
            model,
            eval_loader,
            num_classes,
            device,
            desc=f"{dataset_name.upper()} F{fold} Ep{epoch + 1} Eval",
        )
        if ep_res["score"] > best_score:
            best_score = ep_res["score"]
            best_epoch = epoch + 1
            best_primary_res = ep_res
            torch.save(model.state_dict(), ckpt_path)

        logger.info(
            "[%s F%d Ep %02d/%02d] Loss=%.4f | %s Score=%.2f (UAR=%.2f%%, WAR=%.2f%%) | Best=%.2f (Ep %d)",
            dataset_name.upper(),
            fold,
            epoch + 1,
            epochs,
            avg_loss,
            "Val" if split_mode == "inner_val" else "Test",
            ep_res["score"],
            ep_res["uar"],
            ep_res["war"],
            best_score,
            best_epoch,
        )

    test_res = best_primary_res if split_mode == "full" else None
    if split_mode == "inner_val" and eval_test_after_val:
        model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
        test_res = evaluate_loader(
            model, test_loader, num_classes, device, desc=f"{dataset_name.upper()} F{fold} Test"
        )

    del model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()

    return {
        "fold": fold,
        "best_epoch": best_epoch,
        "val_res": best_primary_res if split_mode == "inner_val" else None,
        "test_res": test_res,
        "ckpt_path": ckpt_path,
    }


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default="")
    cfg_args, _ = config_parser.parse_known_args()

    cfg = {}
    if cfg_args.config:
        if yaml is None:
            raise ImportError("PyYAML is required when passing --config.")
        with open(cfg_args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    parser = argparse.ArgumentParser(parents=[config_parser])
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["dfew", "ferv39k", "mafw11", "mafw43"],
    )
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--split_dir", type=str, default=None)
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs/train")
    parser.add_argument(
        "--model_type",
        type=str,
        default="ior_net",
        choices=["ior_net", "baseline", "ablation"],
    )
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--mem_refiner_layers", type=int, default=None)
    parser.add_argument("--use_odu", action="store_true", default=True)
    parser.add_argument("--no_odu", action="store_false", dest="use_odu")
    parser.add_argument("--use_pgs", action="store_true", default=True)
    parser.add_argument("--no_pgs", action="store_false", dest="use_pgs")
    parser.add_argument(
        "--proto_mode",
        type=str,
        default="etf_learnable",
        choices=["etf_learnable", "random_init", "fixed_etf"],
    )
    parser.add_argument(
        "--saliency_mode",
        type=str,
        default="max_proto",
        choices=["max_proto", "l2_norm", "mean_proto", "none"],
    )
    parser.add_argument(
        "--split_mode", type=str, default="full", choices=["full", "inner_val"]
    )
    parser.add_argument("--inner_val_ratio", type=float, default=0.10)
    parser.add_argument("--eval_test_after_val", action="store_true")
    parser.add_argument("--resume_eval_if_exists", action="store_true")
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
    parser.add_argument("--ckpt_pattern", type=str, default="")
    if cfg:
        parser.set_defaults(**cfg)
    args = parser.parse_args()

    for required_field in ("dataset", "data_root", "split_dir"):
        if getattr(args, required_field) is None:
            parser.error(f"--{required_field} is required (via CLI or --config).")
    return args


def main():
    t0 = time.time()
    args = parse_args()
    spec = DATASET_SPECS[args.dataset]
    max_iter = (
        args.max_iterations if args.max_iterations is not None else spec["K"]
    )
    mem_layers = (
        args.mem_refiner_layers
        if args.mem_refiner_layers is not None
        else spec["L"]
    )
    num_classes = spec["num_classes"]
    folds = args.folds if args.folds else spec["folds"]
    class_names = spec["class_names"]

    run_tag = f"train_{args.dataset}_{args.model_type}_k{max_iter}_l{mem_layers}_s{args.seed}"
    logger = setup_logger(args.output_dir, name=run_tag)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_records = []
    val_scores, val_uars, val_wars = [], [], []
    test_uars, test_wars, test_per_cls = [], [], []
    pooled_targets, pooled_preds = [], []

    for fold in folds:
        if args.ckpt_pattern:
            ckpt_name = args.ckpt_pattern.format(
                dataset=args.dataset,
                model=args.model_type,
                K=max_iter,
                L=mem_layers,
                seed=args.seed,
                fold=fold,
            )
        elif len(folds) == 1 and args.dataset == "ferv39k":
            ckpt_name = (
                f"{args.model_type}_{args.dataset}_K{max_iter}_L{mem_layers}_Seed{args.seed}.pth"
            )
        else:
            ckpt_name = (
                f"{args.model_type}_{args.dataset}_K{max_iter}_L{mem_layers}_Seed{args.seed}_Fold{fold}.pth"
            )
        ckpt_path = os.path.join(args.output_dir, ckpt_name)

        rec = train_single_fold(
            dataset_name=args.dataset,
            data_root=args.data_root,
            split_dir=args.split_dir,
            pretrained_path=args.pretrained_path or "",
            fold=fold,
            max_iterations=max_iter,
            mem_refiner_layers=mem_layers,
            num_classes=num_classes,
            split_mode=args.split_mode,
            inner_val_ratio=args.inner_val_ratio,
            epochs=args.epochs,
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
            model_type=args.model_type,
            use_odu=args.use_odu,
            use_pgs=args.use_pgs,
            proto_mode=args.proto_mode,
            saliency_mode=args.saliency_mode,
            eval_test_after_val=args.eval_test_after_val,
            resume_eval_if_exists=args.resume_eval_if_exists,
            device=device,
            logger=logger,
        )
        fold_records.append(rec)

        if rec["val_res"] is not None:
            val_scores.append(rec["val_res"]["score"])
            val_uars.append(rec["val_res"]["uar"])
            val_wars.append(rec["val_res"]["war"])
        if rec["test_res"] is not None:
            test_uars.append(rec["test_res"]["uar"])
            test_wars.append(rec["test_res"]["war"])
            test_per_cls.append(rec["test_res"]["per_class"])
            pooled_targets.append(rec["test_res"]["targets"])
            pooled_preds.append(rec["test_res"]["preds"])

    summary = {
        "dataset": args.dataset,
        "model_type": args.model_type,
        "split_mode": args.split_mode,
        "K": max_iter,
        "L": mem_layers,
        "seed": args.seed,
        "epochs": args.epochs,
        "folds": folds,
        "elapsed_min": float((time.time() - t0) / 60.0),
    }

    if val_scores:
        vs_m, vs_s = mean_std_sample(val_scores)
        vu_m, vu_s = mean_std_sample(val_uars)
        vw_m, vw_s = mean_std_sample(val_wars)
        summary.update(
            {
                "val_score_mean": float(vs_m),
                "val_score_std": float(vs_s),
                "val_uar_mean": float(vu_m),
                "val_uar_std": float(vu_s),
                "val_war_mean": float(vw_m),
                "val_war_std": float(vw_s),
            }
        )
        logger.info(
            "[%s Inner-Val] Score=%.2f±%.2f | UAR=%.2f±%.2f%% | WAR=%.2f±%.2f%%",
            args.dataset.upper(),
            vs_m,
            vs_s,
            vu_m,
            vu_s,
            vw_m,
            vw_s,
        )

    if test_uars:
        tu_m, tu_s = mean_std_sample(test_uars)
        tw_m, tw_s = mean_std_sample(test_wars)
        cls_m, cls_s = mean_std_sample(test_per_cls, axis=0)
        yt_all = np.concatenate(pooled_targets)
        yp_all = np.concatenate(pooled_preds)
        cls_labels = list(range(num_classes))
        pooled_uar = float(
            recall_score(
                yt_all, yp_all, labels=cls_labels, average="macro", zero_division=0
            )
            * 100.0
        )
        pooled_war = float(accuracy_score(yt_all, yp_all) * 100.0)
        summary.update(
            {
                "test_uar_mean": float(tu_m),
                "test_uar_std": float(tu_s),
                "test_war_mean": float(tw_m),
                "test_war_std": float(tw_s),
                "test_pooled_uar": pooled_uar,
                "test_pooled_war": pooled_war,
                "per_class_mean": cls_m.tolist(),
                "per_class_std": cls_s.tolist(),
            }
        )
        logger.info(
            "[%s Test] UAR=%.2f±%.2f%% (Pooled %.2f%%) | WAR=%.2f±%.2f%% (Pooled %.2f%%)",
            args.dataset.upper(),
            tu_m,
            tu_s,
            pooled_uar,
            tw_m,
            tw_s,
            pooled_war,
        )
        if num_classes <= 11:
            for idx, cname in enumerate(class_names):
                logger.info("  %-14s: %.2f±%.2f%%", cname, cls_m[idx], cls_s[idx])

    json_path = os.path.join(args.output_dir, f"{run_tag}_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved training summary to %s", json_path)


if __name__ == "__main__":
    main()
