import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import dlib
except ImportError:
    dlib = None

ALL_12_CONDITIONS = [
    "Clean",
    "Occ_Background",
    "Occ_Eyes",
    "Occ_Mouth",
    "Occ_Block30",
    "Noise_Gauss",
    "Occ_Temp50",
    "Temp_Burst50",
    "Temp_Freeze50",
    "Temp_Freeze75",
    "Temp_Noise50",
    "Temp_Shuffle",
]

DLIB_CONDITIONS = [
    "Clean",
    "Occ_Background",
    "Occ_Eyes",
    "Occ_Mouth",
]

FALLBACK_BG_KEEP = (26, 20, 198, 204)
FALLBACK_EYES_BOX = (36, 40, 188, 108)
FALLBACK_MOUTH_BOX = (44, 128, 180, 196)


def boxes_from_landmarks_68(pts: np.ndarray, height: int = 224, width: int = 224):
    fx1, fx2 = float(pts[:, 0].min()), float(pts[:, 0].max())
    fy_brow = float(pts[17:27, 1].min())
    fy_chin = float(pts[0:17, 1].max())
    fw = max(fx2 - fx1, 20.0)
    fh = max(fy_chin - fy_brow, 20.0)

    bx1 = int(np.clip(round(fx1 - 0.14 * fw), 0, width - 2))
    bx2 = int(np.clip(round(fx2 + 0.14 * fw), bx1 + 2, width))
    by1 = int(np.clip(round(fy_brow - 0.35 * fh), 0, height - 2))
    by2 = int(np.clip(round(fy_chin + 0.10 * fh), by1 + 2, height))
    bg_keep = (bx1, by1, bx2, by2)

    eye_pts = np.vstack([pts[17:27], pts[36:48]])
    ex1, ex2 = float(eye_pts[:, 0].min()), float(eye_pts[:, 0].max())
    ey1, ey2 = float(eye_pts[:, 1].min()), float(eye_pts[:, 1].max())
    ew, eh = max(ex2 - ex1, 10.0), max(ey2 - ey1, 8.0)
    epx = max(0.12 * ew, 6.0)
    epy_top = max(0.25 * eh, 5.0)
    epy_bot = max(0.35 * eh, 6.0)
    eyes_box = (
        int(np.clip(round(ex1 - epx), 0, width - 2)),
        int(np.clip(round(ey1 - epy_top), 0, height - 2)),
        int(np.clip(round(ex2 + epx), 2, width)),
        int(np.clip(round(ey2 + epy_bot), 2, height)),
    )

    m_pts = pts[48:68]
    mx1, mx2 = float(m_pts[:, 0].min()), float(m_pts[:, 0].max())
    my1, my2 = float(m_pts[:, 1].min()), float(m_pts[:, 1].max())
    mw, mh = max(mx2 - mx1, 8.0), max(my2 - my1, 4.0)
    mpx = max(0.22 * mw, 0.08 * fw, 8.0)
    mpy = max(0.30 * mh, 0.08 * fh, 10.0)
    mouth_box = (
        int(np.clip(round(mx1 - mpx), 0, width - 2)),
        int(np.clip(round(my1 - mpy), 0, height - 2)),
        int(np.clip(round(mx2 + mpx), 2, width)),
        int(np.clip(round(my2 + mpy), 2, height)),
    )
    return bg_keep, eyes_box, mouth_box


def compute_clip_dlib_masks(video_norm: torch.Tensor, detector, predictor):
    _, num_frames, height, width = video_norm.shape
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=video_norm.dtype).view(3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=video_norm.dtype).view(3, 1, 1, 1)
    v01 = torch.clamp(video_norm * std + mean, 0.0, 1.0)
    gray_f = (0.299 * v01[0] + 0.587 * v01[1] + 0.114 * v01[2]) * 255.0
    gray_np = np.ascontiguousarray(gray_f.byte().numpy())

    frame_boxes = [None] * num_frames
    valid_ts = []

    for t in range(num_frames):
        frame_gray = gray_np[t]
        try:
            rects = detector(frame_gray, 0)
            if len(rects) == 0:
                rects = detector(frame_gray, 1)
            if len(rects) > 0:
                def _score(rect):
                    area = max(rect.width(), 1) * max(rect.height(), 1)
                    cx = 0.5 * (rect.left() + rect.right())
                    cy = 0.5 * (rect.top() + rect.bottom())
                    dist = ((cx - width * 0.5) ** 2 + (cy - height * 0.5) ** 2) ** 0.5
                    return area - 40.0 * dist

                best_rect = max(rects, key=_score)
                shape = predictor(frame_gray, best_rect)
                pts = np.array(
                    [(shape.part(i).x, shape.part(i).y) for i in range(68)],
                    dtype=np.float32,
                )
                frame_boxes[t] = boxes_from_landmarks_68(pts, height, width)
                valid_ts.append(t)
        except Exception:
            pass

    n_det = len(valid_ts)
    clip_hit = 1 if n_det > 0 else 0

    mask_bg = np.zeros((num_frames, height, width), dtype=np.uint8)
    mask_eyes = np.zeros((num_frames, height, width), dtype=np.uint8)
    mask_mouth = np.zeros((num_frames, height, width), dtype=np.uint8)

    for t in range(num_frames):
        if frame_boxes[t] is not None:
            bg_keep, eyes_box, mouth_box = frame_boxes[t]
        elif clip_hit == 1:
            nearest_t = min(valid_ts, key=lambda vt: abs(vt - t))
            bg_keep, eyes_box, mouth_box = frame_boxes[nearest_t]
        else:
            bg_keep = FALLBACK_BG_KEEP
            eyes_box = FALLBACK_EYES_BOX
            mouth_box = FALLBACK_MOUTH_BOX

        mask_bg[t, :, :] = 1
        bx1, by1, bx2, by2 = bg_keep
        mask_bg[t, by1:by2, bx1:bx2] = 0

        ex1, ey1, ex2, ey2 = eyes_box
        mask_eyes[t, ey1:ey2, ex1:ex2] = 1

        mx1, my1, mx2, my2 = mouth_box
        mask_mouth[t, my1:my2, mx1:mx2] = 1

    return mask_bg, mask_eyes, mask_mouth, n_det, clip_hit


class DlibLandmarkMaskWrapper(Dataset):
    def __init__(self, base_dataset: Dataset, predictor_path: str):
        if dlib is None:
            raise ImportError("dlib is required for landmark-guided occlusion masks.")
        self.base_dataset = base_dataset
        self.predictor_path = predictor_path
        self._detector = None
        self._predictor = None

    def _ensure_dlib(self):
        if self._detector is None:
            self._detector = dlib.get_frontal_face_detector()
            self._predictor = dlib.shape_predictor(self.predictor_path)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        item = self.base_dataset[idx]
        if item is None:
            return None
        video, label = item
        self._ensure_dlib()
        m_bg, m_eyes, m_mouth, n_det, clip_hit = compute_clip_dlib_masks(
            video, self._detector, self._predictor
        )
        return (
            video,
            int(label),
            torch.from_numpy(m_bg),
            torch.from_numpy(m_eyes),
            torch.from_numpy(m_mouth),
            int(n_det),
            int(clip_hit),
        )


def collate_fn_dlib(batch):
    batch = [x for x in batch if x is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def build_corruptions_batch(
    videos_norm: torch.Tensor,
    m_bg: torch.Tensor,
    m_eyes: torch.Tensor,
    m_mouth: torch.Tensor,
    seed_base: int,
    batch_start_idx: int,
    device: torch.device,
    include_all_12: bool = True,
):
    batch_size, channels, num_frames, height, width = videos_norm.shape
    mean = torch.tensor(
        [0.485, 0.456, 0.406], device=device, dtype=videos_norm.dtype
    ).view(1, 3, 1, 1, 1)
    std = torch.tensor(
        [0.229, 0.224, 0.225], device=device, dtype=videos_norm.dtype
    ).view(1, 3, 1, 1, 1)
    black_val = (0.0 - mean) / std

    out = {
        "Clean": videos_norm,
        "Occ_Background": torch.where(m_bg.unsqueeze(1).bool(), black_val, videos_norm),
        "Occ_Eyes": torch.where(m_eyes.unsqueeze(1).bool(), black_val, videos_norm),
        "Occ_Mouth": torch.where(m_mouth.unsqueeze(1).bool(), black_val, videos_norm),
    }

    if not include_all_12:
        return out

    x_block = videos_norm.clone()
    for b in range(batch_size):
        rng = np.random.RandomState(seed_base + 10000 + batch_start_idx + b)
        y0 = int(rng.randint(0, height - 122 + 1))
        x0 = int(rng.randint(0, width - 122 + 1))
        x_block[b : b + 1, :, :, y0 : y0 + 122, x0 : x0 + 122] = black_val
    out["Occ_Block30"] = x_block

    x_temp50 = videos_norm.clone()
    for b in range(batch_size):
        rng = np.random.RandomState(seed_base + 20000 + batch_start_idx + b)
        drop_idx = rng.choice(num_frames, size=num_frames // 2, replace=False)
        x_temp50[b : b + 1, :, drop_idx, :, :] = black_val
    out["Occ_Temp50"] = x_temp50

    gen_gauss = torch.Generator(device=device)
    gen_gauss.manual_seed(seed_base + 30000 + batch_start_idx)
    x_01 = videos_norm * std + mean
    noise = torch.randn(
        videos_norm.shape, generator=gen_gauss, device=device, dtype=videos_norm.dtype
    )
    out["Noise_Gauss"] = (torch.clamp(x_01 + noise * 0.18, 0.0, 1.0) - mean) / std

    for cond_key, offset, span in (
        ("Temp_Freeze50", 40000, 8),
        ("Temp_Freeze75", 50000, 12),
    ):
        x_freeze = videos_norm.clone()
        for b in range(batch_size):
            rng = np.random.RandomState(seed_base + offset + batch_start_idx + b)
            t0 = int(rng.randint(0, num_frames - span + 1))
            anchor = x_freeze[b : b + 1, :, t0 : t0 + 1, :, :].clone()
            x_freeze[b : b + 1, :, t0 : t0 + span, :, :] = anchor.expand(
                -1, -1, span, -1, -1
            )
        out[cond_key] = x_freeze

    x_burst = videos_norm.clone()
    for b in range(batch_size):
        rng = np.random.RandomState(seed_base + 60000 + batch_start_idx + b)
        t0 = int(rng.randint(0, num_frames - 8 + 1))
        x_burst[b : b + 1, :, t0 : t0 + 8, :, :] = black_val
    out["Temp_Burst50"] = x_burst

    x_noise50 = videos_norm.clone()
    gen_rand = torch.Generator(device=device)
    gen_rand.manual_seed(seed_base + 70000 + batch_start_idx)
    rand_norm = (
        torch.rand(
            (batch_size, channels, num_frames // 2, height, width),
            generator=gen_rand,
            device=device,
            dtype=videos_norm.dtype,
        )
        - mean
    ) / std
    for b in range(batch_size):
        rng = np.random.RandomState(seed_base + 80000 + batch_start_idx + b)
        drop_idx = rng.choice(num_frames, size=num_frames // 2, replace=False)
        x_noise50[b, :, drop_idx, :, :] = rand_norm[b]
    out["Temp_Noise50"] = x_noise50

    x_shuffle = videos_norm.clone()
    for b in range(batch_size):
        rng = np.random.RandomState(seed_base + 90000 + batch_start_idx + b)
        perm = rng.permutation(num_frames)
        x_shuffle[b] = x_shuffle[b, :, perm, :, :].clone()
    out["Temp_Shuffle"] = x_shuffle

    return out
