import glob
import logging
import os
import random

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

MAFW_LABEL_MAP = {
    "anger": 0,
    "anxiety": 1,
    "contempt": 2,
    "disappointment": 3,
    "disgust": 4,
    "fear": 5,
    "happiness": 6,
    "helplessness": 7,
    "neutral": 8,
    "sadness": 9,
    "surprise": 10,
}


class ConsistentVideoTransform:
    def __init__(self, is_train: bool = True, size: int = 224):
        self.is_train = is_train
        self.size = size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, video_tensor: torch.Tensor) -> torch.Tensor:
        video = video_tensor.permute(1, 0, 2, 3)

        if self.is_train:
            if random.random() < 0.5:
                video = torch.flip(video, dims=[3])

            i, j, h, w = transforms.RandomResizedCrop.get_params(
                video, scale=(0.8, 1.0), ratio=(0.75, 1.33)
            )
            video = torch.stack(
                [F.resized_crop(frame, i, j, h, w, [self.size, self.size]) for frame in video]
            )
        else:
            video = torch.stack(
                [F.resize(frame, [self.size, self.size]) for frame in video]
            )

        video = torch.stack(
            [F.normalize(frame, self.mean, self.std) for frame in video]
        )
        return video.permute(1, 0, 2, 3)


class MAFWDataset(Dataset):
    def __init__(
        self,
        frames_root: str,
        txt_root_base: str,
        fold: int = 1,
        mode: str = "train",
        num_frames: int = 16,
        is_train: bool = None,
    ):
        self.frames_root = frames_root
        self.mode = mode
        self.is_train = (mode == "train") if is_train is None else bool(is_train)
        self.num_frames = num_frames

        txt_path = os.path.join(txt_root_base, f"set_{fold}", f"{mode}.txt")
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Label file not found: {txt_path}")

        self.data_list = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(" ")
                if len(parts) >= 2:
                    vid_name = parts[0].replace(".mp4", "")
                    label_str = parts[1].strip()
                    if label_str in MAFW_LABEL_MAP:
                        self.data_list.append((vid_name, MAFW_LABEL_MAP[label_str]))

        logger.info(
            "[MAFW-11 Fold-%d | %s] Loaded %d samples.",
            fold,
            mode.upper(),
            len(self.data_list),
        )

        self.resize_size = 256 if self.is_train else 224
        self.pre_transform = transforms.Compose(
            [
                transforms.Resize((self.resize_size, self.resize_size)),
                transforms.ToTensor(),
            ]
        )
        self.video_transform = ConsistentVideoTransform(
            is_train=self.is_train, size=224
        )

    def __len__(self) -> int:
        return len(self.data_list)

    def get_indices(self, total_frames: int) -> np.ndarray:
        if total_frames < self.num_frames:
            indices = np.arange(total_frames)
            repeat = int(np.ceil(self.num_frames / total_frames))
            return np.tile(indices, repeat)[: self.num_frames]

        indices = np.zeros(self.num_frames, dtype=int)
        interval = total_frames / self.num_frames
        for i in range(self.num_frames):
            start = int(interval * i)
            end = min(int(interval * (i + 1)), total_frames)
            if start >= end:
                start = end - 1

            if self.is_train:
                idx = np.random.randint(start, end) if end > start else start
            else:
                idx = (start + end) // 2
            indices[i] = idx
        return indices

    def __getitem__(self, idx: int):
        vid_name, label = self.data_list[idx]
        video_dir = os.path.join(self.frames_root, vid_name)

        n_frames_path = os.path.join(video_dir, "n_frames")
        total_frames = 0
        if os.path.exists(n_frames_path):
            try:
                with open(n_frames_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        total_frames = int(content)
            except Exception:
                pass

        if total_frames == 0:
            total_frames = len(glob.glob(os.path.join(video_dir, "*.png")))
            if total_frames == 0:
                return torch.zeros(3, self.num_frames, 224, 224), label

        indices = self.get_indices(total_frames)
        frames_list = []
        for i in indices:
            img_path = os.path.join(video_dir, f"{int(i):04d}.png")
            try:
                with Image.open(img_path) as img:
                    frames_list.append(self.pre_transform(img.convert("RGB")))
            except Exception:
                frames_list.append(
                    torch.zeros(3, self.resize_size, self.resize_size)
                )

        video_tensor = torch.stack(frames_list).permute(1, 0, 2, 3)
        video_final = self.video_transform(video_tensor)
        return video_final, label
