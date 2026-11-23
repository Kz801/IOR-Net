import glob
import logging
import os
import random

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

FERV39K_LABEL_MAP = {
    "Happy": 0,
    "Sad": 1,
    "Neutral": 2,
    "Angry": 3,
    "Surprise": 4,
    "Disgust": 5,
    "Fear": 6,
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


class FERV39KDataset(Dataset):
    def __init__(
        self,
        csv_root_dir: str,
        clip_root: str,
        split_type: str = "train",
        is_train: bool = True,
        num_frames: int = 16,
        max_frames: int = None,
    ):
        self.clip_root = clip_root
        self.is_train = is_train
        self.num_frames = max_frames if max_frames is not None else num_frames
        self.max_frames = self.num_frames

        self.transform = ConsistentVideoTransform(is_train=is_train, size=224)
        self.to_tensor = transforms.ToTensor()

        csv_path = os.path.join(csv_root_dir, f"{split_type}_All.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Label CSV file not found: {csv_path}")

        try:
            self.data_info = pd.read_csv(
                csv_path, sep=" ", header=None, names=["path", "label"]
            )
        except Exception:
            self.data_info = pd.read_csv(
                csv_path, sep=None, engine="python", header=None, names=["path", "label"]
            )

        logger.info(
            "[%s] FERV39k (%s) loaded: %d samples from %s",
            "TRAIN" if is_train else "EVAL",
            split_type,
            len(self.data_info),
            self.clip_root,
        )

    def __len__(self) -> int:
        return len(self.data_info)

    def get_indices(self, total_frames: int) -> np.ndarray:
        if total_frames < self.num_frames:
            all_indices = np.arange(total_frames)
            repeat_times = int(np.ceil(self.num_frames / total_frames))
            return np.tile(all_indices, repeat_times)[: self.num_frames]

        indices = []
        split_size = total_frames / self.num_frames
        for i in range(self.num_frames):
            start = int(i * split_size)
            end = int((i + 1) * split_size)
            if end <= start:
                end = start + 1

            if self.is_train:
                idx = np.random.randint(start, min(end, total_frames))
            else:
                idx = min((start + end) // 2, total_frames - 1)
            indices.append(idx)

        return np.array(indices, dtype=int)

    def _get_dummy_output(self):
        return torch.zeros(3, self.num_frames, 224, 224), 0

    def __getitem__(self, idx: int):
        if idx >= len(self.data_info):
            return None

        row = self.data_info.iloc[idx]
        rel_path = str(row["path"]).strip()
        label_str = str(row["label"]).strip()
        label = FERV39K_LABEL_MAP.get(label_str, 0)

        video_folder = os.path.join(self.clip_root, rel_path)
        image_paths = sorted(glob.glob(os.path.join(video_folder, "*.jpg")))
        if not image_paths:
            image_paths = sorted(glob.glob(os.path.join(video_folder, "*.png")))

        if not image_paths:
            return self._get_dummy_output()

        indices = self.get_indices(len(image_paths))
        frames = []
        try:
            for i in indices:
                with Image.open(image_paths[i]) as img:
                    frames.append(self.to_tensor(img.convert("RGB")))
        except Exception as exc:
            logger.error("Error loading images in %s: %s", rel_path, exc)
            return self._get_dummy_output()

        if not frames:
            return self._get_dummy_output()

        raw_video_tensor = torch.stack(frames).permute(1, 0, 2, 3)
        final_video_tensor = self.transform(raw_video_tensor)
        return final_video_tensor, label
