import glob
import logging
import os
import random

import pandas as pd
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

DFEW_CLASS_NAMES = [
    "Happy",
    "Sad",
    "Neutral",
    "Anger",
    "Surprise",
    "Disgust",
    "Fear",
]


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


class DFEWDataset(Dataset):
    def __init__(
        self,
        split_csv_dir: str,
        clip_root: str,
        fold_num: int,
        split_type: str,
        is_train: bool = False,
        num_frames: int = 16,
    ):
        self.clip_root = clip_root
        self.is_train = is_train
        self.num_frames = num_frames

        self.transform = ConsistentVideoTransform(is_train=is_train, size=224)
        self.to_tensor = transforms.ToTensor()

        csv_folder = (
            "train(single-labeled)"
            if split_type == "train"
            else "test(single-labeled)"
        )
        csv_path = os.path.join(split_csv_dir, csv_folder, f"set_{fold_num}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Label CSV file not found: {csv_path}")

        self.data_info = pd.read_csv(csv_path)
        logger.info(
            "[%s] DFEW Fold-%d loaded: %d samples from %s",
            "TRAIN" if is_train else "EVAL",
            fold_num,
            len(self.data_info),
            self.clip_root,
        )

    def __len__(self) -> int:
        return len(self.data_info)

    def __getitem__(self, idx: int):
        if idx >= len(self.data_info):
            return None

        video_name = str(self.data_info.loc[idx, "video_name"]).zfill(5)
        label = int(self.data_info.loc[idx, "label"]) - 1

        video_folder = os.path.join(self.clip_root, video_name)
        image_files = sorted(glob.glob(os.path.join(video_folder, "*.jpg")))
        if len(image_files) != self.num_frames:
            return None

        frames = []
        try:
            for img_path in image_files:
                with Image.open(img_path) as img:
                    frames.append(self.to_tensor(img.convert("RGB")))
        except Exception as exc:
            logger.error("Error loading frames for %s: %s", video_name, exc)
            return None

        raw_video_tensor = torch.stack(frames).permute(1, 0, 2, 3)
        final_video_tensor = self.transform(raw_video_tensor)
        return final_video_tensor, label


DFEWRawDataset = DFEWDataset
