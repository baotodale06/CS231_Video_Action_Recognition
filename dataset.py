from pathlib import Path
from typing import Optional, Tuple, Dict, List
import numpy as np
import re
from PIL import Image
import random


import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

class VideoTransform:
    def __init__(self, mode="train", img_size=224):
        self.is_train = (mode=='train')
        self.image_size = img_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [T, C, H, W]
        if self.is_train:
            # random r3siz3d crop
            h, w = frames.shape[-2:]
            scale = random.uniform(0.8, 1.0)
            new_h, new_w = int(h*scale), int(w*scale)
            frames = TF.resize(frames, [new_h, new_w], interpolation=transforms.InterpolationMode.BILINEAR)

            # random crop
            i = random.randint(0, max(0, new_h - self.image_size))
            j = random.randint(0, max(0, new_w - self.image_size))
            frames = TF.crop(frames, i, j, min(self.image_size, new_h), min(self.image_size, new_w))
            frames = TF.resize(frames, [self.image_size, self.image_size], interpolation=transforms.InterpolationMode.BILINEAR)

            # horizontal flip (left <--> right)
            if random.random() < 0.5:
                frames = TF.hflip(frames)

        else: #val
            frames = TF.resize(frames, [self.image_size, self.image_size], interpolation=transforms.InterpolationMode.BILINEAR)

        normalized = [TF.normalize(frame, self.mean, self.std) for frame in frames]
        return torch.stack(normalized)

class HMDB51Dataset(Dataset):
    def __init__(self, root: str, split: str, num_frames: int = 16, frame_stride: int = 1,
                 image_size: int = 224, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42):
        super().__init__()

        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Data root not found: {self.root}")

        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        if not self.classes:
            raise RuntimeError(f"No class folders in {self.root}")

        self.class_to_id = {name: id for id, name in enumerate(self.classes)}

        # group videos by (class, base_video_name)
        grouped_samples: Dict[Tuple[str, str], List[Tuple[List[Path], int]]] = {}
        for cls in self.classes:
            cls_dir = self.root / cls
            for video_dir in sorted([d for d in cls_dir.iterdir() if d.is_dir()]):
                frame_paths = sorted([p for p in video_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
                if not frame_paths:
                    continue
                group_key = (cls, self._base_video_name(video_dir.name))
                grouped_samples.setdefault(group_key, []).append((frame_paths, self.class_to_id[cls]))

        if not grouped_samples:
            raise RuntimeError(f"No frame folders found inside {self.root}")

        # split groups
        group_values = list(grouped_samples.values())
        rng = np.random.RandomState(seed)
        group_indices = np.arange(len(group_values))
        rng.shuffle(group_indices)

        split_train = int(len(group_indices) * (1-val_ratio-test_ratio))
        split_val = int(len(group_indices) * (1-val_ratio))

        if split == "train":
            selected_groups = group_indices[:split_train]
        elif split == "val":
            selected_groups = group_indices[split_train:split_val]
        elif split == "test":
            selected_groups = group_indices[split_val:]
        else:
            raise ValueError(f"Unknown split: {split}")

        samples: List[Tuple[List[Path], int]] = []

        for id in selected_groups:
            samples.extend(group_values[int(id)])

        if not samples:
            raise RuntimeError(f"Selected split has no samples, adjust ratio or check data folders")

        self.samples = samples
        self.split = split
        self.num_frames = num_frames
        self.frame_stride = max(1, frame_stride)
        self.transform = VideoTransform(mode="train" if split == "train" else "val", img_size=image_size)
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.samples)

    def _select_indices(self, total: int) -> torch.Tensor:
        if total <= 0:
            raise ValueError("Video folder has no frames")

        if total == 1:
            return torch.zeros(self.num_frames, dtype=torch.long)

        steps = max(self.num_frames*self.frame_stride, self.num_frames)
        grid = torch.linspace(0, total-1, steps=steps)
        ids = grid[:: self.frame_stride].long()

        if ids.numel() < self.num_frames:
            pad = ids.new_full((self.num_frames - ids.numel(),), ids[-1].item())
            ids = torch.cat([ids, pad], dim=0)

        return ids[:self.num_frames]

    @staticmethod
    def _base_video_name(name: str) -> str:
        match = re.match(r"(.+)_\d+$", name)
        return match.group(1) if match else name

    def __getitem__(self, id: int) -> Tuple[torch.Tensor, int]:
        frame_paths, label = self.samples[id]
        total = len(frame_paths)
        ids = self._select_indices(total)

        frames = []
        for i in ids:
            path = frame_paths[int(i.item())]
            with Image.open(path) as img:
                img = img.convert("RGB")
                frames.append(self.to_tensor(img))

        video = torch.stack(frames)
        video = self.transform(video)
        return video, label

def collate_fn(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    videos = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return videos, labels