"""
Dataset utilities for training sign recognition.

Features:
- Unzip dataset archive to a workspace folder
- Provide a PyTorch Dataset that extracts features using MediaPipeExtractor
"""

import os
import zipfile
import json
from typing import List, Tuple
from pathlib import Path

import torch
from torch.utils.data import Dataset
import numpy as np

from .video_helpers import list_videos_in_dir
from ..models.mediapipe_extractor import MediaPipeExtractor


def extract_zip(zip_path: str, dest_dir: str) -> str:
    """Extract `zip_path` into `dest_dir` and return the extraction root."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dest)
    return str(dest)


class SignVideoDataset(Dataset):
    """PyTorch Dataset that lazily extracts MediaPipe features for videos.

    Expects dataset layout like:
      root/<label>/*.mp4
    """

    def __init__(self, root_dir: str, max_frames: int = 64, cache_features: bool = False):
        self.root_dir = root_dir
        self.max_frames = max_frames
        self.cache_features = cache_features

        # Build list of (video_path, label)
        self.samples: List[Tuple[str, str]] = []
        for label in sorted(os.listdir(root_dir)):
            label_dir = os.path.join(root_dir, label)
            if not os.path.isdir(label_dir):
                continue
            videos = list_videos_in_dir(label_dir)
            for v in videos:
                self.samples.append((v, label))

        self.labels = sorted({lbl for _, lbl in self.samples})
        self.label_to_idx = {l: i for i, l in enumerate(self.labels)}

        # extractor is not pickle-friendly; create when needed
        self.extractor = None
        self._feature_cache = {} if cache_features else None

    def _ensure_extractor(self):
        if self.extractor is None:
            self.extractor = MediaPipeExtractor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        if self.cache_features and video_path in self._feature_cache:
            features = self._feature_cache[video_path]
        else:
            self._ensure_extractor()
            features = self.extractor.extract_video(video_path, self.max_frames)
            if self.cache_features:
                self._feature_cache[video_path] = features

        # convert to torch tensor
        x = torch.from_numpy(features).float()
        y = torch.tensor(self.label_to_idx[label], dtype=torch.long)
        return x, y

    def close(self):
        if self.extractor is not None:
            self.extractor.close()


def write_metadata(out_path: str, labels: List[str], max_frames: int, landmark_dim: int):
    data = {
        'idx_to_label': {i: l for i, l in enumerate(labels)},
        'num_classes': len(labels),
        'max_frames': max_frames,
        'landmark_dim': landmark_dim
    }
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)
