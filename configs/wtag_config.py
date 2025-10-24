from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Config:

    path: str = '/Users/leonid/Desktop/misc/Datasets/W_dataset/W_images'
    train_test_ratio = 0.3
    loss_weight = 'mean'
    hl_features = False
    ll_features = True
    experiment_name = 'ML4W'
    train_files: list[str] = ['train.npy']
    test_files: list[str] = ['val.npy']
