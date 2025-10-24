from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    path: str = '/Users/leonid/Desktop/misc/Datasets/JetNet/'
    train_test_ratio = 0.3
    n_epoch = 100
    accumulation_step = 6
    resume_training = False  # True
    do_test = True
    save_model = True
    seed = 3407
    loss_weight = 'mean'
    hl_features = False
    ll_features = True
    experiment_name = 'ML4Jet'
