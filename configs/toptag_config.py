from __future__ import annotations
from dataclasses import dataclass
@dataclass
class Config:
    path: str = '/Users/leonid/Desktop/misc/Datasets/TopLandscape/'
    train_test_ratio:float = 0.3
    loss_weight:str = 'mean'
    hl_features:bool = False
    ll_features:bool = True
    experiment_name = 'ML4Jet'
    model:str = 'Mixer' 
    moe:bool = True  # False
    train_files = ['train.h5',]
    test_files = ['train_file.parquet']
    lr:float = 1e-4
    alpha:float = 1e-6
    regularization:float = 1e-6
    batch_size:int = 256

    