from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class Config:
    dataset:str = "jetnet"
    root_dir = "./JetNet/"
    path: str = '/Users/leonid/Desktop/misc/Datasets/JetNet/'
    moe:bool = True  # False
    rw_dim:int=4
    lap_dim:int=4
    n_patches:int = 16
    lr:float = 1e-4
    alpha:float = 1e-6
    regularization:float = 1e-6
    batch_size:int = 256
    optimizer = "adamw"
    train_test_ratio = 0.3
    n_epoch = 100
    accumulation_step = 6
    resume_training = False  # True
    do_test = True
    save_model = True
    seed = 3407
    loss_weight = 'mean'
    patch_rw_dim:int=4
    n_patches:int=16
    rw_dim:int=4
    lap_dim:int=4

    
@dataclass
class MixerConfig:
    
    nfeat_node:int=8
    nfeat_edge:int=3
    dropout:float=0.0
    mlpmixer_dropout:float=0.0
    patch_rw_dim:int=0
    nhid:int=64
    nlayer_gnn:int=0
    n_patches:int=16
    nout:int=2
    rw_dim:int=4
    lap_dim:int=4
    nlayer_mlpmixer:int=8
    token_dim:list[int]=field(default_factory=lambda: [512, 512, 256, 64, 64, 64, 64, 64])
    channel_dim:list[int]=field(default_factory=lambda: [512, 512, 256, 256, 128, 64, 64, 64])
    
@dataclass
class TrainingConfig:
    loss:str
    gamma:int
    max_epoch:200
    seed:0
    

    