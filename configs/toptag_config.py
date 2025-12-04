from __future__ import annotations
from dataclasses import dataclass, field
@dataclass
class Config:
    dataset:str = "top_quark"
    input_path: str = '/Users/leonid/Desktop/misc/Datasets/TopLandscape/'
    root_dir:str = './top_quark'
    train_test_ratio:float = 0.3
    loss_weight:str = 'mean'
    hl_features:bool = False
    ll_features:bool = True
    experiment_name = 'ML4Jet'
    model:str = 'Mixer' 
    moe:bool = True  # False
    train_files = ['train.h5',]
    test_files = ['val.h5']
    lr:float = 1e-4
    alpha:float = 1e-6
    regularization:float = 1e-6
    batch_size:int = 128
    optimizer = "adamw"
    scheduler = 'cosine_warmup_annealing'
    
    
@dataclass
class MixerConfig:
    
    nfeat_node:int=14
    rw_dim:int=0
    lap_dim:int=0
    patch_rw_dim:int=0
    dropout:float=0.0
    mlpmixer_dropout:float=0.0
    
    nfeat_edge:int=1
    nhid:int=64
    nlayer_gnn:int=0
    n_patches:int=32
    nout:int=2
    bias = False
    nlayer_mlpmixer:int=5
    token_dim:list[int]=field(default_factory=lambda: [128, 128, 64, 64, 64 ])
    channel_dim:list[int]=field(default_factory=lambda: [128, 128, 64, 64, 64 ])
    
@dataclass
class TrainingConfig:
    loss:str="focal_loss"
    gamma:int=0
    max_epoch:int=200
    seed:int=42
    

    
