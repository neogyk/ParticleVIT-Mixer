from __future__ import annotations
from dataclasses import dataclass, field
@dataclass
class Config:
    dataset:str = "quark_gluon"
    root_dir = "./QG/"
    path: str = '/Users/leonid/Desktop/misc/Datasets/QG/'
    train_test_ratio:float = 0.3
    loss_weight:str = 'mean'
    hl_features:bool = False
    ll_features:bool = True
    experiment_name = 'ML4Jet'
    model:str = 'Mixer' 
    moe:bool = True  # False
    rw_dim:int=4
    lap_dim:int=4
    n_patches:int = 16
    train_files = [
        'QG_jets_1.npz',
        'QG_jets_10.npz',
        'QG_jets_11.npz',
        'QG_jets_12.npz',
        'QG_jets_13.npz'
    ]

    test_files = ['QG_jets.npz']
    lr:float = 1e-4
    alpha:float = 1e-6
    regularization:float = 1e-6
    batch_size:int = 256
    optimizer = "adamw"
    
@dataclass
class MixerConfig:
    
    nfeat_node:int=21
    nfeat_edge:int=4
    #rw_dim:int=0
    #lap_dim:int=0
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
    

    