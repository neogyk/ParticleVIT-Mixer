from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Config:
    path: str = '/Users/leonid/Desktop/Datasets/'
    train_test_ratio = 0.3
    loss_weight = 'mean'
    hl_features = False
    ll_features = True
    experiment_name = 'JetClass'
    coordinates = ['part_deta', 'part_dphi']
    variables = ['part_px', 'part_py', 'part_pz', 'part_energy',
                 'part_deta', 'part_dphi', 'part_d0val', 'part_d0err',
                 'part_dzval', 'part_dzerr', 'part_charge', 'jet_pt', 'jet_eta', 'jet_phi', 'jet_energy',
                 'jet_nparticles', 'jet_sdmass', 'jet_tau1', 'jet_tau2', 'jet_tau3', 'jet_tau4',
                 'aux_genpart_eta', 'aux_genpart_phi', 'aux_genpart_pid', 'aux_genpart_pt', 'aux_truth_match']
    
    categorical = ['part_isChargedHadron', 'part_isNeutralHadron', 'part_isPhoton', 'part_isElectron', 'part_isMuon']
    labels = ['label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q',
              'label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl']
    
    train_files = ['JetClass_example_100k.root']
    test_files = ['JetClass_example_100k.root']


    dataset:str = "jetclass"
    path: str = '/Users/leonid/Desktop/misc/Datasets/JetClass/'
    train_test_ratio:float = 0.3
    loss_weight:str = 'mean'
    hl_features:bool = False
    ll_features:bool = True
    experiment_name = 'ML4Jet'
    model:str = 'Mixer' 
    moe:bool = True  # False
    rw_dim:int=4
    lap_dim:int=4
    n_patches:int = 8
    train_files = ['JetClass_example_100k.root']
    test_files = []
    lr:float = 1e-4
    alpha:float = 1e-6
    regularization:float = 1e-6
    batch_size:int = 256
    optimizer = "adamw"
    
@dataclass
class MixerConfig:
    
    nfeat_node:int=21
    nfeat_edge:int=4
    dropout:float=0.0
    mlpmixer_dropout:float=0.0
    patch_rw_dim:int=0
    nhid:int=64
    nlayer_gnn:int=0
    n_patches:int=8
    nout:int=2
    rw_dim:int=4
    lap_dim:int=4
    nlayer_mlpmixer:int=8
    token_dim:list[int]=field(default_factory=lambda: [128, 128, 64, 64, 64, 64, 64, 64])
    channel_dim:list[int]=field(default_factory=lambda: [512, 512, 256, 256, 128, 64, 64, 64])
    
@dataclass
class TrainingConfig:
    loss:str="focal_loss"
    gamma:int=0
    max_epoch:int=200
    seed:int=42
    


    
