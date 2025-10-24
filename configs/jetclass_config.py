from __future__ import annotations
from dataclasses import dataclass


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
                 'part_dzval', 'part_dzerr', 'part_charge',
                 'part_isChargedHadron', 'part_isNeutralHadron', 'part_isPhoton', 'part_isElectron', 'part_isMuon',
                 , 'jet_pt', 'jet_eta', 'jet_phi', 'jet_energy', 'jet_nparticles', 'jet_sdmass', 'jet_tau1', 'jet_tau2', 'jet_tau3', 'jet_tau4',
                 'aux_genpart_eta', 'aux_genpart_phi', 'aux_genpart_pid', 'aux_genpart_pt', 'aux_truth_match']

    labels = ['label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q',
              'label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl']
    train_files = ['JetClass_example_100k.root']
    test_files = ['JetClass_example_100k.root']
