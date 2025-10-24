from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass
class Config:

    path: str = '/Users/leonid/Desktop/misc/Datasets/QG/'
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
    train_files = [
        'QG_jets_withbc_10.npz',
        'QG_jets_13.npz',
        #        "QG_jets.npz",
        #        "QG_jets_12.npz",
        #        "QG_jets_withbc_2.npz",
        #        "QG_jets_withbc_0.npz",
        #        "QG_jets_10.npz",
        #        "QG_jets_11.npz",
        #        "QG_jets_withbc_1.npz",
        #        "QG_jets_8.npz",
    ]

    test_files = ['QG_jets_withbc_12.npz']
