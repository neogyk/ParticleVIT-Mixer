from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass
class Config:

    path: str = '/Users/leonid/Desktop/FusionAI/JetTagger/Jets/datasets/QG/'
    # /Users/leonid/Desktop/FusionAI/Jets/datasets/QG/*.npz"
    train_test_ratio = 0.3
    loss_weight = 'mean'
    hl_features = False
    ll_features = True
    experiment_name = 'ML4Jet'
    train_files = [
        'QG_jets_withbc_3.npz',
        'QG_jets_13.npz',
        'QG_jets.npz',
        'QG_jets_12.npz',
        'QG_jets_withbc_2.npz',
        'QG_jets_withbc_0.npz',
        'QG_jets_10.npz',
        'QG_jets_11.npz',
        'QG_jets_withbc_1.npz',
        'QG_jets_8.npz',
    ]

    test_files = ['QG_jets_withbc_5.npz', 'QG_jets_15.npz', 'QG_jets_14.npz']
