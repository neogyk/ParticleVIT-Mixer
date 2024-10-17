import dataclasses
from dataclasses import dataclass

@dataclass
class Config():

    path:str = "/Users/leonid/Desktop/FusionAI/Jets/datasets/QG/*.npz"
    train_test_ratio = 0.3
    loss_weight = "mean"
    hl_features = False
    ll_features = True
    experiment_name = "ML4Jet"
