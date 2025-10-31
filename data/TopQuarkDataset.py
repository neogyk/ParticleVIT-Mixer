from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader.dataloader import DataLoader
from graph_mixer.data_processing import GraphPartitionTransform
from torch_geometric.utils import random
import pdb

jet_vars = ["jet_pt", "jet_eta", "jet_phi", "jet_energy", "jet_mass", "jet_nparticles"]
part_vars = [
    "part_px",
    "part_py",
    "part_pz",
    "part_energy",
    "part_deta",
    "part_dphi",
]
pos_var = ["part_deta", "part_dphi"]
num_particle_var = "jet_nparticles"
label = "label"


# graph partitioning: https://github.com/JosephDoUrden/Graph-Partitioning-Algorithms-Comparative-Study
class JetTopTagDataset(torch.utils.data.Dataset):
    def __init__(self, input_path, scale=True, mode="train", apply_norm=False):
        self.input_path = input_path
        self.mode = mode
        self.apply_norm = apply_norm
        if mode == "train":
            self.file = pd.read_parquet(
                f"{self.input_path}/train_file.parquet",
            )
        elif mode == "test":
            self.file = pd.read_parquet(f"{self.input_path}/test_file.parquet")

        self.label = self.file[label]
        if mode == "train":
            wrong_idx = 456856
            self.file.drop(wrong_idx)
        idx = torch.randperm(self.file.shape[0]).numpy()
        self.x = self.file[part_vars].loc[idx]
        self.pos = self.file[pos_var].loc[idx]
        self.num_particle = self.file[num_particle_var].loc[idx]
        self.label = self.label.loc[idx]
        self.jet_vars = self.file[jet_vars].loc[idx]
        if self.apply_norm:
            self.rot_normalization = torch_geometric.transforms.NormalizeRotation()
            self.scale_norm = torch_geometric.transforms.NormalizeScale()
        del self.file
        print(
            f"Initialized the dataset with the dimentionality: {self.label.shape[0]}",
        )

    def __len__(self):
        """ """
        if self.mode == "train":
            return self.label.shape[0]
        else:
            return self.label.shape[0]

    def __getitem__(self, idx):
        """ """
        num_particle = self.num_particle[idx]
        event = (
            torch.Tensor(np.concatenate(self.x.loc[idx].values))
            .view(
                num_particle,
                len(part_vars),
            )
            .to(dtype=torch.float)
        )
        pos = (
            torch.Tensor(np.concatenate(self.pos.loc[idx].values))
            .view(
                num_particle,
                len(pos_var),
            )
            .to(dtype=torch.float)
        )

        jet = torch.Tensor(self.jet_vars.loc[idx].values)
        j_features = torch.stack(
            [torch.full([num_particle], i) for i in jet],
         ).T
        # j_features[:, 0] = torch.log(j_features[0])
        # j_features[:, 3] = torch.log(j_features[3])
        dR = torch.sqrt(event[:, 4] ** 2 + event[:, 5] ** 2)
        pt = torch.sqrt(event[:, 0] ** 2 + event[:, 1] ** 2)
        
        event = Data(
            x=torch.hstack(
                [event, dR.view(-1, 1), pt.view(-1, 1), j_features],
            ).view(num_particle, len(jet_vars)+len(part_vars) + 2),
            pos=pos,
            label=torch.Tensor([self.label[idx]])
        )
        # event.jet_features = jet.view(1, 6)
        # event.jet_features[:, 0] = torch.log(jet[0])
        # event.jet_features[:, 3] = torch.log(jet[3])
        knn_graph = torch_geometric.transforms.knn_graph.KNNGraph(k=8, loop=True)
        gpatrches = GraphPartitionTransform(n_patches=8, metis=False)
        # apply positional encoding
        event = knn_graph(event)
        event = gpatrches(event)
        return event
