import torch
import numpy as np
import torch_geometric
from torch_geometric.data import Data
import polars as pl
import tqdm
from graph_vit.data_processing import GraphPartitionTransform
from graph_vit import PositionalEncodingTransform
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import augmentation


class TopQuarkDataset(torch.utils.data.Dataset):

    def __init__(self, input_path, train=False):
        """ """
        self.input_path = input_path
        df = pl.read_parquet(input_path)
        self.label = df["label"].to_torch()
        self.index = np.random.permutation(np.arange(self.label.size()[0]))
        self.label = self.label[self.index]
        self.len = self.label.size()[0]

        jet_variables = [
            "jet_pt",
            "jet_eta",
            "jet_phi",
            "jet_energy",
            "jet_mass",
            "jet_nparticles",
        ]
        part_variales = ["part_energy", "part_deta", "part_dphi"]

        self.part_variables = df[part_variales].to_numpy()[self.index]

        self.jet_df = df[jet_variables].to_torch()[self.index]

        # self.scaler = StandardScaler()
        # self.jet_df = self.scaler.fit_transform(self.jet_df)#.to_torch()
        # self.part_variables = self.scaler.fit_transform(self.part_variables)

        self.train = train

        # if self.train:
        #    self.idx = torch.randperm(self.label.shape[0])
        #    self.label = self.label[self.idx]
        #    self.data = torch.Tensor(self.data)[self.idx]
        # else:
        #    self.label = self.label
        #    self.data = torch.Tensor(self.data)

    def edge_features(self, event):
        return

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        part_tensor = torch.stack(
            [torch.Tensor(self.part_variables[idx][i]) for i in range(0, 3)]
        ).T

        # part_pt = torch.Tensor(self.part_variables[idx][0])).view(part_tensor.size()[0],1)
        jet_nparticles = part_tensor.size()[0]

        jet_tensor = torch.stack(
            [torch.full([jet_nparticles], i) for i in self.jet_df[idx]]
        ).T
        x = torch.concatenate([jet_tensor, part_tensor], axis=-1)
        event = Data(x=x.double(), pos=part_tensor.double(), label=self.label[idx])

        event.x, perm = augmentation.shuffle_node(event.x, event.batch)
        # event.x = augmentation.mask_feature(event.x)

        label = self.label[idx]
        knn_graph = torch_geometric.transforms.knn_graph.KNNGraph(k=8)
        pre_transform = PositionalEncodingTransform(rw_dim=0, lap_dim=6)
        gpatrches = GraphPartitionTransform(n_patches=3, metis=False, patch_rw_dim=4)
        event = knn_graph(event)
        event = pre_transform(event)
        event = gpatrches(event)

        return event, label
