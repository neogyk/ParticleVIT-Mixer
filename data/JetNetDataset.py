# import pdb
from __future__ import annotations
import gc
from glob import glob
import h5py
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader.dataloader import DataLoader
import random
from graph_mixer import PositionalEncodingTransform
from graph_mixer.data_processing import GraphPartitionTransform, RicciFlowPartition


class JetNetDataset(torch.utils.data.Dataset):

    def __init__(
        self, input_path: str, train: bool = False
    ) -> torch.utils.data.Dataset:
        """JetNet Dataset downloaded from:


        Args:
            input_path (str): path to directory, which contains data
            train (bool, optional): Flag which indicates, whether dataset will be used for training or testing. Defaults to False.
        #Features:

        """
        self.input_path = input_path
        print(self.input_path)

        labels = []
        jet_features = []  # [pt, theta, mass, number of particles]
        # 4 is the number of particle features, in order: ['theta','phi, 'pt']
        particle_features = []

        label_map = {
            "q": 0,
            "w": 1,
            "g": 2,
            "t": 3,
            "z": 4,
        }

        labels = []
        for file in glob(self.input_path + "*.hdf5"):
            label_size = torch.Tensor(
                h5py.File(file)["particle_features"].__array__().shape[0],
            )
            jet_features.append(h5py.File(file)["jet_features"].__array__())
            particle_features.append(
                h5py.File(file)["particle_features"].__array__(),
            )
            label = file.split("/")[-1].strip(".hdf5")
            labels.append(torch.full_like(label_size, label_map[label]))

        self.label = torch.concatenate(labels)
        self.label = self.label  # [idx]
        self.jet_features = np.concatenate(jet_features)  # [idx]
        self.jet_features = torch.Tensor(self.jet_features)
        self.particle_features = torch.Tensor(
            np.concatenate(particle_features),
        )  # [idx]
        # self.normalize = NormalizeFeatures()
        _shape = [
            torch.Tensor(
                self.particle_features[idx, :][
                    (self.particle_features[idx, :][:, 0] != 0), :
                ]
            )[:, :3].size()[0]
            for idx in range(self.particle_features.shape[0] - 1)
        ]
        input_ids = list(range(self.label.size()[0]))
        self.sorted_input_ids = sorted(zip(input_ids, _shape), key=lambda x: x[1])

    def __len__(self) -> int:
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.particle_features.shape[0]

    def __getitem__(self, idx: int):  # ->tuple(torch.Tensor, torch.Tensor):
        """_summary_

        Args:
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        # idx = self.sorted_input_ids[idx][0]
        idx = random.randint(0, self.particle_features.shape[0] - 1)
        pos = self.particle_features[idx, :]
        pos = torch.Tensor(pos[(pos[:, 0] != 0), :])[:, :3]
        n_particle = pos.size()[0]
        jet_pt = torch.log(torch.full([n_particle], self.jet_features[idx][0])).view(
            -1, 1
        )

        j_features = torch.stack(
            [torch.full([n_particle], i) for i in self.jet_features[idx][1:]]
        ).T
        # j_features = torch.stack([jet_pt, j_features])
        dR = torch.sqrt((pos[:, 0]) ** 2 + (pos[:, 1]) ** 2).view(-1, 1)
        # cos_eta = torch.cos(pos[:, 0])
        event = Data(
            pos=pos.view(pos.shape),
            label=self.label[idx],
        )
        event.x = torch.concatenate((event.pos, dR, jet_pt, j_features), dim=1)
        knn_graph = torch_geometric.transforms.knn_graph.KNNGraph(
            k=8, loop=True, num_workers=12
        )
        event = knn_graph(event)

        edge_index = event.edge_index
        dR = torch.sqrt(
            (event.x[:, 0][edge_index[0]]) ** 2 + (event.x[:, 1][edge_index[1]]) ** 2,
        )
        k_T = (
            torch.min(
                event.x[:, 2][edge_index[0]],
                event.x[:, 2][edge_index[1]],
            )
            * dR
        )
        z = torch.min(event.x[:, 2][edge_index[0]], event.x[:, 2][edge_index[1]]) / (
            event.x[:, 2][edge_index[0]] + event.x[:, 2][edge_index[1]]
        )
        event.edge_attr = torch.stack([dR, k_T, z]).T
        pre_transform = PositionalEncodingTransform(rw_dim=4, lap_dim=6)
        gpatrches = GraphPartitionTransform(n_patches=4, metis=False, patch_rw_dim=4)
        # ricchie_patcher = RicciFlowPartition()
        # event = ricchie_patcher(event.edge_index,event.batch, event.x)
        event = gpatrches(event)
        event = pre_transform(event)
        return event
