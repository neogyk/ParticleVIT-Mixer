from __future__ import annotations
import os
import awkward as ak
import torch
import torch_geometric
import vector
from torch_geometric.data import Data
from torch_geometric.loader.dataloader import DataLoader
import pdb
import numpy as np
import pandas as pd

vector.register_awkward()

# Reconstruction  features
variables = [
    "reco_cand_p4s",
    "reco_jet_p4s",
    "reco_cand_charge",
    "reco_cand_pdg",
    "reco_cand_dxy",
    "reco_cand_dz",
    "reco_cand_dxy_err",
    "reco_cand_dz_err",
]

# Generator Features
labels = [
    "gen_jet_p4s",
    "gen_jet_tau_decaymode",
    "gen_jet_tau_p4s",
]


def _p4_from_ptetaphim(pt, eta, phi, mass):
    """_summary_

    Args:
        pt (_type_): _description_
        eta (_type_): _description_
        phi (_type_): _description_
        mass (_type_): _description_

    Returns:
        _type_: _description_
    """
    return vector.zip({"pt": pt, "eta": eta, "phi": phi, "mass": mass})


def to_p4(p4_obj):
    """_summary_

    Args:
        p4_obj (_type_): _description_

    Returns:
        _type_: _description_
    """
    return vector.awk(
        ak.zip(
            {
                "mass": p4_obj["tau"],
                "x": p4_obj["x"],
                "y": p4_obj["y"],
                "z": p4_obj["z"],
            },
        ),
    )


class JetTauIdDataset(torch.utils.data.Dataset):

    def __init__(self, input_path: str, n_rows: 10000) -> torch.utils.data.Dataset:
        """Tau Id dataset:
        #['reco_cand_p4s',
        # 'reco_cand_charge',
        # 'reco_cand_pdg',
        # 'reco_jet_p4s',
        # 'reco_cand_dxy',
        # 'reco_cand_dz',
        # 'reco_cand_dxy_err',
        # 'reco_cand_dz_err',
        # 'gen_jet_p4s',
        # 'gen_jet_tau_decaymode',
        # 'gen_jet_tau_p4s']

        #Link to the dataset:

        Args:
            input_path (str): _description_
            train (bool, optional): _description_. Defaults to False.

        """
        mode: str = "train"
        self.input_path: str = input_path

        self.processes: list = [
            f"z_{mode}.parquet",
            f"zh_{mode}.parquet",
            f"qq_{mode}.parquet",
        ]

        zh_path: str = os.path.join(input_path, "zh_train.parquet")
        z_path: str = os.path.join(input_path, "z_train.parquet")
        qq_path: str = os.path.join(input_path, "qq_train.parquet")
        self.n_rows = n_rows
        pdb.set_trace()
        self.zh_data = pd.read_parquet(
            zh_path
        )  # ak.to_arrow_table(ak.from_parquet(zh_path)).to_pandas()
        self.z_data = pd.read_parquet(
            z_path
        )  # ak.to_arrow_table(ak.from_parquet(z_path)).to_pandas()
        self.qq_data = pd.read_parquet(
            qq_path
        )  # ak.to_arrow_table(ak.from_parquet(z_path)).to_pandas()
        self.len: int = self.zh_data.shape[0]
        print(f"Init dataset with the shape {self.len}")

    def compute_obs(self, event):

        res = torch.stack([torch.Tensor(i) for i in list_features])
        jet_features = torch.stack(
            [torch.full([jet_nparticles], i) for i in jet_features],
        )
        pos = torch.stack([torch.Tensor(i) for i in [pt, deta, dphi]]).T
        cat = torch.stack([torch.Tensor(i) for i in cat_features])
        res = torch.concatenate((res, cat, jet_features)).T

        event = Data(
            x=res.to(dtype=torch.double).view(res.shape),
            pos=pos.to(dtype=torch.double).view(pos.shape),
        )
        # Compute the Edge features:
        knn_graph = torch_geometric.transforms.knn_graph.KNNGraph(
            k=8,
            loop=True,
            num_workers=12,
        )  # 4)
        event = knn_graph(event)

        edge_index = event.edge_index
        res = res.T

        dR = torch.sqrt(
            (res[1][edge_index[0]]) ** 2 + (res[2][edge_index[1]]) ** 2,
        )
        k_T = torch.min(res[0][edge_index[0]], res[0][edge_index[1]]) * dR
        z = torch.min(res[0][edge_index[0]], res[0][edge_index[1]]) / (
            res[0][edge_index[0]] + res[0][edge_index[1]]
        )
        m_2 = -torch.abs(res[0][edge_index[0]] - res[0][edge_index[1]]) ** 2 + (
            res[3][edge_index[0]] + res[3][edge_index[1]]
        )

        edge_data = torch.stack([dR, k_T, z, m_2])

        event.edge_attr = edge_data.T

        # pre_transform = PositionalEncodingTransform(rw_dim=0, lap_dim=6)
        # gpatrches = GraphPartitionTransform(n_patches=32, metis=False, patch_rw_dim=4)
        # event = gpatrches(event)
        # event = pre_transform(event)

        return event

    def __len__(self) -> int:
        """_summary_

        Returns:
            _type_: _description_
        """

        return self.len

    def __getitem__(self, idx: int):  # ->tuple(torch.Tensor, torch.Tensor):
        """_summary_

        Args:
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        event = self.zh_data.take([idx]).to_dict()
        dm = event["gen_jet_tau_decaymode"]
        gen_tau = event["gen_jet_tau_p4s"]
        reco_tau = event["reco_cand_p4s"]  # [0]
        charge = event["reco_cand_charge"][0]
        pdg_id = event["reco_cand_pdg"][0]
        reco_jet = event["reco_jet_p4s"][0]
        reco_cand_dxy = event["reco_cand_dxy"][0]
        reco_cand_dz = event["reco_cand_dz"][0]
        reco_cand_dz_err = event["reco_cand_dz_err"][0]
        reco_cand_dxy_err = event["reco_cand_dxy_err"][0]
        num_candidates: int = len(charge)
        # reco_cand_p4s = np.array(list(map(to_p4,df.loc[0]['reco_cand_p4s'])
        j_features = torch.stack(
            [
                torch.full([num_candidates], i)
                for i in [
                    num_candidates,
                    vector.zip(reco_jet).eta,
                    vector.zip(reco_jet).phi,
                ]
            ],
        ).T
        particle_features = torch.Tensor(
            [
                pdg_id,
                reco_particle_featurescand_dxy,
                reco_cand_dz,
                reco_cand_dz_err,
                reco_cand_dxy_err,
            ]
        )
        _particle_pos = []
        for i_candidate in range(num_candidates):
            pdb.set_trace()
            # convert tot the pt, eta, phi, mass
            eta, phi = (
                vector.zip(reco_tau[i_candidate]).eta,
                vector.zip(reco_tau[i_candidate]).phi,
            )
            x, y, z, tau = vector.zip(reco_tau[i_candidate])
            _particle_pos.append([eta, phi, x, y, z, tau])
        # Extract the Weights
        event = Data(x=particle_features, pos=_particle_pos)
        knn_graph = torch_geometric.transforms.knn_graph.KNNGraph(k=18, loop=True)
        # gpatrches = GraphPartitionTransform(n_patches=8, metis=False)
        event = knn_graph(event)
        event.label = torch.Tensor([label])
        # Create the event -> fill the jet features
        # Add the position
        return event, event.label


if __name__ == "__main__":
    # define the config file
    input_path = "/Users/leonid/Desktop/misc/Datasets/TauId/"
    dataset = JetTauIdDataset(input_path)
    dl = DataLoader(dataset, batch_size=32)
    for event, label in dl:
        break
