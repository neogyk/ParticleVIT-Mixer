from __future__ import annotations

import gc
import os
from glob import glob
import pdb
import awkward as ak
import numpy as np
import pandas as pd
import torch
import torch_geometric
import vector
from torch_geometric.data import Data
from torch_geometric.loader.dataloader import DataLoader
from graph_mixer import PositionalEncodingTransform
from graph_mixer.data_processing import GraphPartitionTransform

vector.register_awkward()


# warnings.filterwarnings("error", category=torch.UserWarning)
def _p4_from_ptetaphim(pt, eta, phi, mass):
    """ """
    return vector.zip({"pt": pt, "eta": eta, "phi": phi, "mass": mass})


class JetQGDataset(torch.utils.data.Dataset):
    """Jet Quark-Gluon Dataset

    :param torch:
    :type torch:
    """

    def __init__(
        self, config, train: bool, n_patche: int = 8
    ) -> torch.utils.data.Dataset:
        """_summary_

        Args:
            input_path (str): _description_
            train (bool, optional): _description_. Defaults to False.
        """
        # if the dataset doesn't exists:
        # zenodo_get RECORD_ID_OR_DOI
        self.input_path = config.path
        print("Input Path:", config.path)
        self.len = 0
        if train:
            files = config.train_files
        else:
            files = config.test_files

        self.data = [
            np.load(os.path.join(self.input_path, inp_path))["X"] for inp_path in files
        ]
        self.label = [
            np.load(os.path.join(self.input_path, inp_path))["y"] for inp_path in files
        ]
        self.len = sum([i.shape[0] for i in self.label])
        print(f"Init dataset with the shape {self.len}")
        # sorted_input_ids = sorted(zip(input_ids, train_targets), key=lambda x: len(x[0]))

        return

    # @numba.jit(nopython=True)
    def compute_obs(self, event):
        # compute the jet "mass" as an angularity with exponent 2
        # it's easier for the network to predict the log of the observable, shifted and scaled
        # obs = torch.log10(event[:,0]*(event[:,1:3]**2).sum()/event[:,:,0].sum()).detach().numpy()
        obs = np.log10(
            (event[:, 0] * (event[:, 1:3] ** 2).sum(1) / event[:, 0].sum()).sum(),
        )
        pt, eta, phi, mass = (
            event[:, 0],
            event[:, 1],
            event[:, 2],
            np.zeros_like(event[:, 0]),
        )

        p4 = _p4_from_ptetaphim(pt, eta, phi, mass)
        px, py, pz, energy = p4.x, p4.y, p4.z, p4.energy

        jet_p4 = ak.sum(p4)

        _jet_etasign = np.sign(eta)
        _jet_etasign[_jet_etasign == 0] = 1
        deta = (p4.eta - eta) * _jet_etasign
        dphi = p4.deltaphi(jet_p4)

        part_log_pt = np.log(p4.pt)
        part_e_log = np.log(energy)

        part_pt = p4.pt
        part_e = energy

        n_particles = event.shape[0]
        jet_pt = jet_p4.pt
        jet_eta = jet_p4.eta
        jet_phi = jet_p4.phi
        jet_energy = jet_p4.energy
        jet_mass = jet_p4.mass
        jet_nparticles = n_particles

        _jet_etasign = np.sign(eta)
        _jet_etasign[_jet_etasign == 0] = 1
        deta = (p4.eta - jet_eta) * _jet_etasign
        dphi = p4.deltaphi(jet_p4)

        PID = event[:, 3]
        part_pid = PID
        part_isCHPlus = ak.values_astype(
            (PID == 211) + (PID == 321) + (PID == 2212),
            "float32",
        )
        part_isCHMinus = ak.values_astype(
            (PID == -211) + (PID == -321) + (PID == -2212),
            "float32",
        )
        part_isNeutralHadron = ak.values_astype(
            (PID == 130) + (PID == 2112) + (PID == -2112),
            "float32",
        )
        part_isPhoton = ak.values_astype(PID == 22, "float32")
        part_isEPlus = ak.values_astype(PID == -11, "float32")
        part_isEMinus = ak.values_astype(PID == 11, "float32")
        part_isMuPlus = ak.values_astype(PID == -13, "float32")
        part_isMuMinus = ak.values_astype(PID == 13, "float32")

        part_isChargedHadron = part_isCHPlus + part_isCHMinus
        part_isElectron = part_isEPlus + part_isEMinus
        part_isMuon = part_isMuPlus + part_isMuMinus

        part_charge = (part_isCHPlus + part_isEPlus + part_isMuPlus) - (
            part_isCHMinus + part_isEMinus + part_isMuMinus
        )

        # 6
        cat_features = [
            part_isNeutralHadron.to_numpy(),
            part_isPhoton.to_numpy(),
            part_isChargedHadron.to_numpy(),
            part_isElectron.to_numpy(),
            part_isMuon.to_numpy(),
            part_charge.to_numpy(),
        ]

        # 15
        list_features = [
            pt,
            eta,
            phi,
            energy.to_numpy(),
            deta.to_numpy(),
            dphi.to_numpy(),
            px.to_numpy(),
            py.to_numpy(),
            pz.to_numpy(),
            part_log_pt,
            part_e_log,
            np.log(part_pt / jet_pt),
            np.log(part_e / jet_energy),
            np.sqrt(deta.to_numpy() ** 2 + dphi.to_numpy() ** 2),
            part_pid,
        ]

        # 7
        jet_features = [
            jet_eta,
            jet_phi,
            jet_pt,
            jet_mass,
            jet_energy,
            jet_nparticles,
            obs,
        ]
        return list_features, jet_features, cat_features

    # @numba.jit(nopython=True)
    def arrays2event(self, list_features, jet_features, cat_features):

        res = torch.stack([torch.Tensor(i) for i in list_features])
        jet_features = torch.Tensor(jet_features).view(1, 7)
        # torch.stack(
        #    [torch.full([jet_features[-2]], i) for i in jet_features],
        # )
        pos = torch.stack(
            [
                torch.Tensor(i)
                for i in [
                    list_features[0],
                    list_features[1],
                    list_features[2],
                ]
            ]
        ).T
        cat = torch.stack([torch.Tensor(i) for i in cat_features])
        res = torch.concatenate((res, cat)).T

        event = Data(
            x=res.to(dtype=torch.float).view(res.shape),
            pos=pos.to(dtype=torch.float).view(pos.shape),
        )
        event.jet_features = jet_features
        # Compute the Edge features:
        knn_graph = torch_geometric.transforms.knn_graph.KNNGraph(
            k=16,
            loop=True,
            num_workers=12,
        )
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

        pre_transform = PositionalEncodingTransform(rw_dim=4, lap_dim=4)
        gpatrches = GraphPartitionTransform(
            n_patches=8, metis=False, patch_rw_dim=4, num_hops=1
        )
        event = gpatrches(event)
        event = pre_transform(event)
        return event

    def preprocess(self, event):

        event = torch.Tensor(event[(event[:, 0] != 0), :]).detach().numpy()
        # yphi_avg = np.average(event[:,1:3], axis=0)
        # event[:,1:3] -= torch.Tensor(yphi_avg)
        # event[:,0] /= event[:,0].sum()

        return event

    def __len__(self) -> int:
        """_summary_

        Returns:
            _type_: _description_
        """

        return 10000  # self.len

    def __getitem__(self, idx: int):  # ->tuple(torch.Tensor, torch.Tensor):
        """_summary_

        Args:
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        if idx >= 100000:
            i_file = idx // 100000
            i = idx % 100000
        else:
            i_file = 0
            i = idx
        label = self.label[i_file][i]
        event = self.data[i_file][i]
        event = self.preprocess(event)
        list_features, jet_features, cat_features = self.compute_obs(event)
        event = self.arrays2event(list_features, jet_features, cat_features)
        label = torch.Tensor(np.array(label))
        event.label = label
        return event
