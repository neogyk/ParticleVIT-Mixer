from __future__ import annotations

import os
import awkward as ak
import numpy as np
import torch
import torch_geometric
import vector
from numba import jit
from torch_geometric.data import Data
from graph_mixer import PositionalEncodingTransform
from graph_mixer.data_processing import GraphPartitionTransform

vector.register_awkward()

# warnings.filterwarnings("error", category=torch.UserWarning)


def _p4_from_ptetaphim(pt, eta, phi, mass):
    """ """
    return vector.zip({"pt": pt, "eta": eta, "phi": phi, "mass": mass})


class JetQGDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        input_path: str,
        train: bool,
        config,
    ) -> torch.utils.data.Dataset:
        """_summary_

        Args:
            input_path (str): _description_
            train (bool, optional): _description_. Defaults to False.
        """
        # if the dataset doesn't exists:
        # zenodo_get RECORD_ID_OR_DOI

        self.input_path = input_path
        self.len = 0
        if train:
            files = config.train_files
        else:
            files = config.test_files
        print("Input Path:", input_path)

        self.data = [
            np.load(os.path.join(self.input_path, inp_path))["X"] for inp_path in files
        ]
        self.label = [
            np.load(os.path.join(self.input_path, inp_path))["y"] for inp_path in files
        ]
        self.len = sum([i.shape[0] for i in self.label])
        print(f"Init dataset with the shape {self.len}")
        return

    @jit
    def compute_obs(self, event):
        # compute the jet "mass" as an angularity with exponent 2
        # it's easier for the network to predict the log of the observable, shifted and scaled
        event = event.detach().numpy()
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
        res = torch.stack([torch.Tensor(i) for i in list_features])
        jet_features = torch.stack(
            [torch.full([jet_nparticles], i) for i in jet_features],
        )
        pos = torch.stack([torch.Tensor(i) for i in [pt, deta, dphi]]).T
        cat = torch.stack([torch.Tensor(i) for i in cat_features])
        res = torch.concatenate((res, cat, jet_features)).T

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

        return pos, res, edge_data

    def convert2torch_geometric(self, res, pos, edge_data):

        event = Data(
            x=res.to(dtype=torch.float).view(res.shape),
            pos=pos.to(dtype=torch.float).view(pos.shape),
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

        event.edge_attr = edge_data.T

        pre_transform = PositionalEncodingTransform(rw_dim=0, lap_dim=6)
        gpatrches = GraphPartitionTransform(
            n_patches=32,
            metis=False,
            patch_rw_dim=4,
        )
        event = gpatrches(event)
        event = pre_transform(event)

        return event

    def preprocess(self, event):

        event = torch.Tensor(event[(event[:, 0] != 0), :])

        # pdb.set_trace()

        # yphi_avg = np.average(event[:,1:3], axis=0)
        # event[:,1:3] -= torch.Tensor(yphi_avg)
        # event[:,0] /= event[:,0].sum()

        return event

    def __len__(self) -> int:
        """_summary_

        Returns:
            _type_: _description_
        """

        return self.len

    def add_prediction(self, tensor: np.array, store=True):
        """_summary_

        Args:
            tensor (np.array): _description_
            store (bool, optional): _description_. Defaults to True.
        """
        # self.file.close()
        # file = h5py.File(self.input_path, mode='a')
        # file.create_dataset('prediction', data=tensor)
        # file.close()
        return

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
        res, pos, edge_data = self.compute_obs(event)
        event = self.convert2torch_geometric(res, pos, edge_data)
        label = torch.Tensor(np.array(label))
        event.label = label
        return event, label
