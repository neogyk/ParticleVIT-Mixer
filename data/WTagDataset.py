from __future__ import annotations
import pdb
import torch
import awkward as ak
import numpy as np
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader.dataloader import DataLoader

# from graph_vit.data_processing import GraphPartitionTransform

npoints: int = 40  # npoint=(Number of pixels+1) of the image
N_pixels: int = np.power(npoints, 2)

# input image dimensions
img_rows, img_cols = npoints, npoints
ncolors: int = 1


def expand_array(image):

    image = image.reshape(110, 3)

    expandedimage = np.zeros((img_rows, img_cols, ncolors), dtype="float32")

    expandedimage[
        image[:, 0].astype("int"),
        image[:, 1].astype(
            "int",
        ),
    ] = image[
        :, 2
    ].reshape((-1, ncolors))
    expandedimage = np.transpose(expandedimage, (2, 0, 1))

    return expandedimage


class WTagDataset(torch.utils.data.Dataset):

    def __init__(self, input_path: str) -> torch.utils.data.Dataset:
        """_summary_

        Args:
            input_path (str): _description_
            train (bool, optional): _description_. Defaults to False.
        """
        self.representation: str = "image"  # image
        self.input_path: str = input_path

        self.mode: str = "train"
        self.len: int = 0
        self.data: np.array = np.load(self.input_path + f"/{self.mode}.npy")
        # self.hlf_data:np.array = np.load(self.input_path + f"/highlevel_features.npy")
        self.X_train = np.apply_along_axis(expand_array, 1, self.data[:, :330])
        self.y_train: np.array = self.data[:, 330]
        self.m_train: np.array = self.data[:, 331]
        self.pt_train: np.array = self.data[:, 332]
        self.len: int = self.data.shape[0]
        print(f"Init dataset with the shape {self.len}")
        return

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
        # gpatrches = GraphPartitionTransform(
        #    n_patches=32, metis=False, patch_rw_dim=4,
        # )
        # event = gpatrches(event)
        # event = pre_transform(event)

        return event

    def rho(self, m_j, p_j):
        return torch.log(self.m_j**2, self.p_j**2)

    def dR(self, phi, eta):
        dr = torch.sqrt(phi**2 + eta**2)
        return dr

    def pT(self, p_X, p_Y):
        return torch.log(torch.sqrt(self.p_X**2 + self.p_Y**2))

    def preprocess(self, event):
        event = torch.Tensor(event[(event[:, 0] != 0), :])
        return event

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int):  # ->tuple(torch.Tensor, torch.Tensor):

        x_train = torch.Tensor(self.X_train[idx].reshape(1, -1))
        y_train = self.y_train[idx]
        m_train = self.m_train[idx]
        pt_train = self.pt_train[idx]
        event = Data(
            x=x_train[(x_train[:] != 0)],
            pos=x_train[(x_train[:] != 0)],
            label=torch.Tensor([y_train]),
        )
        knn_graph = torch_geometric.transforms.knn_graph.KNNGraph(k=18, loop=True)
        # gpatrches = GraphPartitionTransform(n_patches=8, metis=False)
        event = knn_graph(event)
        # event = gpatrches(event)
        return event


if __name__ == "__main__":
    input_path = "/Users/leonid/Desktop/misc/Datasets/W_dataset/W_images/"
    wtag_ds = WTagDataset(input_path=input_path)
    for data in iter(wtag_ds):
        print(data)
        pdb.set_trace()
