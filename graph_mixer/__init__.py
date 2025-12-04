from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from einops.layers.torch import Rearrange
from torch.nn.modules import activation
from torch_geometric.nn import global_add_pool
from torch_scatter import scatter
from pe import *
from soft_moe import SoftMoELayerWrapper
from graph_mixer.mamba import MambaSSM
from graph_mixer.hrm import HRM
from x_transformers import Encoder

torch.autograd.set_detect_anomaly(True)
import pdb
import torch

spectral_norm = False


class TeLU(nn.Module):
    def __init__(self):
        """
        Init method.
        https://arxiv.org/pdf/2402.02790
        """
        super().__init__()  # init the base class

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return input * torch.nn.functional.tanh(torch.exp(input))


class PositionalEncodingTransform:
    def __init__(self, rw_dim=0, lap_dim=0):
        super().__init__()
        self.rw_dim = rw_dim
        self.lap_dim = lap_dim

    def __call__(self, data):
        if self.rw_dim > 0:
            data.rw_pos_enc = RWSE(
                data.edge_index,
                self.rw_dim,
                data.num_nodes,
            )
        if self.lap_dim > 0:
            data.lap_pos_enc = LapPE(
                data.edge_index,
                self.lap_dim,
                data.num_nodes,
            )
        return data


class MLP(nn.Module):
    def __init__(
        self, nin, nout, nlayer=2, with_final_activation=True, with_norm=True, bias=True
    ):
        super().__init__()
        n_hid = nin
        self.layers = nn.ModuleList(
            [
                (
                    torch.nn.utils.spectral_norm(
                        nn.Linear(
                            nin if i == 0 else n_hid,
                            n_hid if i < nlayer - 1 else nout,
                            bias=bias,
                        )
                    )
                    if spectral_norm
                    else nn.Linear(
                        nin if i == 0 else n_hid,
                        n_hid if i < nlayer - 1 else nout,
                        bias=bias,
                    )
                )
                for i in range(nlayer)
            ]
        )

        self.norms = nn.ModuleList(
            [
                (
                    nn.RMSNorm(n_hid if i < nlayer - 1 else nout, eps=1e-8)
                    if with_norm
                    else Identity()
                )
                for i in range(nlayer)
            ]
        )

        self.activations = nn.ModuleList(
            [torch.nn.GELU(approximate="tanh") for i in range(nlayer)]
        )
        self.nlayer = nlayer
        self.with_final_activation = with_final_activation
        self.residual = nin == nout

    def reset_parameters(self):
        for layer, norm in zip(self.layers, self.norms):
            layer.reset_parameters()
            norm.reset_parameters()

    def forward(self, x):
        previous_x = x
        for i, (layer, norm, activation) in enumerate(
            zip(self.layers, self.norms, self.activations)
        ):
            x = layer(x)
            if i < self.nlayer - 1:
                x = norm(x)
            if not self.with_final_activation:
                pass
            else:
                x = activation(x)
        if self.residual:
            x = x + previous_x
        return x


class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0, is_mlp=True):

        super().__init__()
        self.is_mlp = is_mlp
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            (
                torch.nn.utils.spectral_norm(nn.Linear(in_dim, hidden_dim))
                if spectral_norm
                else nn.Linear(in_dim, hidden_dim)
            ),
            torch.nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            (
                torch.nn.utils.spectral_norm(nn.Linear(hidden_dim, out_dim))
                if spectral_norm
                else nn.Linear(hidden_dim, out_dim)
            ),
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):
    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.0):
        super().__init__()
        self.token_mix = nn.Sequential(
            nn.RMSNorm(dim, eps=1e-8),
            Rearrange("b p d -> b d p"),
            FeedForward(num_patch, token_dim, out_dim=num_patch),
            Rearrange("b d p -> b p d"),
        )
        self.channel_mix = nn.Sequential(
            nn.RMSNorm(dim, eps=1e-8),
            FeedForward(dim, channel_dim, out_dim=dim),
        )

    def forward(self, x):
        token_mixer = self.token_mix(x) + x
        channel_mixer = self.channel_mix(token_mixer) + token_mixer
        return channel_mixer


class MambaMixerBlock(nn.Module):
    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.0):
        super().__init__()
        self.token_mix = nn.Sequential(
            nn.RMSNorm(dim, eps=1e-8),
            Rearrange("b p d -> b d p"),
            MambaSSM(
                dim=num_patch,
                h_dim=token_dim,
                d_state=token_dim,
                d_time=4,
                kernel_size=2,
            ),
        )
        self.rearrange = Rearrange("b d p -> b p d")

        self.channel_mix = nn.Sequential(
            nn.RMSNorm(dim, eps=1e-8),
            MambaSSM(
                dim=dim, h_dim=channel_dim, d_state=token_dim, d_time=16, kernel_size=2
            ),
        )

    def forward(self, x):
        token_mixer, cache = self.token_mix(x)
        token_mixer = self.rearrange(token_mixer) + x
        channel_mixer, cache = self.channel_mix(token_mixer)

        return channel_mixer + token_mixer


class MLPMixer(nn.Module):
    def __init__(
        self,
        nhid,
        nlayer,
        n_patches,
        dropout=0,
        with_final_norm=False,
        token_dim=128,
        channel_dim=128,
    ):
        super().__init__()
        self.use_mixer = False
        self.token_dim = token_dim
        self.channel_dim = channel_dim
        self.n_patches: int = n_patches
        self.with_final_norm: bool = with_final_norm
        self.rms_norm_mixer_blocks = nn.ModuleList(
            [torch.nn.RMSNorm(nhid) for _ in range(nlayer)]
        )
        self.mixer_blocks = nn.ModuleList(
            [
                (
                    MixerBlock(
                        nhid,
                        self.n_patches,
                        token_dim=token_dim[_],
                        channel_dim=channel_dim[_],
                        dropout=dropout,
                    )
                    if self.use_mixer
                    else MambaMixerBlock(
                        nhid,
                        self.n_patches,
                        token_dim=token_dim[_],
                        channel_dim=channel_dim[_],
                        dropout=dropout,
                    )
                )
                for _ in range(nlayer)
            ],
        )
        if self.with_final_norm:
            self.layer_norm = nn.RMSNorm(nhid, eps=1e-8)
        self.moe = SoftMoELayerWrapper(
            dim=nhid,
            slots_per_expert=4,
            num_experts=8,
            layer=FeedForward,
            in_dim=nhid,
            hidden_dim=nhid,
            out_dim=nhid,
        )

    def forward(self, x):
        # cache = (None, torch.zeros(nhid, 32, 2 - 1))
        prev = x
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
            prev = x
            x += prev

        x = self.moe(x) + x
        if self.with_final_norm:
            x = self.layer_norm(x)
        return x


def LinearEncoder(nin, nhid, grad=False):
    lin_encoder = torch.nn.Sequential(
        nn.BatchNorm1d(nin),
        nn.LayerNorm(nin),
        nn.Linear(nin, nhid),
        nn.GELU(approximate="tanh"),
    )
    for param in lin_encoder.parameters():
        param.requires_grad = grad

    return lin_encoder


class GCNConv(nn.Module):
    def __init__(self, nin, bias=True):
        super().__init__()
        self.nn = MLP(nin, nin, 1, True, bias=bias)
        self.activation = torch.nn.ReLU()
        self.layer = gnn.CGConv(nin, nin, bias=bias, batch_norm=True, aggr="max")

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        x = self.layer(x, edge_index, edge_attr)
        x = self.activation(x)
        x = self.nn(x)
        return x


class GNN(nn.Module):
    def __init__(self, nin, nout, nlayer_gnn, dropout=0.5, res=True):
        super().__init__()
        self.dropout = dropout
        self.res = res
        self.convs = nn.ModuleList([GCNConv(nin, nin) for _ in range(nlayer_gnn)])
        self.norms = nn.ModuleList(
            [nn.RMSNorm(nin, eps=1e-8) for _ in range(nlayer_gnn)]
        )
        self.activations = nn.ModuleList([TeLU() for i in range(nlayer_gnn)])
        self.output_encoder = torch.nn.utils.spectral_norm(nn.Linear(nin, nout))

    def forward(self, x, edge_index, edge_attr):
        previous_x = x
        for layer, norm, activation in zip(self.convs, self.norms, self.activations):
            x = layer(x, edge_index, edge_attr)
            x = norm(x)
            x = activation(x)
            x = F.dropout(x, self.dropout, training=self.training)
            if self.res:
                x = x + previous_x
                previous_x = x
        x = self.output_encoder(x)
        return x


class GraphMLPMixer(nn.Module):
    def __init__(
        self,
        nfeat_node,
        nfeat_edge,
        nhid,
        nout,
        nlayer_gnn,
        nlayer_mlpmixer,
        rw_dim: int = 0,
        lap_dim: int = 0,
        dropout: float = 0.3,
        mlpmixer_dropout: float = 0.1,
        res: bool = True,
        pooling: str = "mean",
        n_patches: int = 4,
        patch_rw_dim: int = 4,
        token_dim: list[int] = None,
        channel_dim: list[int] = None,
        concat_ops: str = "mul",
    ):
        super().__init__()
        self.dropout = dropout
        self.use_rw = rw_dim > 0
        self.use_lap = lap_dim > 0
        self.pooling = pooling
        self.res = res
        self.patch_rw_dim = patch_rw_dim
        if self.use_rw:
            self.rw_encoder = MLP(rw_dim, nhid, 1)
        if self.use_lap:
            self.lap_encoder = MLP(lap_dim, nhid, 1)
        if self.patch_rw_dim > 0:
            self.patch_rw_encoder = MLP(self.patch_rw_dim, nhid, 1)
        self.concat_ops = concat_ops
        self.type = "mixer"
        self.input_encoder = LinearEncoder(nfeat_node, nhid, grad=True)
        self.edge_encoder = LinearEncoder(nfeat_edge, nhid, grad=True)
        # encoder_layer = nn.TransformerEncoderLayer(d_model=nhid, activation=torch.nn.functional.gelu, nhead=8)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        ################################################################################################
        #                       Encoder Modules:
        ################################################################################################
        self.gnns = nn.ModuleList(
            [
                GNN(
                    nin=nhid,
                    nout=nhid,
                    nlayer_gnn=1,
                    dropout=dropout,
                    res=res,
                )
                for _ in range(nlayer_gnn)
            ]
        )
        self.U = nn.ModuleList(
            [
                MLP(
                    nhid,
                    nhid,
                    nlayer=1,
                    with_final_activation=True,
                )
                for _ in range(nlayer_gnn - 1)
            ],
        )

        self.reshape = Rearrange("(B p) d ->  B p d", p=n_patches)
        ################################################################################################
        #                       Mixer Modules:
        ################################################################################################
        if self.type == "mixer":
            self.encoder = MLPMixer(
                nhid=nhid,
                dropout=mlpmixer_dropout,
                nlayer=nlayer_mlpmixer,
                n_patches=n_patches,
                token_dim=token_dim,
                channel_dim=channel_dim,
            )
        elif self.type == "mamba":
            print("Mamba")

        elif self.type == "hrm":
            self.encoder = HRM(
                networks=[
                    dict(
                        dim=64,
                        depth=3,
                        attn_dim_head=8,
                        heads=4,
                        use_rmsnorm=True,
                        rotary_pos_emb=True,
                        pre_norm=False,
                    ),
                    dict(
                        dim=64,
                        depth=4,
                        attn_dim_head=8,
                        heads=4,
                        use_rmsnorm=True,
                        rotary_pos_emb=True,
                        pre_norm=False,
                    ),
                    Encoder(
                        dim=64,
                        depth=4,
                        attn_dim_head=8,
                        heads=4,
                        use_rmsnorm=True,
                        rotary_pos_emb=True,
                        pre_norm=False,
                    ),
                ],
                causal=True,
                num_tokens=64,
                dim=64,
                reasoning_steps=4,
            )

        ################################################################################################
        #                       Decoder Modules:
        ################################################################################################
        if self.type == "hrm":
            self.output_decoder = torch.nn.Linear(nhid, nout)
        else:
            self.output_decoder = MLP(nhid, nout, nlayer=1, with_final_activation=False)

    def forward(self, data):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = self.input_encoder(data.x.squeeze())
        ################################################################################################
        #                       Node PE
        ################################################################################################
        if self.use_rw:
            x += self.rw_encoder(data.rw_pos_enc)
        if self.use_lap:
            x += self.lap_encoder(data.lap_pos_enc)
        edge_attr = data.edge_attr
        if edge_attr is None:
            edge_attr = data.edge_index.new_zeros(
                data.edge_index.size(-1),
                1,
            ).to(dtype=torch.float)
        edge_attr = self.edge_encoder(edge_attr)
        ################################################################################################
        #                      Patch Encoder
        ################################################################################################
        x = x[data.subgraphs_nodes_mapper]
        e = edge_attr[data.subgraphs_edges_mapper]
        edge_index = data.combined_subgraphs
        batch_x = data.subgraphs_batch
        for i, gnn in enumerate(self.gnns):
            if i > 0:
                subgraph = scatter(x, batch_x, dim=0, reduce=self.pooling)[batch_x]
                x = x + self.U[i - 1](subgraph)
                x = scatter(
                    x,
                    data.subgraphs_nodes_mapper,
                    dim=0,
                    reduce="mean",
                )[data.subgraphs_nodes_mapper]
            x = gnn(x, edge_index, e)

        subgraph_x = scatter(x, batch_x, dim=0, reduce=self.pooling)
        # Patch PE
        if self.patch_rw_dim > 0:
            subgraph_x += self.patch_rw_encoder(data.patch_pe)

        mixer_x = self.reshape(subgraph_x)

        # Encoder
        # mixer_x = self.transformer_encoder(mixer_x)
        mixer_x = self.encoder(mixer_x)
        # Global Average Pooling
        x = (mixer_x * data.mask.unsqueeze(-1)).sum(1) / data.mask.sum(1, keepdim=True)
        # Readout
        x = self.output_decoder(x)
        return x
