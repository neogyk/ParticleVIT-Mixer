import torch
from torch import nn
import torch_geometric.nn as gnn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from einops.layers.torch import Rearrange
from torch_scatter import scatter
from pe import *

BN = True


class PositionalEncodingTransform(object):
    def __init__(self, rw_dim=0, lap_dim=0):
        super().__init__()
        self.rw_dim = rw_dim
        self.lap_dim = lap_dim

    def __call__(self, data):
        if self.rw_dim > 0:
            data.rw_pos_enc = RWSE(
                data.edge_index, self.rw_dim, data.num_nodes)
        if self.lap_dim > 0:
            data.lap_pos_enc = LapPE(
                data.edge_index, self.lap_dim, data.num_nodes)
        return data

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

    def reset_parameters(self):
        pass

class MLP(nn.Module):
    def __init__(self, nin, nout, nlayer=2, with_final_activation=True, with_norm=BN, bias=True):
        super().__init__()
        n_hid = nin
        self.layers = nn.ModuleList([nn.Linear(nin if i == 0 else n_hid,
                                     n_hid if i < nlayer-1 else nout,
                                     # TODO: revise later
                                               bias=True if (i == nlayer-1 and not with_final_activation and bias)
                                               or (not with_norm) else False)  # set bias=False for BN
                                     for i in range(nlayer)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(n_hid if i < nlayer-1 else nout) if with_norm else Identity()
                                    for i in range(nlayer)])
        self.nlayer = nlayer
        self.with_final_activation = with_final_activation
        self.residual = (nin == nout)  # TODO: test whether need this

    def reset_parameters(self):
        for layer, norm in zip(self.layers, self.norms):
            layer.reset_parameters()
            norm.reset_parameters()

    def forward(self, x):
        previous_x = x
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = layer(x)
            if i < self.nlayer-1 or self.with_final_activation:
                x = norm(x)
                x = F.gelu(x)

        if self.residual:
            x = x + previous_x
        return x



class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.LazyBatchNorm1d(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LazyBatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.):
        super().__init__()
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b p d -> b d p'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d p -> b p d'),
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x


class MLPMixer(nn.Module):
    def __init__(self,
                 nhid,
                 nlayer,
                 n_patches,
                 dropout=0,
                 with_final_norm=True
                 ):
        super().__init__()
        self.n_patches = n_patches
        self.with_final_norm = with_final_norm
        self.mixer_blocks = nn.ModuleList(
            [MixerBlock(nhid, self.n_patches, nhid, nhid, dropout=dropout) for _ in range(nlayer)])
        if self.with_final_norm:
            self.layer_norm = nn.LayerNorm(nhid)

    def forward(self, x, coarsen_adj, mask):
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        if self.with_final_norm:
            x = self.layer_norm(x)
        return x


def LinearEncoder(nin, nhid):
    return nn.Linear(nin, nhid, dtype=torch.float)

class GCNConv(nn.Module):
    def __init__(self, nin, nout, bias=True):
        super().__init__()
        self.nn = MLP(nin, nout, 2, True, bias=bias)
        #self.layer = gnn.CGConv(nin, nin, bias=bias)
        self.layer = gnn.TransformerConv(nin, nin,concat=False, heads=4, edge_dim=nin)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        #x = self.layer(x, edge_index)
        return self.nn(F.gelu(self.layer(x, edge_index, edge_attr)))

class GNN(nn.Module):
    def __init__(self,
                    nin,
                    nout,
                    nlayer_gnn,
                    gnn_type,
                    bn=BN,
                    dropout=0.5,
                    res=True):
        super().__init__()
        self.dropout = dropout
        self.res = res

        self.convs = nn.ModuleList([GCNConv(
            nin, nin, bias=not bn) for _ in range(nlayer_gnn)])
        self.norms = nn.ModuleList(
            [nn.BatchNorm1d(nin) if bn else Identity() for _ in range(nlayer_gnn)])
        self.output_encoder = nn.Linear(nin, nout)

    def forward(self, x, edge_index, edge_attr):
        previous_x = x
        for layer, norm in zip(self.convs, self.norms):
            x = layer(x, edge_index, edge_attr)
            x = norm(x)
            x = F.gelu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            if self.res:
                x = x + previous_x
                previous_x = x

        x = self.output_encoder(x)
        return x


class GraphMLPMixer(nn.Module):

    def __init__(self,
                 nfeat_node, nfeat_edge,
                 nhid, nout,
                 nlayer_gnn,
                 nlayer_mlpmixer,
                 gnn_type="GCNConv",
                 gMHA_type='MLPMixer',
                 rw_dim=0,
                 lap_dim=0,
                 dropout=0,
                 mlpmixer_dropout=0.1,
                 bn=True,
                 res=True,
                 pooling='mean',
                 n_patches=32,
                 patch_rw_dim=0):

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

        self.input_encoder = LinearEncoder(nfeat_node, nhid)
        self.edge_encoder = LinearEncoder(nfeat_edge, nhid)

        self.gnns = nn.ModuleList([GNN(nin=nhid,
                                        nout=nhid, nlayer_gnn=4, gnn_type=gnn_type,
                                        bn=bn, dropout=dropout, res=res) for _ in range(nlayer_gnn)])
        self.U = nn.ModuleList(
            [MLP(nhid, nhid, nlayer=1, with_final_activation=True) for _ in range(nlayer_gnn-1)])

        self.reshape = Rearrange('(B p) d ->  B p d', p=n_patches)
        #TODO getattr(gMHA_wrapper, gMHA_type) replaced with MLPMixer

        self.transformer_encoder = MLPMixer(nhid=nhid,
            dropout=mlpmixer_dropout, nlayer=nlayer_mlpmixer, n_patches=n_patches)

        self.output_decoder = MLP(
            nhid, nout, nlayer=4, with_final_activation=True)

    def forward(self, data):
        x = self.input_encoder(data.x.squeeze())
        # Node PE
        if self.use_rw:
            x += self.rw_encoder(data.rw_pos_enc)
        if self.use_lap:
            x += self.lap_encoder(data.lap_pos_enc)
        edge_attr = data.edge_attr
        if edge_attr is None:
            edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1), 1).to(dtype=torch.float)
        edge_attr = self.edge_encoder(edge_attr)
        # Patch Encoder
        x = x[data.subgraphs_nodes_mapper]
        e = edge_attr[data.subgraphs_edges_mapper]

        edge_index = data.combined_subgraphs
        batch_x = data.subgraphs_batch
        for i, gnn in enumerate(self.gnns):
            if i > 0:
                subgraph = scatter(x, batch_x, dim=0,
                                   reduce=self.pooling)[batch_x]
                x = x + self.U[i-1](subgraph)
                x = scatter(x, data.subgraphs_nodes_mapper,
                            dim=0, reduce='mean')[data.subgraphs_nodes_mapper]
            x = gnn(x, edge_index, e)
        subgraph_x = scatter(x, batch_x, dim=0, reduce=self.pooling)

        # Patch PE
        if self.patch_rw_dim > 0:
            subgraph_x += self.patch_rw_encoder(data.patch_pe)
        mixer_x = self.reshape(subgraph_x)

        # MLPMixer
        mixer_x = self.transformer_encoder(mixer_x, data.coarsen_adj if hasattr(
            data, 'coarsen_adj') else None, ~data.mask)

        # Global Average Pooling
        x = (mixer_x * data.mask.unsqueeze(-1)).sum(1) / \
            data.mask.sum(1, keepdim=True)

        # Readout
        x = self.output_decoder(x)
        return x
