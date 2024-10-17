import numpy as np
from graph_vit import GraphMLPMixer
import pandas as pd
import torch_geometric
from torch_geometric.data import Data
import torch
import h5py
import tqdm
import hdf5plugin
import pdb

class JeQGDataset(torch.utils.data.Dataset):
    def __init__(self, input_path, train=False):
        #"/kaggle/input/topquarkzenodo/train.h5"
        self.input_path = input_path
        self.file = h5py.File(input_path)
        ar = np.array(self.file['table/table'][:])
        self.data = ar['values_block_0']
        self.label = torch.Tensor(ar['values_block_1'][:,1])
        self.train = train
        if self.train:
            self.idx = torch.randperm(self.label.shape[0])
            self.label = self.label[self.idx]
            self.data = torch.Tensor(self.data)[self.idx]
        else:
            self.label = self.label
            self.data = torch.Tensor(self.data)

        self.file.close()

    def __len__(self):
        return self.data.size()[0]
    
    def add_prediction(self, tensor:np.array, store=True):
        #store the input tensor 
#        self.file.close()
        file = h5py.File(self.input_path, mode='a')
        file.create_dataset('prediction', data=tensor)
        file.close()
        return

    def __getitem__(self,idx):
        event = self.data[idx]
        event = Data(x=event.to(dtype=torch.float).view(201,4),
                    pos=event.to(dtype=torch.float).view(201,4), label=self.label[idx])

        label = self.label[idx]
        knn_graph = torch_geometric.transforms.knn_graph.KNNGraph(k=4)
        gpatrches  = GraphPartitionTransform(n_patches=3,metis=False)
        event = knn_graph(event)

        event = gpatrches(event)
        return event, label
