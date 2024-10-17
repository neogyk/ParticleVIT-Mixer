from config import Config
import numpy as np
import pdb
from graph_vit import GraphMLPMixer
import pandas as pd
import torch_geometric
from torch_geometric.data import Data
import torch
import h5py
import tqdm
import hdf5plugin
import pdb
import argparse
from data.QGDataset import JetQGDataset
#Preprocess the input data
from graph_vit.data_processing import *
from torch_geometric.loader.dataloader import DataLoader
from config import Config
#Pass the dataset as the input

class JeTopQuarkDataset(torch.utils.data.Dataset):
    def __init__(self, input_path):
        #"/kaggle/input/topquarkzenodo/train.h5"
        file = h5py.File(input_path)
        ar = np.array(file['table/table'][:])
        self.data = ar['values_block_0']

        self.label = torch.Tensor(ar['values_block_1'][:,1])

        self.idx = torch.randperm(self.label.shape[0])
        self.label = self.label[self.idx]
        self.data = torch.Tensor(self.data)[self.idx]
        #np.random.shuffle(self.data)
        #np.random.shuffle(self.label)

    def __len__(self):
        return self.data.size()[0]

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



#test_ds = JetQGDataset(input_path="C:/Users/leonid_didukh/Desktop/NIPS/Jets/*withbc*.npz")
#train_dl = DataLoader(test_ds, batch_size=32, shuffle=False)
#for event, label in test_dl:
#    break

graph_mixer = GraphMLPMixer(nfeat_node=4,
                            nfeat_edge=1,
                             nhid=256,
                             nout=1,
                             nlayer_gnn=10,#5
                             n_patches=12,#3
                             nlayer_mlpmixer=3)


graph_mixer = graph_mixer.to(dtype=torch.float)

#train_path = "/eos/user/l/ledidukh/TopQuark/train.h5"
#test_path = "/eos/user/l/ledidukh/TopQuark/test.h5"

#train_ds = JeTopQuarkDataset(train_path)
#test_ds = JeTopQuarkDataset(test_path)


print(graph_mixer)
config = Config()
train_ds = JetQGDataset(input_path=config.path)
train_dl = DataLoader(train_ds, batch_size=512, shuffle=True)

#test_ds = JetQGDataset(input_path=config.path)
#test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)


n_epoch = 10
device = torch.device('cpu')
model = graph_mixer.to(device)
optimizer = torch.optim.AdamW(graph_mixer.parameters(), lr=1e-3)
loss_fn = torch.nn.BCELoss(reduction='none')

prev_loss = None
do_test = False
graph_mixer.train()
i=0
for epoch in range(n_epoch):
    loss_train = 0
    acc_train = 0
    for event, label in tqdm.tqdm(train_dl):
        event = event.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        logits = model.forward(event)
        output = torch.nn.functional.sigmoid(logits)
        if i == 0:
            loss = loss_fn(output.ravel(), label).sum()
            prev_loss = loss_fn(output.ravel(), label).detach()

        else:
            loss = torch.div(loss_fn(output.ravel(), label), 2*prev_loss).sum()

        loss.backward()
        optimizer.step()

        acc_train += torch.sum(torch.round(output.ravel()) == label)/len(train_ds)
        acc_batch = torch.sum(torch.round(output.ravel()) == label)/label.size()[0]
        loss_train+=loss.item()/len(train_ds)
        print(f"Epoch: {epoch} | Accuracy: {acc_train} | Accuracy/Batch {acc_batch} | Loss: {loss_train}")
    acc_val  = 0
    loss_val = 0
    torch.save(model.state_dict(), "./gViT/model.pt")
    if do_test:
        for event, label in tqdm.tqdm(test_dl):
            with torch.no_grad():
                event = event.to(device)
                label = label.to(device)
                logits = model.forward(event)
                output = torch.nn.functional.sigmoid(logits)
                loss = loss_fn(output.ravel(), label)
                acc_val += torch.sum(torch.round(output.ravel()) == label)/len(test_ds)
                loss_val += loss.item()/len(test_ds)
        print(f"Validation: Acuracy: {acc_val} | Loss: {loss_val}")
