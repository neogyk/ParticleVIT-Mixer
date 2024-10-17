import os
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
from graph_vit.data_processing import *
from torch_geometric.loader.dataloader import DataLoader
from config import Config
from loss import LBELoss, FocalLoss
import random
from torch_lr_finder import LRFinder
from sophia_opt import SophiaG
from torcheval.metrics import BinaryAccuracy

#torch.cuda.empty_cache()
seed = 3407
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

set_seed(seed)
lr_finder = True



graph_mixer = GraphMLPMixer(nfeat_node=28,#5,
                            rw_dim=0,#32,
                            lap_dim=6, dropout=0.135, mlpmixer_dropout=0.105, patch_rw_dim=4,
                            nfeat_edge=4,
                            nhid=64,
                            nout=1,
                            nlayer_gnn=5,#5
                            n_patches=32,
                            nlayer_mlpmixer=6)

graph_mixer = graph_mixer.to(dtype=torch.float)
device = torch.device("cpu")
graph_mixer = graph_mixer.to(device)

graph_mixer.train()


print(graph_mixer)
config = Config()
print("Input Path:", config.path)


def seed_worker(worker_id):
    """PyTorch implementation to fix the seeds
    Args:
        worker_id ():
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


train_ds = JetQGDataset(input_path=config.path)
train_dl = DataLoader(train_ds,
                        batch_size=256,
                        shuffle=True)

#test_ds = JetQGDataset(input_path=config.path)
#test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)


n_epoch = 100
resume_training = False#True
#optimizer = torch.optim.AdamW(graph_mixer.parameters(), lr=5e-3, weight_decay=1e-1, amsgrad=False, betas=(0.8, 0.89), fused=False)
optimizer = SophiaG(graph_mixer.parameters(), lr=2e-3, betas=(0.965, 0.99), rho=0.01, weight_decay=1e-1)
loss_fn = FocalLoss(gamma=2)
#loss_fn = torch.nn.CrossEntropyLoss()
#kl_loss = torch.nn.KLDivLoss(reduce="batchmean")
prev_loss = None
do_test = False
save_model = True
i=0



PATH = "./gViT/model_4.pt"
if resume_training:
    checkpoint = torch.load(PATH, weights_only=True)
    graph_mixer.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    last_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Loaded the model {last_epoch} | {loss}")


metric = BinaryAccuracy()


last_epoch = 0
for epoch in range(n_epoch):
    loss_train = 0
    acc_train = 0
    for event, label in tqdm.tqdm(train_dl):

        event = event.to(device)
        label = event.label.to(device)
        graph_mixer.zero_grad()
        optimizer.zero_grad()

        logits = graph_mixer.forward(event)
        output = torch.nn.functional.sigmoid(logits)
        l2_lambda = 1e-8
        l2_reg = torch.tensor(0.).to(device)
        l2_reg = sum(p.pow(2).sum() for p in graph_mixer.parameters())
        loss = loss_fn(output.ravel(), label)# + kl_loss(output.ravel(), label)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(graph_mixer.parameters(), max_norm=1.0, norm_type=2)
        optimizer.step()
        metric.update(output.ravel(), label)
        acc = metric.compute().detach()
        acc_train += torch.sum(torch.round(output.ravel()) == label)/len(train_ds)
        acc_batch = torch.sum(torch.round(output.ravel()) == label)/label.size()[0]
        loss_train += loss.item()/len(train_ds)
        print(f"Accuracy/Batch {acc} | Loss: {loss_train}")

    print(f"Epoch: {epoch} | Accuracy: {acc_train} | Accuracy/Batch {acc_batch} | Loss: {loss_train}")
    acc_val  = 0
    loss_val = 0

    if save_model:
        torch.save({'epoch': epoch,
                    'model_state_dict': graph_mixer.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss},f"./gViT/model_{epoch}.pt")
    if do_test:
        for event, label in tqdm.tqdm(test_dl):
            with torch.no_grad():
                event = event.to(device)
                label = label.to(device)
                logits = model.forward(event)
                output = torch.nn.functional.softmax(logits)
                loss = loss_fn(logits.ravel(), label)
                acc_val += torch.sum(torch.round(output.ravel()) == label)/len(test_ds)
                loss_val += loss.item()/len(test_ds)
        print(f"Validation: Acuracy: {acc_val} | Loss: {loss_val}")
