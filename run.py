import os
from config import Config
import numpy as np
import random
from graph_vit import GraphMLPMixer
import pandas as pd
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader.dataloader import DataLoader
import torch
import h5py
import tqdm
from data.QGDataset import JetQGDataset
from graph_vit.data_processing import *
from config import Config
from loss import FocalLoss
from sophia_opt import SophiaG
from torcheval.metrics import BinaryAccuracy
from torcheval.metrics import BinaryAUROC
import wandb
import pdb


wandb.init()
torch.cuda.empty_cache()
seed = 3407


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


set_seed(seed)


# Define the model:
graph_mixer = GraphMLPMixer(
    nfeat_node=28,  # 5,
    rw_dim=0,  # 32,
    lap_dim=6,
    dropout=0.135,
    gMHA_type="MLPMixer",
    mlpmixer_dropout=0.105,
    patch_rw_dim=4,
    nfeat_edge=4,
    nhid=64,
    nout=1,
    nlayer_gnn=5,  # 5
    n_patches=32,
    nlayer_mlpmixer=6,
)

graph_mixer = graph_mixer.to(dtype=torch.float)
device = torch.device("cpu")
graph_mixer = graph_mixer.to(device)
graph_mixer.train()


print(graph_mixer)

config = Config()
print("Input Path:", config.path)


train_ds = JetQGDataset(input_path=config.path, train=True)
train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)

test_ds = JetQGDataset(input_path=config.path, train=False)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)


for event, label in tqdm.tqdm(train_dl):
    pdb.set_trace()
    break

n_epoch = 100
resume_training = False  # True

# Choose the optimizer:

# optimizer = torch.optim.AdamW(graph_mixer.parameters(), lr=5e-3, weight_decay=1e-1, amsgrad=False, betas=(0.8, 0.89), fused=False)
optimizer = SophiaG(
    graph_mixer.parameters(), lr=2e-3, betas=(0.965, 0.99), rho=0.01, weight_decay=1e-1
)

# Define the loss function:
loss_fn = FocalLoss(gamma=2)
loss_bce = torch.nn.BCEWithLogitsLoss()
kl_loss = torch.nn.KLDivLoss(reduce="batchmean")

# Metrics
# Accuracy
acc_metric = BinaryAccuracy()
# ROC AUC
rocauc_metric = BinaryAUROC()

prev_loss = None
do_test = True
save_model = True
i = 0


PATH = "./gViT/model_4.pt"
if resume_training:
    checkpoint = torch.load(PATH, weights_only=True)
    graph_mixer.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    last_epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print(f"Loaded the model {last_epoch} | {loss}")


# Training Process:
last_epoch = 0
for epoch in range(1, n_epoch):
    loss_train = 0
    acc_train = 0
    iter = 0

    for event, label in tqdm.tqdm(train_dl):
        iter += 1
        event = event.to(device)
        label = event.label.to(device)
        graph_mixer.zero_grad()
        optimizer.zero_grad()

        logits = graph_mixer.forward(event)
        output = torch.nn.functional.sigmoid(logits)
        l2_lambda = 1e-8
        l2_reg = torch.tensor(0.0).to(device)
        l2_reg = sum(p.pow(2).sum() for p in graph_mixer.parameters())
        # loss = loss_fn(output.ravel(), label)# + kl_loss(output.ravel(), label)
        loss = loss_bce(output.ravel(), label) + l2_reg
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            graph_mixer.parameters(), max_norm=1.0, norm_type=2
        )
        optimizer.step()
        acc_metric.update(output.ravel(), label)
        acc = acc_metric.compute().detach()
        acc_train += torch.sum(torch.round(output.ravel()) == label) / len(train_ds)
        acc_batch = torch.sum(torch.round(output.ravel()) == label) / label.size()[0]
        loss_train += loss.item() / len(train_ds)

        # roc auc
        rocauc_metric.update(output.ravel(), label)
        roc_auc = rocauc_metric.compute()
        print(f"Accuracy/Batch {acc} | Loss: {loss_train} | ROC AUC {roc_auc}")

        wandb.log({"Train Accuracy": acc})
        wandb.log({"Train ROC AUC": roc_auc})
        wandb.log({"Train Loss": loss_train})
        wandb.log({"epoch": epoch})
        wandb.log({"iter": i * epoch})

    print(
        f"Epoch: {epoch} | Accuracy: {acc_train} | Accuracy/Batch {acc_batch} | Loss: {loss_train}"
    )
    acc_val = 0
    loss_val = 0

    if do_test:
        for event, label in tqdm.tqdm(test_dl):
            with torch.no_grad():
                event = event.to(device)
                label = label.to(device)
                logits = model.forward(event)
                output = torch.nn.functional.softmax(logits)
                loss = loss_fn(logits.ravel(), label)
                acc_val += torch.sum(torch.round(output.ravel()) == label) / len(
                    test_ds
                )
                loss_val += loss.item() / len(test_ds)
        print(f"Validation: Acuracy: {acc_val} | Loss: {loss_val}")
    if save_model:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": graph_mixer.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            f"./gViT/model_{epoch}.pt",
        )
