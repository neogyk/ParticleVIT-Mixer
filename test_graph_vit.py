import numpy as np
import os
from graph_vit import GraphMLPMixer
import pdb
import random
import pandas as pd
import torch_geometric
from torch_geometric.data import Data
import torch
import tqdm

resume_training = True
from data.TopQuarkDataset import TopQuarkDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torcheval.metrics import BinaryAccuracy
from torcheval.metrics import BinaryAUROC
import wandb
from lightning.fabric import Fabric
from sophia_opt import SophiaG
from wasam import WASAM
from loss import FocalLoss
from torch.profiler import profile, record_function, ProfilerActivity


torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)

run = wandb.init()


seed = 0


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

graph_mixer = GraphMLPMixer(
    nfeat_node=9,
    rw_dim=0,
    lap_dim=6,
    dropout=0.1,
    gMHA_type="MLPMixer",
    mlpmixer_dropout=0.3,
    patch_rw_dim=4,
    nfeat_edge=1,
    nhid=64,
    nout=2,
    nlayer_gnn=2,
    n_patches=3,
    nlayer_mlpmixer=4,
)
graph_mixer = graph_mixer.to(dtype=torch.float)
graph_mixer.init_weights()

data_path = "/eos/user/l/ledidukh/Tagging/TopTag/TopLandscape/"
train_path = f"{data_path}/train_file.parquet"
test_path = f"{data_path}test_file.parquet"

train_ds = TopQuarkDataset(train_path)
test_ds = TopQuarkDataset(test_path)


print(graph_mixer)
PATH = "graph_mixer_2.pt"
# Resume the training:
if resume_training:
    checkpoint = torch.load(PATH, weights_only=False)
    graph_mixer.load_state_dict(checkpoint)
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # last_epoch = checkpoint["epoch"]
    # loss = checkpoint["loss"]
    # print(f"Loaded the model {last_epoch} | {loss}")

# Preprocess the input data
from graph_vit.data_processing import *
from torch_geometric.loader.dataloader import DataLoader

batch_size = 2 * 256
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

n_epoch = 10
l2_lambda = 1e-2

# base_optimizer = torch.optim.Adam#(graph_mixer.parameters(), lr=1e-3, weight_decay=1e-5)
# optimizer = SophiaG(
#    graph_mixer.parameters(), lr=2e-3, betas=(0.965, 0.99), rho=0.4, weight_decay=1e-5
# )

# optimizer = SAM(base_optimizer=base_optimizer, params=graph_mixer.parameters(), rho=0.5, lr=1e-3, weight_decay=1e-5)
base_optimizer = torch.optim.SGD(
    graph_mixer.parameters(), lr=0.01, momentum=0.9
)  # define an optimizer for the "sharpness-aware" update

optimizer = WASAM(
    graph_mixer.parameters(), base_optimizer, rho=0.5, lr=0.01, momentum=0.9
)

fabric = Fabric(accelerator="cuda", precision="32-true")
fabric.launch()
graph_mixer, optimizer = fabric.setup(graph_mixer, optimizer)
train_dl = fabric.setup_dataloaders(train_dl)


model_parameters = filter(lambda p: p.requires_grad, graph_mixer.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Number of the parameters:", params)


scheduler = ReduceLROnPlateau(optimizer, "min")
# loss_fn = torch.nn.BCEWithLogitsLoss()
loss_fn = FocalLoss(gamma=1.0)

# Replace it with the cross entropy loss function

# Metrics
acc_metric = BinaryAccuracy()
rocauc_metric = BinaryAUROC()

graph_mixer.train()

run.watch(graph_mixer)
for epoch in range(n_epoch):
    loss_train = 0
    acc_train = 0
    for event, label in tqdm.tqdm(train_dl):
        optimizer.zero_grad()
        # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA,]) as prof:
        #    with record_function("gMixer_train"):
        logits = graph_mixer.forward(event)
        output = torch.argmax(logits, dim=-1)
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
        output = torch.nn.functional.softmax(logits, dim=1)
        label = torch.nn.functional.one_hot(event.label.to(torch.int64), num_classes=2)
        l2_reg = sum(p.pow(2).sum() for p in graph_mixer.parameters())
        l1_reg = sum(torch.norm(p).sum() for p in graph_mixer.parameters())
        loss = loss_fn(logits.float(), label.float()) + l2_lambda / (2) * (
            l2_reg + l1_reg
        )
        # loss = torch.nn.functional.cross_entropy(logits.(), label.float())
        try:
            fabric.backward(loss)
        except Exception:
            pdb.set_trace()
        torch.nn.utils.clip_grad_norm_(
            graph_mixer.parameters(), max_norm=1.0, norm_type=2
        )
        torch.nn.utils.clip_grad_value_(graph_mixer.parameters(), 1.0)

        optimizer.first_step(zero_grad=True)
        torch.autograd.set_detect_anomaly(True)
        logits = graph_mixer(event)
        loss = loss_fn(logits.float(), label.float())
        # loss = torch.nn.functional.cross_entropy(logits.(), label.float())
        fabric.backward(loss)
        optimizer.second_step(zero_grad=True)
        torch.autograd.set_detect_anomaly(True)
        # scheduler.step(loss)
        with torch.no_grad():
            acc_metric.update(output.argmax(dim=1), event.label)
            acc = acc_metric.compute().detach()

            loss_train += loss.item() / len(train_ds)
            # roc auc
            rocauc_metric.update(output.argmax(dim=1), event.label)
            roc_auc = rocauc_metric.compute()
            print(f"Accuracy/Batch {acc} | Loss: {loss.item()} | ROC AUC {roc_auc}")
            wandb.log({"Train Accuracy": acc})
            # wandb.log({"Train ROC AUC": roc_auc})
            wandb.log({"Train Loss": loss.item()})
            wandb.log({"epoch": epoch})
            # wandb.log({"iter": iter*epoch})
            acc_metric.reset()
            rocauc_metric.reset()
        torch.cuda.empty_cache()
    print(f"Epoch: {epoch} | Accuracy: {acc_train} | Loss: {loss_train}")
    # acc_val  = 0
    # loss_val = 0

    # torch.save(model.state_dict(), "./gViT/model.pt")
    # for event, label in tqdm.tqdm(test_dl):
    #    with torch.no_grad():
    #        event = event.to(device)
    #        label = label.to(device)
    #        logits = model.forward(event)
    #        output = torch.nn.functional.sigmoid(logits)
    #        loss = loss_fn(output.ravel(), label)
    #        acc_val += torch.sum(torch.round(output.ravel()) == label)/len(test_ds)
    #        loss_val += loss.item()/len(test_ds)
    # print(f"Validation: Acuracy: {acc_val} | Loss: {loss_val}")
