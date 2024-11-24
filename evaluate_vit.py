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
from sklearn.metrics import roc_auc_score


class JeTopQuarkDataset(torch.utils.data.Dataset):
    def __init__(self, input_path, train=False):
        # "/kaggle/input/topquarkzenodo/train.h5"
        self.input_path = input_path
        self.file = h5py.File(input_path)
        ar = np.array(self.file["table/table"][:])
        self.data = ar["values_block_0"]
        self.label = torch.Tensor(ar["values_block_1"][:, 1])
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

    def add_prediction(self, tensor: np.array, store=True):
        # store the input tensor
        #        self.file.close()
        file = h5py.File(self.input_path, mode="a")
        file.create_dataset("prediction", data=tensor)
        file.close()
        return

    def __getitem__(self, idx):
        event = self.data[idx]
        event = Data(
            x=event.to(dtype=torch.float).view(201, 4),
            pos=event.to(dtype=torch.float).view(201, 4),
            label=self.label[idx],
        )

        label = self.label[idx]
        knn_graph = torch_geometric.transforms.knn_graph.KNNGraph(k=4)
        gpatrches = GraphPartitionTransform(n_patches=3, metis=False)
        event = knn_graph(event)

        event = gpatrches(event)
        return event, label


device = torch.device("cuda")
graph_mixer = GraphMLPMixer(
    nfeat_node=4,
    nfeat_edge=1,
    nhid=256,
    nout=1,
    nlayer_gnn=5,
    n_patches=3,
    nlayer_mlpmixer=3,
)


graph_mixer = graph_mixer.to(dtype=torch.float)

train_path = "/eos/user/l/ledidukh/TopQuark/train.h5"
test_path = "/eos/user/l/ledidukh/TopQuark/test_bu.h5py"
val_path = "/eos/user/l/ledidukh/TopQuark/val.h5"
# train_ds = JeTopQuarkDataset(train_path)
test_ds = JeTopQuarkDataset(test_path)
val_ds = JeTopQuarkDataset(val_path)

print(graph_mixer)

graph_mixer.load_state_dict(torch.load("./gViT/model.pt"))

# Preprocess the input data
from graph_vit.data_processing import *
from torch_geometric.loader.dataloader import DataLoader

batch_size = 256
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
# test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
# val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

model = graph_mixer.to(device)


n_parameters = np.sum([parameter for parameter in model.parameters()])
print("N parameters: ", n_parameters)
loss_fn = torch.nn.BCELoss()
# graph_mixer.train()
epoch = "Evaluation"
loss_train = 0
acc_train = 0
# Evaluate the Training Dataset:
train_pred = []
n_events = 100
# for event, label in tqdm.tqdm(train_dl):
#    break
#    with torch.no_grad():
#        event = event.to(device)
#        logits = model.forward(event)
#        output = torch.nn.functional.sigmoid(logits)
#        label = label.to(device)
#        loss = loss_fn(output.ravel(), label)
#        train_pred.append(output.ravel())
#        acc_train += torch.sum(torch.round(output.ravel()) == label)/len(train_ds)
#        loss_train+=loss.item()/len(train_ds)
#        print(f"Epoch: {epoch} | Accuracy: {acc_train} | Loss: {loss_train}")
# pdb.set_trace()
#        roc_auc = roc_auc_score(np.round(output.ravel().detach().cpu().numpy()), label.detach().cpu().numpy())
# print("ROC AUC: ", roc_auc)
# if n_events==100:break

# acc_test  = 0
# loss_test = 0
# pred_test = []
# Evaluate the Test Data
# for event, label in tqdm.tqdm(test_dl):
#    with torch.no_grad():
#        event = event.to(device)
#        label = label.to(device)
#        logits = model.forward(event)
#        output = torch.nn.functional.sigmoid(logits)
#        pred_test.append(output.ravel().detach().cpu().numpy())
# loss = loss_fn(output.ravel(), label)
# acc_test += torch.sum(torch.round(output.ravel()) == label)/len(test_ds)
# loss_test += loss.item()/len(test_ds)
# roc_auc = roc_auc_score(np.round(output.ravel().detach().cpu().numpy()), label.detach().cpu().numpy())
# print("ROC AUC: ", roc_auc)
# pdb.set_trace()
# pred_test = np.concatenate(pred_test)
# test_ds.add_prediction(pred_test, store=True)
# print(f"Test: Acuracy: {acc_test} | Loss: {loss_test}")

acc_val = 0
loss_val = 0
pred_val = []
# Evaluate the Validation  Data
for event, label in tqdm.tqdm(train_dl):
    with torch.no_grad():
        event = event.to(device)
        label = label.to(device)
        logits = model.forward(event)
        output = torch.nn.functional.sigmoid(logits)
        pred_val.append(output.ravel().detach().cpu().numpy())
        # loss = loss_fn(output.ravel(), label)
        # acc_val += torch.sum(torch.round(output.ravel()) == label)/len(test_ds)
        # loss_val += loss.item()/len(test_ds)
        # x = torch.round(output.ravel()).detach().cpu().numpy()
        # y =  label.detach().cpu().numpy()
        # roc_auc = roc_auc_score(x, y)
#        print("ROC AUC: ", roc_auc)
# break
pred_val = np.concatenate(pred_val)
df = pd.DataFrame()
df["prediction"] = pred_val
df["labels"] = train_ds.label.detach().numpy()
df.to_csv("pred_train.csv", index=False)

# np.save("pred_val.npy", pred_val)
# print(f"Validation: Acuracy: {acc_val} | Loss: {loss_val}")
# pdb.set_trace()
# val_ds.add_prediction(pred_val, store=True)
