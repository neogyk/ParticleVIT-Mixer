from __future__ import annotations

import argparse
import importlib

import pytorch_lightning as pl
import torch
import torchmetrics
import tqdm
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lion_pytorch import Lion
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.loader.dataloader import DataLoader
from torch_geometric.loader.imbalanced_sampler import ImbalancedSampler
from torchmetrics.classification import MulticlassConfusionMatrix
from torchsampler import ImbalancedDatasetSampler

import wandb
from augmentation import graph_augmentation
from configs.jetnet_config import Config
from data.JetNetDataset import JetNetDataset
from graph_vit import GraphMLPMixer
from graph_vit.data_processing import *
from graph_vit.loss import FocalLoss
from utils import set_seed
from utils import weight_init

parser = argparse.ArgumentParser(
    prog="Graph MLP Mixer for Jet Tagging",
    description="The script allows to train and infer the model using the input config file",
)

parser.add_argument("config_path")
args = parser.parse_args()
config_path = args.config_path
config = importlib.util.spec_from_file_location("Config", config_path)
config = Config()

# TODO extract from the config file
n_epoch = 100
accumulation_step = 6
resume_training = False  # True
prev_loss = None
do_test = True
save_model = True
i = 0


class gMLPMixer(pl.LightningModule):
    """ """

    def __init__(self):
        super().__init__()
        self.model = GraphMLPMixer(
            nfeat_node=7,
            rw_dim=0,
            lap_dim=6,
            dropout=0.3,
            gMHA_type="MLPMixer",
            mlpmixer_dropout=0.4,
            patch_rw_dim=4,
            nfeat_edge=3,
            nhid=256,
            nlayer_gnn=1,
            n_patches=8,
            nout=5,
            nlayer_mlpmixer=4,
        )
        self.save_hyperparameters()
        self.model.apply(weight_init)

        self.loss_fn = FocalLoss(gamma=5)
        self.train_acc = torchmetrics.classification.Accuracy(
            task="multiclass",
            num_classes=5,
        )
        self.valid_acc = torchmetrics.classification.Accuracy(
            task="multiclass",
            num_classes=5,
        )
        self.train_auc = torchmetrics.AUROC(task="multiclass", num_classes=5)
        self.valid_auc = torchmetrics.AUROC(task="multiclass", num_classes=5)

        self.confmat = MulticlassConfusionMatrix(num_classes=5, normalize="true")

        self.val_confmat = MulticlassConfusionMatrix(num_classes=5, normalize="true")
        return

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):

        x, y = batch, batch.label
        logits = self.model(x)
        y = y.long()
        y = torch.nn.functional.one_hot(y, 5)

        class_weights = (
            1
            / torch.bincount(batch.label.int())
            / (torch.bincount(batch.label.int()).sum())
        )

        output = torch.nn.functional.softmax(logits, dim=1)
        loss = self.loss_fn(
            output,
            y.float(),
            weight=torch.Tensor(class_weights),
        ).mean()
        auc = self.train_auc(output, y.argmax(dim=1))

        self.log("train/loss", loss, prog_bar=True)
        acc = self.train_acc(output.argmax(dim=1), y.argmax(dim=1))
        self.log("train/acc", acc, prog_bar=True)
        conf_matrix = self.val_confmat.update(output.argmax(dim=1), y.argmax(dim=1))
        fig_, ax_ = self.val_confmat.plot()
        self.log("train/auc", auc, prog_bar=True)
        self.logger.log_image("Confusion Matrix", [fig_])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch, batch.label
        logits = self.model(x)
        y = y.long()
        y = torch.nn.functional.one_hot(y, 5)
        output = torch.nn.functional.softmax(logits, dim=1)
        auc = self.train_auc(output, y.argmax(dim=1))
        class_weights = (
            1
            / torch.bincount(batch.label.int())
            / (torch.bincount(batch.label.int()).sum())
        )
        loss = self.loss_fn(
            output,
            y.float(),
            weight=torch.Tensor(class_weights),
        ).mean()
        valid_acc = self.valid_acc(output.argmax(dim=1), output.argmax(dim=1))
        self.log("val/auc", auc, prog_bar=True)
        self.log("val/loss", loss, prog_bar=True)
        self.valid_acc(output, y)
        self.log("valid/acc", valid_acc, prog_bar=True)
        return loss

    def configure_optimizers(self):

        optimizer = Lion(self.model.parameters(), lr=1e-4, weight_decay=1e-2)

        # scheduler = {
        #    "scheduler": torch.optim.lr_scheduler.StepLR(
        #        optimizer,
        #        #mode="min",
        #        step_size=100,
        #        gamma = 0.99,
        #        verbose=True,
        #    ),
        #    "monitor": "train/loss",
        #    "interval": "step",
        #    "frequency": 1,
        # }
        return {"optimizer": optimizer}  # , "lr_scheduler": scheduler}


def train(graph_mixer, train_dl, test_dl):
    wandb_logger = WandbLogger(project="JetNet_Lion")
    trainer = pl.Trainer(
        default_root_dir="./JetNet_Lion/",
        max_epochs=100,
        deterministic=True,
        accumulate_grad_batches=1,
        enable_checkpointing=True,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="value",
        logger=wandb_logger,
        accelerator="cpu",
        check_val_every_n_epoch=10,
    )

    trainer.fit(
        graph_mixer,
        train_dl,
        test_dl,
        # ckpt_path="./JetNet_Lion/k6mhva77/checkpoints/epoch=3-step=27500.ckpt"
        # /JetNet_Lion/u9tfosr0/checkpoints/epoch=1-step=13750.ckpt"
    )
    return trainer


if __name__ == "__main__":
    # Training Process:
    set_seed(17)
    # Define the model:
    print(config.path)

    train_ds = JetNetDataset(input_path=config.path, train=True)
    evens = list(range(0, len(train_ds), 2))
    odds = list(range(1, len(train_ds), 2))
    trainset_1 = torch.utils.data.Subset(train_ds, evens + odds[: int(len(odds) / 2)])
    trainset_2 = torch.utils.data.Subset(train_ds, odds[int(len(odds) / 2) :])

    train_dl = DataLoader(
        trainset_1,
        batch_size=256,
        shuffle=True,
        pin_memory=True,
        num_workers=7,
    )
    test_dl = DataLoader(
        trainset_2,
        batch_size=128,
        shuffle=True,
        pin_memory=True,
        num_workers=7,
    )

    graph_mixer = gMLPMixer()

    trainer = train(graph_mixer, train_dl, test_dl)
