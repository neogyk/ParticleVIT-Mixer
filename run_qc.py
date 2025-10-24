from __future__ import annotations
import argparse
import importlib
import pytorch_lightning as pl
import torch
import torchmetrics
from lightning.pytorch.loggers import WandbLogger
from lion_pytorch import Lion
from torch_ema import ExponentialMovingAverage
from torch_geometric.loader.dataloader import DataLoader
from configs.qg_config import Config
from data.QGDataset import JetQGDataset
from graph_mixer import GraphMLPMixer
from graph_mixer.data_processing import *
from graph_mixer.loss import FocalLoss
from utils import seed_worker
from utils import set_seed
from utils import weight_init
from adam_atan2_pytorch import AdamAtan2
from adabelief_pytorch import AdaBelief

parser = argparse.ArgumentParser(
    prog="Graph MLP Mixer for Jet Tagging",
    description="The script allows to train and infer the model using the input config file",
)
parser.add_argument("config")
args = parser.parse_args()
config_path = args.config
config = importlib.util.spec_from_file_location("Config", config_path)
config = Config()
alpha = 1e-6


class gMLPMixer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = GraphMLPMixer(
            nfeat_node=21,
            nfeat_jet=7,
            rw_dim=0,
            lap_dim=0,
            dropout=0,
            gMHA_type="MLPMixer",
            mlpmixer_dropout=0,
            patch_rw_dim=0,
            nfeat_edge=4,
            nhid=64,
            nlayer_gnn=0,
            n_patches=8,
            nout=2,
            nlayer_mlpmixer=5,
            token_dim=[512, 256, 128, 64, 64],
            channel_dim=[64, 64, 64, 64, 64],
        )
        print(self.model)
        self.save_hyperparameters()

        self.model.apply(weight_init)
        self.loss_fn = FocalLoss(gamma=5)
        self.strict_loading = True
        self.train_acc = torchmetrics.classification.BinaryAccuracy()
        self.valid_acc = torchmetrics.classification.BinaryAccuracy()
        self.train_auc = torchmetrics.classification.BinaryAUROC()
        self.valid_auc = torchmetrics.classification.BinaryAUROC()

        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.995)
        self.confmat = torchmetrics.ConfusionMatrix(
            normalize="true",
            task="binary",
            num_classes=2,
        )
        self.val_confmat = torchmetrics.ConfusionMatrix(
            normalize="true",
            task="binary",
            num_classes=2,
        )

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch, batch.label
        logits = self.model(x)
        # class_weights = ( torch.bincount(batch.label.int())
        #    / (torch.bincount(batch.label.int()).sum())
        # )
        class_weights = torch.Tensor([0.5, 0.5])
        output = torch.nn.functional.softmax(logits, dim=1)
        l1_norm = 0
        l2_norm = 0
        for param in self.model.parameters():
            l1_norm += alpha * torch.norm(torch.abs(param), 1)
            l2_norm += alpha * torch.norm(torch.abs(param**2), 1)
        loss = self.loss_fn(
            logits, y.long(), weight=torch.Tensor(class_weights)
        ) + alpha * (l1_norm + l2_norm)
        loss = loss.mean()
        auc = self.train_auc(output.argmax(dim=1), y)
        self.log("train/loss", loss, prog_bar=True)
        acc = self.train_acc(output.argmax(dim=1), y)
        self.log("train/acc", acc, prog_bar=True)
        conf_matrix = self.confmat.update(output.argmax(dim=1), y)
        fig_, ax_ = self.confmat.plot()
        self.log("train/auc", auc, prog_bar=True)
        self.log("lr", self.optimizers()._optimizer.param_groups[-1]["lr"])
        self.logger.log_image("Confusion Matrix", [fig_])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch, batch.label
        logits = self.model(x)
        output = torch.nn.functional.softmax(logits, dim=1)
        auc = self.train_auc(output.argmax(dim=1), y)
        class_weights = (
            1
            / torch.bincount(batch.label.int())
            / (torch.bincount(batch.label.int()).sum())
        )
        loss = self.loss_fn(logits, y.long(), weight=torch.Tensor(class_weights)).mean()
        valid_acc = self.valid_acc(output.argmax(dim=1), y)
        self.log("val/auc", auc, prog_bar=True)
        self.log("val/loss", loss, prog_bar=True)
        self.log("valid/acc", valid_acc, prog_bar=True)
        conf_matrix = self.val_confmat.update(output.argmax(dim=1), y)
        fig_, ax_ = self.val_confmat.plot()
        self.logger.log_image("Validation Confusion Matrix", [fig_])
        return loss

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        for p in range(len(optimizer.param_groups[0]["params"])):
            if optimizer.param_groups[0]["params"][p].grad is not None:
                size = optimizer.param_groups[0]["params"][p].grad.size()
                optimizer.param_groups[0]["params"][p].grad += 1e-3 * torch.normal(
                    torch.zeros(size), torch.ones(size)
                )
        optimizer.step(closure=optimizer_closure)
        # self.ema.update(self.model.parameters())

    def configure_optimizers(self):
        # optimizer =  AdamAtan2( self.model.parameters(),
        #    lr=1e-3,
        #    weight_decay=1e-2)
        optimizer = AdaBelief(
            self.model.parameters(),
            lr=1e-4,
            eps=1e-16,
            betas=(0.9, 0.999),
            weight_decouple=True,
            rectify=False,
        )
        # scheduler1 = torch.optim.lr_scheduler.CyclicLR(optimizer, factor=0.1, total_iters=10000)

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=100, T_mult=2, eta_min=1e-6, last_epoch=-1
            ),
            "monitor": "train/loss",
            "interval": "step",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def train(graph_mixer, train_dl, test_dl):

    wandb_logger = WandbLogger(project="QG_Lion")
    # wandb_logger.watch(graph_mixer, log_graph=True)
    trainer = pl.Trainer(
        default_root_dir="./QG_Lion/",
        max_epochs=100,
        precision="32-true",
        deterministic=True,
        enable_checkpointing=True,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        logger=wandb_logger,
        accelerator="cpu",
        check_val_every_n_epoch=1,
    )
    trainer.fit(graph_mixer, train_dl, test_dl)
    # ckpt_path="./QG_Lion/u3uqcyec/checkpoints/epoch=7-step=25000.ckpt")
    # ckpt_path="./QG_Lion/ljktf5vs/checkpoints/epoch=1-step=6250.ckpt")
    return trainer


if __name__ == "__main__":
    # Training Process:
    g = set_seed(42)
    mode = "train"
    # Define the model:
    train_ds = JetQGDataset(config=config, train=True)
    test_ds = JetQGDataset(config=config, train=False)
    train_dl = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        generator=g,
        worker_init_fn=seed_worker,
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=256,
        shuffle=False,
        pin_memory=True,
        generator=g,
        worker_init_fn=seed_worker,
    )
    graph_mixer = gMLPMixer()
    if mode == "train":
        trainer = train(graph_mixer, train_dl, test_dl)
    else:
        for data in test_dl:
            with torch.no_grad():
                result = graph_mixer(data)
                output = torch.nn.functional.softmax(result[0], dim=1)
                pdb.set_trace()
