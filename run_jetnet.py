from __future__ import annotations
import pdb
import wandb
import argparse
import importlib
import torch
import torchmetrics
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger
from torch_ema import ExponentialMovingAverage
from torch_geometric.loader.dataloader import DataLoader
from torchmetrics.classification import MulticlassConfusionMatrix
from torch_geometric.profile import *
from augmentation import graph_augmentation
from configs.jetnet_config import Config
from data.JetNetDataset import JetNetDataset
from graph_vit import GraphMLPMixer
from graph_vit.data_processing import *
from graph_vit.loss import FocalLoss
from utils import seed_worker
from utils import set_seed
from utils import weight_init
from optimizer import SophiaG
from adam_atan2_pytorch import AdamAtan2
from optimizer import Muon
from muon import MuonWithAuxAdam
import hydra
import omegaconf
from loss import CDW_CELoss

torch.autograd.set_detect_anomaly(True)
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
resume_training = True  # True
prev_loss = None
do_test = True
save_model = True
i = 0
alpha = 1e-8  # -5


def loglikelihood(logits):
    probs = torch.nn.functional.log_softmax(logits, dim=1)
    return -probs


def HESS(model: torch.nn.Module, loss: torch.Tensor):
    hess = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            grad = torch.autograd.grad(
                loss, param, retain_graph=True, create_graph=True, allow_unused=True
            )[0].reshape(-1)
            hess_vec = []
            for i in range(len(param)):
                hess_vec.append(
                    torch.autograd.grad(
                        grad[i],
                        param,
                        retain_graph=True,
                        create_graph=True,
                        allow_unused=True,
                    )[0].reshape(-1)
                )
            Hessian = torch.stack(hess_vec)
    return Hessian


def FIM(model: torch.nn.Module, data: torch.Tensor):
    logits = model.forward(data)
    llk = loglikelihood(logits).sum(dim=0).mean()
    fim = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fim[name] = torch.zeros_like(param)

            fim[name] = torch.autograd.grad(
                llk, param, create_graph=True, retain_graph=True
            )
    return fim


class gMLPMixer(pl.LightningModule):
    """_summary_

    Args:
        pl (_type_): _description_
    """
    def __init__(self):
        super().__init__()
        self.model = GraphMLPMixer(
            nfeat_node=8,
            rw_dim=4,
            lap_dim=6,
            dropout=0.1,
            gMHA_type="MLPMixer",
            mlpmixer_dropout=0.0,
            patch_rw_dim=4,
            nfeat_edge=3,
            nhid=64,
            nlayer_gnn=0,
            n_patches=4,
            nout=5,
            nlayer_mlpmixer=8,
            token_dim=[128, 128, 64, 64, 64, 64, 64, 64],
            channel_dim=[512, 512, 256, 256, 128, 64, 64, 64],
        )
        self.save_hyperparameters()
        self.model.apply(weight_init)
        # self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.9)
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
        self.make_log = True
        return

    def forward(self, x: torch.Tensor):
        """_summary_

        Args:
            x (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        return self.model(x)

    def training_step(self, batch):
        """_summary_

        Args:
            batch (_type_): _description_
            batch_idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        x, y = batch, batch.label
        logits, aux_logits = self.model(x)
        y = y.long()
        class_weights = torch.Tensor([0.2, 0.2, 0.2, 0.2, 0.2])
        output = torch.nn.functional.softmax(logits, dim=1)
        loss = self.loss_fn(
            logits, y.float(), weight=torch.Tensor(class_weights)
        ).mean()
        l2_norm = 0
        l1_norm = 0
        i = 0
        for param in self.model.parameters():
            i += 1
            l1_norm += alpha * i * torch.norm(torch.abs(param), 1)
            l2_norm += alpha * i * torch.norm(torch.abs(param**2), 1)
        loss += l2_norm + l1_norm
        auc = self.train_auc(output, y)
        if self.make_log:
            self.log("train/loss", loss, prog_bar=True)
            acc = self.train_acc(output.argmax(dim=1), y)
            self.log("train/acc", acc, prog_bar=True)
            conf_matrix = self.val_confmat.update(output.argmax(dim=1), y)
            fig_, ax_ = self.val_confmat.plot()
            self.log("train/auc", auc, prog_bar=True)
            self.logger.log_image("Confusion Matrix", [fig_])
            self.log("lr", self.optimizers()._optimizer.param_groups[-1]["lr"])
        return loss

    def validation_step(self, batch):
        """_summary_
        Args:
            batch (_type_): _description_
            batch_idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        x, y = batch, batch.label
        logits, aux_logits = self.model(x)
        y = y.long()
        output = torch.nn.functional.softmax(logits, dim=1)
        auc = self.train_auc(output, y)
        class_weights = torch.Tensor([0.2, 0.2, 0.2, 0.2, 0.2])
        loss = self.loss_fn(
            logits, y.float(), weight=torch.Tensor(class_weights)
        ).mean()
        valid_acc = self.valid_acc(output.argmax(dim=1), y)
        if self.make_log:
            self.log("val/auc", auc, prog_bar=True)
            self.log("val/loss", loss, prog_bar=True)
            self.valid_acc(output, y)
            self.log("valid/acc", valid_acc, prog_bar=True)
        return loss

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # Get the lr; t; gamma
        for p in range(len(optimizer.param_groups[0]["params"])):
            if optimizer.param_groups[0]["params"][p].grad is not None:
                size = optimizer.param_groups[0]["params"][p].grad.size()
                optimizer.param_groups[0]["params"][p].grad += alpha * torch.normal(
                    torch.zeros(size), torch.ones(size)
                )
        optimizer.step(closure=optimizer_closure)
        # self.ema.update(self.model.parameters())

    def configure_optimizers(self):
        """
        Returns:
            _type_: _description_
        """
        optimizer = AdamAtan2(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        # optimizer = Muon(self.model.parameters(),  lr=1e-6, weight_decay=1e-5)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.9,
                patience=100,
                threshold=0.0001,
                threshold_mode="rel",
                cooldown=0,
                min_lr=0,
                eps=1e-08,
            ),
            "monitor": "train/loss",
            "interval": "step",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def train(graph_mixer, train_dl, test_dl):
    wandb_logger = WandbLogger(project="JetNet_Muon")
    trainer = pl.Trainer(
        default_root_dir="./JetNet_Muon/",
        max_epochs=1000,
        precision="32-true",
        deterministic=True,
        accumulate_grad_batches=1,
        enable_checkpointing=True,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        logger=wandb_logger,
        accelerator="cpu",
        check_val_every_n_epoch=100,
    )

    trainer.fit(
        graph_mixer,
        train_dl,
        test_dl,
        # ckpt_path="./JetNet_Muon/8hycdao8/checkpoints/epoch=99-step=3000.ckpt"
        # ckpt_path="./JetNet_AdamMini_150/hhklowkp/checkpoints/epoch=79-step=320.ckpt"
        # ckpt_path = "./JetNet_AdamMini_150/ah1xoayh/checkpoints/epoch=19-step=4700.ckpt"
        # ckpt_path = "./JetNet_AdamMini_150/5dlxrmbi/checkpoints/epoch=1-step=10314.ckpt"
        # ckpt_path="./JetNet_AdamMini_150/i7m5a7u5/checkpoints/epoch=6-step=36099.ckpt"
        # ckpt_path="./JetNet_AdamMini_150/j72am1ne/checkpoints/epoch=31-step=7520.ckpt"
        # ckpt_path="./JetNet_AdamMini_150/xagqza2o/checkpoints/epoch=59-step=14100.ckpt"
        # ckpt_path="./JetNet_AdamMini_150/4uys2uml/checkpoints/epoch=99-step=23500.ckpt"
    )
    return trainer


if __name__ == "__main__":
    g = set_seed(0)
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
        generator=g,
        worker_init_fn=seed_worker,
    )
    test_dl = DataLoader(
        trainset_2,
        batch_size=128,
        shuffle=False,
        pin_memory=False,
        generator=g,
        worker_init_fn=seed_worker,
    )
    graph_mixer = gMLPMixer()
    trainer = train(graph_mixer, train_dl, test_dl)
