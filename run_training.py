from __future__ import annotations
import argparse
import importlib
import torch
import torchmetrics
from graph_mixer import GraphMLPMixer
from graph_mixer.data_processing import *
from graph_mixer.loss import FocalLoss
from utils import seed_worker, set_seed, weight_init

from adam_atan2_pytorch import AdamAtan2
from adabelief_pytorch import AdaBelief
from lion_pytorch import Lion
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger
from torch_geometric.loader.dataloader import DataLoader

from data import (
    TopQuarkDataset,
    JetClassDataset,
    QGDataset,
    JetNetDataset,
    WTagDataset,
)
from dataclasses import asdict


def get_the_arguments():

    parser = argparse.ArgumentParser(
        prog="Graph MLP Mixer for Jet Tagging",
        description="The script allows to train and infer the model using the input config file",
    )

    parser.add_argument("config")
    args = parser.parse_args()
    config_path = args.config
    config = importlib.util.spec_from_file_location("Config", config_path)
    # TODO realize this classes and inport from path:
    from configs.toptag_config import Config, MixerConfig
    config = Config()
    mixer_config = MixerConfig()
    return config, mixer_config


class gMLPMixer(pl.LightningModule):
    def __init__(self, mixer_config, training_config):
        super().__init__()
        self.config = training_config

        self.model = GraphMLPMixer(**{k: v for k, v in asdict(mixer_config).items()})

        self.save_hyperparameters()

        self.model.apply(weight_init)

        self.loss_fn = FocalLoss(gamma=5.0)
        self.train_acc = torchmetrics.classification.BinaryAccuracy()
        self.valid_acc = torchmetrics.classification.BinaryAccuracy()
        self.train_auc = torchmetrics.classification.BinaryAUROC()
        self.valid_auc = torchmetrics.classification.BinaryAUROC()
        # self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.995)
        self.confmat = torchmetrics.ConfusionMatrix(
            normalize="true", task="binary", num_classes=2
        )
        self.val_confmat = torchmetrics.ConfusionMatrix(
            normalize="true", task="binary", num_classes=2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """_summary_

        Args:
            batch (_type_): _description_
            batch_idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        x, y = batch, batch.label
        try:
            logits = self.model(x)
        except Exception:
            pdb.set_trace()
        logits = torch.nan_to_num(logits, 1e-16)
        class_weights = [
            0.5,
            0.5,
        ]  # batch.label.size(0)/(torch.histc(batch.label,  bins=2)*2)
        noise_std = 0.001
        output = torch.nn.functional.softmax(logits, dim=1)
        l1_norm = 0
        l2_norm = 0
        i = 0
        _len = len(list(self.model.parameters()))
        for param in self.model.parameters():
            i += 1
            l1_norm += self.config.alpha * (_len - i) * torch.norm(torch.abs(param), 1)
            l2_norm += (
                self.config.alpha * (_len - i) * torch.norm(torch.abs(param**2), 1)
            )
        loss = self.loss_fn(
            logits, y.long(), weight=torch.Tensor(class_weights)
        ) + self.config.alpha * (l1_norm + l2_norm)
        loss += torch.rand_like(loss) * noise_std
        if torch.isnan(loss.mean()):
            pdb.set_trace()
        loss = torch.nan_to_num(loss, 1e-16)
        loss = loss.mean()
        auc = self.train_auc(output.argmax(dim=1), y)
        self.log("train/loss", loss.detach(), prog_bar=True)
        acc = self.train_acc(output.argmax(dim=1), y)
        self.log("train/acc", acc, prog_bar=True)
        conf_matrix = self.confmat.update(output.argmax(dim=1), y)
        fig_, ax_ = self.confmat.plot()
        self.log("train/auc", auc, prog_bar=True)
        self.log("lr", self.optimizers()._optimizer.param_groups[-1]["lr"])
        self.logger.log_image("Validation Confusion Matrix", [fig_])
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch, batch.label
        logits = self.model(x)
        output = torch.nn.functional.softmax(logits, dim=1)
        auc = self.train_auc(output.argmax(dim=1), y)
        class_weights = torch.histc(batch.label, bins=2) / batch.label.size(0)
        loss = self.loss_fn(logits, y.long(), weight=torch.Tensor(class_weights)).mean()
        valid_acc = self.valid_acc(output.argmax(dim=1), y)
        self.log("val/auc", auc, prog_bar=True)
        self.log("val/loss", loss, prog_bar=True)
        self.log("valid/acc", valid_acc, prog_bar=True)
        conf_matrix = self.val_confmat.update(output.argmax(dim=1), y)
        fig_, ax_ = self.val_confmat.plot()
        self.logger.log_image("Validation Confusion Matrix", [fig_])
        return loss

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        optimizer = optimizer.optimizer
        optimizer.step(closure=optimizer_closure)

    def configure_optimizers(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        if self.config.optimizer == "lion":
            optimizer = Lion(self.model.parameters(), lr=4e-4, weight_decay=1e-2)
        elif self.config.optimizer == "adam":
            optimizer = AdamAtan2(self.model.parameters(), lr=1e-6, weight_decay=1e-2)
        elif self.config.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=1e-3, weight_decay=1e-5
            )
        elif self.config.optimizer == "adabelief":
            optimizer = AdaBelief(
                self.model.parameters(),
                lr=1e-6,
                eps=1e-16,
                betas=(0.9, 0.999),
                weight_decouple=True,
                rectify=False,
            )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.9,
                patience=100,
                threshold=0.000001,
                threshold_mode="rel",
                cooldown=0,
                min_lr=1e-8,
                eps=1e-08,
            ),
            "monitor": "train/loss",
            "interval": "step",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def train(graph_mixer, train_dl, test_dl):
    wandb_logger = WandbLogger(project="TopQuark_Muon")
    # code_artifact = wandb.Artifact(name="executable",type="code")
    # wandb_logger.use_artifact("./run_topquark.py", artifact_type="code")
    wandb_logger.watch(graph_mixer, log="all")  ##log_graph=True)
    trainer = pl.Trainer(
        default_root_dir="./TopQuark_Muon/",
        max_epochs=200,
        precision="32",
        deterministic=True,
        enable_checkpointing=True,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        gradient_clip_algorithm="norm",
        logger=wandb_logger,
        accelerator="cpu",
        detect_anomaly=True,
        check_val_every_n_epoch=2,
    )
    trainer.fit(graph_mixer, train_dl, test_dl)
    return trainer


def load_dataset(config):
    if config.dataset == "top_quark":
        train_ds = TopQuarkDataset.JetTopTagDataset(input_path=config.path)
        test_ds = TopQuarkDataset.JetTopTagDataset(input_path=config.path, mode="test")
    elif config.dataset == "quark_gluon":
        train_ds = QGDataset.JetQGDataset(input_path=config.path)
        test_ds = QGDataset.JetQGDataset(input_path=config.path, mode="test")
    elif config.dataset == "w_boson":
        train_ds = WTagDataset.WTagDataset(input_path=config.path)
        test_ds = WTagDataset.WTagDataset(input_path=config.path, mode="test")
    elif config.dataset == "jetnet":
        train_ds = JetNetDataset.JetNetDataset(input_path=config.path)
        test_ds = JetNetDataset.JetNetDataset(input_path=config.path, mode="test")
    elif config.dataset == "jetclass":
        train_ds = JetClassDataset.JetClassDataset(input_path=config.path)
        test_ds = JetClassDataset.JetClassDataset(input_path=config.path, mode="test")
    
    train_dl = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        generator=g,
        worker_init_fn=seed_worker,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=True,
        generator=g,
        worker_init_fn=seed_worker,
    )
    return train_dl, test_dl

if __name__ == "__main__":
    g = set_seed(0)
    config, mixer_config = get_the_arguments()
    train_dl, test_dl = load_dataset(config=config)
    graph_mixer = gMLPMixer(mixer_config, config)
    trainer = train(graph_mixer, train_dl, test_dl)
