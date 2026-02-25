from __future__ import annotations
import argparse, sys
import pdb
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
from matplotlib import pyplot as plt
from dataclasses import asdict
from data import (
    TopQuarkDataset,
    JetClassDataset,
    QGDataset,
    JetNetDataset,
    WTagDataset,
)
import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity, record_function


def get_the_arguments():
    parser = argparse.ArgumentParser(
        prog="Graph MLP Mixer for Jet Tagging",
        description="The script allows to train and infer the model using the input config file",
    )

    parser.add_argument("config")
    args = parser.parse_args()
    config_path = args.config

    config_spec = importlib.util.spec_from_file_location("Config", config_path)
    config = importlib.util.module_from_spec(config_spec)
    sys.modules["Config"] = config
    config_spec.loader.exec_module(config)
    # Return 3 config one is about to the data, second for Neural Network parameter and third for Training Process
    return config.Config(), config.MixerConfig(), config.TrainingConfig()


class gMLPMixer(pl.LightningModule):
    def __init__(self, mixer_config, training_config):
        super().__init__()
        self.config = training_config
        self.mixer_config = mixer_config
        self.model = GraphMLPMixer(**{k: v for k, v in asdict(mixer_config).items()})
        self.save_hyperparameters()
        self.model.apply(weight_init)
        self.loss_fn = FocalLoss(gamma=2.0)
        self.train_acc = torchmetrics.classification.Accuracy(
            num_classes=mixer_config.nout,
            task="binary" if mixer_config.nout == 2 else "multiclass",
        )
        self.valid_acc = torchmetrics.classification.Accuracy(
            num_classes=mixer_config.nout,
            task="binary" if mixer_config.nout == 2 else "multiclass",
        )
        self.train_auc = torchmetrics.classification.AUROC(
            num_classes=mixer_config.nout,
            task="binary" if mixer_config.nout == 2 else "multiclass",
        )
        self.valid_auc = torchmetrics.classification.AUROC(
            num_classes=mixer_config.nout,
            task="binary" if mixer_config.nout == 2 else "multiclass",
        )
        # self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.995)
        self.confmat = torchmetrics.ConfusionMatrix(
            normalize="true",
            task="binary" if mixer_config.nout == 1 else "multiclass",
            num_classes=mixer_config.nout,
        )
        self.val_confmat = torchmetrics.ConfusionMatrix(
            normalize="true",
            task="binary" if mixer_config.nout == 1 else "multiclass",
            num_classes=mixer_config.nout,
        )
        # self.qat_quantizer = Int8DynActInt4WeightQATQuantizer()
        # self.model = self.qat_quantizer.prepare(self.model)

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

        logits = self.model(x)
        logits = torch.nan_to_num(logits, 1e-16)
        class_weights = batch.label.size(0) / (
            torch.histc(batch.label, bins=self.mixer_config.nout)
        )
        output = torch.nn.functional.softmax(logits, dim=1)
        noise_std = 1e-4
        # l1_norm = torch.Tensor([torch.norm(torch.abs(param), 1) for param in self.model.parameters()])
        # l2_norm = torch.pow(l1_norm,2)
        loss = self.loss_fn(logits, y.long(), weight=torch.Tensor(class_weights))

        loss += torch.rand_like(loss) * noise_std
        if torch.isnan(loss.mean()):
            pdb.set_trace()
        sch = self.lr_schedulers()
        loss = torch.nan_to_num(loss, 1e-16)
        loss = (
            loss.mean()
        )  # + self.config.alpha *(torch.sum(l1_norm) + torch.sum(l2_norm))
        if batch_idx % 10000:
            auc = self.train_auc(output, y.to(dtype=torch.int))
            self.log(
                "train/loss",
                loss.detach(),
                prog_bar=True,
                batch_size=self.config.batch_size,
            )
            acc = self.train_acc(output, y)
            self.log("train/acc", acc, prog_bar=True, batch_size=self.config.batch_size)
            conf_matrix = self.confmat.update(output.argmax(dim=1), y)
            fig_, ax_ = self.confmat.plot()
            self.log("train/auc", auc, prog_bar=True, batch_size=self.config.batch_size)
            self.log("lr", self.optimizers()._optimizer.param_groups[-1]["lr"])
            self.logger.log_image("Train Confusion Matrix", [fig_])
            plt.close()
        sch.step()
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch, batch.label
        logits = self.model(x)
        output = torch.nn.functional.softmax(logits, dim=1)
        class_weights = torch.histc(
            batch.label, bins=self.mixer_config.nout
        ) / batch.label.size(0)
        # pdb.set_trace()
        loss = self.loss_fn(logits, y.long(), weight=torch.Tensor(class_weights)).mean()
        if batch_idx % 10000:
            auc = self.valid_auc(output, y.to(dtype=torch.int))

            valid_acc = self.valid_acc(output, y)
            self.log("val/auc", auc, prog_bar=True, batch_size=self.config.batch_size)
            self.log("val/loss", loss, prog_bar=True, batch_size=self.config.batch_size)
            self.log(
                "valid/acc", valid_acc, prog_bar=True, batch_size=self.config.batch_size
            )
            conf_matrix = self.val_confmat.update(output.argmax(dim=1), y)
            fig_, ax_ = self.val_confmat.plot()
            plt.close()
            self.logger.log_image("Validation Confusion Matrix", [fig_])
        return loss

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        optimizer = optimizer.optimizer
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def configure_optimizers(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        if self.config.optimizer == "lion":
            optimizer = Lion(self.model.parameters(), lr=4e-4, weight_decay=1e-2)
        elif self.config.optimizer == "adam":
            optimizer = AdamAtan2(self.model.parameters(), lr=1e-3, weight_decay=1e-2)
        elif self.config.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=1e-3,
                weight_decay=1e-5,
                betas=(0.89, 0.99),
                # lr=1e-4, weight_decay=1e-2
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
        # Select Optimizer
        print("Scheduler:", config.scheduler)
        if config.scheduler == "ReduceOnPlateau":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=0.5,
                    patience=10,
                    threshold=0.0001,
                    threshold_mode="rel",
                    cooldown=0,
                    min_lr=1e-8,
                    eps=1e-08,
                ),
                "monitor": "train/loss",
                "interval": "step",
                "frequency": 1,
            }
        elif config.scheduler == "cosine_warmup_annealing":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=1, T_mult=8, eta_min=1e-8, last_epoch=-1
                ),
                "monitor": "train/loss",
                "interval": "step",
                "frequency": 1,
            }
        else:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=5e-3,
                total_steps=2000,
                epochs=1,
                steps_per_epoch=200,
                pct_start=0.3,
                anneal_strategy="cos",
                cycle_momentum=True,
                base_momentum=0.9,
                max_momentum=0.99,
                div_factor=0.68,
                final_div_factor=10000.0,
                three_phase=True,
                last_epoch=-1,
            )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def train(config, graph_mixer, train_dl, test_dl):
    wandb_logger = WandbLogger(project=config.dataset)
    wandb_logger.watch(graph_mixer, log="all")  ##log_graph=True)

    trainer = pl.Trainer(
        default_root_dir=config.root_dir,
        max_epochs=200,
        precision="32",
        deterministic=False,
        enable_checkpointing=True,
        gradient_clip_val=0.25,
        accumulate_grad_batches=1,
        gradient_clip_algorithm="norm",
        logger=wandb_logger,
        accelerator="cpu",
        detect_anomaly=False,
        check_val_every_n_epoch=3,
    )
    trainer.fit(
        graph_mixer, train_dl, test_dl
    )  # ckpt_path="./quark_gluon/v7ucneiw/checkpoints/epoch=4-step=3910.ckpt")
    return trainer


def load_dataset(config):
    if config.dataset == "top_quark":
        train_ds = TopQuarkDataset.JetTopTagDataset(config=config)
        test_ds = TopQuarkDataset.JetTopTagDataset(config=config, mode="test")
    elif config.dataset == "quark_gluon":
        train_ds = QGDataset.JetQGDataset(config=config)
        test_ds = QGDataset.JetQGDataset(config=config, mode="test")
    elif config.dataset == "w_boson":
        train_ds = WTagDataset.WTagDataset(config=config)
        test_ds = WTagDataset.WTagDataset(config=config, mode="test")
    elif "jetnet" in config.dataset:
        _train_ds = JetNetDataset.JetNetDataset(config=config)
        evens = list(range(0, len(_train_ds), 2))
        odds = list(range(1, len(_train_ds), 2))
        train_ds = torch.utils.data.Subset(
            _train_ds, evens + odds[: int(len(odds) / 2)]
        )
        test_ds = torch.utils.data.Subset(_train_ds, odds[int(len(odds) / 2) :])
    elif config.dataset == "jetclass":
        train_ds = JetClassDataset.JetClassDataset(config=config)
        test_ds = JetClassDataset.JetClassDataset(config=config, mode="test")

    train_dl = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        generator=g,
        worker_init_fn=seed_worker,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
        generator=g,
        worker_init_fn=seed_worker,
    )
    return train_dl, test_dl


if __name__ == "__main__":
    config, mixer_config, train_config = get_the_arguments()
    g = set_seed(config.seed)
    train_dl, test_dl = load_dataset(config=config)
    graph_mixer = gMLPMixer(mixer_config, config)
    # torch.serialization.safe_globals([mixer_config, config])
    # graph_mixer = gMLPMixer.load_from_checkpoint("./quark_gluon/ghe2tz9a/checkpoints/epoch=0-step=782.ckpt", weights_only=False)

    # if argparse.train:
    trainer = train(config, graph_mixer, train_dl, test_dl)
    # elif argparse.validate:
    #    trainer = train(config, graph_mixer, train_dl, test_dl)
