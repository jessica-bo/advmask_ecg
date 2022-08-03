from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Sequence, Tuple
import os, sys 

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../utils')
from utils.metrics import calculate_auc, calculate_acc, weighted_mean
from backbones.resnet import resnet1d18

class LinearModel(pl.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        n_classes: int,
        max_epochs: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        extra_optimizer_args: dict,
        **kwargs,
    ):
        """Implements linear evaluation.

        Args:
            backbone (nn.Module): backbone architecture for feature extraction.
            n_classes (int): number of classes in the dataset.
            max_epochs (int): total number of epochs.
            batch_size (int): batch size.
            optimizer (str): optimizer to use.
            lr (float): learning rate.
            weight_decay (float): weight decay.
            extra_optimizer_args (dict): extra optimizer arguments.
        """

        super().__init__()

        self.backbone = backbone
        if self.backbone == None: 
            self.backbone = resnet1d18()

        feat_in = self.backbone.features_size
        self.classifier = nn.Linear(feat_in, n_classes)  # type: ignore

        # training related
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.extra_optimizer_args = extra_optimizer_args

        # all the other parameters
        self.extra_args = kwargs

        for param in self.backbone.parameters():
            param.requires_grad = False

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds basic linear arguments.

        Args:
            parent_parser (ArgumentParser): argument parser that is used to create a
                argument group.

        Returns:
            ArgumentParser: same as the argument, used to avoid errors.
        """

        parser = parent_parser.add_argument_group("linear")

        # general train
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--lr", type=float, default=0.1)
        parser.add_argument("--weight_decay", type=float, default=0.0001)
        parser.add_argument("--num_workers", type=int, default=0)

        # wandb
        parser.add_argument("--name", type=str)
        parser.add_argument("--project", type=str)
        parser.add_argument("--entity", type=str)
        parser.add_argument("--wandb", action='store_true', type=bool, default=False)
        # parser.add_argument("--offline", action='store_true', type=bool, default=False)

        # # optimizer
        # SUPPORTED_OPTIMIZERS = ["sgd", "adam"]

        # parser.add_argument("--optimizer", choices=SUPPORTED_OPTIMIZERS,
        #                     type=str, default="sgd")


        return parent_parser

    def forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs forward pass of the frozen backbone and the linear layer for evaluation.

        Args:
            X (torch.tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing features and logits.
        """

        with torch.no_grad():
            feats = self.backbone.encoder(X)

        logits = self.classifier(feats)
        return {"logits": logits, "feats": feats}

    def configure_optimizers(self) -> Tuple[List, List]:
        """Configures the optimizer for the linear layer.

        Raises:
            ValueError: if the optimizer is not in (sgd, adam).
            ValueError: if the scheduler is not in not in (warmup_cosine, cosine, reduce, step,
                exponential).

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """

        optimizer = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.extra_optimizer_args,
        )

        return optimizer

    def shared_step(
        self, batch: Tuple) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs operations that are shared between the training nd validation steps.

        Args:
            batch (Tuple): a batch of images in the tensor format.
        Returns:
            Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
                batch size, loss, accuracy @1 and accuracy @5.
        """

        X, target = batch
        batch_size = X.size(0)

        out = self(X)["logits"]

        loss = F.cross_entropy(out, target)

        acc = calculate_acc(out, target)
        auc = calculate_auc(out, target)
        return batch_size, loss, acc, auc

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        """Performs the training step for the linear eval.

        Args:
            batch (torch.Tensor): a batch of images in the tensor format.
        Returns:
            torch.Tensor: cross-entropy loss between the predictions and the ground truth.
        """

        # set encoder to eval mode
        self.backbone.eval()

        _, loss, acc, auc = self.shared_step(batch)

        log = {"train_loss": loss, "accuracy": acc, "AUC": auc}
        self.log_dict(log, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: torch.Tensor) -> Dict[str, Any]:
        """Performs the validation step for the linear eval.

        Args:
            batch (torch.Tensor): a batch of images in the tensor format.
        Returns:
            Dict[str, Any]:
                dict with the batch_size (used for averaging),
                the classification loss and accuracies.
        """

        batch_size, loss, acc, auc = self.shared_step(batch)

        results = {
            "batch_size": batch_size,
            "val_loss": loss,
            "accuracy": acc,
            "AUC": auc,
        }
        return results

    def validation_epoch_end(self, outs: List[Dict[str, Any]]):
        """Averages the losses and accuracies of all the validation batches.
        This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.

        Args:
            outs (List[Dict[str, Any]]): list of outputs of the validation step.
        """

        val_loss = weighted_mean(outs, "val_loss", "batch_size")
        acc = weighted_mean(outs, "accuracy", "batch_size")
        auc = weighted_mean(outs, "AUC", "batch_size")

        log = {"val_loss": val_loss, "accuracy": acc, "AUC": auc}
        self.log_dict(log, sync_dist=True)
