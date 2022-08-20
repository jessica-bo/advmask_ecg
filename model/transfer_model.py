from argparse import ArgumentParser
from tarfile import TarError
from typing import Any, Dict, List, Optional, Sequence, Tuple
import os, sys 

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
logging.basicConfig(level=logging.NOTSET)

sys.path.append('../utils')
from utils.metrics import weighted_mean, evaluate_single

sys.path.append('../data')
from data.cinc2021.utils_cinc2021 import evaluate_scores

from .backbones import BACKBONES

# import logging
# logging.basicConfig(level=logging.NOTSET)

class TransferModel(pl.LightningModule):
    def __init__(
        self,
        encoder,
        target_type,
        n_classes,
        max_epochs,
        batch_size,
        lr,
        weight_decay,
        finetune,
        **kwargs,
    ):
        """Implements linear evaluation.

        Args:
            encoder (nn.Module): backbone architecture for feature extraction.
            n_classes (int): number of classes in the dataset.
            max_epochs (int): total number of epochs.
            batch_size (int): batch size.
            optimizer (str): optimizer to use.
            lr (float): learning rate.
            weight_decay (float): weight decay.
        """

        super().__init__()
        self.save_hyperparameters()

        self.encoder = encoder
        self.target_type = target_type
        feat_in = self.encoder.embedding_dim
        self.classifier = nn.Linear(feat_in, n_classes)  # type: ignore

        # training related
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay

        self.finetune = finetune
        if not self.finetune:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if self.target_type == 'multilabel':
            self.loss_fn = torch.nn.BCELoss() 
        elif self.target_type == "single":
            self.loss_fn = F.cross_entropy
        
        self.eval_fn = evaluate_single

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
        parser.add_argument("--weight_decay", type=float, default=0)
        parser.add_argument("--num_workers", type=int, default=9)
        parser.add_argument("--embedding_dim", type=int, default=512)

        # wandb
        parser.add_argument("--name", type=str)
        parser.add_argument("--project", type=str)
        parser.add_argument("--entity", type=str)
        parser.add_argument("--wandb", action='store_true', default=False)

        return parent_parser

    def forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs forward pass of the frozen encoder and the linear layer for evaluation.

        Args:
            X (torch.tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing features and logits.
        """
        with torch.no_grad():
            feats = self.encoder(X)

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

        if self.finetune:
            optimizer = torch.optim.Adam(
                list(self.classifier.parameters()) + list(self.encoder.parameters()),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(
                self.classifier.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
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
        target = target.type(torch.float) if self.target_type == "multilabel" else target
        loss = self.loss_fn(out, target)

        auc, acc = self.eval_fn(target.cpu().detach().numpy(), out.cpu().detach().numpy())
        
        return batch_size, loss, acc, auc

    def shared_epoch_end(self, outs: List[Dict[str, Any]]):
        """Averages the losses and accuracies of all the validation batches.
        This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.

        Args:
            outs (List[Dict[str, Any]]): list of outputs of the validation step.
        """

        loss = weighted_mean(outs, "loss", "batch_size")
        acc = weighted_mean(outs, "acc", "batch_size")
        auc = weighted_mean(outs, "auc", "batch_size")

        return loss, acc, auc

    def training_step(self, batch: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Performs the training step for the linear eval.

        Args:
            batch (torch.Tensor): a batch of images in the tensor format.
        Returns:
            torch.Tensor: cross-entropy loss between the predictions and the ground truth.
        """

        # set encoder to eval mode
        self.encoder.eval()
        batch_size, loss, acc, auc = self.shared_step(batch)

        results = {
            "batch_size": batch_size,
            "loss": loss,
            "acc": acc,
            "auc": auc,
        }
        return results

    def training_epoch_end(self, outs: List[Dict[str, Any]]):
        loss, acc, auc = self.shared_epoch_end(outs)
        print("Finished validation with accuracy {}, AUC {}".format(acc, auc))

        log = {"transfer/train_loss": loss, "transfer/train_acc": acc, "transfer/train_auc": auc}
        self.log_dict(log, on_epoch=True, sync_dist=True)

    def validation_step(self, batch: torch.Tensor, *args, **kwargs) -> Dict[str, Any]:
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
            "loss": loss,
            "acc": acc,
            "auc": auc,
        }
        return results

    def validation_epoch_end(self, outs: List[Dict[str, Any]]):
        val_loss, acc, auc = self.shared_epoch_end(outs)
        print("Finished validation with accuracy {}, AUC {}".format(acc, auc))

        log = {"transfer/val_loss": val_loss, "transfer/val_acc": acc, "transfer/val_auc": auc}
        self.log_dict(log, sync_dist=True)

    @torch.no_grad()
    def test_step(self, batch: torch.Tensor, *args, **kwargs) -> Dict[str, Any]:
        """Performs the test step for the linear eval.

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
            "loss": loss,
            "acc": acc,
            "auc": auc,
        }
        return results

    def test_epoch_end(self, outs: List[Dict[str, Any]]):
        test_loss, acc, auc = self.shared_epoch_end(outs)
        print("Finished test with accuracy {}, AUC {}".format(acc, auc))

        log = {"transfer/test_loss": test_loss, "transfer/test_acc": acc, "transfer/test_auc": auc}
        self.log_dict(log, sync_dist=True)
