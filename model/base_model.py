"""
Adapted from @YugeTen 
Source: https://github.com/YugeTen/adios/blob/main/src/methods/base.py
        https://github.com/YugeTen/adios/blob/main/src/methods/simclr.py
"""

from argparse import ArgumentParser
from typing import Any, Dict, List, Sequence
import sys

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../utils')
from utils.losses import simclr_loss_fn
from utils.metrics import weighted_mean, evaluate_single

sys.path.append('../data')
from data.cinc2021.utils_cinc2021 import evaluate_scores

from .backbones import BACKBONES

class BaseModel(pl.LightningModule):
    def __init__(
        self, 
        encoder_name,
        n_classes,
        target_type,
        max_epochs,
        batch_size,
        lr,
        weight_decay,
        temperature,
        proj_hidden_dim,
        output_dim,
        positive_pairing,
        simclr_loss_only,
        **kwargs):
        super().__init__()

        self.save_hyperparameters()
        self.encoder_name = encoder_name
        self.encoder = BACKBONES[self.encoder_name](**kwargs)
        print("Loaded {} backbone.".format(self.encoder_name))
                        
        self.lr = lr
        self.max_epochs = max_epochs
        self.temperature = temperature
        self.positive_pairing = positive_pairing
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.target_type = target_type
        self.simclr_loss_only = simclr_loss_only # do no include classification loss 

        self.projector = nn.Sequential(
            nn.Linear(self.encoder.embedding_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
        )
        self.classifier = nn.Linear(self.encoder.embedding_dim, n_classes)

        if target_type == 'multilabel':
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
            self.eval_fn = evaluate_scores
        elif target_type == "single":
            self.loss_fn = F.cross_entropy
            self.eval_fn = evaluate_single

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("base")

        # general train
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--weight_decay", type=float, default=0)
        parser.add_argument("--num_workers", type=int, default=9)
        parser.add_argument("--embedding_dim", type=int, default=512)
        parser.add_argument("--simclr_loss_only", action='store_true', default=False)

        # wandb
        parser.add_argument("--name", type=str)
        parser.add_argument("--project", type=str)
        parser.add_argument("--entity", type=str)
        parser.add_argument("--wandb", action='store_true', default=False)

        parser.add_argument("--encoder_name", type=str, default='resnet')
        parser.add_argument("--output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)
        parser.add_argument("--temperature", type=float, default=0.1)

        return parent_parser

    @property
    def learnable_params(self) -> Dict[str, Any]:
        learnable_params = list(self.encoder.parameters()) + list(self.projector.parameters()) + list(self.classifier.parameters())
        return {"encoder": learnable_params}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.learnable_params['encoder'],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer 

    def forward(self, x):
        feats = self.encoder(x)
        logits = self.classifier(feats.detach())
        z = self.projector(feats)
        return {"logits": logits, "feats": feats, "z": z}
        
    def shared_step(self, X: torch.Tensor, targets: torch.Tensor) -> Dict:
        batch_size = X.size(0)

        out = self.forward(X)
        logits, feats, z = out["logits"], out["feats"], out["z"]     

        targets = targets.type(torch.float) if self.target_type == "multilabel" else targets
        class_loss = self.loss_fn(logits + 1e-10, targets.type(torch.float))

        auc, acc = self.eval_fn(targets.detach().cpu().numpy(), logits.detach().cpu().numpy())

        return {
            "class_loss": class_loss,
            "batch_size": batch_size,
            "logits": logits,
            "feats": feats,
            "z": z,
            "acc": acc,
            "auc": auc,
        }

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for SimCLR and supervised SimCLR reusing BaseModel training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.n_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of SimCLR loss and classification loss.
        """
        X, targets = batch
        targets = targets['labels'] if isinstance(targets, dict) else targets
        outs = [self.shared_step(X[:,:,:,i], targets) for i in range(X.shape[-1])]
        z1, z2 = [out["z"] for out in outs]

        classification_loss = sum(out["class_loss"] for out in outs) / 2
        nce_loss = simclr_loss_fn(latent_embeddings=[z1, z2], 
                                  positive_pairing=self.positive_pairing, 
                                  temperature=self.temperature)

        if self.simclr_loss_only: 
            loss = nce_loss # don't need to accumulate batches since Trainer handles it 
        else:
            loss = (nce_loss + classification_loss) 

        metrics = {
            "nce_loss": nce_loss,
            "class_loss": classification_loss,
            "loss": loss,
            "batch_size": outs[0]["batch_size"],
            "acc": sum(out["acc"] for out in outs) / 2,
            "auc": sum(out["auc"] for out in outs) / 2,
        }

        return metrics

    def training_epoch_end(self, outs: List[Dict[str, Any]]):
        class_loss = weighted_mean(outs, "class_loss", "batch_size")
        acc = weighted_mean(outs, "acc", "batch_size")
        auc = weighted_mean(outs, "auc", "batch_size")
        nce_loss = weighted_mean(outs, "nce_loss", "batch_size")
        
        metrics = {"train_class_loss": class_loss, 
                   "train_acc": acc, 
                   "train_auc": auc,
                   "train_nce_loss": nce_loss}

        self.log_dict(metrics, on_epoch=True, on_step=False, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        X, targets = batch
        targets = targets['labels'] if isinstance(targets, dict) else targets
        out = self.shared_step(X, targets)

        metrics = {
            "batch_size": out["batch_size"],
            "val_class_loss": out["class_loss"],
            "val_acc": out["acc"],
            "val_auc": out["auc"]
        }

        return metrics

    def validation_epoch_end(self, outs: List[Dict[str, Any]]):
        """Averages the losses and accuracies of all the validation batches.
        This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.

        Args:
            outs (List[Dict[str, Any]]): list of outputs of the validation step.
        """
        val_class_loss = weighted_mean(outs, "val_class_loss", "batch_size")
        acc = weighted_mean(outs, "val_acc", "batch_size")
        auc = weighted_mean(outs, "val_auc", "batch_size")
        print("Finished validation with accuracy {}, AUC {}".format(acc, auc))
        
        metrics = {
            "val_class_loss": val_class_loss,
            "val_acc": acc,
            "val_auc": auc
        }

        self.log_dict(metrics, on_epoch=True, on_step=False, sync_dist=True)
