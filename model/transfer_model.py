from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Sequence, Tuple
import sys 

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../utils')
from utils.metrics import weighted_mean, evaluate_single


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
        """
        Linear evaluation (LE) or finetuning (FT).
        """
        super().__init__()

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
            print("Freezing encoder params...")
            for param in self.encoder.parameters():
                param.requires_grad = False

        if self.target_type == 'multilabel':
            self.loss_fn = torch.nn.BCELoss() 
        elif self.target_type == "single":
            self.loss_fn = F.cross_entropy
        elif self.target_type == "binary":
            self.loss_fn = torch.nn.BCEWithLogitsLoss() 
        elif self.target_type == "regression":
            self.loss_fn = torch.nn.L1Loss()
        
        self.eval_fn = evaluate_single

        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
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
        parser.add_argument("--wandb_key", type=str, default="your_key")

        return parent_parser

    def configure_optimizers(self):
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

    def forward(self, X):
        """
        Performs forward pass of the encoder and the linear layer for evaluation.
        """
        with torch.no_grad():
            feats = self.encoder(X)

        logits = self.classifier(feats)
        return {"logits": logits, "feats": feats}

    def shared_step(self, batch):
        """
        Performs operations that are shared between the training nd validation steps.
        """
        X, target = batch
        batch_size = X.size(0)

        out = self(X)["logits"]
        target = target.type(torch.float) if (self.target_type in ["multilabel", "binary", "regression"]) else target
        loss = self.loss_fn(out.squeeze(), target)

        auc, acc = self.eval_fn(target.cpu().detach().numpy(), out.cpu().detach().numpy(), classification=self.target_type)
        
        return batch_size, loss, acc, auc

    def shared_epoch_end(self, outs):
        """
        Averages the losses and accuracies of all the validation batches.
        """
        loss = weighted_mean(outs, "loss", "batch_size")
        acc = weighted_mean(outs, "acc", "batch_size")
        auc = weighted_mean(outs, "auc", "batch_size")

        return loss, acc, auc

    def training_step(self, batch, *args, **kwargs):
        """
        Performs the training step for transfer.
        """
        # set encoder to eval mode
        if not self.finetune:
            self.encoder.eval()
        batch_size, loss, acc, auc = self.shared_step(batch)

        results = {
            "batch_size": batch_size,
            "loss": loss,
            "acc": acc,
            "auc": auc,
        }
        return results

    def training_epoch_end(self, outs):
        loss, acc, auc = self.shared_epoch_end(outs)

        log = {"transfer/train_loss": loss, "transfer/train_acc": acc, "transfer/train_auc": auc}
        self.log_dict(log, on_epoch=True, sync_dist=True)

    def validation_step(self, batch, *args, **kwargs):
        """
        Performs the validation step for transfer.
        """
        batch_size, loss, acc, auc = self.shared_step(batch)
        results = {
            "batch_size": batch_size,
            "loss": loss,
            "acc": acc,
            "auc": auc,
        }
        return results

    def validation_epoch_end(self, outs):
        val_loss, acc, auc = self.shared_epoch_end(outs)
        print("Finished validation with accuracy {}, AUC {}".format(acc, auc))

        log = {"transfer/val_loss": val_loss, "transfer/val_acc": acc, "transfer/val_auc": auc}
        self.log_dict(log, sync_dist=True)

    @torch.no_grad()
    def test_step(self, batch, *args, **kwargs):
        """
        Performs the test step for transfer.
        """
        batch_size, loss, acc, auc = self.shared_step(batch)
        results = {
            "batch_size": batch_size,
            "loss": loss,
            "acc": acc,
            "auc": auc,
        }
        return results

    def test_epoch_end(self, outs):
        test_loss, acc, auc = self.shared_epoch_end(outs)

        log = {"transfer/test_loss": test_loss, "transfer/test_acc": acc, "transfer/test_auc": auc}
        self.log_dict(log, sync_dist=True)