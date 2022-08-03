from argparse import ArgumentParser
from functools import partial
from typing import Any, Callable, Dict, List, Sequence, Tuple
from convnet import ConvNet

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_NETWORKS = {
            "convnet": ConvNet,
            # "resnet50": resnet50,
            # "vit_tiny": vit_tiny,
        }

class BaseModel(pl.LightningModule):
    def __init__(
        self, 
        args, 
        n_classes,
        target_type):
        super().__init__()

        self.save_hyperparameters()
        self.encoder = ConvNet(args)
        self.classifier = nn.Linear(self.features_size, n_classes)
        
        self.target_type = target_type

        self.metric_keys = ['acc'] if self.target_type == 'single' \
            else ['f1_macro', 'f1_micro', 'f1_weighted']
            
        if self.target_type == '2-way':
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.target_type = nn.CrossEntropyLoss()


    def forward(self, x):
        feats = self.encoder(x)
        logits = self.classifier(feats.detach())
        return {"logits": logits, "feats": feats}

    def _shared_step(self, X: torch.Tensor, targets: torch.Tensor) -> Dict:
        out = self._base_forward(X)
        logits, feats = out["logits"], out["feats"]

        loss = self.loss_fn(logits, targets, **self.loss_args)
        results = self.metric_fn(logits, targets, **self.metric_args)

        return {
            "loss": loss,
            "logits": logits,
            "feats": feats,
            **results
        }

    def training_step(self, batch, batch_idx):
        _, X, targets = batch
        targets = targets['labels'] if isinstance(targets, dict) else targets
        X = [X] if isinstance(X, torch.Tensor) else X

        # check that we received the desired number of crops
        assert len(X) == self.n_crops + self.n_small_crops

        outs = [self._shared_step(x, targets) for x in X[: self.n_crops]]

        # collect data
        logits = [out["logits"] for out in outs]
        feats = [out["feats"] for out in outs]

        # loss and stats
        loss = sum(out["loss"] for out in outs) / self.n_crops
        metrics = {"train_class_loss": loss}

        for key in self.metric_keys:
            metrics.update({f"train_{key}": sum(out[key] for out in outs) / self.n_crops})

        if self.multicrop:
            feats.extend([self.encoder(x) for x in X[self.n_crops :]])

        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        if not self.disable_knn_eval:
            self.knn(
                train_features=torch.cat(outs["feats"][: self.num_crops]).detach(),
                train_targets=targets.repeat(self.num_crops).detach(),
            )

        return {"loss": loss, "feats": feats, "logits": logits}

    def validation_step(self, batch, batch_idx):
        X, targets = batch
        targets = targets['labels'] if isinstance(targets, dict) else targets
        batch_size = targets.size(0)

        out = self._shared_step(X, targets)

        if not self.disable_knn_eval and not self.trainer.running_sanity_check:
            self.knn(test_features=out.pop("feats").detach(), test_targets=targets.detach())

        metrics = {
            "batch_size": batch_size,
            "val_loss": out["loss"]
        }
        for key in self.metric_keys:
            metrics.update({f"val_{key}": out[key]})

        return metrics

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.learnable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.extra_optimizer_args,
        )
        return optimizer 

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("base")
        # parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--encoder", choices=SUPPORTED_NETWORKS.keys(), type=str, default='resnet18')
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser

    
    @property
    def learnable_params(self):

        return [
            {"name": "encoder", "params": self.encoder.parameters()},
            {
                "name": "classifier",
                "params": self.classifier.parameters(),
                "lr": self.classifier_lr,
                "weight_decay": 0,
            },
        ]

