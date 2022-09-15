import numpy as np
import os, sys
from functools import partial
from argparse import ArgumentParser
from typing import Any, Dict, List, Tuple, Callable, Sequence

import torch
from scipy.signal import stft, istft

from .base_model import BaseModel

sys.path.append('../data')
from data.augs import Normalize


class AdversarialModel(BaseModel):
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
        adv_lr,
        train_mask_interval,
        augmentation_model,
        accumulate_grad_batches,
        **kwargs):

        super().__init__(
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
            **kwargs)

        self.accumulate_grad_batches = 1 if accumulate_grad_batches==None else accumulate_grad_batches
        self.lr = lr * self.accumulate_grad_batches
        
        self.train_mask_interval = train_mask_interval
        self.adv_lr = adv_lr * self.accumulate_grad_batches
        self.normalize = Normalize()

        self.augmenter = augmentation_model
        
        self.save_hyperparameters()
        
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parent_parser = super(AdversarialModel, AdversarialModel).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("adversarial")
        parser.add_argument("--adv_lr", type=float, default=0.001)
        parser.add_argument("--train_mask_interval", type=int, default=1)
        return parent_parser

    @property
    def automatic_optimization(self) -> bool:
        return False

    @property
    def learnable_params(self) -> Dict[str, Any]:
        encoder_learnable_params = super().learnable_params["encoder"]
        adv_learnable_params = list(self.augmenter.parameters()) 
        return {"encoder": encoder_learnable_params, "augmenter": adv_learnable_params}

    def configure_optimizers(self) -> Tuple[List, List]:
        optimizer = [torch.optim.Adam(
            self.learnable_params['encoder'],
            lr=self.lr,
            weight_decay=self.weight_decay,
        ),
        torch.optim.Adam(
                self.learnable_params['augmenter'],
                lr=self.adv_lr,
                weight_decay=self.weight_decay,
            )]

        return optimizer

    def forward(self, X: torch.tensor, *args, **kwargs) -> Dict[str, Any]:
        out = super().forward(X, *args, **kwargs)
        z = self.projector(out["feats"])
        return {**out, "z": z}

    def flip_grad(self, status):
        """
        Sets requires_grad of inpainter (inference) model as True or False.
        """
        for param in self.encoder.parameters():
            param.requires_grad = status
        for param in self.projector.parameters():
            param.requires_grad = status
        for param in self.classifier.parameters():
            param.requires_grad = status

    def classifier_forward(self, batch: Sequence[Any]) -> torch.Tensor:
        """
        Forward function for classification.
        """
        X, targets = batch
        x_orig = X[:,:,:,0]
        out = super().shared_step(x_orig, targets)

        class_loss = out["class_loss"]
        batch_size = out["batch_size"]
        acc = out["acc"]
        auc = out["auc"]
        return class_loss, batch_size, acc, auc

