import numpy as np
import os, sys
from functools import partial
from argparse import ArgumentParser
from typing import Any, Dict, List, Tuple, Callable, Sequence

import wandb
import PIL
import matplotlib.pyplot as plt
# import tslearn
from dtw import dtw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

sys.path.append('../utils')
from utils.losses import simclr_loss_fn
from utils.metrics import weighted_mean, Entropy, evaluate_single
from .base_model import BaseModel
from .backbones.mlp import mlp

sys.path.append('../data')
from data.cinc2021.utils_cinc2021 import evaluate_scores
from data.augs import Normalize, AdvAugmentation, ADVAUG_OUTPUTS

from .backbones import BACKBONES


class AdvMLPModel(BaseModel):
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
        mask_lr,
        train_mask_interval,
        accumulate_grad_batches=4,
        advaug_name="gaussian",
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
            accumulate_grad_batches=None,
            **kwargs)

        self.save_hyperparameters()
        self.encoder_name = encoder_name
        self.encoder = BACKBONES[self.encoder_name](**kwargs)

        self.accumulate_grad_batches = 1 if accumulate_grad_batches==None else accumulate_grad_batches
        self.max_epochs = max_epochs
        self.temperature = temperature
        self.lr = lr * self.accumulate_grad_batches
        self.positive_pairing = positive_pairing
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.simclr_loss_only = simclr_loss_only
        
        self.train_mask_interval = train_mask_interval
        self.mask_lr = mask_lr * self.accumulate_grad_batches
        self.entropy = Entropy()

        self.advaug_name = advaug_name
        self.transform = AdvAugmentation(self.advaug_name) #AdvGaussian() #
        print("Using adversarial augmentation {}".format(self.advaug_name))
        self.normalize = Normalize()

        # simclr projector
        self.projector = nn.Sequential(
            nn.Linear(self.encoder.embedding_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
        )

        self.classifier = nn.Linear(self.encoder.embedding_dim, n_classes)
        self.augmenter = mlp(noutputs=ADVAUG_OUTPUTS[self.advaug_name])
        
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parent_parser = super(AdvMLPModel, AdvMLPModel).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("adversarial")

        # adios args
        parser.add_argument("--mask_lr", type=float, default=0.001)
        parser.add_argument("--train_mask_interval", type=int, default=1)
        parser.add_argument("--advaug_name", type=str, default="gaussian")

        return parent_parser

    @property
    def learnable_params(self) -> Dict[str, Any]:
        encoder_learnable_params = list(self.encoder.parameters()) + list(self.projector.parameters()) + list(self.classifier.parameters())
        augment_learnable_params = list(self.augmenter.parameters()) 

        return {"encoder": encoder_learnable_params, "augment": augment_learnable_params}

    def configure_optimizers(self) -> Tuple[List, List]:
        optimizer = [torch.optim.Adam(
            self.learnable_params['encoder'],
            lr=self.lr,
            weight_decay=self.weight_decay,
        ),
        torch.optim.Adam(
                self.learnable_params['augment'],
                lr=self.mask_lr,
                weight_decay=self.weight_decay,
            )]
        return optimizer

    @property
    def automatic_optimization(self) -> bool:
        return False

    def forward(self, X: torch.tensor, *args, **kwargs) -> Dict[str, Any]:
        out = super().forward(X, *args, **kwargs)
        z = self.projector(out["feats"])
        return {**out, "z": z}

    def flip_grad(self, status):
        """Sets requires_grad of inpainter (inference) model as True or False.

        Args:
            status (bool): determines whether requires_grad is True or False.
        """
        for param in self.encoder.parameters():
            param.requires_grad = status
        for param in self.projector.parameters():
            param.requires_grad = status
        for param in self.classifier.parameters():
            param.requires_grad = status

    def training_step(self, batch: Sequence[Any], batch_idx: int): #, optimizer_idx: int):
        """Training step for SimCLR ADIOS.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.n_crops containing batches of images.
            batch_idx (int): index of the batch.
        """

        # get optimiser and scheduler
        opt_e, opt_m = self.optimizers()

        # encoder (inference model) forward
        self.flip_grad(status=True)
        class_loss, batch_size, acc, auc = self.classifier_forward(batch, batch_idx)
        nce_loss = self.encoder_forward(batch, batch_size)

        if self.simclr_loss_only:
            encoder_loss = nce_loss/ self.accumulate_grad_batches  
        else:
            encoder_loss = (nce_loss + class_loss) / self.accumulate_grad_batches

        self.manual_backward(encoder_loss) #, opt_e) # maximise similarity
        if (batch_idx + 1) % self.accumulate_grad_batches==0:
            opt_e.step()
            opt_e.zero_grad()

        # masking model (occlusion model) forward
        if batch_idx % self.train_mask_interval == 0:
            log_image = True if batch_idx==0 else False
            self.flip_grad(status=False)
            mask_nce_loss, distance = self.mask_forward(batch, batch_size, log_image)
            combined_mask_loss = (-mask_nce_loss + distance) / self.accumulate_grad_batches

            # Maximize NCE loss == make dissimilar and minimize distance 
            self.manual_backward(-combined_mask_loss)
            if (batch_idx + 1) % self.accumulate_grad_batches==0:
                opt_m.step()
                opt_m.zero_grad()

        metrics = {
            "nce_loss": nce_loss,
            "class_loss": class_loss,
            "aug_distance": distance,
            "batch_size": batch_size,
            "acc": acc,
            "auc": auc,
        }
        
        return metrics

    def training_epoch_end(self, outs: List[Dict[str, Any]]):
        super().training_epoch_end(outs)
        aug_dis = weighted_mean(outs, "aug_distance", "batch_size")
        self.log_dict({"train_aug_distance": aug_dis}, on_epoch=True, on_step=False, sync_dist=True)

    def classifier_forward(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        X, targets = batch
        x_orig = X[:,:,:,0]
        out = super().shared_step(x_orig, targets)

        class_loss = out["class_loss"]
        batch_size = out["batch_size"]
        acc = out["acc"]
        auc = out["auc"]
        return class_loss, batch_size, acc, auc

    def encoder_forward(self, batch: Sequence[Any], actual_batch_size: int) -> torch.Tensor:
        """Forward function for encoder (inference model).

         Args:
             batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                 [X] is a list of size self.n_crops containing batches of images.
         """
        X, targets = batch
        x_orig = X[:,:,:,0]
        aug_params = self.augmenter(x_orig) 
        x_transformed = self.transform(x_orig, aug_params)
        x_transformed = self.normalize(x_transformed)

        feats1, feats2 = self.encoder(x_orig), self.encoder(x_transformed) 
        z1, z2 = self.projector(feats1), self.projector(feats2)

        # compute similarity between mask and no mask
        nce_loss = simclr_loss_fn(latent_embeddings=[z1, z2], 
                                  positive_pairing=self.positive_pairing, 
                                  temperature=self.temperature)
        return nce_loss

    def mask_forward(self, batch: Sequence[Any], actual_batch_size: int, log_image: bool) -> torch.Tensor:
        """Forward function for masking model (occlusion model).
        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.n_crops containing batches of images.
        """
        X, targets = batch
        x_orig = X[:,:,:,0]
        aug_params = self.augmenter(x_orig) #[bs, nparams]

        x_transformed = self.transform(x_orig, aug_params)
        x_transformed = self.normalize(x_transformed)

        feats1, feats2 = self.encoder(x_orig), self.encoder(x_transformed) 
        z1, z2 = self.projector(feats1), self.projector(feats2)

        # compute similarity between mask and no mask
        mask_nce_loss = simclr_loss_fn(latent_embeddings=[z1, z2], 
                                       positive_pairing=self.positive_pairing, 
                                       temperature=self.temperature)

        if log_image: self.log_augs(x_orig[0], x_transformed[0], targets[0], aug_params[0])

        # aug_distance = torch.Tensor([tslearn.metrics.dtw(orig, aug) for orig, aug in zip(x_orig, x_transformed)]).sum().cuda()
        aug_distance = torch.tensor([dtw(orig.detach().cpu().numpy(), aug.detach().cpu().numpy()).normalizedDistance for orig, aug in zip(x_orig, x_transformed)]).mean().cuda().requires_grad_()

        return mask_nce_loss, aug_distance 

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        # call parent class to get results from standard metrics
        base_metrics = super().validation_step(batch, batch_idx)
        return {**base_metrics}

    def validation_epoch_end(self, outs: List[Dict[str, Any]]):
        super().validation_epoch_end(outs)

    def log_augs(self, x_orig, x_transformed, target, augs):

        fig, axs = plt.subplots(2, 1, figsize=(15, 12))

        axs[0].plot(x_orig.squeeze().transpose(0,1).detach().cpu().numpy(), 'red')
        axs[1].plot(x_transformed.squeeze().transpose(0,1).detach().cpu().numpy(), 'blue')

        fig.canvas.draw()
        image = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
        wandb_image = wandb.Image(image, caption="Label {}, augs {}".format(target.detach().cpu().numpy(), augs.detach().cpu().numpy()))

        NoneType = type(None)
        if not isinstance(self.logger, NoneType):
            self.logger.experiment.log({"masked_image": wandb_image})

