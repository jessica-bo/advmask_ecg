import numpy as np
import os, sys
from functools import partial
from argparse import ArgumentParser
from typing import Any, Dict, List, Tuple, Callable, Sequence

import wandb
import PIL
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../utils')
from utils.losses import simclr_loss_fn
from utils.metrics import weighted_mean, Entropy, evaluate_single
from .base_model import BaseModel
from .backbones.unet import unet1Dsmall 

sys.path.append('../data')
from data.cinc2021.utils_cinc2021 import evaluate_scores

class AdversarialModel(BaseModel):
    def __init__(
        self, 
        encoder,
        console_log,
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
        mask_lr,
        train_mask_interval,
        nmasks,
        alpha_sparsity,
        ratio,
        **kwargs):
        
        super().__init__(
            encoder,
            console_log,
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
            **kwargs)

        self.console_log = console_log
        self.save_hyperparameters()

        self.encoder = encoder
        self.max_epochs = max_epochs
        self.temperature = temperature
        self.lr = lr
        self.mask_lr = mask_lr
        self.ratio = ratio
        self.train_mask_interval = train_mask_interval
        self.alpha1 = alpha_sparsity
        self.alpha2 = 0
        self.nmasks = nmasks
        self.positive_pairing = positive_pairing
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.entropy = Entropy()

        # simclr projector
        self.projector = nn.Sequential(
            nn.Linear(self.encoder.embedding_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
        )

        self.classifier = nn.Linear(self.encoder.embedding_dim, n_classes)

        self.metric_keys = ['acc', 'auc'] 
        if target_type == 'multilabel':
            self.loss_fn = torch.nn.BCELoss() #torch.nn.BCEWithLogitsLoss()#F.binary_cross_entropy_with_logits
            self.eval_fn = evaluate_scores
        elif target_type == "single":
            self.loss_fn = F.cross_entropy
            self.eval_fn = evaluate_single

        self.mask_encoder = unet1Dsmall(input_dim=12, embedding_dim=self.encoder.embedding_dim, kernel_size=7, depth=2, nmasks=self.nmasks)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parent_parser = super(AdversarialModel, AdversarialModel).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("adversarial")

        # adios args
        parser.add_argument("--mask_lr", type=float, default=0.001)
        parser.add_argument("--ratio", type=float, default=1)
        parser.add_argument("--train_mask_interval", type=int, default=1)
        parser.add_argument("--nmasks", type=int, default=1)
        parser.add_argument("--alpha_sparsity", type=float, default=1.)
        parser.add_argument("--alpha_entropy", type=float, default=0.)

        return parent_parser

    @property
    def learnable_params(self) -> Dict[str, Any]:
        encoder_learnable_params = list(self.encoder.parameters()) + list(self.projector.parameters()) + list(self.classifier.parameters())
        mask_learnable_params = list(self.mask_encoder.parameters()) 

        return {"encoder": encoder_learnable_params, "mask": mask_learnable_params}

    def configure_optimizers(self) -> Tuple[List, List]:
        optimizer = [torch.optim.Adam(
            self.learnable_params['encoder'],
            lr=self.lr,
            weight_decay=self.weight_decay,
        ),
        torch.optim.Adam(
                self.learnable_params['mask'],
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
        opt_e.zero_grad()
        nce_loss, _ = self.encoder_forward(batch, False)
        class_loss, batch_size, acc, auc = self.classifier_forward(batch, batch_idx)
        encoder_loss = class_loss + nce_loss
        self.manual_backward(encoder_loss) #, opt_e) # maximise similarity
        opt_e.step()

        if batch_idx % self.train_mask_interval == 0:
            # masking model (occlusion model) forward
            log_image = True if batch_idx==0 else False
            self.flip_grad(status=False)
            opt_m.zero_grad()
            mask_nce_loss, mask_loss = self.mask_forward(batch, log_image)
            self.manual_backward(-(mask_nce_loss+mask_loss)) #, opt_m) # minimise similarity
            opt_m.step()

        metrics = {
            "nce_loss": nce_loss,
            "class_loss": class_loss,
            "mask_loss": mask_loss,
            "batch_size": batch_size,
            "acc": acc,
            "auc": auc,
        }
        
        return metrics

    def training_epoch_end(self, outs: List[Dict[str, Any]]):
        super().training_epoch_end(outs)
        mask_loss = weighted_mean(outs, "mask_loss", "batch_size")
        self.log_dict({"train_mask_loss": mask_loss}, on_epoch=True, on_step=False, sync_dist=True)

    def classifier_forward(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        X, targets = batch
        x_orig = X[:,:,:,0]
        out = super().shared_step(x_orig, targets)

        class_loss = out["class_loss"]
        batch_size = out["batch_size"]
        acc = out["acc"]
        auc = out["auc"]
        return class_loss, batch_size, acc, auc

    def encoder_forward(self, batch: Sequence[Any], log_image: bool) -> torch.Tensor:
        """Forward function for encoder (inference model).

         Args:
             batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                 [X] is a list of size self.n_crops containing batches of images.
         """
        X, targets = batch
        x_orig = X[:,:,:,0]
        x_transformed = X[:,:,:,1]
        masks = self.mask_encoder(x_transformed)

        # randomly choose one mask for each sample in the batch
        chosen_mask = torch.randint(low=0, high=self.nmasks, size=(self.batch_size,))
        mask = torch.stack([masks[idx, c, ...] for idx, c in enumerate(chosen_mask)], dim=0).unsqueeze(1)
        mask = (mask > 0.5).float()
        feats1, feats2 = self.encoder(x_orig), self.encoder(x_transformed*(1-mask)) #if mask=0, then nothing is masked
        z1 = self.projector(feats1)
        z2 = self.projector(feats2)

        # compute similarity between mask and no mask
        nce_loss = simclr_loss_fn(latent_embeddings=[z1, z2], 
                                  positive_pairing=self.positive_pairing, 
                                  temperature=self.temperature)

        if log_image: self.log_mask(x_orig[0], mask[0].squeeze())

        return nce_loss, masks

    def mask_forward(self, batch: Sequence[Any], log_image: bool) -> torch.Tensor:
        """Forward function for masking model (occlusion model).

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.n_crops containing batches of images.
        """
        
        # X, targets = batch
        # x_orig = X[:,:,:,0]
        # x_transformed = X[:,:,:,1]
        # masks = self.mask_encoder(x_transformed)

        # # randomly choose one mask for each sample in the batch
        # chosen_mask = torch.randint(low=0, high=self.nmasks, size=(self.batch_size, ))
        # mask = torch.stack([masks[i, c, ...] for i, c in enumerate(chosen_mask)], dim=0).unsqueeze(1)
        # feats1, feats2 = self.encoder(x_orig), self.encoder(x_transformed*(1-mask))
        # z1 = self.projector(feats1)
        # z2 = self.projector(feats2)

        # # compute similarity between mask and no mask
        # mask_nce_loss = simclr_loss_fn(latent_embeddings=[z1, z2], 
        #                       positive_pairing=self.positive_pairing, 
        #                       temperature=self.temperature)

        mask_nce_loss, masks = self.encoder_forward(batch, log_image)

        # compute mask penalty
        mask_penalty = masks.sum([-1]) / 5000
        mask_loss = -(self.alpha1 * (1 / (torch.sin(self.ratio + mask_penalty * np.pi) + 1e-10)).mean(0).sum(0))
        mask_loss += (self.alpha2 * self.entropy(mask_penalty))

        return mask_nce_loss, mask_loss

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        # call parent class to get results from standard metrics
        base_metrics = super().validation_step(batch, batch_idx)
        return {**base_metrics}

    def validation_epoch_end(self, outs: List[Dict[str, Any]]):
        super().validation_epoch_end(outs)

    def log_mask(self, x_orig, mask):

        if self.nmasks == 1:
            fig = plt.figure(figsize=(12, 6))
            plt.plot(x_orig.squeeze().transpose(0,1).cpu().detach().numpy(), 'red')

            final_mask = 1 - mask
            final_mask = final_mask.cpu().detach().numpy()
            for j in range(5000-1):
                if final_mask[j] == 0:
                    plt.axvspan(j, j+1, ymin=0, ymax=1, alpha=0.7,zorder=10)
        else:
            fig, axs = plt.subplots(self.nmasks, 1, figsize=(15, 12))

            
            for i in range(self.nmasks):
                axs[i].plot(x_orig.squeeze().transpose(0,1).cpu().detach().numpy(), 'red')
                final_mask_i = final_mask[i].cpu().detach().numpy()
                
                for j in range(5000-1):
                    if final_mask_i[j] == 0:
                        axs[i].axvspan(j, j+1, ymin=0, ymax=1, alpha=0.7,zorder=10)

        fig.canvas.draw()
        image = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
        wandb_image = wandb.Image(image)
        self.logger.experiment.log({"masked_image": wandb_image})

