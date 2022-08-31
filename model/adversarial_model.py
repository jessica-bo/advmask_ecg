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

from scipy.signal import stft, istft

sys.path.append('../utils')
from utils.losses import simclr_loss_fn
from utils.metrics import weighted_mean, Entropy, evaluate_single
from .base_model import BaseModel
from .backbones.unet import unet1Dsmall, unet1D 
from .backbones import BACKBONES

sys.path.append('../data')
from data.cinc2021.utils_cinc2021 import evaluate_scores
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
        mask_lr,
        train_mask_interval,
        nmasks=1,
        unet_depth=1, 
        unet_large=False, 
        alpha_sparsity=0.1,
        ratio=1,
        fourier=False,
        accumulate_grad_batches=4,
        dropout=False,
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
        
        self.ratio = ratio
        self.train_mask_interval = train_mask_interval
        self.alpha1 = alpha_sparsity
        self.mask_lr = mask_lr * self.accumulate_grad_batches
        self.entropy = Entropy()
        self.unet_depth = unet_depth
        self.unet_large = unet_large
        self.fourier = fourier
        self.fourier_scale = 1.1 if self.positive_pairing=="SimCLR" else 1.2
        self.nmasks = 12 if self.fourier else nmasks
        self.dropout = dropout

        self.normalize = Normalize()

        # simclr projector
        self.projector = nn.Sequential(
            nn.Linear(self.encoder.embedding_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
        )

        self.classifier = nn.Linear(self.encoder.embedding_dim, n_classes)

        # Params 144,246,017
        if self.unet_large:
            self.mask_encoder = unet1D(input_dim=12, embedding_dim=self.encoder.embedding_dim, depth=self.unet_depth, nmasks=self.nmasks, fourier=self.fourier, fourier_scale=self.fourier_scale)
        else:
            self.mask_encoder = unet1Dsmall(input_dim=12, embedding_dim=self.encoder.embedding_dim, depth=self.unet_depth, nmasks=self.nmasks, fourier=self.fourier, fourier_scale=self.fourier_scale)
        
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

        parser.add_argument("--unet_depth", type=int, default=1)
        parser.add_argument("--unet_large", action="store_true", default=False)

        parser.add_argument("--fourier", action="store_true", default=False)
        parser.add_argument("--dropout", type=float, default=0)
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
        class_loss, batch_size, acc, auc = self.classifier_forward(batch, batch_idx)
        nce_loss, _ = self.encoder_forward(batch, batch_size, False)

        if self.simclr_loss_only: 
            encoder_loss = nce_loss/ self.accumulate_grad_batches  
        else:
            encoder_loss = (nce_loss + class_loss) / self.accumulate_grad_batches

        self.manual_backward(encoder_loss) 
        if (batch_idx + 1) % self.accumulate_grad_batches==0:
            opt_e.step()
            opt_e.zero_grad()

        # masking model (occlusion model) forward
        if batch_idx % self.train_mask_interval == 0:
            log_image = True if batch_idx==0 else False
            self.flip_grad(status=False)
            mask_nce_loss, mask_loss = self.mask_forward(batch, batch_size, log_image)

            # Maximize NCE loss == make dissimilar and minimize mask enforced loss 
            combined_mask_loss = (-mask_nce_loss + mask_loss) / self.accumulate_grad_batches
            self.manual_backward(combined_mask_loss) 
            if (batch_idx + 1) % self.accumulate_grad_batches==0:
                opt_m.step()
                opt_m.zero_grad()

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

    def encoder_forward(self, batch: Sequence[Any], actual_batch_size: int, log_image: bool) -> torch.Tensor:
        """Forward function for encoder (inference model).

         Args:
             batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                 [X] is a list of size self.n_crops containing batches of images.
         """
        X, targets = batch
        x_orig = X[:,:,:,0]
        x_transformed = X[:,:,:,1]
        masks = self.mask_encoder(x_transformed)

        thres = 1/self.nmasks

        if self.fourier: 
            mask_reshape = torch.reshape(masks, [masks.shape[0], masks.shape[1], 500, int(x_transformed.shape[-1]/500) + 1])
            _, _, Zxx = stft(x_transformed.cpu(), 500, nperseg=999) # Zxx shape is [bs, 12, 500, 11] for n=5000 and [bs, 12, 500, 6] for n=2500
            Zxx = Zxx * (1-mask_reshape.detach().cpu().numpy())
            _, x_transformed = istft(Zxx, nperseg=999)
            x_transformed = torch.from_numpy(x_transformed[:,:,:x_transformed.shape[-1]]).float().cuda()

        elif self.nmasks==12:
            mask = (masks > thres).float()
            x_transformed = x_transformed * (1-mask)

        else: 
            # randomly choose one mask for each sample in the batch
            chosen_mask = torch.randint(low=0, high=self.nmasks, size=(actual_batch_size,))
            mask = torch.stack([masks[idx, c, ...] for idx, c in enumerate(chosen_mask)], dim=0).unsqueeze(1)
            mask = (mask > thres).float().repeat(1, 12, 1) # 12 layers of the same masks 

            if self.dropout > 0: #0 is no dropout, 1 is full dropout 
                dropout_mask = torch.zeros(mask.shape).cuda()
                survivors = torch.rand(12) >= self.dropout 
                dropout_mask[:, survivors, :] = mask[:, survivors, :]
                mask = dropout_mask

            x_transformed = x_transformed * (1-mask)
        
        x_transformed = self.normalize(x_transformed)
        feats1, feats2 = self.encoder(x_orig), self.encoder(x_transformed) #if mask=0, then nothing is masked
        z1 = self.projector(feats1)
        z2 = self.projector(feats2)

        # compute similarity between mask and no mask
        nce_loss = simclr_loss_fn(latent_embeddings=[z1, z2], 
                                  positive_pairing=self.positive_pairing, 
                                  temperature=self.temperature)

        if log_image: 
            if self.fourier:
                self.log_stft(x_orig[0], x_transformed[0], targets[0])
            else: 
                self.log_mask(x_orig[0], x_transformed[0], targets[0], masks[0].squeeze(), thres)

        return nce_loss, masks

    def mask_forward(self, batch: Sequence[Any], actual_batch_size: int, log_image: bool) -> torch.Tensor:
        """Forward function for masking model (occlusion model).

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.n_crops containing batches of images.
        """
        mask_nce_loss, masks = self.encoder_forward(batch, actual_batch_size, log_image)

        # compute mask penalty
        mask_penalty = masks.sum([-1]) / masks.shape[-1]
        mask_loss = (self.alpha1 * (1 / (torch.sin(self.ratio * mask_penalty * np.pi) + 1e-10)).mean(0).sum(0))

        return mask_nce_loss, mask_loss

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        # call parent class to get results from standard metrics
        base_metrics = super().validation_step(batch, batch_idx)
        return {**base_metrics}

    def validation_epoch_end(self, outs: List[Dict[str, Any]]):
        super().validation_epoch_end(outs)

    def log_mask(self, x_orig, x_trans, target, mask, thres):
        mask = (mask > thres).float()
        final_mask = 1 - mask
        final_mask = final_mask.detach().cpu().numpy()

        if self.nmasks == 1:
            fig = plt.figure(figsize=(12, 6))
            plt.plot(x_trans.squeeze().transpose(0,1).detach().cpu().numpy(), 'red')

            for j in range(len(final_mask)-1):
                if final_mask[j] == 0:
                    plt.axvspan(j, j+1, ymin=0, ymax=1, alpha=0.7, zorder=10)
        else:
            fig, axs = plt.subplots(self.nmasks+1, 1, figsize=(15, 12))

            for i in range(self.nmasks):
                axs[i].plot(x_trans.squeeze().transpose(0,1).detach().cpu().numpy(), 'red')
                final_mask_i = final_mask[i]
                
                for j in range(len(final_mask_i)-1):
                    if final_mask_i[j] == 0:
                        axs[i].axvspan(j, j+1, ymin=0, ymax=1, alpha=0.7,zorder=10)
            axs[-1].plot(x_orig.squeeze().transpose(0,1).detach().cpu().numpy(), 'green')

        fig.canvas.draw()
        image = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
        wandb_image = wandb.Image(image, caption="Label {}".format(target.detach().cpu().numpy()))

        NoneType = type(None)
        if not isinstance(self.logger, NoneType):
            self.logger.experiment.log({"masked_image": wandb_image})

    def log_stft(self, x_orig, x_transformed, target):
        fig, axs = plt.subplots(2, 1, figsize=(15, 12))

        axs[0].plot(x_orig.squeeze().transpose(0,1).detach().cpu().numpy(), 'red')
        axs[1].plot(x_transformed.squeeze().transpose(0,1).detach().cpu().numpy(), 'blue')

        fig.canvas.draw()
        image = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
        wandb_image = wandb.Image(image, caption="Label {}".format(target.detach().cpu().numpy()))

        NoneType = type(None)
        if not isinstance(self.logger, NoneType):
            self.logger.experiment.log({"masked_image": wandb_image})

