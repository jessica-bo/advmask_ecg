import numpy as np
import os, sys
from functools import partial
from argparse import ArgumentParser
from typing import Any, Dict, List, Tuple, Callable, Sequence

import torch
from scipy.signal import stft, istft
from dtw import dtw
import matplotlib.pyplot as plt
import PIL 
import wandb

from .base_model import BaseModel

sys.path.append('../utils')
from utils.losses import simclr_loss_fn
from utils.metrics import weighted_mean, evaluate_single

sys.path.append('../data')
from data.augs import Normalize
from data.style import encode_LL_HH, stylize_feature


class StyleModel(BaseModel):
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
        style_kernel,
        scaling_factor,
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


        self.normalize = Normalize()
        self.style_kernel = style_kernel
        self.scaling_factor = scaling_factor
        
        self.save_hyperparameters()
        
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parent_parser = super(StyleModel, StyleModel).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("style")
        parser.add_argument("--style_kernel", type=int, default=10)
        parser.add_argument("--scaling_factor", type=float, default=1.0)
        return parent_parser


    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """
        Forward function for classification.
        """
        X, targets = batch
        x_orig = X[:,:,:,0]

        LL, HH = encode_LL_HH(x_orig, kernel=self.style_kernel)
        LL_transformed = stylize_feature(LL, scaling_factor=self.scaling_factor)
        x_trans = self.normalize(LL_transformed + HH)
        X[:,:,:,1] = x_trans

        aug_distance = torch.tensor([dtw(orig.detach().cpu().numpy(), aug.detach().cpu().numpy()).normalizedDistance for orig, aug in zip(x_orig, x_trans)]).mean().cuda().requires_grad_()
        
        if batch_idx==0: self.log_style(x_orig[0], x_trans[0], targets[0])

        metrics = super().training_step((X, targets), batch_idx)
        
        metrics["aug_distance"] = aug_distance

        return metrics

    def training_epoch_end(self, outs: List[Dict[str, Any]]):
        super().training_epoch_end(outs)
        aug_distance = weighted_mean(outs, "aug_distance", "batch_size")
        self.log_dict({"train_aug_distance": aug_distance}, on_epoch=True, on_step=False, sync_dist=True)


    def log_style(self, x_orig, x_transformed, target):
        fig, axs = plt.subplots(2, 1, figsize=(15, 12))

        axs[0].plot(x_orig.squeeze().transpose(0,1).detach().cpu().numpy(), 'red')
        axs[1].plot(x_transformed.squeeze().transpose(0,1).detach().cpu().numpy(), 'blue')

        fig.canvas.draw()
        image = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
        wandb_image = wandb.Image(image, caption="Label {}".format(target.detach().cpu().numpy()))

        NoneType = type(None)
        if not isinstance(self.logger, NoneType):
            self.logger.experiment.log({"masked_image": wandb_image})

        plt.close('all')