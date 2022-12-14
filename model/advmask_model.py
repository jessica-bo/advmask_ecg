"""
Adapted from @YugeTen 
Source: https://github.com/YugeTen/adios/blob/main/src/methods/base_adios.py
"""

import numpy as np
import sys

import wandb
import PIL
import matplotlib.pyplot as plt

import torch

from scipy.signal import stft, istft

sys.path.append('../utils')
from utils.losses import simclr_loss_fn
from utils.metrics import weighted_mean 
from .adversarial_model import AdversarialModel
from .backbones.unet import unet1D 


class AdvMaskModel(AdversarialModel):
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
        nmasks,
        unet_depth, 
        alpha_sparsity,
        ratio,
        dropout,
        binarization,
        accumulate_grad_batches=4,
        fourier=False,
        **kwargs):

        # Options for adversarially masking the STFT
        self.fourier = fourier
        self.fourier_scale = 1.1 if positive_pairing=="SimCLR" else 1.2

        self.ratio = ratio
        self.alpha = alpha_sparsity
        self.unet_depth = unet_depth
        self.nmasks = 12 if self.fourier else nmasks
        self.dropout = dropout
        self.binarization = int(binarization)

        augmentation_model = unet1D(input_dim=12, embedding_dim=512, depth=self.unet_depth, nmasks=self.nmasks, fourier=self.fourier, fourier_scale=self.fourier_scale)

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
            adv_lr,
            train_mask_interval,
            augmentation_model,
            accumulate_grad_batches,
            **kwargs)

        self.save_hyperparameters()
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = super(AdvMaskModel, AdvMaskModel).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("advmask")

        parser.add_argument("--ratio", type=float, default=1)
        parser.add_argument("--nmasks", type=int, default=1)
        parser.add_argument("--alpha_sparsity", type=float, default=1.)
        parser.add_argument("--unet_depth", type=int, default=1)
        parser.add_argument("--fourier", action="store_true", default=False)
        parser.add_argument("--dropout", type=float, default=0)
        parser.add_argument("--binarization", type=str, default="50")
        return parent_parser

    def training_epoch_end(self, outs):
        super().training_epoch_end(outs)
        mask_loss = weighted_mean(outs, "mask_loss", "batch_size")
        self.log_dict({"train_mask_loss": mask_loss}, on_epoch=True, on_step=False, sync_dist=True)

    def training_step(self, batch, batch_idx): 
        """
        Training step for adversarial masking pretraining
        """
        opt_e, opt_m = self.optimizers()

        # encoder (inference model) forward
        self.flip_grad(status=True)
        class_loss, batch_size, acc, auc = super().classifier_forward(batch)
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
                # print(self.augmenter.cbr_up2.conv1.bias.grad)
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

    def encoder_forward(self, batch, actual_batch_size, log_image):
        """
        Forward function for encoder (inference model).
        """
        X, targets = batch
        x_orig = X[:,:,:,0]
        x_transformed = X[:,:,:,1]
        masks = self.augmenter(x_transformed)

        thres = 1/self.nmasks

        if self.fourier: 
            mask_reshape = torch.reshape(masks, [masks.shape[0], masks.shape[1], 500, int(x_transformed.shape[-1]/500) + 1])
            _, _, Zxx = stft(x_transformed.cpu(), 500, nperseg=999) # Zxx shape is [bs, 12, 500, 11] for n=5000 and [bs, 12, 500, 6] for n=2500
            Zxx = Zxx * (1-mask_reshape.detach().cpu().numpy())
            _, x_transformed = istft(Zxx, nperseg=999)
            x_transformed = torch.from_numpy(x_transformed[:,:,:x_transformed.shape[-1]]).float().cuda()

        elif self.nmasks==12:
            # if N=12, the output is a sigmoid so we just want to keep any values over 0.5 
            thres = 0.5
            mask = 1. / (1+ torch.exp(-(self.binarization) * (masks-0.5)))
            x_transformed = x_transformed * (1-mask)

        else: 
            # randomly choose one mask for each sample in the batch
            chosen_mask = torch.randint(low=0, high=self.nmasks, size=(actual_batch_size,))
            mask = torch.stack([masks[idx, c, ...] for idx, c in enumerate(chosen_mask)], dim=0).unsqueeze(1)
            mask = 1. / (1+ torch.exp(-(self.binarization) * (mask-0.5))).repeat(1, 12, 1)

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
                self.log_mask(x_orig[0], x_transformed[0], targets[0], mask[0][0].squeeze(), thres)

        return nce_loss, masks

    def mask_forward(self, batch, actual_batch_size, log_image):
        """
        Forward function for masking model (occlusion model).
        """
        mask_nce_loss, masks = self.encoder_forward(batch, actual_batch_size, log_image)

        # compute mask penalty
        mask_penalty = masks.sum([-1]) / masks.shape[-1]
        mask_loss = (self.alpha * (1 / (torch.sin(self.ratio * mask_penalty * np.pi) + 1e-10)).mean(0).sum(0))

        return mask_nce_loss, mask_loss

    def log_mask(self, x_orig, x_trans, target, mask, thres):
        mask_bin = 1 - (mask > thres).float()
        final_mask = 1 - mask
        final_mask = final_mask.detach().cpu().numpy()

        fig, axs = plt.subplots(2, 1, figsize=(15, 12))

        axs[0].plot(x_trans.squeeze().transpose(0,1).detach().cpu().numpy(), 'red')
        axs[1].plot(x_orig.squeeze().transpose(0,1).detach().cpu().numpy(), 'green')
        for j in range(len(mask_bin)-1):
            if mask_bin[j] == 0:
                axs[1].axvspan(j, j+1, ymin=0, ymax=1, alpha=0.7,zorder=10)

        fig.canvas.draw()
        image = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
        wandb_image = wandb.Image(image, caption="Label {}".format(target.detach().cpu().numpy()))

        NoneType = type(None)
        if not isinstance(self.logger, NoneType):
            self.logger.experiment.log({"masked_image": wandb_image})
        
        plt.close('all')

    def log_stft(self, x_orig, x_transformed, target):
        fig, axs = plt.subplots(2, 1, figsize=(15, 12))

        axs[0].plot(x_orig.squeeze().transpose(0, 1).detach().cpu().numpy(), 'red')
        axs[1].plot(x_transformed.squeeze().transpose(0,1).detach().cpu().numpy(), 'blue')

        fig.canvas.draw()
        image = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
        wandb_image = wandb.Image(image, caption="Label {}".format(target.detach().cpu().numpy()))

        NoneType = type(None)
        if not isinstance(self.logger, NoneType):
            self.logger.experiment.log({"masked_image": wandb_image})

        plt.close('all')