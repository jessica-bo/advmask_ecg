import numpy as np
import os
import random 
import torch

from .augs import CollatedTransform
from .dataset_wrapper import DatasetWrapper

class StyleDatasetWrapper(DatasetWrapper):
    def __init__(
        self, 
        data_dir, 
        dataset,
        style_dataset,
        style_alpha,
        method, 
        phase, 
        seed, 
        positive_pairing=None, 
        nleads=12, 
        reduce_dataset="",
        **kwargs):

        self.style_db_dir = os.path.join(data_dir, style_dataset)
        self.style_alpha = style_alpha

        super().__init__(
            data_dir, 
            dataset,
            method, 
            phase, 
            seed, 
            positive_pairing=positive_pairing, 
            nleads=nleads, 
            reduce_dataset=reduce_dataset,
            **kwargs
        )

        self.load_style_data() 

    def load_style_data(self):
        self.styleX = np.load(os.path.join(self.style_db_dir, "seed{}/{}X_{}.npy".format(self.seed, self.reduce_dataset, self.phase)), allow_pickle=True)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample, label = self.X[idx], self.y[idx]
        label = torch.tensor(label,dtype=torch.long)

        random_style_idx = random.randint(0, self.styleX.shape[0]-1)
        style_sample = self.styleX[random_style_idx]

        if self.method == "transfer":
            sample = self.normalize(sample)
            return sample.float(), label

        else:
            nsamples = sample.shape[-1]
            if self.positive_pairing == 'CMSC': 
                combined_sample = torch.empty(self.nleads, int(nsamples/2), 2)
                sample1, sample2 = np.split(sample, 2, axis=1)
                combined_sample[:,:,0] = self.normalize(sample1)
                combined_sample[:,:,1] = self.fourier_mixup(sample2, style_sample)
            elif self.positive_pairing == 'SimCLR': 
                combined_sample = torch.empty(self.nleads, nsamples, 2)
                combined_sample[:,:,0] = self.normalize(sample)
                combined_sample[:,:,1] = self.fourier_mixup(sample, style_sample)
            else: 
                # CMLC, CMSMLC, PCLR not implemented
                raise NotImplementedError
            
            return combined_sample.float(), label

    def fourier_mixup(self, data, style): 
        n = data.shape[-1]
        style = style[:,0:n]

        fftd = np.fft.fft(data)
        fftd_abs, fftd_pha = np.abs(fftd), np.angle(fftd)
        fftd_abs = np.fft.fftshift(fftd_abs, axes=(0, 1))

        ffts = np.fft.fft(style)
        ffts_abs, _ = np.abs(ffts), np.angle(ffts)
        ffts_abs = np.fft.fftshift(ffts_abs)

        fftd_abs_ = np.copy(fftd_abs)
        ffts_abs_ = np.copy(ffts_abs)
            
        fftd_abs = self.style_alpha*ffts_abs_ + (1-self.style_alpha)*fftd_abs_
        fftd_abs = np.fft.ifftshift(fftd_abs)
        fft_comb = fftd_abs * (np.e ** (1j * fftd_pha))
        style_data = np.fft.ifft(fft_comb)

        return torch.Tensor(style_data)