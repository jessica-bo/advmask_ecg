"""
Augmentations partially adapted from @Jwoo5
Source: https://github.com/Jwoo5/fairseq-signals/blob/master/fairseq_signals/data/ecg/augmentations.py

"""

import random
import os
import numpy as np
import math

import torch 
import torchvision
from torchvision import transforms

import scipy
from scipy.spatial.transform import Rotation as R

from scipy.signal import stft, istft

class ToTensor1D:
    def __call__(self, sample):
        return torch.from_numpy(sample)

class Normalize: 
    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = (x - np.min(x))/(np.max(x) - np.min(x) + 1e-8)
        elif isinstance(x,torch.Tensor):
            x = (x - torch.min(x))/(torch.max(x) - torch.min(x) + 1e-8)
        return x

class GaussianNoise:
    def __init__(self, sigma=0.05):
        self.sigma = sigma

    def __call__(self, x):
        size = x.shape[-1]
        gauss_noise = np.random.normal(0,self.sigma,size=size)
        x = x + gauss_noise
        return x

class BaselineWander:
    def __init__(
        self,
        max_amplitude=0.5,
        min_amplitude=0,
        max_freq=0.2,
        min_freq=0.01,
        # k=3,
    ):
        self.max_amplitude = max_amplitude
        self.min_amplitude = min_amplitude
        self.max_freq = max_freq
        self.min_freq = min_freq
        self.k = 3
        self.freq = 500
    
    def __call__(self, sample):
        new_sample = sample.copy()
        csz, tsz = new_sample.shape
        amp_channel = np.random.normal(1, 0.5, size=(csz, 1))
        c = np.array([i for i in range(12)])
        amp_general = np.random.uniform(self.min_amplitude, self.max_amplitude, size=self.k)
        noise = np.zeros(shape=(1, tsz))
        for k in range(self.k):
            noise += self._apply_baseline_wander(tsz) * amp_general[k]
        noise = (noise * amp_channel).astype(np.float32)
        new_sample[c,:] = new_sample[c,:] + noise[c,:]
        return new_sample.astype(float)
        
    def _apply_baseline_wander(self, tsz):
        f = np.random.uniform(self.min_freq, self.max_freq)
        t = np.linspace(0, tsz-1, tsz)
        r = np.random.uniform(0, 2*np.pi)
        noise = np.cos(2*np.pi*f*(t/self.freq)+r)
        return noise

class BaselineShift:
    def __init__(
        self,
        max_amplitude=0.25,
        min_amplitude=0,
        shift_ratio=0.2,
        # num_segment=1,
    ):
        self.max_amplitude = max_amplitude
        self.min_amplitude = min_amplitude
        self.shift_ratio = shift_ratio
        self.num_segment = 1
        self.freq = 500
    
    def __call__(self, sample):
        new_sample = sample.copy()
        csz, tsz = new_sample.shape
        shift_length = tsz * self.shift_ratio
        amp_channel = np.random.choice([1, -1], size=(csz, 1))
        amp_general = np.random.uniform(self.min_amplitude, self.max_amplitude, size=(1,1))
        amp = amp_channel - amp_general
        noise = np.zeros(shape=(csz, tsz))
        for i in range(self.num_segment):
            segment_len = np.random.normal(shift_length, shift_length*0.2)
            t0 = int(np.random.uniform(0, tsz-segment_len))
            t = int(t0+segment_len)
            c = np.array([i for i in range(12)])
            noise[c, t0:t] = 1
        new_sample = new_sample + noise * amp
        return new_sample.astype(float)

class PowerlineNoise:
    def __init__(
        self,
        max_amplitude=0.5,
        min_amplitude=0,
    ):
        self.max_amplitude = max_amplitude
        self.min_amplitude = min_amplitude
        self.freq = 500

    def __call__(self, sample):
        new_sample = sample.copy()
        csz, tsz = new_sample.shape
        amp = np.random.uniform(self.min_amplitude, self.max_amplitude, size=(1,1))
        f = 50 if np.random.uniform(0,1) > 0.5 else 60
        noise = self._apply_powerline_noise(tsz, f)
        new_sample = new_sample + noise * amp
        return new_sample.astype(float)

    def _apply_powerline_noise(self, tsz, f):
        t = np.linspace(0,tsz-1,tsz)
        phase = np.random.uniform(0,2*np.pi)
        noise = np.cos(2*np.pi*f*(t/self.freq)+phase)
        return noise

class EMGNoise:
    def __init__(
        self,
        max_amplitude=0.1,
        min_amplitude=0,
    ):
        self.max_amplitude = max_amplitude
        self.min_amplitude = min_amplitude
    
    def __call__(self, sample):
        new_sample = sample.copy()
        csz, tsz = new_sample.shape
        amp = np.random.uniform(self.min_amplitude, self.max_amplitude, size=(csz, 1))
        noise = np.random.normal(0, 1, [csz, tsz])
        new_sample = new_sample + noise * amp
        return new_sample.astype(float)
    
class RandomLeadsMask(object):
    def __init__(
        self,
        p=0.5,
    ):
        self.p = p
    
    def __call__(self, sample):
        new_sample = torch.zeros(sample.shape)
        survivors = torch.rand(12) >= self.p
        new_sample[survivors] = sample[survivors]
        return new_sample

class RandomBlockMask(object):
    def __init__(
        self,
        mask_ratio,
    ):
        self.mask_ratio = mask_ratio
    
    def __call__(self, sample):
        new_sample = sample.copy()
        sample_size = sample.shape[-1]

        if self.mask_ratio > 0:
            offset = math.floor(sample_size * self.mask_ratio)

            start_indices = np.random.randint(0, sample_size, size=12)
            end_indices = np.array([s + offset if s + offset <= sample_size else sample_size
                                    for s in start_indices])
            leftovers = np.array([s + offset - sample_size if s + offset > sample_size else 0
                                  for s in start_indices])

            for i in range(12):
                new_sample[i, start_indices[i]:end_indices[i]] = 0
                new_sample[i, 0:leftovers[i]] = 0

        return new_sample

class RandomBlockMaskBad(object):
    def __init__(
        self,
        mask_ratio,
    ):
        self.mask_ratio = mask_ratio
    
    def __call__(self, sample):
        new_sample = sample.copy()
        sample_size = sample.shape[-1]

        if self.mask_ratio > 0:
            offset = math.floor(sample_size * self.mask_ratio)

            start_index = np.random.randint(0, sample_size, size=1)
            end_index = start_index + offset if start_index + offset <= sample_size else sample_size
            leftover = start_index + offset - sample_size if start_index + offset > sample_size else 0

            for i in range(12):
                new_sample[i, int(start_index):int(end_index)] = 0
                new_sample[i, 0:int(leftover)] = 0

        return new_sample

class RandomMask(object):
    def __init__(
        self,
        mask_ratio,
    ):
        self.mask_ratio = mask_ratio
    
    def __call__(self, sample):
        mask = np.random.choice([0,1], sample.shape, p=[self.mask_ratio, 1-self.mask_ratio])
        new_sample = sample * mask
        
        return new_sample

class RandomFourier(object):
    
    def __call__(self, sample):
        _, _, Zxx = stft(sample, 500, nperseg=999)
        np_mask = np.random.beta(5, 2, size=Zxx.shape) 
        Zxx = Zxx * np_mask
        _, new_sample = istft(Zxx, nperseg=999)
        new_sample = new_sample[:,:sample.shape[-1]]

        return new_sample

class PeakMask(object):

    def __call__(self, sample):
        mean = np.mean(sample)
        avg_leads = np.mean(sample, axis=0)
        mask = np.where(avg_leads>mean, 0, 1)

        if np.random.rand() > 0.5:
            mask = 1 - mask

        new_sample = mask*sample

        return new_sample


class ThreeKGTransform:
    def __init__(
        self,
        angle=45/90,
        scale=1.5-0.75,
    ):
        # Assuming that inputs are between 0-1
        self.angle = angle*90
        self.scale = scale + 0.75
    
    def _get_other_four_leads(self, I, II):
        """calculate other four leads (III, aVR, aVL, aVF) from the first two leads (I, II)"""
        III = -I + II
        aVR = -(I + II) / 2
        aVL = I - II/2
        aVF = -I/2 + II

        return III, aVR, aVL, aVF

    def __call__(self, sample):
        leads_taken = [0,1,6,7,8,9,10,11]
        other_leads = [2,3,4,5]
        sample = sample[leads_taken]

        D_i = np.array(
            [
                [0.156, -0.009, -0.172, -0.073, 0.122, 0.231, 0.239, 0.193],
                [-0.227, 0.886, 0.057, -0.019, -0.106, -0.022, 0.040, 0.048],
                [0.021, 0.102, -0.228, -0.310, -0.245, -0.063, 0.054, 0.108]
            ]
        )
        D = np.linalg.pinv(D_i)

        vcg = D_i @ sample

        if self.angle:
            angles = np.random.uniform(-self.angle, self.angle, size=3)
            RT = R.from_euler('zyx', angles, degrees=True).as_dcm()
        else:
            RT = np.diag((1,1,1))
        
        if self.scale:
            scales = np.random.uniform(1, self.scale, size=3)
            ST = np.diag(scales)
        
        res = D @ ST @ RT @ vcg

        sample_size = sample.shape[-1]
        ecg = np.zeros((12, sample_size))

        ecg[leads_taken] = res
        ecg[other_leads] = self._get_other_four_leads(res[0], res[1])
        
        return ecg


class CollatedTransform:
    """Adds callable base class to implement different transformation pipelines."""

    def __init__(
        self, 
        gaussian=False, 
        gaussian_sigma=0.05, 
        wander=False, 
        shift=False, 
        powerline=False,
        emg=False, 
        rlm=False, 
        rlm_prob=0.5,
        mask=False,
        mask_ratio=0.2,
        blockmask=False,
        blockmask_ratio=0.2,
        blockmaskbad=False,
        threeKG=False, 
        threeKG_angle=45,
        threeKG_scale=1.5, 
        randfourier=False,
        peakmask=False,
        **kwargs,
    ):
        self.transform = transforms.Compose([])

        if gaussian:
            self.transform = transforms.Compose([self.transform, GaussianNoise(sigma=gaussian_sigma)])
            print("Collating Gaussian with sigma={}.\n".format(gaussian_sigma))
        if wander:
            self.transform = transforms.Compose([self.transform, BaselineWander()])
            print("Collating BaselineWander.\n")
        if shift:
            self.transform = transforms.Compose([self.transform, BaselineShift()])
            print("Collating BaselineShift.\n")
        if powerline:
            self.transform = transforms.Compose([self.transform, PowerlineNoise()])
            print("Collating PowerlineNoise.\n")
        if emg:
            self.transform = transforms.Compose([self.transform, EMGNoise()])
            print("Collating EMGNoise.\n")
        if mask:
            self.transform = transforms.Compose([self.transform, RandomMask(mask_ratio=mask_ratio)])
            print("Collating RandomMask with ratio={}.\n".format(mask_ratio))
        if blockmask:
            self.transform = transforms.Compose([self.transform, RandomBlockMask(mask_ratio=blockmask_ratio)])
            print("Collating RandomBlockMask with ratio={}.\n".format(blockmask_ratio))
        if blockmaskbad:
            self.transform = transforms.Compose([self.transform, RandomBlockMaskBad(mask_ratio=blockmask_ratio)])
            print("Collating RandomBlockMaskBad with ratio={}.\n".format(blockmask_ratio))
        if threeKG:
            self.transform = transforms.Compose([self.transform, ThreeKGTransform(angle=threeKG_angle,
                                                                                  scale=threeKG_scale)])
        if randfourier: 
            self.transform = transforms.Compose([self.transform, RandomFourier()])     
        if peakmask: 
            self.transform = transforms.Compose([self.transform, PeakMask()])        

        self.transform = transforms.Compose([self.transform, 
                                             ToTensor1D(),
                                             Normalize()])
        if rlm:
            self.transform = transforms.Compose([self.transform, RandomLeadsMask(p=rlm_prob)])
            print("Collating RandomLeadsMask with p={}.\n".format(rlm_prob))

    def __call__(self, x):
        return self.transform(x)

    def __repr__(self):
        return str(self.transform)


"""
Adversarial verion of baseline augmentations -- instead of initializing the augmention parameters at the start, 
they are generated adversarially and passed in on each training step. 
"""

class AdvAugmentation: 
    def __init__(self, arg_name):
        self.augmentation = ADVAUGS[arg_name]
        
    def __call__(self, batch, param_arr):
        new_batch = batch.clone()
        for i in range(len(batch)):
            params = param_arr[i].detach().cpu().numpy()
            transformed = self.augmentation(*params)(batch[i].detach().cpu().numpy())
            new_batch[i] = torch.tensor(transformed, requires_grad=True)
            
        return new_batch

ADVAUGS = {
    "gaussian": GaussianNoise,
    "wander": BaselineWander,
    "shift": BaselineShift,
    "powerline": PowerlineNoise,
    "emg": EMGNoise,
    "mask": RandomMask,
    "blockmask": RandomBlockMask,
    "threeKG": ThreeKGTransform,
}

ADVAUG_OUTPUTS = {
    "gaussian": 1,
    "wander": 4,
    "shift": 3,
    "powerline": 2,
    "emg": 2,
    "mask": 1,
    "blockmask": 1,
    "threeKG": 2,
}