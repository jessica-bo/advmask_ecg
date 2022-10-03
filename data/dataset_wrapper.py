import numpy as np
import os
import torch
from torch.utils.data import Dataset

from .augs import CollatedTransform

class DatasetWrapper(Dataset):
    def __init__(
        self, 
        data_dir, 
        dataset,
        method, 
        phase, 
        task,
        seed, 
        positive_pairing=None, 
        nleads=12, 
        reduce_dataset="",
        **kwargs):

        self.db_dir = os.path.join(data_dir, dataset)
        self.positive_pairing = positive_pairing
        self.transform = CollatedTransform(**kwargs)
        self.normalize = CollatedTransform()
        self.nleads = nleads
        self.method = method
        self.seed = seed 
        self.task = task
        self.phase = phase
        self.reduce_dataset = reduce_dataset

        self.load_data() 

    def load_data(self):
        self.X = np.load(os.path.join(self.db_dir, "seed{}{}/{}X_{}.npy".format(self.seed, self.task, self.reduce_dataset, self.phase)), allow_pickle=True)
        self.y = np.squeeze(np.load(os.path.join(self.db_dir, "seed{}{}/{}y_{}.npy".format(self.seed, self.task, self.reduce_dataset, self.phase)), allow_pickle=True))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample, label = self.X[idx], self.y[idx]
        label = torch.tensor(label,dtype=torch.long)

        if self.method == "transfer":
            sample = self.normalize(sample)
            return sample.float(), label

        else:
            nsamples = sample.shape[-1]
            if self.positive_pairing == 'CMSC': 
                combined_sample = torch.empty(self.nleads, int(nsamples/2), 2)
                sample1, sample2 = np.split(sample, 2, axis=1)
                combined_sample[:,:,0] = self.normalize(sample1)
                combined_sample[:,:,1] = self.transform(sample2)
            elif self.positive_pairing == 'SimCLR': 
                combined_sample = torch.empty(self.nleads, nsamples, 2)
                combined_sample[:,:,0] = self.normalize(sample)
                combined_sample[:,:,1] = self.transform(sample)
            else: 
                # CMLC, CMSMLC, PCLR not implemented
                raise NotImplementedError
            
            return combined_sample.float(), label

    