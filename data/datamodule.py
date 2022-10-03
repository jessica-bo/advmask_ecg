import os
from collections import Counter

import torch
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler, RandomSampler

from pytorch_lightning import LightningDataModule

from .dataset_wrapper import DatasetWrapper
from .style_dataset_wrapper import StyleDatasetWrapper

class ECGDataModule(LightningDataModule):
    def __init__(
        self, 
        data_dir, 
        dataset, 
        batch_size, 
        method, 
        seed, 
        positive_pairing, 
        num_workers, 
        do_test=False,
        debug=False,
        reduce_dataset="",
        nleads=12, 
        style_dataset="",
        style_alpha=0.5,
        task="",
        **kwargs
    ):
        super().__init__()

        self.dataset = dataset 
        self.style_dataset = style_dataset
        self.style_alpha = style_alpha
        self.method = method
        self.data_dir = data_dir
        self.debug = debug
        self.reduce_dataset = reduce_dataset
        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.phase = "val" if self.debug else "train"

        if self.style_dataset:
            self.data_train = StyleDatasetWrapper(self.data_dir, self.dataset, self.style_dataset, self.style_alpha, self.method, self.phase, self.task, seed, positive_pairing=positive_pairing, nleads=nleads, reduce_dataset=reduce_dataset, **kwargs)
        else:
            self.data_train = DatasetWrapper(self.data_dir, self.dataset, self.method, self.phase, self.task, seed, positive_pairing=positive_pairing, nleads=nleads, reduce_dataset=reduce_dataset, **kwargs)
   
        self.data_val = DatasetWrapper(self.data_dir, self.dataset, "transfer", "val", self.task, seed, positive_pairing=positive_pairing, nleads=nleads, reduce_dataset=reduce_dataset, **kwargs)

        if do_test:
            self.data_test = DatasetWrapper(self.data_dir, self.dataset, "transfer", "test", self.task, seed, positive_pairing=positive_pairing, nleads=nleads, **kwargs)


    def train_dataloader(self):
        '''returns training dataloader'''
        sampler = self.get_sampler()
        return DataLoader(self.data_train, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        '''returns validation dataloader'''
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        '''returns test dataloader'''
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def get_sampler(self):
        if self.dataset == "chapman" and self.get_type() != "regression":
            class_dist = dict(Counter(self.data_train.y))
            class_weights = [1/v for (k,v) in class_dist.items()]
            sample_weights = torch.Tensor([class_weights[t] for t in self.data_train.y])
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        else:
            sampler = RandomSampler(self.data_train.y)

        if self.debug: 
            return sampler
        else:
            return DistributedSampler(sampler)

    # For Chapman
    def get_nclass(self):
        if self.task == "age":
            return 1
        elif self.task == "gender":
            return 1
        elif self.task == "":
            return 4

    # For Chapman
    def get_type(self):
        if self.task == "age":
            return "regression"
        elif self.task == "gender":
            return "binary"
        elif self.task == "":
            return "single"
