import os, sys
import random
from typing import Any, Callable, Iterable, List, Optional, Sequence, Type, Union
from collections import Counter

import torch
import torchvision
from torch.utils.data import DataLoader, random_split, DistributedSampler, WeightedRandomSampler, RandomSampler

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import pytorch_lightning as pl

from .dataset_wrapper import DatasetWrapper

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
        **kwargs
    ):
        super().__init__()

        self.dataset = dataset
        self.db_dir = os.path.join(data_dir, self.dataset)   
        self.debug = debug
        self.reduce_dataset = reduce_dataset

        if self.debug:
            # Load smaller validation dataset to speed up debugging
            self.data_train = DatasetWrapper(self.db_dir, method, "val", seed, positive_pairing=positive_pairing, nleads=nleads, reduce_dataset=reduce_dataset, **kwargs)
        else:
            self.data_train = DatasetWrapper(self.db_dir, method, "train", seed, positive_pairing=positive_pairing, nleads=nleads, reduce_dataset=reduce_dataset, **kwargs)

        self.data_val = DatasetWrapper(self.db_dir, "transfer", "val", seed, positive_pairing=positive_pairing, nleads=nleads, reduce_dataset=reduce_dataset, **kwargs)

        if do_test:
            self.data_test = DatasetWrapper(self.db_dir, "transfer", "test", seed, positive_pairing=positive_pairing, nleads=nleads, **kwargs)

        self.batch_size = batch_size
        self.num_workers = num_workers
 
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
        if self.dataset == "chapman":
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
