import torch
from functools import partial
from argparse import ArgumentParser
from typing import Any, Dict, List, Tuple, Callable, Sequence

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from src.utils.lars import LARSWrapper
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from src.utils.metrics import average_ari, average_segcover, weighted_mean
from src.methods.base import BaseModel, BaseMomentumModel


def static_lr(
    get_lr: Callable, param_group_indexes: Sequence[int], lrs_to_replace: Sequence[float]
):
    lrs = get_lr()
    for idx, lr in zip(param_group_indexes, lrs_to_replace):
        lrs[idx] = lr
    return lrs


class AdversarialModel(BaseModel):
    def __init__(self, dataset, train_mask_epoch=0, **kwargs):
        super().__init__(**kwargs)
        # add summed, max metrics
        self.extra_metric_keys = []
        self.mask_metric_keys = ["mask_ari", "mask_ari_fg", "mask_msc","mask_msc_fg",
                                 "mask_ssc", "mask_ssc_fg", "mask_map", "mask_sap",
                                 "mask_map_fg", "mask_sap_fg"]
        self.train_mask_epoch = train_mask_epoch

    def configure_optimizers(self):
        optimizer = [optimizer(
                self.learnable_params['encoder'],
                lr=self.lr,
                weight_decay=self.weight_decay,
                **self.extra_optimizer_args,
            ),
            optimizer(
                    self.learnable_params['mask'],
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    **self.extra_optimizer_args,
                )]
        return optimizer 

    @property
    def learnable_params(self) -> Dict[str, Any]:
        encoder_learnable_params = \
            super().learnable_params + [{"params": self.projector.parameters()}]

        mask_learnable_params = [
            {
                "name": "mask_encoder",
                "params": self.mask_encoder.parameters(),
                "lr": self.mask_lr
            },
            {
                "name": "mask_head",
                "params": self.mask_head.parameters(),
                "lr": self.mask_lr
            }
        ]
        return {"encoder": encoder_learnable_params, "mask": mask_learnable_params}