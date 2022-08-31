"""
Adapted from @YugeTen
Source: https://github.com/YugeTen/adios/blob/main/src/utils/checkpointer.py

"""

import json
import os
from argparse import ArgumentParser, Namespace
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class Checkpointer(Callback):
    def __init__(
        self,
        args: Namespace,
        logdir: str = "trained_models",
        frequency: int = 1,
        keep_previous_checkpoints: bool = False,
    ):
        """
        Custom checkpointer callback that stores checkpoints in an easier to access way.
        """
        super().__init__()

        self.args = args
        self.logdir = logdir
        self.frequency = frequency
        self.keep_previous_checkpoints = keep_previous_checkpoints

    @staticmethod
    def add_checkpointer_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("checkpointer")
        parser.add_argument("--checkpoint_frequency", default=1, type=int)
        return parent_parser

    def initial_setup(self, trainer: pl.Trainer):
        self.path = self.logdir
        self.last_ckpt: Optional[str] = None

        # create logging dirs
        if trainer.is_global_zero:
            os.makedirs(self.path, exist_ok=True)

    def save_args(self, trainer: pl.Trainer):
        if trainer.is_global_zero:
            args = vars(self.args)
            json_path = os.path.join(self.path, "args.json")
            json.dump(args, open(json_path, "w"), default=lambda o: "<not serializable>")

    def save(self, trainer: pl.Trainer):
        if trainer.is_global_zero and not trainer.sanity_checking:
            epoch = trainer.current_epoch  # type: ignore
            ckpt_path = os.path.join(self.path, "model-{}.ckpt".format(epoch))

            trainer.save_checkpoint(ckpt_path, weights_only=False)
            if self.last_ckpt and self.last_ckpt != ckpt_path and not self.keep_previous_checkpoints:
                os.remove(self.last_ckpt)
            self.last_ckpt = ckpt_path

    def on_train_start(self, trainer: pl.Trainer, _):
        self.initial_setup(trainer)
        self.save_args(trainer)

    def on_validation_end(self, trainer: pl.Trainer, _):
        epoch = trainer.current_epoch  # type: ignore
        if epoch % self.frequency == 0:
            self.save(trainer)

