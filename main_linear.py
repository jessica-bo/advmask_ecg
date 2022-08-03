import os
import json
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin

from src.args.setup import parse_args_linear

from models.linear_model import LinearModel
from utils.checkpointer import Checkpointer
from setup import METHODS


from .data.datamodule import ECGDataModule

def main():
    args = parse_args_linear()

    if args.pretrained_feature_extractor is not None:
        # build paths
        ckpt_dir = Path(args.pretrained_feature_extractor)
        args_path = ckpt_dir / "args.json"
        ckpt_path = [ckpt_dir / ckpt for ckpt in os.listdir(ckpt_dir) if ckpt.endswith(".ckpt")][0]
    else:
        ckpt_dir = None

    # load arguments
    with open(args_path) as f:
        method_args = json.load(f)

    # build the model
    model = METHODS[method_args["method"]].load_from_checkpoint(
        ckpt_path, strict=False, **method_args
    )

    MethodClass = LinearModel
    model = MethodClass(model, **args.__dict__)

    data_module= ECGDataModule(dataset="chapman", batch_size=32, trial="transfer", seed=1, positive_pairing=None, augmentations="normalize", nleads=12)

    callbacks = []

    # wandb logging
    if args.wandb:
        wandb_logger = WandbLogger(
            name=args.name, project=args.project, entity=args.entity, offline=True
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(args)

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

        # save checkpoint on last epoch only
        ckpt = Checkpointer(
            args,
            logdir=os.path.join(args.pretrained_feature_extractor, "linear"),
            frequency=args.checkpoint_frequency,
        )
        callbacks.append(ckpt)

    trainer = Trainer.from_argparse_args(
        args,
        logger=wandb_logger if args.wandb else None,
        callbacks=callbacks,
        plugins=DDPPlugin(find_unused_parameters=False),
        checkpoint_callback=False,
        terminate_on_nan=True,
        accelerator="ddp",
    )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
