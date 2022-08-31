"""
Adapted from @YugeTen 
Source: https://github.com/YugeTen/adios/blob/main/main_pretrain.py

"""

import os

import torch
torch.cuda.empty_cache()

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.plugins import DDPPlugin

from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from utils.checkpointer import Checkpointer
from setup import parse_args_pretrain, METHODS, NUM_CLASSES, TARGET_TYPE
from data.datamodule import ECGDataModule

def main():
    args = parse_args_pretrain()
    seed_everything(args.seed)
    print(" Beginning pretrain main() with seed {} and arguments {}: \n".format(args.seed, args))

    MethodClass = METHODS[args.method]
    model = MethodClass(n_classes=NUM_CLASSES[args.dataset], 
                        target_type=TARGET_TYPE[args.dataset], 
                        **args.__dict__)
    print(" Loaded {} model.".format(args.method))


    data_module = ECGDataModule(do_test=False,
                               **args.__dict__)
    print(" Loaded datamodule with dataset {}.".format(args.dataset))

    callbacks = []
    if args.simclr_loss_only:
        early_stop = EarlyStopping(monitor="train_nce_loss", mode="min", patience=5) 
        print("Early stopping with train_nce_loss.")
    else:
        early_stop = EarlyStopping(monitor="val_acc", mode="max", patience=5) 
        print("Early stopping with val_acc.")

    callbacks.append(early_stop)

    if args.wandb:
        print("Initiating WandB configs.")

        wandb_logger = WandbLogger(
            name=args.name, project=args.project, entity=args.entity, offline=True
        )
        wandb_logger.log_hyperparams(args)

        ckpt = Checkpointer(
            args,
            logdir=os.path.join(args.checkpoint_dir, args.name, "seed{}".format(args.seed)),
            frequency=args.checkpoint_frequency,
        )
        callbacks.append(ckpt)

    trainer = Trainer.from_argparse_args(
        args,
        logger=wandb_logger if args.wandb else None,
        callbacks=callbacks,
        checkpoint_callback=False,
        terminate_on_nan=True,
        gpus=args.num_devices,
        fast_dev_run=10 if args.debug else False,
        accelerator="gpu",
        strategy="ddp",
        precision=16 if args.method=="adversarial" else 32,
        profiler="simple",
        accumulate_grad_batches=args.accumulate_grad_batches if args.method=="base" else None, # do manually for adversarial 
        replace_sampler_ddp=False,
    )
    
    print("Created Lightning Trainer and starting training.")
    trainer.fit(model=model, datamodule=data_module)

if __name__ == "__main__":
    main()



