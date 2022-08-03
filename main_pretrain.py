import os, sys
import numpy as np
import wandb
import argparse

from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin

from setup import parse_args_pretrain

def main():
    seed = np.random.randint(0, 2**32)
    seed_everything(seed)
    args = parse_args_pretrain()

    MethodClass = METHODS[args.method]
    model = MethodClass(**args.__dict__)

    callbacks = []

    if args.wandb:
        wandb_logger = WandbLogger(
            name=args.name,
            project=args.project,
            entity=args.entity,
            offline=args.offline,
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(args)

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

        # save checkpoint on last epoch only
        # ckpt = Checkpointer(
        #     args,
        #     logdir=os.path.join(args.checkpoint_dir, args.method),
        #     frequency=args.checkpoint_frequency,
        # )
        callbacks.append(ckpt)

        #TODO log images 

    #TODO resume from checkpoint 

    train_dataset = prepare_datasets(
        args.dataset,
        transform,
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        morphology=args.morph,
        load_masks=args.load_masks,
    )
    train_loader = prepare_dataloader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    val_dataset = prepare_datasets(
        args.dataset,
        transform,
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        morphology=args.morph,
        load_masks=args.load_masks,
    )
    val_loader = prepare_dataloader(
        val_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    test_dataset = prepare_datasets(
        args.dataset,
        transform,
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        morphology=args.morph,
        load_masks=args.load_masks,
    )
    test_loader = prepare_dataloader(
        test_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )


    trainer = Trainer.from_argparse_args(
        args,
        logger=wandb_logger if args.wandb else None,
        callbacks=callbacks,
        plugins=DDPPlugin(find_unused_parameters=False),
        checkpoint_callback=False,
        terminate_on_nan=True,
        accelerator="ddp",
        gpus=-1,
        check_val_every_n_epoch=args.validation_frequency,
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(test_dataloaders=test_loader)


if __name__ == "__main__":


    main()



