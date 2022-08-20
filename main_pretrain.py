import os
import json
from pathlib import Path
import torch
torch.cuda.empty_cache()

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from model.transfer_model import TransferModel
from utils.checkpointer import Checkpointer
from setup import parse_args_pretrain, METHODS, NUM_CLASSES, TARGET_TYPE

from data.datamodule import ECGDataModule

# import logging 
import logging.config
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

def main():
    args = parse_args_pretrain()
    seed_everything(args.seed)

    # console_log = logging.getLogger("Lightning")
    # console_log.setLevel(20)
    print(" Beginning pretrain main() with seed {} and arguments {}: \n".format(args.seed, args))

    callbacks = []

    MethodClass = METHODS[args.method]
    model = MethodClass(n_classes=NUM_CLASSES[args.dataset], 
                        target_type=TARGET_TYPE[args.dataset], 
                        **args.__dict__)
                        
    print(" Loaded {} model.".format(args.method))

    data_module= ECGDataModule(data_dir=args.data_dir, 
                               dataset=args.dataset, 
                               batch_size=args.batch_size, 
                               method=args.method, 
                               seed=args.seed, 
                               positive_pairing=args.positive_pairing,
                               nleads=12, 
                               num_workers=args.num_workers, 
                               do_test=False,
                               debug=args.debug)

    print(" Loaded datamodule with dataset {}.".format(args.dataset))

    callbacks = []
    early_stop = EarlyStopping(monitor="val_class_loss", mode="min", patience=10) #TODO 
    callbacks.append(early_stop)

    # wandb logging
    if args.wandb:
        print("Initiating WandB configs.")
        wandb_logger = WandbLogger(
            name=args.name, project=args.project, entity=args.entity, offline=True
        )
        wandb_logger.watch(model, log=None) #, log_freq=100)
        wandb_logger.log_hyperparams(args)

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

        # save checkpoint on last epoch only
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
        fast_dev_run=args.debug,
        accelerator="gpu",
        strategy="ddp",
        precision=16,
        profiler="simple",
        accumulate_grad_batches=args.accumulate_grad_batches if args.method=="base" else None, # disable for adversarial 
        # auto_scale_batch_size='power',
        # auto_lr_find=True,
    )
    
    # TODO cuda error 
    if args.tune:
        tuner = Tuner(trainer)

        print("Finding batch_size...")
        new_batch_size = tuner.scale_batch_size(model, datamodule=data_module)
        print(new_batch_size)

        print("Finding LR...")
        lr_finder = tuner.lr_find(model, datamodule=data_module)
        print(lr_finder.results)
        print(lr_finder.suggestion())

    else:
        print(" Created Lightning Trainer and starting training.")
        trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()



