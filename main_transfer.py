import os
import json
from pathlib import Path
import torch
torch.cuda.empty_cache()

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.plugins import DDPPlugin

from setup import parse_args_transfer

from model.transfer_model import TransferModel
from model.backbones import BACKBONES
from utils.checkpointer import Checkpointer
from setup import METHODS, NUM_CLASSES, TARGET_TYPE
from data.datamodule import ECGDataModule

# import logging 
import logging.config
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

def main():
    args = parse_args_transfer()
    seed_everything(args.seed)
    print(" Beginning transfer main() with seed {} and arguments \n {}: \n".format(args.seed, args))

    if args.pretrained_feature_extractor is not None:
        # build paths
        ckpt_dir = Path(args.pretrained_feature_extractor)
        args_path = ckpt_dir / "args.json"
        ckpt_path = [ckpt_dir / ckpt for ckpt in os.listdir(ckpt_dir) if ckpt.endswith(".ckpt")][0]

        # load arguments
        with open(args_path) as f:
            method_args = json.load(f)
        
        method = method_args["method"]
        original_model = METHODS[method].load_from_checkpoint(ckpt_path, strict=False, **(method_args))        
        backbone = original_model.encoder
        print("Loaded pretrained model {}.".format(args.pretrained_feature_extractor))
        print(ModelSummary(original_model, max_depth=1))

    else:
        backbone = BACKBONES[args.backbone](**vars(args))
        print("Loaded scratch model.")

    MethodClass = TransferModel
    model = MethodClass(encoder=backbone, 
                        n_classes=NUM_CLASSES[args.dataset], 
                        target_type=TARGET_TYPE[args.dataset], 
                        **args.__dict__)
    print(ModelSummary(model, max_depth=1))
    
    data_module= ECGDataModule(method="transfer", 
                               positive_pairing=None, 
                               nleads=12, 
                               do_test=True, 
                               **args.__dict__)

    print(" Loaded datamodule with dataset {}.".format(args.dataset))

    callbacks = []
    early_stop = EarlyStopping(monitor="transfer/val_auc", mode="max", patience=10)
    callbacks.append(early_stop)
    # wandb logging
    if args.wandb:
        print("Initiating WandB configs.")

        wandb_logger = WandbLogger(
            name=args.name, project=args.project, entity=args.entity, offline=True
        )
        # wandb_logger.watch(model, log="gradients", log_freq=100)
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
        # weights_summary="top",
        logger=wandb_logger if args.wandb else None,
        callbacks=callbacks,
        # plugins=DDPPlugin(find_unused_parameters=False),
        checkpoint_callback=False,
        terminate_on_nan=True,
        gpus=-1,
        fast_dev_run=args.debug,
        accelerator="gpu",
        strategy="ddp",
        profiler="simple",
        replace_sampler_ddp=False,
    )
    print(" Created Lightning Trainer and starting training.")

    trainer.fit(model=model, datamodule=data_module)
    trainer.test(model=model, datamodule=data_module)

if __name__ == "__main__":
    main()
