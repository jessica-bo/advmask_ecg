import argparse
import os, sys
import json
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import wandb
import argparse
from pathlib import Path

from model.base_model import BaseModel
from model.advmask_model import AdvMaskModel
from model.advMLP_model import AdvMLPModel
from model.transfer_model import TransferModel

from utils.checkpointer import Checkpointer

METHODS = {
    "base": BaseModel,
    "advmask": AdvMaskModel,
    "advmlp": AdvMLPModel,
    "transfer": TransferModel,
}

NUM_CLASSES = {
    "chapman": 4, 
    "cinc2021": 21,
    "ptbxl": 12,
}

TARGET_TYPE = {
    "chapman": "single", 
    "cinc2021": "multilabel",
    "ptbxl": "multilabel",
}

def none_or_str(value):
    if value == 'None':
        return None
    return value

def parse_args_pretrain():
    """
    Parses dataset, augmentation, pytorch lightning, model specific and additional args.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--method", type=str, default="base", choices=["base", "advmask", "advmlp"])
    parser.add_argument("--positive_pairing", type=str, default="SimCLR", choices=["SimCLR", "CMSC"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--checkpoint_dir", type=str, default="/home/gridsan/ybo/advaug/outputs/")

    parser = Trainer.add_argparse_args(parser)
    dataset_args(parser)
    augmentation_args(parser)

    temp_args, _ = parser.parse_known_args()
    parser = METHODS[temp_args.method].add_model_specific_args(parser)
    temp_args, _ = parser.parse_known_args()

    # add checkpointer args if logging is enabled
    if temp_args.wandb:
        parser = Checkpointer.add_checkpointer_args(parser)

    args = parser.parse_args()

    wandb_args = {
        "config": args,
        "project": args.project,
        "entity": args.entity,
        "dir": os.path.join(args.checkpoint_dir, args.name, "seed{}".format(args.seed))
    }
    if args.name != "none":
        wandb_args["name"] = args.name

    os.environ["WANDB_API_KEY"] = "57578f2c085ea7a785a36d8a38adad6d5e3ee3d5"#args.wandb_key 
    os.environ["WANDB_MODE"] = "offline" if args.wandb else "disabled"
    wandb.init(**wandb_args)

    return args

def parse_args_transfer():
    """
    Parses args for linear training, i.e. feature extractor.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_feature_extractor", type=none_or_str, nargs='?', default=None)
    parser.add_argument("--backbone", type=str, default="resnet")
    parser.add_argument("--finetune", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoint_dir", type=str, default="/home/gridsan/ybo/advaug/outputs/")

    parser = Trainer.add_argparse_args(parser)
    dataset_args(parser)
    parser = METHODS["transfer"].add_model_specific_args(parser)

    temp_args, _ = parser.parse_known_args()
    if temp_args.wandb:
        parser = Checkpointer.add_checkpointer_args(parser)

    args = parser.parse_args()

    wandb_args = {
        "config": args,
        "project": args.project,
        "entity": args.entity,
        "dir": os.path.join(args.checkpoint_dir, args.name, "seed{}".format(args.seed))
    }
    if args.name != "none":
        wandb_args["name"] = args.name

    os.environ["WANDB_API_KEY"] = "57578f2c085ea7a785a36d8a38adad6d5e3ee3d5"#args.wandb_key
    os.environ["WANDB_MODE"] = "offline" if args.wandb else "disabled"
    wandb.init(**wandb_args)

    return args

def dataset_args(parser):
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--data_dir", type=str, default="/home/gridsan/ybo/advaug/data/")
    parser.add_argument("--reduce_dataset", type=str, default="")

def augmentation_args(parser):
    # Gaussian
    parser.add_argument("--gaussian", action="store_true", default=False)
    parser.add_argument("--gaussian_sigma", type=float, default=0.05)

    # BaselineWander
    parser.add_argument("--wander", action="store_true", default=False)

    # BaselineShift
    parser.add_argument("--shift", action="store_true", default=False)

    # PowerlineNoise
    parser.add_argument("--powerline", action="store_true", default=False)

    # EMGNoise
    parser.add_argument("--emg", action="store_true", default=False)

    # RandomLeadMask
    parser.add_argument("--rlm", action="store_true", default=False)
    parser.add_argument("--rlm_prob", type=float, default=0.5)

    # RandomMask
    parser.add_argument("--mask", action="store_true", default=False)
    parser.add_argument("--mask_ratio", type=float, default=0.2)

    # RandomBlockMask
    parser.add_argument("--blockmask", action="store_true", default=False)
    parser.add_argument("--blockmaskbad", action="store_true", default=False)
    parser.add_argument("--blockmask_ratio", type=float, default=0.2)

    # 3KG
    parser.add_argument("--threeKG", action="store_true", default=False)
    parser.add_argument("--threeKG_angle", type=int, default=45)
    parser.add_argument("--threeKG_scale", type=float, default=1.5)
    parser.add_argument("--threeKG_mask", type=float, default=0)

    # RandomFourier
    parser.add_argument("--randfourier", action="store_true", default=False)



