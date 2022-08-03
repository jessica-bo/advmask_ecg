import argparse
import os, sys
import json
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import wandb
import argparse
from pathlib import Path
from argparse import Namespace

from model.base_model import BaseModel
from model.adversarial_model import AdversarialModel
from model.linear_model import LinearModel

from utils.checkpointer import Checkpointer

METHODS = {
    "base": BaseModel,
    "adversarial": AdversarialModel,
    "linear": LinearModel,
}


def parse_args_pretrain() -> argparse.Namespace:
    """Parses dataset, augmentation, pytorch lightning, model specific and additional args.

    First adds shared args such as dataset, augmentation and pytorch lightning args, then pulls the
    model name from the command and proceeds to add model specific args from the desired class. If
    wandb is enabled, it adds checkpointer args. Finally, adds additional non-user given parameters.

    Returns:
        argparse.Namespace: a namespace containing all args needed for pretraining.
    """

    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    # parser = METHODS[temp_args.method].add_model_specific_args(parser)

    parser.add_argument("--train-mode", type=str, default="SimCLR", help="SimCLR, CMSC, CMLC, CMSMLC")
    parser.add_argument("--trial", type=str, default="Pretrain", help="Pretrain, Linear, Fine-Tuning, Random")
    parser.add_argument("--dataset", type=str, default="ptbxl", help="ptbxl, chapman")
    parser.add_argument("--pretrain-dataset", type=str, default="", help="dataset of pretrained model")
    parser.add_argument("--perturbation", type=str, default="", help="string of pertubations")
    parser.add_argument("--mask", action="store_true", default=False)
    parser.add_argument("--thres", type=float, default=0.5)
    parser.add_argument("--ratio", type=float, default=0.8)
    parser.add_argument("--nmasks", type=int, default=1)
    parser.add_argument("--randmask", action="store_true", default=False)
    parser.add_argument("--gaussian", action="store_true", default=False)
    parser.add_argument("--randconv", action="store_true", default=False)
    parser.add_argument("--wandb-disabled", action="store_true", default=False)
    parser.add_argument("--save-disabled", action="store_true", default=False)
    parser.add_argument("--max-seed", type=int, default=1)

    args = parser.parse_args()

    # load pretrained model
    if args.pretrained_dir is not None:
        assert os.path.exists(args.pretrained_dir), \
            f"Pretrained model folder {args.pretrained_dir} does not exist!"
        args_path = os.path.join(args.pretrained_dir, "args.json")
        ckpt_paths = [os.path.join(args.pretrained_dir, ckpt)
                     for ckpt in os.listdir(Path(args.pretrained_dir))
                     if ckpt.endswith(".ckpt")]
        assert os.path.exists(args_path) and len(ckpt_paths)>0, \
            f"Pretrained model folder {args.pretrained_dir} is incomplete! " \
            f"Make sure there is a checkpoint file and args file in the directory."
        args.ckpt_path, args.args_path = ckpt_paths[0], args_path
        # load arguments
        with open(args_path) as f:
            pretrained_args = json.load(f)
        # some args needs to be loaded from pretrained models
        # args = inherit_args_from_pretrained(args.__dict__, pretrained_args)

    wandb_args = {
        "config": args,
        "project": args.project,
        "entity": args.entity,
        "dir": args.wandb_dir
    }
    if args.name != "none":
        wandb_args["name"] = args.name

    wandb.init(**wandb_args)
    c = wandb.config
    
    return Namespace(**c)

def parse_args_linear() -> argparse.Namespace:
    """Parses args for linear training, i.e. feature extractor.

    Returns:
        argparse.Namespace: a namespace containing all args needed for linear probing.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_feature_extractor", type=str, default=None)

    # add shared arguments
    dataset_args(parser)

    # add pytorch lightning trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # linear model
    parser = METHODS["linear"].add_model_specific_args(parser)

    # THIS LINE IS KEY TO PULL WANDB
    temp_args, _ = parser.parse_known_args()

    # add checkpointer args (only if logging is enabled)
    if temp_args.wandb:
        parser = Checkpointer.add_checkpointer_args(parser)

    # parse args
    args = parser.parse_args()

    wandb_args = {
        "config": args,
        "project": args.project,
        "entity": args.entity,
        "dir": args.wandb_dir
    }
    if args.name != "none":
        wandb_args["name"] = args.name

    wandb.init(**wandb_args)
    c = wandb.config

    return Namespace(**c)

def parse_args_finetune() -> argparse.Namespace:
    """Parses args for finetuning, including feature extractor, and validation frequency.

    Returns:
        argparse.Namespace: a namespace containing all args needed for finetuning.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_feature_extractor", type=str)
    parser.add_argument("--validation_frequency", type=int, default=1)

    # add shared arguments
    dataset_args(parser)

    # add pytorch lightning trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # linear model
    parser = METHODS["supervised"].add_model_specific_args(parser)

    # THIS LINE IS KEY TO PULL WANDB
    temp_args, _ = parser.parse_known_args()

    # add checkpointer args (only if logging is enabled)
    # if temp_args.wandb:
    #     parser = Checkpointer.add_checkpointer_args(parser)

    # parse args
    args = parser.parse_args()

    return args

def dataset_args(parser):
    """Adds dataset-related arguments to a parser.

    Args:
        parser (ArgumentParser): parser to add dataset args to.
    """

    parser.add_argument("--dataset", type=str)

    # dataset path
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--train_dir", type=str, default=None)
    parser.add_argument("--val_dir", type=str, default=None)

def augmentations_args(parser):
    """Adds augmentation-related arguments to a parser.

    Args:
        parser (ArgumentParser): parser to add augmentation args to.
    """

    # cropping
    parser.add_argument("--n_crops", type=int, default=2)
    parser.add_argument("--n_small_crops", type=int, default=0)

    # augmentations
    parser.add_argument("--brightness", type=float, nargs="+", default=[0.8])
    parser.add_argument("--contrast", type=float, nargs="+", default=[0.8])
    parser.add_argument("--saturation", type=float, nargs="+", default=[0.8])
    parser.add_argument("--hue", type=float, nargs="+", default=[0.2])
    parser.add_argument("--gaussian_prob", type=float, default=[0.5], nargs="+")
    parser.add_argument("--solarization_prob", type=float, default=[0.0], nargs="+")
    parser.add_argument("--min_scale", type=float, default=[0.08], nargs="+")

    # for imagenet or custom dataset
    parser.add_argument("--size", type=int, default=[224], nargs="+")

    # for custom dataset
    parser.add_argument("--mean", type=float, default=[0.485, 0.456, 0.406], nargs="+")
    parser.add_argument("--std", type=float, default=[0.228, 0.224, 0.225], nargs="+")
