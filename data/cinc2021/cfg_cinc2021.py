"""
Adapted from: torch_ecg
Source: https://github.com/DeepPSP/torch_ecg/blob/af5e3d7ae879602cc4b0388f145a67efb95a6f58/benchmarks/train_crnn_cinc2021/cfg.py
"""

import os
from copy import deepcopy
from typing import List, NoReturn, MutableMapping, NoReturn, Optional, Any
from itertools import repeat
from easydict import EasyDict as ED
import torch
import random 
import numpy as np
from pathlib import Path
from functools import partial

import utils_cinc2021 

from torch_ecg.utils.utils_nn import adjust_cnn_filter_lengths
from torch_ecg.model_configs import ECG_CRNN_CONFIG
# from torch_ecg_bak.torch_ecg.utils.utils_nn import adjust_cnn_filter_lengths

__all__ = [
    "BaseCfg",
    "PlotCfg",
    "SpecialDetectorCfg",
    "TrainCfg",
    "TrainCfg_ns",
    "ModelCfg",
    "ModelCfg_ns",
    "ModelArchCfg",
]


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_ONE_MINUTE_IN_MS = 60 * 1000


# names of the 12 leads
Standard12Leads = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]
InferiorLeads = [
    "II",
    "III",
    "aVF",
]
LateralLeads = [
    "I",
    "aVL",
] + [f"V{i}" for i in range(5, 7)]
SeptalLeads = [
    "aVR",
    "V1",
]
AnteriorLeads = [f"V{i}" for i in range(2, 5)]
ChestLeads = [f"V{i}" for i in range(1, 7)]
PrecordialLeads = ChestLeads
LimbLeads = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
]


# settings from official repo
twelve_leads = (
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)
six_leads = ("I", "II", "III", "aVR", "aVL", "aVF")
four_leads = ("I", "II", "III", "V2")
three_leads = ("I", "II", "V2")
two_leads = ("I", "II")
lead_sets = (twelve_leads, six_leads, four_leads, three_leads, two_leads)


BaseCfg = ED()
BaseCfg.db_dir = "/home/gridsan/ybo/advaug/data/cinc2021/raw"
BaseCfg.log_dir = os.path.join(_BASE_DIR, "log")
BaseCfg.model_dir = os.path.join(_BASE_DIR, "saved_models")
# os.makedirs(BaseCfg.log_dir, exist_ok=True)
# os.makedirs(BaseCfg.model_dir, exist_ok=True)
BaseCfg.fs = 500
BaseCfg.torch_dtype = "float"  # "double"


SpecialDetectorCfg = ED()
SpecialDetectorCfg.leads_ordering = deepcopy(Standard12Leads)
SpecialDetectorCfg.pr_fs_lower_bound = 47  # Hz
SpecialDetectorCfg.pr_spike_mph_ratio = (
    15  # ratio to the average amplitude of the signal
)
SpecialDetectorCfg.pr_spike_mpd = 300  # ms
SpecialDetectorCfg.pr_spike_prominence = 0.3
SpecialDetectorCfg.pr_spike_prominence_wlen = 120  # ms
SpecialDetectorCfg.pr_spike_inv_density_threshold = (
    2500  # inverse density (1/density), one spike per 2000 ms
)
SpecialDetectorCfg.pr_spike_leads_threshold = 7 / 12  # proportion
SpecialDetectorCfg.axis_qrs_mask_radius = 70  # ms
SpecialDetectorCfg.axis_method = "2-lead"  # can also be "3-lead"
SpecialDetectorCfg.brady_threshold = _ONE_MINUTE_IN_MS / 60  # ms, corr. to 60 bpm
SpecialDetectorCfg.tachy_threshold = _ONE_MINUTE_IN_MS / 100  # ms, corr. to 100 bpm
SpecialDetectorCfg.lqrsv_qrs_mask_radius = 60  # ms
SpecialDetectorCfg.lqrsv_ampl_bias = (
    0.02  # mV, TODO: should be further determined by resolution, etc.
)
SpecialDetectorCfg.lqrsv_ratio_threshold = 0.8
SpecialDetectorCfg.prwp_v3_thr = 0.3  # mV

# special classes using special detectors
_SPECIAL_CLASSES = ["Brady", "LAD", "RAD", "PR", "LQRSV"]


# configurations for visualization
PlotCfg = ED()
# default const for the plot function in dataset.py
# used only when corr. values are absent
# all values are time bias w.r.t. corr. peaks, with units in ms
PlotCfg.p_onset = -40
PlotCfg.p_offset = 40
PlotCfg.q_onset = -20
PlotCfg.s_offset = 40
PlotCfg.qrs_radius = 60
PlotCfg.t_onset = -100
PlotCfg.t_offset = 60


def _assign_classes(cfg: ED, special_classes: List[str]) -> NoReturn:
    """ """
    cfg.special_classes = deepcopy(special_classes)
    cfg.tranche_class_weights = ED(
        {
            t: utils_cinc2021.get_class_weight(
                t,
                exclude_classes=cfg.special_classes,
                scored_only=True,
                threshold=20,
                min_weight=cfg.min_class_weight,
            )
            for t in [
                # "A",
                # "B",
                # "AB",
                # "E",
                "F",
                "G",
            ]
        }
    )
    cfg.tranche_classes = ED(
        {t: sorted(list(t_cw.keys())) for t, t_cw in cfg.tranche_class_weights.items()}
    )

    cfg.class_weights = utils_cinc2021.get_class_weight(
        tranches="ABEFG",
        exclude_classes=cfg.special_classes,
        scored_only=True,
        threshold=20,
        min_weight=cfg.min_class_weight,
    )
    cfg.classes = sorted(list(cfg.class_weights.keys()))


# training configurations for machine learning and deep learning
TrainCfg = ED()
TrainCfg.torch_dtype = BaseCfg.torch_dtype

# configs of files
TrainCfg.db_dir = BaseCfg.db_dir
TrainCfg.log_dir = BaseCfg.log_dir
TrainCfg.model_dir = BaseCfg.model_dir
TrainCfg.final_model_name = None
TrainCfg.checkpoints = os.path.join(_BASE_DIR, "checkpoints")
# os.makedirs(TrainCfg.checkpoints, exist_ok=True)
TrainCfg.keep_checkpoint_max = 20

TrainCfg.leads = deepcopy(twelve_leads)

# configs of training data
TrainCfg.fs = BaseCfg.fs
TrainCfg.data_format = "channel_first"

TrainCfg.train_ratio = 0.8
TrainCfg.min_class_weight = 0.5
TrainCfg.tranches_for_training = ""  # one of "", "AB", "E", "F", "G"

# assign classes, class weights, tranche classes, etc.
_assign_classes(TrainCfg, _SPECIAL_CLASSES)

# configs of signal preprocessing
TrainCfg.normalize = ED(
    method="z-score",
    mean=0.0,
    std=1.0,
)
# frequency band of the filter to apply, should be chosen very carefully
TrainCfg.bandpass = None  # [-np.inf, 45]
# TrainCfg.bandpass = [-np.inf, 45]
# TrainCfg.bandpass = ED(
#     lowcut=0.5,
#     highcut=60,
# )

# configs of data aumentation
# TrainCfg.label_smooth = ED(
#     prob=0.8,
#     smoothing=0.1,
# )
TrainCfg.label_smooth = False
TrainCfg.random_masking = False
TrainCfg.stretch_compress = False  # stretch or compress in time axis
TrainCfg.mixup = ED(
    prob=0.6,
    alpha=0.3,
)

# configs of training epochs, batch, etc.
TrainCfg.n_epochs = 50
# TODO: automatic adjust batch size according to GPU capacity
# https://stackoverflow.com/questions/45132809/how-to-select-batch-size-automatically-to-fit-gpu
TrainCfg.batch_size = 64
# TrainCfg.max_batches = 500500

# configs of optimizers and lr_schedulers
TrainCfg.optimizer = "adamw_amsgrad"  # "sgd", "adam", "adamw"
TrainCfg.momentum = 0.949  # default values for corresponding PyTorch optimizers
TrainCfg.betas = (0.9, 0.999)  # default values for corresponding PyTorch optimizers
TrainCfg.decay = 1e-2  # default values for corresponding PyTorch optimizers

TrainCfg.learning_rate = 1e-4  # 1e-3
TrainCfg.lr = TrainCfg.learning_rate

TrainCfg.lr_scheduler = "one_cycle"  # "one_cycle", "plateau", "burn_in", "step", None
TrainCfg.lr_step_size = 50
TrainCfg.lr_gamma = 0.1
TrainCfg.max_lr = 2e-3  # for "one_cycle" scheduler, to adjust via expriments

TrainCfg.burn_in = 400
TrainCfg.steps = [5000, 10000]

TrainCfg.early_stopping = ED()  # early stopping according to challenge metric
TrainCfg.early_stopping.min_delta = 0.001  # should be non-negative
TrainCfg.early_stopping.patience = 10

# configs of loss function
# TrainCfg.loss = "BCEWithLogitsLoss"
# TrainCfg.loss = "BCEWithLogitsWithClassWeightLoss"
TrainCfg.loss = "AsymmetricLoss"  # "FocalLoss"
TrainCfg.loss_kw = ED(gamma_pos=0, gamma_neg=0.2, implementation="deep-psp")
TrainCfg.flooding_level = (
    0.0  # flooding performed if positive, typically 0.45-0.55 for cinc2021?
)

TrainCfg.monitor = "challenge_metric"

TrainCfg.log_step = 20
TrainCfg.eval_every = 20

# configs of model selection
# "resnet_nature_comm_se", "multi_scopic_leadwise", "vgg16", "vgg16_leadwise",
TrainCfg.cnn_name = "resnet_nature_comm_bottle_neck_se"
TrainCfg.rnn_name = "none"  # "none", "lstm"
TrainCfg.attn_name = "none"  # "none", "se", "gc", "nl"

# configs of inputs and outputs
# almost all records have duration >= 8s, most have duration >= 10s
# use `utils.utils_signal.ensure_siglen` to ensure signal length
TrainCfg.input_len = int(500 * 10.0)
# tolerance for records with length shorter than `TrainCfg.input_len`
TrainCfg.input_len_tol = int(0.2 * TrainCfg.input_len)
TrainCfg.sig_slice_tol = 0.4  # None, do no slicing
TrainCfg.siglen = TrainCfg.input_len


# constants for model inference
_bin_pred_thr = 0.5
# `bin_pred_look_again_tol` is used when no prob is greater than `bin_pred_thr`,
# then the prediction would be the one with the highest prob.,
# along with those with prob. no less than the highest prob. minus `bin_pred_look_again_tol`
_bin_pred_look_again_tol = 0.03
_bin_pred_nsr_thr = 0.1


TrainCfg.bin_pred_thr = _bin_pred_thr
TrainCfg.bin_pred_look_again_tol = _bin_pred_look_again_tol
TrainCfg.bin_pred_nsr_thr = _bin_pred_nsr_thr


# the no special classes version

TrainCfg_ns = deepcopy(TrainCfg)
_assign_classes(TrainCfg_ns, [])


_BASE_MODEL_CONFIG = deepcopy(ECG_CRNN_CONFIG)
_BASE_MODEL_CONFIG.cnn.multi_scopic_leadwise.block.batch_norm = "group_norm"  # False

# detailed configs for 12-lead, 6-lead, 4-lead, 3-lead, 2-lead models
# mostly follow from torch_ecg.torch_ecg.model_configs.ecg_crnn
ModelArchCfg = ED()

ModelArchCfg.twelve_leads = deepcopy(_BASE_MODEL_CONFIG)

# TODO: add adjustifications for "leadwise" configs for 6,4,3,2 leads models
ModelArchCfg.six_leads = deepcopy(_BASE_MODEL_CONFIG)
ModelArchCfg.six_leads.cnn.vgg16_leadwise.groups = 6
_base_num_filters = 6 * 6  # 12 * 4
ModelArchCfg.six_leads.cnn.vgg16_leadwise.num_filters = [
    _base_num_filters * 4,
    _base_num_filters * 8,
    _base_num_filters * 16,
    _base_num_filters * 32,
    _base_num_filters * 32,
]
ModelArchCfg.six_leads.cnn.resnet_leadwise.groups = 6
ModelArchCfg.six_leads.cnn.resnet_leadwise.init_num_filters = 6 * 8  # 12 * 8
ModelArchCfg.six_leads.cnn.multi_scopic_leadwise.groups = 6
_base_num_filters = 6 * 6  # 12 * 4
ModelArchCfg.six_leads.cnn.multi_scopic_leadwise.num_filters = [
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
]
ModelArchCfg.six_leads.cnn.densenet_leadwise.groups = 6
ModelArchCfg.six_leads.cnn.densenet_leadwise.init_num_filters = 6 * 8  # 12 * 8
ModelArchCfg.six_leads.cnn.xception_leadwise.groups = 6
_base_num_filters = 6 * 2  # 12 * 2
ModelArchCfg.six_leads.cnn.xception_vanilla.entry_flow = ED(
    init_num_filters=[_base_num_filters * 4, _base_num_filters * 8],
    init_filter_lengths=3,
    init_subsample_lengths=[2, 1],
    num_filters=[
        _base_num_filters * 16,
        _base_num_filters * 32,
        _base_num_filters * 91,
    ],
    filter_lengths=3,
    subsample_lengths=2,
    subsample_kernels=3,
)
ModelArchCfg.six_leads.cnn.xception_vanilla.middle_flow = ED(
    num_filters=list(repeat(_base_num_filters * 91, 8)),
    filter_lengths=3,
)
ModelArchCfg.six_leads.cnn.xception_vanilla.exit_flow = ED(
    final_num_filters=[_base_num_filters * 182, _base_num_filters * 256],
    final_filter_lengths=3,
    num_filters=[[_base_num_filters * 91, _base_num_filters * 128]],
    filter_lengths=3,
    subsample_lengths=2,
    subsample_kernels=3,
)

ModelArchCfg.four_leads = deepcopy(_BASE_MODEL_CONFIG)
ModelArchCfg.four_leads.cnn.vgg16_leadwise.groups = 4
_base_num_filters = 6 * 4  # 12 * 4
ModelArchCfg.four_leads.cnn.vgg16_leadwise.num_filters = [
    _base_num_filters * 4,
    _base_num_filters * 8,
    _base_num_filters * 16,
    _base_num_filters * 32,
    _base_num_filters * 32,
]
ModelArchCfg.four_leads.cnn.resnet_leadwise.groups = 4
ModelArchCfg.four_leads.cnn.resnet_leadwise.init_num_filters = 6 * 6  # 12 * 8
ModelArchCfg.four_leads.cnn.multi_scopic_leadwise.groups = 4
_base_num_filters = 6 * 4  # 12 * 4
ModelArchCfg.four_leads.cnn.multi_scopic_leadwise.num_filters = [
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
]
ModelArchCfg.four_leads.cnn.densenet_leadwise.groups = 4
ModelArchCfg.four_leads.cnn.densenet_leadwise.init_num_filters = 6 * 6  # 12 * 8
ModelArchCfg.four_leads.cnn.xception_leadwise.groups = 4
_base_num_filters = 6 * 2  # 12 * 2
ModelArchCfg.four_leads.cnn.xception_vanilla.entry_flow = ED(
    init_num_filters=[_base_num_filters * 4, _base_num_filters * 8],
    init_filter_lengths=3,
    init_subsample_lengths=[2, 1],
    num_filters=[
        _base_num_filters * 16,
        _base_num_filters * 32,
        _base_num_filters * 91,
    ],
    filter_lengths=3,
    subsample_lengths=2,
    subsample_kernels=3,
)
ModelArchCfg.four_leads.cnn.xception_vanilla.middle_flow = ED(
    num_filters=list(repeat(_base_num_filters * 91, 8)),
    filter_lengths=3,
)
ModelArchCfg.four_leads.cnn.xception_vanilla.exit_flow = ED(
    final_num_filters=[_base_num_filters * 182, _base_num_filters * 256],
    final_filter_lengths=3,
    num_filters=[[_base_num_filters * 91, _base_num_filters * 128]],
    filter_lengths=3,
    subsample_lengths=2,
    subsample_kernels=3,
)

ModelArchCfg.three_leads = deepcopy(_BASE_MODEL_CONFIG)
ModelArchCfg.three_leads.cnn.vgg16_leadwise.groups = 3
_base_num_filters = 3 * 8  # 12 * 4
ModelArchCfg.three_leads.cnn.vgg16_leadwise.num_filters = [
    _base_num_filters * 4,
    _base_num_filters * 8,
    _base_num_filters * 16,
    _base_num_filters * 32,
    _base_num_filters * 32,
]
ModelArchCfg.three_leads.cnn.resnet_leadwise.groups = 3
ModelArchCfg.three_leads.cnn.resnet_leadwise.init_num_filters = 3 * 12  # 12 * 8
ModelArchCfg.three_leads.cnn.multi_scopic_leadwise.groups = 3
_base_num_filters = 3 * 8  # 12 * 4
ModelArchCfg.three_leads.cnn.multi_scopic_leadwise.num_filters = [
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
]
ModelArchCfg.three_leads.cnn.densenet_leadwise.groups = 3
ModelArchCfg.three_leads.cnn.densenet_leadwise.init_num_filters = 3 * 12  # 12 * 8
ModelArchCfg.three_leads.cnn.xception_leadwise.groups = 3
_base_num_filters = 3 * 4  # 12 * 2
ModelArchCfg.three_leads.cnn.xception_vanilla.entry_flow = ED(
    init_num_filters=[_base_num_filters * 4, _base_num_filters * 8],
    init_filter_lengths=3,
    init_subsample_lengths=[2, 1],
    num_filters=[
        _base_num_filters * 16,
        _base_num_filters * 32,
        _base_num_filters * 91,
    ],
    filter_lengths=3,
    subsample_lengths=2,
    subsample_kernels=3,
)
ModelArchCfg.three_leads.cnn.xception_vanilla.middle_flow = ED(
    num_filters=list(repeat(_base_num_filters * 91, 8)),
    filter_lengths=3,
)
ModelArchCfg.three_leads.cnn.xception_vanilla.exit_flow = ED(
    final_num_filters=[_base_num_filters * 182, _base_num_filters * 256],
    final_filter_lengths=3,
    num_filters=[[_base_num_filters * 91, _base_num_filters * 128]],
    filter_lengths=3,
    subsample_lengths=2,
    subsample_kernels=3,
)

ModelArchCfg.two_leads = deepcopy(_BASE_MODEL_CONFIG)
ModelArchCfg.two_leads.cnn.vgg16_leadwise.groups = 3
_base_num_filters = 2 * 12  # 12 * 4
ModelArchCfg.two_leads.cnn.vgg16_leadwise.num_filters = [
    _base_num_filters * 4,
    _base_num_filters * 8,
    _base_num_filters * 16,
    _base_num_filters * 32,
    _base_num_filters * 32,
]
ModelArchCfg.two_leads.cnn.resnet_leadwise.groups = 2
ModelArchCfg.two_leads.cnn.resnet_leadwise.init_num_filters = 2 * 16  # 12 * 8
ModelArchCfg.two_leads.cnn.multi_scopic_leadwise.groups = 2
_base_num_filters = 2 * 8  # 12 * 4
ModelArchCfg.two_leads.cnn.multi_scopic_leadwise.num_filters = [
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
]
ModelArchCfg.two_leads.cnn.densenet_leadwise.groups = 2
ModelArchCfg.two_leads.cnn.densenet_leadwise.init_num_filters = 2 * 12  # 12 * 8
ModelArchCfg.two_leads.cnn.xception_leadwise.groups = 3
_base_num_filters = 2 * 6  # 12 * 2
ModelArchCfg.two_leads.cnn.xception_vanilla.entry_flow = ED(
    init_num_filters=[_base_num_filters * 4, _base_num_filters * 8],
    init_filter_lengths=3,
    init_subsample_lengths=[2, 1],
    num_filters=[
        _base_num_filters * 16,
        _base_num_filters * 32,
        _base_num_filters * 91,
    ],
    filter_lengths=3,
    subsample_lengths=2,
    subsample_kernels=3,
)
ModelArchCfg.two_leads.cnn.xception_vanilla.middle_flow = ED(
    num_filters=list(repeat(_base_num_filters * 91, 8)),
    filter_lengths=3,
)
ModelArchCfg.two_leads.cnn.xception_vanilla.exit_flow = ED(
    final_num_filters=[_base_num_filters * 182, _base_num_filters * 256],
    final_filter_lengths=3,
    num_filters=[[_base_num_filters * 91, _base_num_filters * 128]],
    filter_lengths=3,
    subsample_lengths=2,
    subsample_kernels=3,
)

# configurations for building deep learning models
# terminologies of stanford ecg repo. will be adopted
ModelCfg = ED()
ModelCfg.torch_dtype = BaseCfg.torch_dtype
ModelCfg.fs = BaseCfg.fs
ModelCfg.spacing = 1000 / ModelCfg.fs
ModelCfg.bin_pred_thr = _bin_pred_thr
ModelCfg.bin_pred_look_again_tol = _bin_pred_look_again_tol
ModelCfg.bin_pred_nsr_thr = _bin_pred_nsr_thr

ModelCfg.special_classes = deepcopy(_SPECIAL_CLASSES)
ModelCfg.dl_classes = deepcopy(TrainCfg.classes)
ModelCfg.tranche_classes = deepcopy(TrainCfg.tranche_classes)
ModelCfg.full_classes = ModelCfg.dl_classes + ModelCfg.special_classes

ModelCfg.dl_siglen = TrainCfg.siglen

ModelCfg.cnn_name = TrainCfg.cnn_name
ModelCfg.rnn_name = TrainCfg.rnn_name
ModelCfg.attn_name = TrainCfg.attn_name

# model architectures configs
ModelCfg.update(ModelArchCfg)
for ld in ["twelve_leads", "six_leads", "four_leads", "three_leads", "two_leads"]:
    adjust_cnn_filter_lengths(ModelCfg[ld], ModelCfg.fs)
    ModelCfg[ld].cnn.name = ModelCfg.cnn_name
    ModelCfg[ld].rnn.name = ModelCfg.rnn_name
    ModelCfg[ld].attn.name = ModelCfg.attn_name
    # ModelCfg[ld].clf = ED()
    # ModelCfg[ld].clf.out_channels = [
    # # not including the last linear layer, whose out channels equals n_classes
    # ]
    # ModelCfg[ld].clf.bias = True
    # ModelCfg[ld].clf.dropouts = 0.0
    # ModelCfg[ld].clf.activation = "mish"  # for a single layer `SeqLin`, activation is ignored


# the no special classes version
ModelCfg_ns = deepcopy(ModelCfg)
ModelCfg_ns.special_classes = []
ModelCfg_ns.dl_classes = deepcopy(TrainCfg_ns.classes)
ModelCfg_ns.tranche_classes = deepcopy(TrainCfg_ns.tranche_classes)
ModelCfg_ns.full_classes = ModelCfg_ns.dl_classes + ModelCfg_ns.special_classes


_PROJECT_ROOT = Path(__file__).parent.resolve()
_PROJECT_CACHE = Path("~").expanduser() / ".cache" / "torch_ecg"
_PROJECT_CACHE.mkdir(parents=True, exist_ok=True)
_DATA_CACHE = _PROJECT_CACHE / "data"
_DATA_CACHE.mkdir(parents=True, exist_ok=True)


class CFG(dict):
    """
    this class is created in order to renew the `update` method,
    to fit the hierarchical structure of configurations
    Examples
    --------
    >>> c = CFG(hehe={"a":1,"b":2})
    >>> c.update(hehe={"a":-1})
    >>> c
    {'hehe': {'a': -1, 'b': 2}}
    >>> c.__update__(hehe={"a":-10})
    >>> c
    {'hehe': {'a': -10}}
    """

    __name__ = "CFG"

    def __init__(self, *args, **kwargs) -> NoReturn:
        """ """
        if len(args) > 1:
            raise TypeError(f"expected at most 1 arguments, got {len(args)}")
        elif len(args) == 1:
            d = args[0]
            assert isinstance(d, MutableMapping)
        else:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            try:
                setattr(self, k, v)
            except Exception:
                dict.__setitem__(self, k, v)
        # Class attributes
        exclude_fields = ["update", "pop"]
        for k in self.__class__.__dict__:
            if (
                not (k.startswith("__") and k.endswith("__"))
                and k not in exclude_fields
            ):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name: str, value: Any) -> NoReturn:
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x) if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super().__setattr__(name, value)
        super().__setitem__(name, value)

    __setitem__ = __setattr__

    def update(
        self, new_cfg: Optional[MutableMapping] = None, **kwargs: Any
    ) -> NoReturn:
        """
        the new hierarchical update method
        Parameters
        ----------
        new_cfg : MutableMapping, optional
            the new configuration, by default None
        kwargs : Any, optional
            key value pairs, by default None
        """
        _new_cfg = new_cfg or CFG()
        if len(kwargs) > 0:  # avoid RecursionError
            _new_cfg.update(kwargs)
        for k in _new_cfg:
            # if _new_cfg[k].__class__.__name__ in ["dict", "EasyDict", "CFG"] and k in self:
            if isinstance(_new_cfg[k], MutableMapping) and k in self:
                self[k].update(_new_cfg[k])
            else:
                try:
                    setattr(self, k, _new_cfg[k])
                except Exception:
                    dict.__setitem__(self, k, _new_cfg[k])

    def pop(self, key: str, default: Optional[Any] = None) -> Any:
        """
        the updated pop method
        Parameters
        ----------
        key : str
            the key to pop
        default : Any, optional
            the default value, by default None
        """
        if key in self:
            delattr(self, key)
        return super().pop(key, default)


DEFAULTS = CFG()

DEFAULTS.log_dir = _PROJECT_CACHE / "log"
DEFAULTS.checkpoints = _PROJECT_CACHE / "checkpoints"
DEFAULTS.model_dir = _PROJECT_CACHE / "saved_models"
DEFAULTS.prefix = "TorchECG"

DEFAULTS.torch_dtype = torch.float32  # torch.float64, torch.float16
DEFAULTS.str_dtype = str(DEFAULTS.torch_dtype).replace("torch.", "")
DEFAULTS.np_dtype = np.dtype(DEFAULTS.str_dtype)
DEFAULTS.dtype = DEFAULTS.torch_dtype

DEFAULTS.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

DEFAULTS.eps = 1e-7

DEFAULTS.SEED = 42
DEFAULTS.RNG = np.random.default_rng(seed=DEFAULTS.SEED)
DEFAULTS.RNG_sample = partial(DEFAULTS.RNG.choice, replace=False, shuffle=False)
DEFAULTS.RNG_randint = partial(DEFAULTS.RNG.integers, endpoint=True)


def set_seed(seed: int) -> NoReturn:
    """
    set the seed of the random number generator
    Parameters
    ----------
    seed: int,
        the seed to be set
    """

    global DEFAULTS
    DEFAULTS.SEED = seed
    DEFAULTS.RNG = np.random.default_rng(seed=seed)
    DEFAULTS.RNG_sample = partial(DEFAULTS.RNG.choice, replace=False, shuffle=False)
    DEFAULTS.RNG_randint = partial(DEFAULTS.RNG.integers, endpoint=True)

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)