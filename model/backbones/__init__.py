from .resnet import resnet1d18
from .convnet import convnet1d
from .transformer import transformer_d2_h4_dim64l

BACKBONES = {
    "resnet": resnet1d18, # Params 4,111,488
    "convnet": convnet1d,
    "transformer": transformer_d2_h4_dim64l,
}
