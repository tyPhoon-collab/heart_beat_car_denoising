import torch.nn as nn
from dataset.randomizer import (
    AddUniformNoiseRandomizer,
    SampleShuffleRandomizer,
    PhaseShuffleRandomizer,
)
from loss.weighted import WeightedLoss
from loss.weighted_combined import WeightedCombinedLoss
from loss.combine import CombinedLoss
from models.pixel_shuffle_auto_encoder import PixelShuffleAutoencoder
from models.pixel_shuffle_auto_encoder_transformer import (
    PixelShuffleAutoencoderTransformer,
)
from models.wave_u_net_enhance import WaveUNetEnhance
from models.wave_u_net_enhance_transformer import WaveUNetEnhanceTransformer
from models.auto_encoder import Autoencoder
from models.wave_u_net_enhance_two_stage_transformer import (
    WaveUNetEnhanceTwoStageTransformer,
)
from denoising_diffusion_pytorch import GaussianDiffusion1D


MODEL = [
    WaveUNetEnhance,
    WaveUNetEnhanceTransformer,
    WaveUNetEnhanceTwoStageTransformer,
    Autoencoder,
    PixelShuffleAutoencoder,
    PixelShuffleAutoencoderTransformer,
    GaussianDiffusion1D,
]
LOSS_FN = [
    nn.L1Loss,
    nn.SmoothL1Loss,
    CombinedLoss,
    WeightedLoss,
    WeightedCombinedLoss,
]
RANDOMIZER = [
    SampleShuffleRandomizer,
    PhaseShuffleRandomizer,
    AddUniformNoiseRandomizer,
]

MODEL_NAMES = [model.__name__ for model in MODEL]
LOSS_FN_NAMES = [loss_fn.__name__ for loss_fn in LOSS_FN]
RANDOMIZER_NAMES = [randomizer.__name__ for randomizer in RANDOMIZER]
