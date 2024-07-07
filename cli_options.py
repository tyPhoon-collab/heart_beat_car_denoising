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


MODEL = {
    "WaveUNetEnhance": WaveUNetEnhance,
    "WaveUNetEnhanceTransformer": WaveUNetEnhanceTransformer,
    "WaveUNetEnhanceTwoStageTransformer": WaveUNetEnhanceTwoStageTransformer,
    "Autoencoder": Autoencoder,
    "PixelShuffleAutoencoder": PixelShuffleAutoencoder,
    "PixelShuffleAutoencoderTransformer": PixelShuffleAutoencoderTransformer,
}

LOSS_FN = {
    "L1Loss": nn.L1Loss,
    "SmoothL1Loss": nn.SmoothL1Loss,
    "CombinedLoss": CombinedLoss,
    "WeightedLoss": WeightedLoss,
    "WeightedCombinedLoss": WeightedCombinedLoss,
}

RANDOMIZER = {
    "SampleShuffleRandomizer": SampleShuffleRandomizer,
    "PhaseShuffleRandomizer": PhaseShuffleRandomizer,
    "AddUniformNoiseRandomizer": AddUniformNoiseRandomizer,
}

MODEL_NAMES = [key for key in MODEL]
LOSS_FN_NAMES = [key for key in LOSS_FN]
RANDOMIZER_NAMES = [key for key in RANDOMIZER]
