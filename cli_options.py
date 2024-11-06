from enum import Enum
import torch.nn as nn
from dataset.dataset import NoisyHeartbeatDataset
from dataset.factory import DatasetFactory
from dataset.randomizer import (
    AddUniformNoiseRandomizer,
    Randomizer,
    SampleShuffleRandomizer,
    PhaseShuffleRandomizer,
)
from loss.weighted import WeightedLoss
from loss.weighted_combined import WeightedCombinedLoss
from loss.combine import CombinedLoss
from models.gaussian_diffusion import GaussianDiffusion
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


class CLIModel(Enum):
    WaveUNetEnhance = "WaveUNetEnhance"
    WaveUNetEnhanceTransformer = "WaveUNetEnhanceTransformer"
    WaveUNetEnhanceTwoStageTransformer = "WaveUNetEnhanceTwoStageTransformer"
    Autoencoder = "Autoencoder"
    PixelShuffleAutoencoder = "PixelShuffleAutoencoder"
    PixelShuffleAutoencoderTransformer = "PixelShuffleAutoencoderTransformer"
    GaussianDiffusion = "GaussianDiffusion"


class CLILossFn(Enum):
    L1Loss = "L1Loss"
    SmoothL1Loss = "SmoothL1Loss"
    CombinedLoss = "CombinedLoss"
    WeightedLoss = "WeightedLoss"
    WeightedCombinedLoss = "WeightedCombinedLoss"


class CLIRandomizer(Enum):
    SampleShuffleRandomizer = "SampleShuffleRandomizer"
    PhaseShuffleRandomizer = "PhaseShuffleRandomizer"
    AddUniformNoiseRandomizer = "AddUniformNoiseRandomizer"


class CLIDataFolder(Enum):
    Raw240219 = "Raw240219"
    Raw240517 = "Raw240517"
    Raw240826 = "Raw240826"


def build_cli_model(args) -> nn.Module:
    match args.model:
        case CLIModel.WaveUNetEnhance:
            return WaveUNetEnhance()
        case CLIModel.WaveUNetEnhanceTransformer:
            return WaveUNetEnhanceTransformer(
                num_encoder_layers=args.num_encoder_layers
            )
        case CLIModel.WaveUNetEnhanceTwoStageTransformer:
            return WaveUNetEnhanceTwoStageTransformer(
                num_encoder_layers=args.num_encoder_layers
            )
        case CLIModel.Autoencoder:
            return Autoencoder()
        case CLIModel.PixelShuffleAutoencoder:
            return PixelShuffleAutoencoder()
        case CLIModel.PixelShuffleAutoencoderTransformer:
            return PixelShuffleAutoencoderTransformer()
        case CLIModel.GaussianDiffusion:
            return GaussianDiffusion(
                nn.MSELoss()
            )  # criterionは仮置き。CLIで上書きされる

    raise ValueError(f"Invalid model: {args.model}")


def build_cli_loss_fn(loss_fn: CLILossFn) -> nn.Module:
    match loss_fn:
        case CLILossFn.L1Loss:
            return nn.L1Loss()
        case CLILossFn.SmoothL1Loss:
            return nn.SmoothL1Loss()
        case CLILossFn.CombinedLoss:
            return CombinedLoss()
        case CLILossFn.WeightedLoss:
            return WeightedLoss()
        case CLILossFn.WeightedCombinedLoss:
            return WeightedCombinedLoss()


def build_cli_randomizer(randomizer: CLIRandomizer) -> Randomizer:
    match randomizer:
        case CLIRandomizer.SampleShuffleRandomizer:
            return SampleShuffleRandomizer()
        case CLIRandomizer.PhaseShuffleRandomizer:
            return PhaseShuffleRandomizer()
        case CLIRandomizer.AddUniformNoiseRandomizer:
            return AddUniformNoiseRandomizer()


def build_cli_data_folder(
    data_folder: CLIDataFolder, **kwargs
) -> NoisyHeartbeatDataset:
    match data_folder:
        case CLIDataFolder.Raw240219:
            return DatasetFactory.create_240219(**kwargs)
        case CLIDataFolder.Raw240517:
            return DatasetFactory.create_240517_filtered(**kwargs)
        case CLIDataFolder.Raw240826:
            return DatasetFactory.create_240826_filtered(**kwargs)
