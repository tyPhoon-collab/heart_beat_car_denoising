from typing import Any

from torch.nn.modules import Module
from models.wave_u_net_enhance_transformer import WaveUNetEnhanceTransformer
from ray import tune

from parameter_tuning.tune_base import TuneBase


class TuneWaveUNetEnhanceTransformer(TuneBase):
    def get_param_space(self) -> dict[str, Any]:
        return {
            "num_encoder_layers": tune.grid_search([2, 4, 6]),
            "lr": tune.grid_search([0.000025]),
            "batch_size": tune.grid_search([64]),
            "epoch_size": tune.grid_search([200]),
            "gain": tune.grid_search([1]),
            "with_progressive_gain": tune.grid_search([False, True]),
        }

    def build_model(self, config) -> Module:
        return WaveUNetEnhanceTransformer(
            num_encoder_layers=config["num_encoder_layers"],
        )
