from typing import Any

from models.wave_u_net_enhance_two_stage_transformer import (
    WaveUNetEnhanceTwoStageTransformer,
)
from ray import tune

from parameter_tuning.tune_base import TuneBase

import torch.nn as nn


class TuneWaveUNetEnhanceTwoStageTransformer(TuneBase):
    def get_param_space(self) -> dict[str, Any]:
        return {
            "time_d_model": tune.grid_search([40]),
            "time_nhead": tune.grid_search([20, 40]),
            "num_encoder_layers": tune.grid_search([2, 4, 6]),
            "lr": tune.grid_search([0.000025]),
            "batch_size": tune.grid_search([64]),
            "epoch_size": tune.grid_search([200]),
            "gain": tune.grid_search([1]),
            "with_progressive_gain": tune.grid_search([False, True]),
        }

    def build_model(self, config) -> nn.Module:
        return WaveUNetEnhanceTwoStageTransformer(
            time_d_model=config["time_d_model"],
            time_nhead=config["time_nhead"],
            num_encoder_layers=config["num_encoder_layers"],
        )
