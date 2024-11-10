from config import Config
from utils.gain_controller import (
    ConstantGainController,
    GainController,
    ProgressiveGainController,
)


class GainControllerFactory:
    @classmethod
    def config(cls, c: Config) -> GainController:
        return (
            ProgressiveGainController(
                epoch_index_to=c.train.progressive_end_epoch - 1,
                min_gain=c.train.progressive_min_gain,
                max_gain=c.gain,
            )
            if c.train.progressive_gain
            else ConstantGainController(gain=c.gain)
        )
