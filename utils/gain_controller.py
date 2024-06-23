from abc import ABC, abstractmethod
from dataclasses import dataclass

from utils.epoch_sensitive import EpochSensitive


class GainController(ABC):
    @abstractmethod
    def get_gain(self) -> float:
        pass


@dataclass
class ConstantGainController(GainController):
    gain: float = 1.0

    def get_gain(self):
        return self.gain

    def __str__(self) -> str:
        return f"ConstantGainController(gain={self.gain})"


@dataclass
class ProgressiveGainController(GainController, EpochSensitive):
    """
    epochの値によって、min_gainからmax_gainの間で線形に変化させる
    """

    epoch_index_from: int = 0
    epoch_index_to: int = 0
    min_gain: float = 0.0
    max_gain: float = 1.0
    gain: float = 1.0

    def get_gain(self):
        return self.gain

    def __str__(self) -> str:
        return f"ProgressiveGainController(epoch=[{self.epoch_index_from}, {self.epoch_index_to}], gain=[{self.min_gain}, {self.max_gain}])"  # noqa

    def on_start_epoch(self, epoch_idx):
        self.set_gain_from_epoch(epoch_idx)

    def set_gain_from_epoch(self, epoch_idx: int):
        self.gain = self.__calculate_gain(epoch_idx)
        print(f"Set gain: {self.gain} (epoch: {epoch_idx})")

    def __calculate_gain(self, epoch: int) -> float:
        if epoch <= self.epoch_index_from:
            return self.min_gain
        elif epoch >= self.epoch_index_to:
            return self.max_gain
        else:
            # Calculate the gain linearly between min_gain and max_gain
            progress = (epoch - self.epoch_index_from) / (
                self.epoch_index_to - self.epoch_index_from
            )
            return self.min_gain + (self.max_gain - self.min_gain) * progress
