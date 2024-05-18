from dataclasses import dataclass


@dataclass
class GainController:
    """
    epochの値によって、min_gainからmax_gainの間で線形に変化させる
    """

    epoch_from: int = 0
    epoch_to: int = 0
    min_gain: float = 0.0
    max_gain: float = 1.0
    gain: float = 1.0

    def reset(self):
        self.gain = 1.0

    def set_gain(self, gain: float):
        self.gain = gain

    def set_gain_from_epoch(self, epoch: int):
        self.gain = self.__calculate_gain(epoch)
        print(f"Set gain: {self.gain}")

    def __calculate_gain(self, epoch: int) -> float:
        if epoch <= self.epoch_from:
            return self.min_gain
        elif epoch >= self.epoch_to:
            return self.max_gain
        else:
            # Calculate the gain linearly between min_gain and max_gain
            progress = (epoch - self.epoch_from) / (self.epoch_to - self.epoch_from)
            return self.min_gain + (self.max_gain - self.min_gain) * progress
