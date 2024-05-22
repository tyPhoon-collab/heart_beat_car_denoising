from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import ArrayLike
from scipy.fft import fft, ifft


class Randomizer(ABC):
    @abstractmethod
    def shuffle(self, array: ArrayLike) -> np.ndarray:
        pass


class SampleShuffleRandomizer(Randomizer):
    def shuffle(self, array: ArrayLike) -> np.ndarray:
        return np.random.permutation(array)


@dataclass
class PhaseShuffleRandomizer(Randomizer):
    angle_randomizer: Randomizer = field(
        default_factory=lambda: SampleShuffleRandomizer()
    )

    def shuffle(self, array: ArrayLike) -> np.ndarray:
        array_fft = fft(array)
        magnitude = np.abs(array_fft)  # type: ignore
        random_phase = np.exp(1j * self.angle_randomizer.shuffle(np.angle(array_fft)))  # type: ignore
        return ifft(magnitude * random_phase).real  # type: ignore
