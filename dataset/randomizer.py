from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import ArrayLike
from scipy.fft import fft, ifft


class Randomizer(ABC):
    @abstractmethod
    def shuffle(self, array: ArrayLike) -> np.ndarray:
        pass


class NumpyRandomShuffleRandomizer(Randomizer):
    def shuffle(self, array: ArrayLike) -> np.ndarray:
        np.random.shuffle(array)
        return array  # type: ignore


@dataclass
class PhaseShuffleRandomizer(Randomizer):
    fft_randomizer: Randomizer = field(
        default_factory=lambda: NumpyRandomShuffleRandomizer()
    )

    def shuffle(self, array: ArrayLike) -> np.ndarray:
        array_fft = fft(array)
        magnitude = np.abs(array_fft)
        random_phase = np.exp(1j * self.fft_randomizer.shuffle(np.angle(array_fft)))
        return ifft(magnitude * random_phase).real
