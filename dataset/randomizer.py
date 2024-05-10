from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import ArrayLike


class Randomizer(ABC):
    @abstractmethod
    def shuffle(self, array: ArrayLike) -> np.ndarray:
        pass


class NumpyRandomShuffleRandomizer(Randomizer):
    def shuffle(self, array: ArrayLike) -> np.ndarray:
        np.random.shuffle(array)
        return array  # type: ignore
