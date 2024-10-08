from abc import ABC, abstractmethod
from numpy.typing import ArrayLike


class EvaluationLogger(ABC):
    @abstractmethod
    def on_data(self, noisy: ArrayLike, clean: ArrayLike, output: ArrayLike):
        pass

    def on_average_loss(self, loss: float):
        pass
