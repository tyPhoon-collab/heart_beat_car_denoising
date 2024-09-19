from abc import ABC, abstractmethod
from typing import Any


Loss = float


class Tune(ABC):
    @abstractmethod
    def get_param_space(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def train(self, config) -> Loss:
        pass
