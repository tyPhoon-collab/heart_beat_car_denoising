from abc import ABC, abstractmethod
from math import inf

from utils.epoch_sensitive import EpochSensitive


class ModelSaveValidator(ABC):
    @abstractmethod
    def validate(self, loss_item: float) -> bool:
        pass

    @property
    @abstractmethod
    def suffix(self) -> str:
        pass


class BestModelSaveValidator(ModelSaveValidator, EpochSensitive):
    def __init__(self, epoch_index_from: int = 0) -> None:
        super().__init__()
        self.lowest_loss = inf
        self.epoch_index_from = epoch_index_from
        self.enable = True

    @property
    def suffix(self) -> str:
        return "best"

    def validate(self, loss_item: float) -> bool:
        if not self.enable:
            return False

        if loss_item < self.lowest_loss:
            self.lowest_loss = loss_item
            return True

        return False

    def on_start_epoch(self, epoch_idx):
        if epoch_idx == 0:
            self.lowest_loss = inf

        self.enable = epoch_idx >= self.epoch_index_from
