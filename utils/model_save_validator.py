from abc import ABC, abstractmethod
from dataclasses import dataclass
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


@dataclass
class AnyCompositeModelSaveValidator(ModelSaveValidator, EpochSensitive):
    validators: list[ModelSaveValidator]

    @property
    def suffix(self) -> str:
        return self._suffix

    def validate(self, loss_item: float) -> bool:
        for v in self.validators:
            if v.validate(loss_item):
                self._suffix = v.suffix
                return True

        return False

    def on_start_epoch(self, epoch_idx):
        for v in self.validators:
            if isinstance(v, EpochSensitive):
                v.on_start_epoch(epoch_idx)


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


@dataclass
class SpecificEpochModelSaveValidator(ModelSaveValidator, EpochSensitive):
    epoch_index: int
    suffix_label: str

    @classmethod
    def last(cls, epoch_size: int):
        return SpecificEpochModelSaveValidator(
            epoch_index=epoch_size - 1,
            suffix_label="last",
        )

    @property
    def suffix(self) -> str:
        return self.suffix_label

    def validate(self, loss_item: float) -> bool:
        if not self.enable:
            return False

        return True

    def on_start_epoch(self, epoch_idx):
        self.enable = epoch_idx == self.epoch_index
