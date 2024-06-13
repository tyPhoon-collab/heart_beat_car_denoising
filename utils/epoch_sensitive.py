from abc import ABC, abstractmethod


class EpochSensitive(ABC):
    @abstractmethod
    def on_start_epoch(self, epoch_idx):
        pass
