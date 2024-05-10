from abc import ABC, abstractmethod


class TrainingLogger(ABC):
    @abstractmethod
    def on_start(self):
        """トレーニング開始時に呼ばれます。"""
        pass

    @abstractmethod
    def on_batch_end(self, batch_idx, loss):
        """各バッチの処理が終わった後に呼ばれます。"""
        pass

    @abstractmethod
    def on_epoch_end(self, epoch_idx, epoch_loss):
        """各エポックの終了時に呼ばれます。"""
        pass

    @abstractmethod
    def on_finish(self):
        """トレーニングが正常に終了した時に呼ばれます。"""
        pass
