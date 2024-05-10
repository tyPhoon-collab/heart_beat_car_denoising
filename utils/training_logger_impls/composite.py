from attr import dataclass
from utils.training_logger import TrainingLogger


@dataclass
class CompositeLogger(TrainingLogger):
    loggers: list[TrainingLogger] = []

    def add_logger(self, logger):
        self.loggers.append(logger)

    def remove_logger(self, logger):
        self.loggers.remove(logger)

    def on_start(self):
        for logger in self.loggers:
            logger.on_start()

    def on_batch_end(self, batch_idx, loss):
        for logger in self.loggers:
            logger.on_batch_end(batch_idx, loss)

    def on_epoch_end(self, epoch_idx, epoch_loss):
        for logger in self.loggers:
            logger.on_epoch_end(epoch_idx, epoch_loss)

    def on_finish(self):
        for logger in self.loggers:
            logger.on_finish()
