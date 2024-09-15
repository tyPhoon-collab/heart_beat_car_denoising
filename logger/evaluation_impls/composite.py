from attr import dataclass
from numpy.typing import ArrayLike
from logger.evaluation_logger import EvaluationLogger


@dataclass
class CompositeEvaluationLogger(EvaluationLogger):
    loggers: list[EvaluationLogger] = []

    def add_logger(self, logger):
        self.loggers.append(logger)

    def remove_logger(self, logger):
        self.loggers.remove(logger)

    def on_data(self, noisy: ArrayLike, clean: ArrayLike, output: ArrayLike):
        for logger in self.loggers:
            logger.on_data(noisy, clean, output)

    def on_average_loss(self, loss: float):
        for logger in self.loggers:
            logger.on_average_loss(loss)
