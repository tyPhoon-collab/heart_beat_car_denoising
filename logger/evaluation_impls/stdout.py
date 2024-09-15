from logger.evaluation_logger import EvaluationLogger
from numpy.typing import ArrayLike


class StdoutEvaluationLogger(EvaluationLogger):
    def on_data(self, noisy: ArrayLike, clean: ArrayLike, output: ArrayLike):
        pass

    def on_average_loss(self, loss: float):
        print("Average loss:", loss)
