from dataclasses import dataclass
from logger.evaluation_logger import EvaluationLogger
from numpy.typing import ArrayLike

from utils.plot import plot_three_signals


@dataclass
class FigureEvaluationLogger(EvaluationLogger):
    filename: str

    def on_data(self, noisy: ArrayLike, clean: ArrayLike, output: ArrayLike):
        plot_three_signals(
            noisy,
            clean,
            output,
            "Noisy",
            "Clean",
            "Output",
            filename=self.filename,
        )