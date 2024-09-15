from dataclasses import dataclass
from logger.evaluation_logger import EvaluationLogger
from numpy.typing import ArrayLike

from plot.plot_plotly import show_plotly_signals


@dataclass
class PlotlyEvaluationLogger(EvaluationLogger):
    filename: str

    def on_data(self, noisy: ArrayLike, clean: ArrayLike, output: ArrayLike):
        show_plotly_signals(
            [noisy, clean, output],
            ["Noisy", "Clean", "Output"],
            filename=self.filename,
        )

    def on_average_loss(self, loss: float):
        pass
