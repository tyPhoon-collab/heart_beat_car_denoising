from config import EvalConfig, LoggingConfig
from logger.evaluation_impls.audio import AudioEvaluationLogger
from logger.evaluation_impls.composite import CompositeEvaluationLogger
from logger.evaluation_impls.figure import FigureEvaluationLogger
from logger.evaluation_impls.plotly import PlotlyEvaluationLogger
from logger.evaluation_impls.stdout import StdoutEvaluationLogger
from logger.evaluation_logger import EvaluationLogger


class EvaluationLoggerFactory:
    @classmethod
    def config(cls, logging: LoggingConfig, c: EvalConfig) -> EvaluationLogger:
        loggers = []

        if logging.stdout:
            loggers.append(StdoutEvaluationLogger())
        if c.figure_filename:
            loggers.append(FigureEvaluationLogger(filename=c.figure_filename))
        if c.audio_filename:
            loggers.append(
                AudioEvaluationLogger(
                    sample_rate=1000,
                    audio_filename=c.audio_filename,
                    clean_audio_filename=c.clean_audio_filename,
                    noisy_audio_filename=c.noisy_audio_filename,
                )
            )
        if c.html_filename:
            loggers.append(PlotlyEvaluationLogger(filename=c.html_filename))

        return CompositeEvaluationLogger(loggers)
