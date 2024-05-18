from logging import warning
import os
from logger.training_impls.composite import CompositeTrainingLogger
from logger.training_impls.discord import DiscordLogger
from logger.training_impls.neptune import NeptuneLogger
from logger.training_impls.noop import NoopTrainingLogger
from logger.training_logger import TrainingLogger


class TrainingLoggerFactory:
    @classmethod
    def noop(cls) -> TrainingLogger:
        return NoopTrainingLogger()

    @classmethod
    def remote(cls) -> TrainingLogger:
        if not cls.__is_enable_env():
            return cls.noop()

        return CompositeTrainingLogger([NeptuneLogger(), DiscordLogger()])

    @classmethod
    def __is_enable_env(cls) -> bool:
        logging_env_value = os.getenv("REMOTE_LOGGING")
        is_enable = logging_env_value == "1"
        if not is_enable:
            warning(
                "Logging is disabled. If you want to enable logging, set REMOTE_LOGGING=1 in .env."
                f"REMOTE_LOGGING is currently set to {logging_env_value}."
            )
        return is_enable
