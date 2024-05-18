from logging import warning
import os
from logger.impls.composite import CompositeLogger
from logger.impls.discord import DiscordLogger
from logger.impls.neptune import NeptuneLogger
from logger.impls.noop import NoopLogger
from logger.training_logger import TrainingLogger


class TrainingLoggerFactory:
    @classmethod
    def noop(cls) -> TrainingLogger:
        return NoopLogger()

    @classmethod
    def remote(cls) -> TrainingLogger:
        if not cls.__is_enable_env():
            return cls.noop()

        return CompositeLogger([NeptuneLogger(), DiscordLogger()])

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
