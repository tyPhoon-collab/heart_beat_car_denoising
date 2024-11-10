from logging import warning
import os
from logger.training_impls.composite import CompositeTrainingLogger
from logger.training_impls.discord import DiscordLogger
from logger.training_impls.neptune import NeptuneLogger
from logger.training_impls.noop import NoopTrainingLogger
from logger.training_impls.stdout import StdoutTrainingLogger
from logger.training_logger import TrainingLogger
from config import LoggingConfig, SecretConfig


class TrainingLoggerFactory:
    @classmethod
    def noop(cls) -> TrainingLogger:
        return NoopTrainingLogger()

    @classmethod
    def stdout(cls) -> TrainingLogger:
        return StdoutTrainingLogger()

    @classmethod
    def config(cls, config: LoggingConfig, secret: SecretConfig) -> TrainingLogger:
        r = config.remote

        loggers = []

        if config.stdout:
            loggers.append(StdoutTrainingLogger())
        if r and config.neptune:
            if not secret.neptune_project_name or not secret.neptune_api_token:
                raise ValueError(
                    "Neptune project name and API token must be provided in secret config."
                )

            loggers.append(
                NeptuneLogger(
                    secret.neptune_project_name,
                    secret.neptune_api_token,
                    config.neptune_save_model_state,
                )
            )
        if r and config.discord:
            if not secret.discord_webhook_url:
                raise ValueError(
                    "Discord webhook URL must be provided in secret config."
                )

            loggers.append(DiscordLogger(secret.discord_webhook_url))

        return CompositeTrainingLogger(loggers)

    @classmethod
    def tune(cls) -> TrainingLogger:
        return CompositeTrainingLogger(
            [
                StdoutTrainingLogger(),
                NoopTrainingLogger(),
            ]
        )

    @classmethod
    def __is_enable_remote_logging(cls) -> bool:
        logging_env_value = os.getenv("REMOTE_LOGGING")
        is_enable = logging_env_value == "1"
        if not is_enable:
            warning(
                "Logging is disabled. If you want to enable logging, set REMOTE_LOGGING=1 in .env."
                f"REMOTE_LOGGING is currently set to {logging_env_value}."
            )
        return is_enable
