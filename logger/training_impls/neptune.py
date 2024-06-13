import os
from logging import warning
from matplotlib import pyplot as plt
import neptune
from logger.training_logger import TrainingLogger
from functools import wraps


def _check_enable(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.enable:
            return
        return func(self, *args, **kwargs)

    return wrapper


class NeptuneLogger(TrainingLogger):
    def __init__(self) -> None:
        super().__init__()
        self._initialize_neptune_settings()

    def _initialize_neptune_settings(self):
        self.project_name = os.getenv("NEPTUNE_PROJECT_NAME")
        self.api_token = os.getenv("NEPTUNE_API_TOKEN")

        self.enable = self._check_neptune_configuration()
        self.enable_saving_model_state = os.getenv("NEPTUNE_SAVE_MODEL_STATE") == "1"

    def _check_neptune_configuration(self):
        if not self.project_name or not self.api_token:
            warning(
                "NEPTUNE_PROJECT_NAME or NEPTUNE_API_TOKEN environment variable is not set. "
                "Neptune logger will be ignored."
            )
            return False
        return True

    @_check_enable
    def on_start(self, params: dict):
        self.run = neptune.init_run(project=self.project_name, api_token=self.api_token)
        self.run["model/parameters"] = params

    @_check_enable
    def on_batch_end(self, batch_idx: int, loss: float):
        self.run[f"train/batch_{batch_idx}/loss"].log(loss)

    @_check_enable
    def on_epoch_end(self, epoch_idx: int, epoch_loss: float):
        self.run["train/epoch_loss"].log(epoch_loss)
        self.run["train/validation_image"].append(plt.gcf())

    @_check_enable
    def on_finish(self):
        self.run.stop()

    @_check_enable
    def on_model_saved(self, path: str):
        if self.enable_saving_model_state:
            self.run["model/weights"].upload(path)
