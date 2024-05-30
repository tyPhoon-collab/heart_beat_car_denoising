from logging import warning
import os
from matplotlib import pyplot as plt
import neptune
from logger.training_logger import TrainingLogger


class NeptuneLogger(TrainingLogger):
    def __init__(self) -> None:
        super().__init__()
        self.project_name = os.getenv("NEPTUNE_PROJECT_NAME")
        self.api_token = os.getenv("NEPTUNE_API_TOKEN")

        if self.project_name is None or self.api_token is None:
            warning(
                "NEPTUNE_PROJECT_NAME or NEPTUNE_API_TOKEN environment variable is not set."
                "neptune logger will be ignored."
            )
            self.enabled = False
        else:
            self.enabled = True

    def on_start(self, params: dict):
        if not self.enabled:
            return

        self.run = neptune.init_run(
            project=self.project_name,
            api_token=self.api_token,
        )
        self.run["model/parameters"] = params

    def on_batch_end(self, batch_idx, loss):
        if not self.enabled:
            return

        self.run[f"train/batch_{batch_idx}/loss"].log(loss)

    def on_epoch_end(self, epoch_idx, epoch_loss):
        if not self.enabled:
            return

        self.run["train/epoch_loss"].log(epoch_loss)
        self.run["train/validation_image"].append(plt.gcf())

    def on_finish(self):
        if not self.enabled:
            return

        self.run.stop()
