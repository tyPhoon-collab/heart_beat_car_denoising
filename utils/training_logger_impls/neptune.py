from logging import warn
import os
import neptune
from utils.training_logger import TrainingLogger


class NeptuneLogger(TrainingLogger):
    def on_start(self):
        project_name = os.getenv("NEPTUNE_PROJECT_NAME")
        api_token = os.getenv("NEPTUNE_API_TOKEN")

        self.run: neptune.Run | None = None

        if project_name is None or api_token is None:
            warn(
                "NEPTUNE_PROJECT_NAME or NEPTUNE_API_TOKEN environment variable is not set."
                "neptune logger will be ignored."
            )
        else:
            self.run = neptune.init_run(
                project=project_name,
                api_token=api_token,
            )

    def on_batch_end(self, batch_idx, loss):
        if self.run is None:
            return

        self.run[f"train/batch_{batch_idx}/loss"].log(loss)

    def on_epoch_end(self, epoch_idx, epoch_loss):
        if self.run is None:
            return

        self.run[f"train/epoch_{epoch_idx}/loss"].log(epoch_loss)

    def on_finish(self):
        if self.run is None:
            return

        self.run.stop()
