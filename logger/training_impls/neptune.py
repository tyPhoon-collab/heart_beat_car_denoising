from dataclasses import dataclass
import os
from matplotlib import pyplot as plt
import neptune
from logger.training_logger import TrainingLogger


@dataclass()
class NeptuneLogger(TrainingLogger):
    project_name: str
    api_token: str
    save_model_state: bool

    def on_start(self, params: dict):
        self.run = neptune.init_run(project=self.project_name, api_token=self.api_token)
        self.run["model/parameters"] = params

    def on_batch_end(self, batch_idx: int, loss: float):
        self.run[f"train/batch_{batch_idx}/loss"].log(loss)

    def on_epoch_end(self, epoch_idx: int, epoch_loss: float):
        self.run["train/epoch_loss"].log(epoch_loss)
        self.run["train/validation_image"].append(plt.gcf())

    def on_finish(self):
        self.run.stop()

    def on_model_saved(self, path: str):
        if self.save_model_state:
            self.run[f"model/{os.path.basename(path)}"].upload(path)
