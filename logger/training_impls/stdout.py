import json
from logger.training_logger import TrainingLogger


class StdoutTrainingLogger(TrainingLogger):
    def on_start(self, params: dict):
        print(json.dumps(params, indent=4))
        print("Start training")

    def on_batch_end(self, batch_idx, loss):
        # print(f"Batch {batch_idx} loss: {loss}")
        pass

    def on_epoch_end(self, epoch_idx, epoch_loss):
        print(f"Epoch {epoch_idx + 1} loss: {epoch_loss.item():.5f}")

    def on_finish(self):
        print("Finish training")

    def on_model_saved(self, path: str):
        print(f"Model saved to {path}")
