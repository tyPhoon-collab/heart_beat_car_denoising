from logger.training_logger import TrainingLogger


class NoopTrainingLogger(TrainingLogger):
    def on_start(self, params: dict):
        pass

    def on_batch_end(self, batch_idx, loss):
        pass

    def on_epoch_end(self, epoch_idx, epoch_loss):
        pass

    def on_finish(self):
        pass

    def on_model_saved(self, path: str):
        pass
