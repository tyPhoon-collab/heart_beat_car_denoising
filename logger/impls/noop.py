from logger.training_logger import Params, TrainingLogger


class NoopLogger(TrainingLogger):
    def on_start(self, params: Params):
        pass

    def on_batch_end(self, batch_idx, loss):
        pass

    def on_epoch_end(self, epoch_idx, epoch_loss):
        pass

    def on_finish(self):
        pass
