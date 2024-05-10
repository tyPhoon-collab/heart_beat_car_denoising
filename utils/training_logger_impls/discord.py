from utils.notificator import (
    send_discord_notification,
    send_discord_notification_on_error,
)
from utils.training_logger import TrainingLogger


class DiscordLogger(TrainingLogger):
    def on_start(self):
        send_discord_notification_on_error()
        send_discord_notification("Training started.")

    def on_batch_end(self, batch_idx, loss):
        pass  # Typically, we might not want to send batch-level updates to Discord.

    def on_epoch_end(self, epoch_idx, epoch_loss):
        pass  # Optionally send updates per epoch if needed.

    def on_finish(self):
        send_discord_notification("Training finished successfully.")
