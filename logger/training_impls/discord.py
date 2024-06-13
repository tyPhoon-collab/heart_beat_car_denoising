from logging import warning
import os
import sys
import traceback

import requests
from logger.training_logger import TrainingLogger


class DiscordLogger(TrainingLogger):
    def __init__(self) -> None:
        super().__init__()

        self.webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
        if self.webhook_url is None:
            warning("Discord webhook URL is not set. Skipping notification.")

    def on_start(self, params: dict):
        self.send_discord_notification_on_error()
        self.send_discord_notification("Training started.")

    def on_batch_end(self, batch_idx, loss):
        pass  # Typically, we might not want to send batch-level updates to Discord.

    def on_epoch_end(self, epoch_idx, epoch_loss):
        pass  # Optionally send updates per epoch if needed.

    def on_finish(self):
        self.send_discord_notification("Training finished successfully.")

    def on_model_saved(self, path: str):
        pass

    def send_discord_notification(self, message):
        if self.webhook_url is None:
            return

        data = {"content": message}
        response = requests.post(self.webhook_url, json=data)
        if response.status_code != 204:
            warning(f"Failed to send notification, status code: {response.status_code}")

    def send_discord_notification_on_error(self):
        def custom_excepthook(exc_type, exc_value, exc_traceback):
            # 例外メッセージとスタックトレースを取得
            error_message = str(exc_value)
            stack_trace = "".join(
                traceback.format_exception(exc_type, exc_value, exc_traceback)
            )
            notification_message = f"Unhandled exception occurred: {error_message}\nStack Trace:\n{stack_trace}"
            self.send_discord_notification(notification_message)
            # 元のsys.__excepthook__を呼び出して、標準エラーにも出力
            sys.__excepthook__(exc_type, exc_value, exc_traceback)

        # 例外フックの設定
        sys.excepthook = custom_excepthook
