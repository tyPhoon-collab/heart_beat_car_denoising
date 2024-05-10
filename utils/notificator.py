from logging import warn
import os
import sys
import traceback
import requests


def send_discord_notification(message):
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

    if webhook_url is None:
        warn("Discord webhook URL is not set. Skipping notification.")
        return

    data = {"content": message}
    response = requests.post(webhook_url, json=data)
    if response.status_code != 204:
        warn(f"Failed to send notification, status code: {response.status_code}")


def send_discord_notification_on_error():
    def custom_excepthook(exc_type, exc_value, exc_traceback):
        # 例外メッセージとスタックトレースを取得
        error_message = str(exc_value)
        stack_trace = "".join(
            traceback.format_exception(exc_type, exc_value, exc_traceback)
        )
        notification_message = f"Unhandled exception occurred: {error_message}\nStack Trace:\n{stack_trace}"
        send_discord_notification(notification_message)
        # 元のsys.__excepthook__を呼び出して、標準エラーにも出力
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    # 例外フックの設定
    sys.excepthook = custom_excepthook


if __name__ == "__main__":
    send_discord_notification("Hello, World!")
