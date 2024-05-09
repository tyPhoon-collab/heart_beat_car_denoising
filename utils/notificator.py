import sys
import traceback
import requests

WEBHOOK_URL = "https://discord.com/api/webhooks/1238032800851034113/1WcDXN9edWCL_VNHeRVs7cCzH03k90EMUgggDpRSVYG2q5cIFQfZ5dAIEmdFCUCINwwb"  # noqa


def send_discord_notification(message):
    data = {"content": message}
    response = requests.post(WEBHOOK_URL, json=data)
    if response.status_code == 204:
        print("Notification sent to Discord successfully!")
    else:
        print(f"Failed to send notification, status code: {response.status_code}")


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
