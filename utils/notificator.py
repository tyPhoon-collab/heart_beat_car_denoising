import requests

WEBHOOK_URL = "https://discord.com/api/webhooks/1238032800851034113/1WcDXN9edWCL_VNHeRVs7cCzH03k90EMUgggDpRSVYG2q5cIFQfZ5dAIEmdFCUCINwwb"  # noqa


def send_discord_notification(message):
    data = {"content": message}
    response = requests.post(WEBHOOK_URL, json=data)
    if response.status_code == 204:
        print("Notification sent to Discord successfully!")
    else:
        print(f"Failed to send notification, status code: {response.status_code}")


if __name__ == "__main__":
    send_discord_notification("Hello, World!")
