from logging import warning
from dotenv import load_dotenv
import torch


def get_torch_device() -> torch.device:
    # GPUが利用可能かどうかをチェック
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU is not available, using CPU instead.")

    return device


def load_local_dotenv():
    ret = load_dotenv(override=True)
    if not ret:
        warning("Could not load .env file.")
