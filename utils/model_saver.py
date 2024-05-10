from abc import ABC, abstractmethod
import os
import torch
import torch.nn as nn
from datetime import datetime


class ModelSaver(ABC):
    @abstractmethod
    def save(self, model: nn.Module, epoch: int):
        pass


class WithDateModelSaver(ModelSaver):
    def __init__(self, base_directory: str):
        today = datetime.now().strftime(
            "%Y-%m-%d"
        )  # 今日の日付を YYYY-MM-DD 形式で取得
        self.save_directory = os.path.join(base_directory, today)
        os.makedirs(self.save_directory, exist_ok=True)  # ディレクトリがなければ作成

    def save(self, model: nn.Module, epoch: int):
        path = os.path.join(self.save_directory, f"model_weights_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), path)
        print(f"Model saved to {path}")
