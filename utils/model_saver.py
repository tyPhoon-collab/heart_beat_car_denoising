from abc import ABC, abstractmethod
import os
import torch
import torch.nn as nn
from datetime import datetime


class ModelSaver(ABC):
    @abstractmethod
    def save(self, model: nn.Module, *, suffix: str | None = None) -> str:
        pass


class SimpleModelSaver(ModelSaver):
    def __init__(self, base_directory: str):
        self.save_directory = base_directory

    def save(self, model: nn.Module, *, suffix: str | None = None):
        os.makedirs(self.save_directory, exist_ok=True)  # ディレクトリがなければ作成

        suffix = f"_{suffix}" if suffix else ""
        path = os.path.join(self.save_directory, f"model_weights{suffix}.pth")
        torch.save(model.state_dict(), path)

        return path


class WithDateModelSaver(ModelSaver):
    def __init__(self, base_directory: str):
        today = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.save_directory = os.path.join(base_directory, today)

    def save(self, model: nn.Module, *, suffix: str | None = None):
        os.makedirs(self.save_directory, exist_ok=True)  # ディレクトリがなければ作成

        suffix = f"_{suffix}" if suffix else ""
        path = os.path.join(self.save_directory, f"model_weights{suffix}.pth")
        torch.save(model.state_dict(), path)

        return path


class WithIdModelSaver(ModelSaver):
    def __init__(self, base_directory: str, id: str):
        self.save_directory = os.path.join(base_directory, id)

    def save(self, model: nn.Module, *, suffix: str | None = None):
        os.makedirs(self.save_directory, exist_ok=True)  # ディレクトリがなければ作成

        suffix = f"_{suffix}" if suffix else ""
        path = os.path.join(self.save_directory, f"model_weights{suffix}.pth")
        torch.save(model.state_dict(), path)

        return path
