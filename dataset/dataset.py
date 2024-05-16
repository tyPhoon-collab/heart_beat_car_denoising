from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset

from .loader import Loader, MatLoader
from .randomizer import Randomizer
from .sampling_rate_converter import SamplingRateConverter


@dataclass
class NoisyHeartbeatDataset(Dataset):
    clean_file_path: str
    noisy_file_path: str
    sampling_rate_converter: SamplingRateConverter
    randomizer: Randomizer
    train: bool = True  # FashionMNISTなどのデータセットを参考にしたプロパティ
    train_split_ratio: float = 0.6
    split_duration_second: float = 5.0

    def __post_init__(self):
        self.split_sample_points = int(
            self.sampling_rate_converter.output_rate * self.split_duration_second
        )
        self.clean_data, self.noisy_data = self.__partition_data(
            self.__load_and_preprocess(self.clean_file_path),
            self.__load_and_preprocess(self.noisy_file_path),
        )

    def __load_and_preprocess(self, file_path: str):
        columns = ["Time", "ECG", "ch1z", "ch2z", "ch3z", "ch4z", "ch5z", "ch6z"]

        loader: Loader = MatLoader(file_path, columns)
        data = loader.load()["ch1z"]
        return self.__preprocess(data)

    def __preprocess(self, data):
        return self.sampling_rate_converter.convert(data)

    def __partition_data(self, clean_data: np.ndarray, noisy_data: np.ndarray):
        total_samples = min(len(clean_data), len(noisy_data))

        # 短い方に合わせる
        clean_data = clean_data[:total_samples]
        noisy_data = noisy_data[:total_samples]

        split_samples = int(self.train_split_ratio * total_samples)

        if self.train:
            clean_data = clean_data[:split_samples]
            noisy_data = noisy_data[:split_samples]
        else:
            clean_data = clean_data[split_samples:]
            noisy_data = noisy_data[split_samples:]

        return clean_data, noisy_data

    def __len__(self):
        return len(self.clean_data) - self.split_sample_points

    def __getitem__(self, idx: int):
        start = idx
        end = idx + self.split_sample_points

        clean = self.clean_data[start:end]
        randomized_noisy = self.__randomize(self.noisy_data[start:end])
        return (
            self.__to_tensor(clean + randomized_noisy).unsqueeze(0),
            self.__to_tensor(clean).unsqueeze(0),
        )

    def __to_tensor(self, data):
        return torch.tensor(data, dtype=torch.float32)

    def __randomize(self, data):
        return self.randomizer.shuffle(data)
