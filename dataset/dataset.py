from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.gain_controller import GainController

from .randomizer import Randomizer
from .sampling_rate_converter import SamplingRateConverter


@dataclass
class NoisyHeartbeatDataset(Dataset):
    clean_data: np.ndarray
    noisy_data: np.ndarray  # noisyデータ。データの都合上、ノイズ単体ではない
    sampling_rate_converter: SamplingRateConverter
    randomizer: Randomizer | None = (
        None  # 指定した場合、getitemのときにnoisy_dataをランダマイズ
    )
    train: bool = True  # FashionMNISTなどのデータセットを参考にしたプロパティ
    train_split_ratio: float = 0.6
    split_samples: int = 5120
    stride_samples: int = 32
    gain_controller: GainController | None = None

    def __post_init__(self):
        self.clean_data, self.noisy_data = self.__partition_data(
            self.__preprocess(self.clean_data),
            self.__preprocess(self.noisy_data),
        )

    @property
    def sample_rate(self):
        return self.sampling_rate_converter.output_rate

    def __preprocess(self, data):
        return self.sampling_rate_converter.convert(data)

    def __partition_data(self, clean_data: np.ndarray, noisy_data: np.ndarray):
        total_samples = min(len(clean_data), len(noisy_data))

        # 短い方に合わせる
        clean_data = clean_data[:total_samples]
        noisy_data = noisy_data[:total_samples]

        using_data_samples_length = int(self.train_split_ratio * total_samples)

        if self.train:
            clean_data = clean_data[:using_data_samples_length]
            noisy_data = noisy_data[:using_data_samples_length]
        else:
            clean_data = clean_data[using_data_samples_length:]
            noisy_data = noisy_data[using_data_samples_length:]

        return clean_data, noisy_data

    def __len__(self):
        return (len(self.clean_data) - self.split_samples) // self.stride_samples + 1

    def __getitem__(self, idx: int):
        start, end = self._get_start_end(idx)

        clean = self.clean_data[start:end]
        noise = self._randomize(self.noisy_data[start:end])

        gain = self.gain_controller.get_gain() if self.gain_controller else 1.0

        return (
            self._to_tensor(clean + noise * gain).unsqueeze(0),
            self._to_tensor(clean).unsqueeze(0),
        )

    def _get_start_end(self, idx: int):
        start = idx * self.stride_samples
        end = start + self.split_samples
        return start, end

    def _to_tensor(self, data):
        return torch.tensor(data, dtype=torch.float32)

    def _randomize(self, data):
        if self.randomizer is None:
            return data
        return self.randomizer.shuffle(data)
