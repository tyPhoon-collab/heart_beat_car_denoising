from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset

from loader import Loader, MatLoader
from utils.visualize import plot_two_signals
from randomizer import NumpyRandomShuffleRandomizer, Randomizer
from sampling_rate_converter import SamplingRateConverter, ScipySamplingRateConverter


@dataclass
class NoisyHeartbeatDataset(Dataset):
    clean_file_path: str
    noisy_file_path: str
    sampling_rate_converter: SamplingRateConverter
    randomizer: Randomizer
    train: bool = True  # デフォルトを True に設定
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

        ratio = self.train_split_ratio

        if self.train:
            clean_data = clean_data[: int(ratio * total_samples)]
            noisy_data = noisy_data[: int(ratio * total_samples)]
        else:
            clean_data = clean_data[int(ratio * total_samples) :]
            noisy_data = noisy_data[int(ratio * total_samples) :]

        return clean_data, noisy_data

    def __len__(self):
        return len(self.clean_data) - self.split_sample_points

    def __getitem__(self, idx):
        start, end = idx, idx + self.split_sample_points
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


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train_dataset = NoisyHeartbeatDataset(
        clean_file_path="data/Stop.mat",
        noisy_file_path="data/100km.mat",
        # noisy_file_path="data/Stop.mat",
        sampling_rate_converter=ScipySamplingRateConverter(
            input_rate=32000,
            output_rate=1000,
        ),
        randomizer=NumpyRandomShuffleRandomizer(),
    )
    test_dataset = NoisyHeartbeatDataset(
        clean_file_path="data/Stop.mat",
        noisy_file_path="data/100km.mat",
        # noisy_file_path="data/Stop.mat",
        sampling_rate_converter=ScipySamplingRateConverter(
            input_rate=32000,
            output_rate=1000,
        ),
        randomizer=NumpyRandomShuffleRandomizer(),
        train=False,
    )

    print(len(train_dataset))
    print(len(test_dataset))

    dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
    )

    noisy, clean = next(iter(dataloader))

    plot_two_signals(noisy[0][0], clean[0][0], "Noisy", "Clean")
