from dataclasses import dataclass
import torch
from torch.utils.data import Dataset

from loader import Loader, MatLoader
from visualize import plot_two_signals
from randomizer import NumpyRandomShuffleRandomizer, Randomizer
from sampling_rate_converter import SamplingRateConverter, ScipySamplingRateConverter


@dataclass
class NoisyHeartbeatDataset(Dataset):
    clean_file_path: str
    noisy_file_path: str
    # loader_builder: TODO ローダーをカスタマイズできるようにする。現状はMatLoaderを使っている
    sampling_rate_converter: SamplingRateConverter
    randomizer: Randomizer
    split_duration_second: float = 5.0

    def __post_init__(self):
        columns = ["Time", "ECG", "ch1z", "ch2z", "ch3z", "ch4z", "ch5z", "ch6z"]
        self.clean_data = self.__load_and_preprocess(self.clean_file_path, columns)
        self.noisy_data = self.__load_and_preprocess(self.noisy_file_path, columns)
        self.split_sample_points = int(
            self.sampling_rate_converter.output_rate * self.split_duration_second
        )

    def __load_and_preprocess(self, file_path: str, columns):
        loader: Loader = MatLoader(file_path, columns)
        data = loader.load()["ch1z"]
        return self.__preprocess(data)

    def __preprocess(self, data):
        return self.sampling_rate_converter.convert(data)

    def __len__(self):
        return (
            min(len(self.clean_data), len(self.noisy_data)) - self.split_sample_points
        )

    def __getitem__(self, idx):
        start, end = idx, idx + self.split_sample_points

        clean = self.clean_data[start:end]
        randomized_noisy = self.__randomize(self.noisy_data[start:end])
        return (
            self.__to_tensor(clean + randomized_noisy),
            self.__to_tensor(clean),
        )

    def __to_tensor(self, data):
        return torch.tensor(data, dtype=torch.float32)

    def __randomize(self, data):
        return self.randomizer.shuffle(data)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = NoisyHeartbeatDataset(
        clean_file_path="data/Stop.mat",
        noisy_file_path="data/100km.mat",
        # noisy_file_path="data/Stop.mat",
        sampling_rate_converter=ScipySamplingRateConverter(
            input_rate=32000,
            output_rate=1000,
        ),
        randomizer=NumpyRandomShuffleRandomizer(),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
    )

    for noisy, clean in dataloader:
        plot_two_signals(noisy[0], clean[0], "Noisy", "Clean")
        break  # サンプルプロットのため、最初のバッチだけ処理する
