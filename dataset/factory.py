from torch.utils.data import DataLoader
from dataset.dataset import NoisyHeartbeatDataset
from dataset.randomizer import NumpyRandomShuffleRandomizer
from dataset.sampling_rate_converter import ScipySamplingRateConverter


class DatasetFactory:
    @classmethod
    def build(
        cls,
        clean_file_path="data/Stop.mat",
        noisy_file_path="data/100km.mat",
        input_rate=32000,
        output_rate=1024,
        randomizer=None,
        train=True,
    ):
        return NoisyHeartbeatDataset(
            clean_file_path=clean_file_path,
            noisy_file_path=noisy_file_path,
            sampling_rate_converter=ScipySamplingRateConverter(
                input_rate=input_rate,
                output_rate=output_rate,
            ),
            randomizer=randomizer or NumpyRandomShuffleRandomizer(),
            train=train,
        )

    @classmethod
    def train(cls, dataset):
        return cls.build(dataset, train=True)

    @classmethod
    def test(cls, dataset):
        return cls.build(dataset, train=False)


class DataLoaderFactory:
    @classmethod
    def build(cls, dataset, batch_size=1, shuffle=True):
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    @classmethod
    def train(cls, dataset, batch_size=1):
        return cls.build(dataset, batch_size, True)

    @classmethod
    def test(cls, dataset, batch_size=1):
        return cls.build(dataset, batch_size, False)
