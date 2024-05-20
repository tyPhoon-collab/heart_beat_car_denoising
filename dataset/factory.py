from torch.utils.data import DataLoader, Dataset
from dataset.dataset import NoisyHeartbeatDataset
from dataset.randomizer import SampleShuffleRandomizer
from dataset.sampling_rate_converter import ScipySamplingRateConverter


class DatasetFactory:
    @classmethod
    def create(
        cls,
        clean_file_path="data/Stop.mat",
        noisy_file_path="data/100km.mat",
        input_rate=32000,
        output_rate=1024,
        randomizer=None,
        gain_controller=None,
        train=True,
    ):
        return NoisyHeartbeatDataset(
            clean_file_path=clean_file_path,
            noisy_file_path=noisy_file_path,
            sampling_rate_converter=ScipySamplingRateConverter(
                input_rate=input_rate,
                output_rate=output_rate,
            ),
            randomizer=randomizer or SampleShuffleRandomizer(),
            gain_controller=gain_controller,
            train=train,
        )

    @classmethod
    def create_train(cls, *, randomizer=None, gain_controller=None):
        return cls.create(
            train=True,
            randomizer=randomizer,
            gain_controller=gain_controller,
        )

    @classmethod
    def create_test(cls, *, randomizer=None):
        return cls.create(train=False, randomizer=randomizer)


class DataLoaderFactory:
    @classmethod
    def create(cls, dataset: Dataset, batch_size=1, shuffle=True):
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    @classmethod
    def create_train(cls, dataset: Dataset, batch_size=1, shuffle=True):
        return cls.create(dataset, batch_size, shuffle)

    @classmethod
    def create_test(cls, dataset: Dataset, batch_size=1):
        return cls.create(dataset, batch_size, False)
