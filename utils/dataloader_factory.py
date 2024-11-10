from config import Config
from torch.utils.data import DataLoader

from dataset.factory import DatasetFactory
from dataset.randomizer import Randomizer
from utils.gain_controller import GainController


class DataLoaderFactory:
    @classmethod
    def config(
        cls, c: Config, randomizer: Randomizer | None, gain_controller: GainController
    ) -> tuple[DataLoader, DataLoader]:
        train_dataloader = cls._internal_dataloader(
            c,
            train=True,
            randomizer=randomizer,
            gain_controller=gain_controller,
        )
        val_dataloader = cls._internal_dataloader(
            c,
            train=False,
            randomizer=randomizer,
            gain_controller=gain_controller,
        )

        return train_dataloader, val_dataloader

    @classmethod
    def _internal_dataloader(
        cls,
        c: Config,
        train: bool,
        randomizer: Randomizer | None,
        gain_controller: GainController,
        base_dir: str = "",
    ) -> DataLoader:
        kwargs = {
            "base_dir": base_dir,
            "train": train,
            "split_samples": c.split,
            "stride_samples": c.stride,
            "randomizer": randomizer,
            "gain_controller": gain_controller,
        }

        dataset = cls._internal_dataset(c, kwargs)

        dataloader = DataLoader(
            dataset,
            batch_size=c.batch,
            shuffle=train,
        )

        return dataloader

    @classmethod
    def _internal_dataset(cls, c, kwargs):
        match c.data:
            case "Raw240219":
                return DatasetFactory.create_240219(**kwargs)
            case "Raw240517":
                return DatasetFactory.create_240517_filtered(**kwargs)
            case "Raw240826":
                return DatasetFactory.create_240826_filtered(**kwargs)

        raise ValueError(f"Invalid data folder: {c.data}")
