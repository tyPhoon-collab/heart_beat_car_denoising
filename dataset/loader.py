from dataclasses import dataclass, field
import pandas as pd
import scipy.io
from abc import ABC, abstractmethod


class Loader(ABC):
    @abstractmethod
    def load(self) -> pd.DataFrame:
        pass


@dataclass
class MatLoader(Loader):
    """
    .mat to pandas DataFrame. Simple converter.
    """

    file_path: str
    columns: list[str]
    data_key: str = "data"

    def load(self):
        # .matファイルを読み込む
        mat = scipy.io.loadmat(self.file_path)
        data = mat.get(self.data_key)

        if data is None:
            raise ValueError(
                f"The key '{self.data_key}' does not exist in the loaded .mat file"
            )

        # データフレームを作成
        df = pd.DataFrame(data)

        # カラム名を検証し、設定する
        if len(df.columns) != len(self.columns):
            raise ValueError(
                "The number of columns in the .mat file does not match the required columns"
            )

        # カラム名を設定する
        df.columns = self.columns

        return df


_cached_dict = dict()


@dataclass
class GlobalCacheableLoader(Loader):
    loader: Loader
    logging: bool = True

    def __post_init__(self):
        assert not isinstance(self.loader, GlobalCacheableLoader)

    def load(self):
        key = self.loader_key

        cache_data = _cached_dict.get(key)

        if cache_data is None:
            cache_data = self.loader.load()
            _cached_dict[key] = cache_data
        elif self.logging:
            print(f"Loaded from cache: {key}")

        return cache_data

    @property
    def loader_key(self):
        if isinstance(self.loader, MatLoader):
            return (
                f"{self.loader.file_path}:{self.loader.data_key},{self.loader.columns}"
            )
        else:
            return self.loader.__class__.__name__


def cacheable(func):
    def wrapper(*args, **kwargs):
        loader = func(*args, **kwargs)
        cacheable_loader = GlobalCacheableLoader(loader)
        print(
            f"Converted: {loader.__class__.__name__} to {cacheable_loader.__class__.__name__}"
        )
        return cacheable_loader

    return wrapper


if __name__ == "__main__":
    loader = MatLoader(
        "data/240517_Rawdata/HS_data.mat",
        # ["Time", "ECG", "ch1z", "ch2z", "ch3z", "ch4z", "ch5z", "ch6z"],
        ["ch1z"] * 36,
        data_key="HS_data",
    )
    data = loader.load()
    print(data)
