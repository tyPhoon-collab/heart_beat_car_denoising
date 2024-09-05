from dataclasses import dataclass
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
    data_key: str
    columns: list[str] | None = None

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
        if self.columns is not None:
            if len(df.columns) != len(self.columns):
                raise ValueError(
                    "The number of columns in the .mat file does not match the required columns"
                )

            # カラム名を設定する
            df.columns = self.columns

        return df


@dataclass
class ConcatColumnsLoader(Loader):
    loader: Loader
    column_name: str = "ch1z"

    def load(self):
        df = self.loader.load()
        combined_df = pd.DataFrame({self.column_name: df.values.flatten()})

        return combined_df


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
        "data/240826_Rawdata/Noise_data_100km.mat",
        "data",
    )
    data = loader.load()
    print(data)
