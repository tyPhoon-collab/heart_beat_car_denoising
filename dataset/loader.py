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


@dataclass
class CacheableLoader(Loader):
    loader: Loader
    cache_data: pd.DataFrame | None = field(default=None, init=False)

    def load(self):
        if self.cache_data is None:
            self.cache_data = self.loader.load()
        return self.cache_data


if __name__ == "__main__":
    loader = MatLoader(
        "data/240517_Rawdata/HS_data.mat",
        # ["Time", "ECG", "ch1z", "ch2z", "ch3z", "ch4z", "ch5z", "ch6z"],
        ["ch1z"] * 36,
        data_key="HS_data",
    )
    data = loader.load()
    print(data)


def cacheable(func):
    def wrapper(*args, **kwargs):
        loader = func(*args, **kwargs)
        cacheable_loader = CacheableLoader(loader)
        print(
            f"Converted: {loader.__class__.__name__} to {cacheable_loader.__class__.__name__}"
        )
        return cacheable_loader

    return wrapper
