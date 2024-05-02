from dataclasses import dataclass
import pandas as pd
import scipy.io
from abc import ABC, abstractmethod

from visualize import plot_signal


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


# test
if __name__ == "__main__":

    def show(file_path):
        loader = MatLoader(
            file_path,
            ["Time", "ECG", "ch1z", "ch2z", "ch3z", "ch4z", "ch5z", "ch6z"],
        )
        data = loader.load()
        print(data)
        plot_signal(data["ch1z"], "ch1z")

    show("data/Stop.mat")
    show("data/100km.mat")
