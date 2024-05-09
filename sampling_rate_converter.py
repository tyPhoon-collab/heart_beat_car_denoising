from abc import ABC, abstractmethod
from dataclasses import dataclass
from scipy.signal import resample
from numpy.typing import ArrayLike
from loader import MatLoader
from utils.visualize import plot_two_signals
import numpy as np


@dataclass
class SamplingRateConverter(ABC):
    input_rate: int
    output_rate: int

    @abstractmethod
    def convert(self, input_signal: ArrayLike) -> np.ndarray:
        pass


@dataclass
class ScipySamplingRateConverter(SamplingRateConverter):
    def convert(self, input_signal: ArrayLike) -> np.ndarray:
        input_length = len(input_signal)  # type: ignore # 入力信号の長さを取得
        output_length = int(
            input_length * self.output_rate / self.input_rate
        )  # 出力信号の長さを計算
        output_signal = resample(
            input_signal, output_length
        )  # サンプリングレートを変更
        return np.array(output_signal)


# test
if __name__ == "__main__":
    # ローダーを設定してMATファイルからデータを読み込む
    loader = MatLoader(
        "data/Stop.mat", ["Time", "ECG", "ch1z", "ch2z", "ch3z", "ch4z", "ch5z", "ch6z"]
    )
    data = loader.load()

    # ch1zチャネルのデータを取得
    ch1z = data["ch1z"]

    # サンプリングレート変換器を作成
    converter = ScipySamplingRateConverter(input_rate=32000, output_rate=1000)
    converted_ch1z = converter.convert(ch1z)

    # 元のデータと変換後のデータをプロット
    plot_two_signals(
        upper=ch1z[: converter.input_rate],
        lower=converted_ch1z[: converter.output_rate],
        upper_label="Original Signal",
        lower_label="Converted Signal",
    )
