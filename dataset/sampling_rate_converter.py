from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from scipy.signal import resample
from numpy.typing import ArrayLike
import numpy as np


@dataclass
class SamplingRateConverter(ABC):
    input_rate: int
    output_rate: int

    @abstractmethod
    def convert(self, input_signal: ArrayLike) -> np.ndarray:
        pass


@dataclass
class NoSamplingRateConverter(SamplingRateConverter):
    rate: int
    input_rate: int = field(init=False)
    output_rate: int = field(init=False)

    def __post_init__(self):
        self.input_rate = self.rate
        self.output_rate = self.rate

    def convert(self, input_signal: ArrayLike) -> np.ndarray:
        if input_signal is np.ndarray:
            return input_signal
        return np.array(input_signal)


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
