from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import ArrayLike
from scipy.fft import fft, ifft


class Randomizer(ABC):
    @abstractmethod
    def shuffle(self, array: ArrayLike) -> np.ndarray:
        pass


class SampleShuffleRandomizer(Randomizer):
    def shuffle(self, array: ArrayLike) -> np.ndarray:
        return np.random.permutation(array)


@dataclass
class PhaseShuffleRandomizer(Randomizer):
    angle_randomizer: Randomizer = field(
        default_factory=lambda: SampleShuffleRandomizer()
    )

    def shuffle(self, array: ArrayLike) -> np.ndarray:
        array_fft = fft(array)
        magnitude = np.abs(array_fft)  # type: ignore
        random_phase = np.exp(1j * self.angle_randomizer.shuffle(np.angle(array_fft)))  # type: ignore
        return ifft(magnitude * random_phase).real  # type: ignore


class PhaseHalfShuffleRandomizer(Randomizer):
    def shuffle(self, array: ArrayLike) -> np.ndarray:
        X = np.fft.fft(array)

        # 振幅と位相を分離
        amplitude = np.abs(X)
        phase = np.angle(X)

        # 位相成分の片側をランダムにシャッフル
        half_point = len(phase) // 2
        shuffled_phase = np.copy(phase)
        shuffled_phase[1:half_point] = np.random.permutation(
            shuffled_phase[1:half_point]
        )

        # 対称性を保つようにする
        shuffled_phase[-half_point + 1 :] = -np.flip(shuffled_phase[1:half_point])

        # シャッフルされた位相を使って新しい複素数列を生成
        shuffled_X = amplitude * np.exp(1j * shuffled_phase)

        # 逆FFTを実行して新しい信号を生成
        new_signal = np.fft.ifft(shuffled_X).real
        return new_signal


class AddUniformNoiseRandomizer(Randomizer):
    def shuffle(self, array: ArrayLike) -> np.ndarray:
        X = np.fft.fft(array)

        # 振幅を取得
        amplitude = np.abs(X)

        # 一様ランダムな位相を生成
        random_phase = np.random.uniform(-np.pi, np.pi, len(X))

        # 対称性を保つようにする
        half_point = len(random_phase) // 2
        random_phase[-half_point + 1 :] = -np.flip(random_phase[1:half_point])

        # ランダムな位相を使って新しい複素数列を生成
        random_X = amplitude * np.exp(1j * random_phase)

        # 逆FFTを実行して新しい信号を生成
        new_signal = np.fft.ifft(random_X).real
        return new_signal
