import numpy as np
from scipy.signal import butter, filtfilt
from scipy.signal import firwin, lfilter


class FIRBandpassFilter:
    def __init__(self, cutoff: tuple[float, float], fs: float):
        self.fs = fs  # サンプリングレート (Hz)
        self.lowcut, self.highcut = cutoff  # バンドパスフィルタの下限周波数 (Hz)
        self.numtaps = 101  # フィルタのタップ数

    def apply(self, data: np.ndarray) -> np.ndarray:
        b = firwin(
            self.numtaps, [self.lowcut, self.highcut], pass_zero=False, fs=self.fs
        )

        # フィルタの適用
        return lfilter(b, 1.0, data)  # type: ignore


class ButterworthLowpassFilter:
    def __init__(self, cutoff: float, fs: float, order: int = 5):
        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        self.b, self.a = self._design_filter()

    def _design_filter(self):
        nyq = 0.5 * self.fs
        normal_cutoff = self.cutoff / nyq
        b, a = butter(self.order, normal_cutoff, btype="low", analog=False)
        return b, a

    def apply(self, data: np.ndarray) -> np.ndarray:
        return filtfilt(self.b, self.a, data)


# サンプルコード
if __name__ == "__main__":
    np.random.seed(0)
    t = np.linspace(0, 1.0, 200)
    data = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.randn(200)

    cutoff = 20
    fs = 200.0
    order = 4
    filter = ButterworthLowpassFilter(cutoff, fs, order)

    filtered_data = filter.apply(data)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(t, data, label="Noisy signal")
    plt.plot(t, filtered_data, label="Filtered signal", linewidth=2)
    plt.xlabel("Time [sec]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.show()
