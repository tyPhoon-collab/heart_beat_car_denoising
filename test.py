import unittest

import numpy as np
from torch.utils.data import DataLoader

from dataset.dataset import NoisyHeartbeatDataset
from dataset.loader import MatLoader
from dataset.randomizer import NumpyRandomShuffleRandomizer
from dataset.sampling_rate_converter import ScipySamplingRateConverter
from utils.plot import plot_signal, plot_spectrogram, plot_two_signals
from utils.sound import save_signal_to_wav_scipy


class TestDataSet(unittest.TestCase):
    def test_data_set(self):
        train_dataset = NoisyHeartbeatDataset(
            clean_file_path="data/Stop.mat",
            noisy_file_path="data/100km.mat",
            # noisy_file_path="data/Stop.mat",
            sampling_rate_converter=ScipySamplingRateConverter(
                input_rate=32000,
                output_rate=1000,
            ),
            randomizer=NumpyRandomShuffleRandomizer(),
        )
        test_dataset = NoisyHeartbeatDataset(
            clean_file_path="data/Stop.mat",
            noisy_file_path="data/100km.mat",
            # noisy_file_path="data/Stop.mat",
            sampling_rate_converter=ScipySamplingRateConverter(
                input_rate=32000,
                output_rate=1000,
            ),
            randomizer=NumpyRandomShuffleRandomizer(),
            train=False,
        )

        print(len(train_dataset))
        print(len(test_dataset))

        dataloader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
        )

        noisy, clean = next(iter(dataloader))

        plot_two_signals(noisy[0][0], clean[0][0], "Noisy", "Clean")


class TestSampleRateConverter(unittest.TestCase):
    def test_scipy_sample_rate_converter(self):
        loader = MatLoader(
            "data/Stop.mat",
            ["Time", "ECG", "ch1z", "ch2z", "ch3z", "ch4z", "ch5z", "ch6z"],
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


class TestLoader(unittest.TestCase):
    def test_mat_loader(self):
        loader = MatLoader(
            "data/Stop.mat",
            ["Time", "ECG", "ch1z", "ch2z", "ch3z", "ch4z", "ch5z", "ch6z"],
        )
        data = loader.load()
        print(data)


class TestVisualize(unittest.TestCase):
    def test_sound_all(self):
        self.convert_to_wav("data/Stop.mat", "stop_32000")
        self.convert_to_wav("data/Idling.mat", "idling_32000")
        self.convert_to_wav("data/100km.mat", "100km_32000")

        self.convert_to_wav("data/Stop.mat", "stop_1024", output_sample_rate=1024)
        self.convert_to_wav("data/Idling.mat", "idling_1024", output_sample_rate=1024)
        self.convert_to_wav("data/100km.mat", "100km_1024", output_sample_rate=1024)

    def test_stft_32000(self):
        single_data = self.load("data/100km.mat", "ch1z")
        plot_spectrogram(single_data, 32000)

    def test_stft_1000(self):
        single_data = self.load("data/Stop.mat", "ch1z")
        # single_data = self.load("data/100km.mat", "ch1z")
        single_data = self.convert_sample_rate(single_data, 32000, 1000)
        plot_spectrogram(single_data[:10000], 1000, ylim=(0, 64))

    def test_show_stop(self):
        self.show("data/Stop.mat")

    def test_show_idling(self):
        self.show("data/Idling.mat")

    def test_show_100km(self):
        self.show("data/100km.mat")

    def convert_to_wav(
        self,
        filename: str,
        output_filename: str,
        *,
        output_sample_rate=32000,
    ):
        single_data = self.load(filename, "ch1z")
        single_data = self.convert_sample_rate(single_data, 32000, output_sample_rate)
        save_signal_to_wav_scipy(
            single_data,
            output_sample_rate,
            f"output/audio/{output_filename}.wav",
        )

    def show(self, file_path: str, ch: str = "ch1z"):
        single_data = self.load(file_path, ch)
        print(single_data.shape)
        print(single_data)
        plot_signal(single_data, ch)

    def load(self, file_path: str, ch: str = "ch1z"):
        loader = MatLoader(
            file_path,
            ["Time", "ECG", "ch1z", "ch2z", "ch3z", "ch4z", "ch5z", "ch6z"],
        )
        data = loader.load()
        single_data = data[ch]
        return single_data.to_numpy()

    def convert_sample_rate(self, signal: np.ndarray, input_rate, output_rate):
        signal = ScipySamplingRateConverter(
            input_rate,
            output_rate,
        ).convert(signal)

        return signal


if __name__ == "__main__":
    unittest.main()
