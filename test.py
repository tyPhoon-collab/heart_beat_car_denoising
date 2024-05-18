import unittest

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.dataset import NoisyHeartbeatDataset
from dataset.loader import MatLoader
from dataset.randomizer import SampleShuffleRandomizer, PhaseShuffleRandomizer
from dataset.sampling_rate_converter import ScipySamplingRateConverter
from models.auto_encoder import Conv1DAutoencoder
from models.pixel_shuffle_auto_encoder import PixelShuffleConv1DAutoencoder
from models.transformer_pixel_shuffle_auto_encoder import (
    PixelShuffleConv1DAutoencoderWithTransformer,
)
from models.wave_u_net import WaveUNet
from utils.gain_controller import GainController
from utils.plot import (
    plot_signal,
    plot_spectrogram,
    plot_three_signals,
    plot_two_signals,
)
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
            randomizer=SampleShuffleRandomizer(),
        )
        test_dataset = NoisyHeartbeatDataset(
            clean_file_path="data/Stop.mat",
            noisy_file_path="data/100km.mat",
            # noisy_file_path="data/Stop.mat",
            sampling_rate_converter=ScipySamplingRateConverter(
                input_rate=32000,
                output_rate=1000,
            ),
            randomizer=SampleShuffleRandomizer(),
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

    def test_data_set_progressive(self):
        train_dataset = NoisyHeartbeatDataset(
            clean_file_path="data/Stop.mat",
            noisy_file_path="data/100km.mat",
            # noisy_file_path="data/Stop.mat",
            sampling_rate_converter=ScipySamplingRateConverter(
                input_rate=32000,
                output_rate=1000,
            ),
            randomizer=PhaseShuffleRandomizer(),
            gain_controller=GainController(epoch_from=0, epoch_to=5),
        )
        test_dataset = NoisyHeartbeatDataset(
            clean_file_path="data/Stop.mat",
            noisy_file_path="data/100km.mat",
            # noisy_file_path="data/Stop.mat",
            sampling_rate_converter=ScipySamplingRateConverter(
                input_rate=32000,
                output_rate=1000,
            ),
            randomizer=PhaseShuffleRandomizer(),
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

    def test_idx_0(self):
        train_dataset = NoisyHeartbeatDataset(
            clean_file_path="data/Stop.mat",
            noisy_file_path="data/100km.mat",
            # noisy_file_path="data/Stop.mat",
            sampling_rate_converter=ScipySamplingRateConverter(
                input_rate=32000,
                output_rate=1024,
            ),
            randomizer=SampleShuffleRandomizer(),
        )

        data = train_dataset[0]
        self.assertEqual(data[0].shape, (1, 5120))
        print(data)

    def test_idx_max(self):
        train_dataset = NoisyHeartbeatDataset(
            clean_file_path="data/Stop.mat",
            noisy_file_path="data/100km.mat",
            # noisy_file_path="data/Stop.mat",
            sampling_rate_converter=ScipySamplingRateConverter(
                input_rate=32000,
                output_rate=1024,
            ),
            randomizer=SampleShuffleRandomizer(),
        )

        data = train_dataset[len(train_dataset) - 1]
        self.assertEqual(data[0].shape, (1, 5120))
        print(data)


class TestModels(unittest.TestCase):
    def test_wave_u_net(self):
        # Create model and example input tensor
        model = WaveUNet()

        example_input = torch.rand(1, 1, 256 * 20)  # (batch_size, channels, length)
        # Get the model output
        output = model(example_input)
        print(output.shape)  # Should be torch.Size([1, 1, 5120])

    def test_transformer_pixel_shuffle_auto_encoder(self):

        # Model instantiation
        model = PixelShuffleConv1DAutoencoderWithTransformer()
        print(model)

        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Dummy data for demonstration
        x = torch.randn(1, 1, 5120)  # Batch size, Channels, Length
        x = x.to(torch.float32)

        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, x)
        print(f"Loss: {loss.item()}")

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def test_pixel_shuffle_auto_encoder(self):
        # Model instantiation
        model = PixelShuffleConv1DAutoencoder()
        print(model)

        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Dummy data for demonstration
        x = torch.randn(1, 1, 5120)  # Batch size, Channels, Length
        x = x.to(torch.float32)

        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, x)
        print(f"Loss: {loss.item()}")

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def test_auto_encoder(self):
        # Model instantiation
        model = Conv1DAutoencoder()
        print(model)

        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Dummy data for demonstration
        x = torch.randn(1, 1, 5120)  # Batch size, Channels, Length
        x = x.to(torch.float32)

        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, x)
        print(f"Loss: {loss.item()}")

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


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
        self.convert_to_wav("data/Stop.mat", "stop_32000.wav")
        self.convert_to_wav("data/Idling.mat", "idling_32000.wav")
        self.convert_to_wav("data/100km.mat", "100km_32000.wav")

        self.convert_to_wav("data/Stop.mat", "stop_1024.wav", output_rate=1024)
        self.convert_to_wav("data/Idling.mat", "idling_1024.wav", output_rate=1024)
        self.convert_to_wav("data/100km.mat", "100km_1024.wav", output_rate=1024)

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

    def test_show_all_randomize(self):
        single_data = self.load("data/100km.mat")
        single_data = self.convert_sample_rate(single_data, 32000, 1000)
        sample_base_shuffled_data = SampleShuffleRandomizer().shuffle(single_data)
        phase_shuffled_data = PhaseShuffleRandomizer().shuffle(single_data)

        plot_three_signals(
            upper=single_data,
            middle=sample_base_shuffled_data,
            lower=phase_shuffled_data,
            upper_label="Original Signal",
            middle_label="Randomized Signal",
            lower_label="Phase Shuffled Signal",
        )

    def convert_to_wav(
        self,
        filename: str,
        output_filename: str,
        *,
        output_rate=32000,
    ):
        single_data = self.load(filename, "ch1z")
        single_data = self.convert_sample_rate(single_data, 32000, output_rate)
        save_signal_to_wav_scipy(single_data, output_rate, output_filename)

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


class TestGainController(unittest.TestCase):
    def test_gain_controller(self):
        controller = GainController(epoch_from=0, epoch_to=4, max_gain=1.1)
        self.assertEqual(controller.gain, 1)

        controller.set_gain_from_epoch(0)
        self.assertEqual(controller.gain, 0)

        controller.set_gain_from_epoch(2)
        self.assertEqual(controller.gain, 0.55)

        controller.set_gain_from_epoch(4)
        self.assertEqual(controller.gain, 1.1)


class TestRandomizer(unittest.TestCase):
    def test_numpy_shuffle(self):
        a = np.random.randn(100)
        rand = SampleShuffleRandomizer().shuffle(a)
        self.assertEqual(len(a), len(rand))
        print(rand)

    def test_phase_shuffle(self):
        a = np.random.randn(100)
        rand = PhaseShuffleRandomizer().shuffle(a)
        self.assertEqual(len(a), len(rand))
        print(rand)


if __name__ == "__main__":
    unittest.main()
