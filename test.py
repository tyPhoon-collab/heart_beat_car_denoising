import unittest

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.factory import DatasetFactory
from dataset.filter import ButterworthLowpassFilter, FIRBandpassFilter
from dataset.loader import MatLoader
from dataset.randomizer import (
    AddUniformNoiseRandomizer,
    SampleShuffleRandomizer,
    PhaseShuffleRandomizer,
    PhaseHalfShuffleRandomizer,
)
from dataset.sampling_rate_converter import ScipySamplingRateConverter
from loss.combine import CombinedLoss, wavelet_transform
from models.auto_encoder import Conv1DAutoencoder
from models.pixel_shuffle_auto_encoder import PixelShuffleConv1DAutoencoder
from models.transformer_pixel_shuffle_auto_encoder import (
    PixelShuffleConv1DAutoencoderWithTransformer,
)
from models.wave_u_net import WaveUNet
from utils.device import get_torch_device
from utils.gain_controller import ProgressiveGainController
from utils.plot import (
    show_signal,
    show_signals,
    show_spectrogram,
    show_wavelet,
)
from utils.sound import save_signal_to_wav_scipy


class TestDataSet(unittest.TestCase):
    def test_data_set(self):
        randomizer = SampleShuffleRandomizer()
        train_dataset = DatasetFactory.create_240219(
            randomizer=randomizer,
        )
        test_dataset = DatasetFactory.create_240219(
            randomizer=randomizer,
            train=False,
        )

        print(len(train_dataset))
        print(len(test_dataset))

        self.assertGreaterEqual(len(train_dataset), len(test_dataset))

        dataloader = DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True,
        )

        noisy, clean = next(iter(dataloader))

        show_signals([noisy[0][0], clean[0][0]], ["Noisy", "Clean"])

    def test_idx_0(self):
        train_dataset = DatasetFactory.create_240219(
            randomizer=SampleShuffleRandomizer(),
        )

        data = train_dataset[0]
        self.assertEqual(data[0].shape, (1, 5120))
        print(data)

    def test_idx_max(self):
        train_dataset = DatasetFactory.create_240219(
            randomizer=SampleShuffleRandomizer(),
        )

        data = train_dataset[len(train_dataset) - 1]
        self.assertEqual(data[0].shape, (1, 5120))
        print(data)


class TestLoss(unittest.TestCase):
    def test_loss(self):
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc = nn.Linear(5120, 5120)  # 入力5120次元、出力5120次元

            def forward(self, x):
                return self.fc(x)

        # モデル、損失関数、最適化手法の初期化
        model = SimpleModel()
        criterion = CombinedLoss(alpha=0.5)
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # ダミーデータの作成
        inputs = torch.randn(1, 1, 5120)
        targets = torch.randn(1, 1, 5120)

        # 学習ループ
        for epoch in range(100):  # 例として100エポック
            optimizer.zero_grad()  # 勾配の初期化
            outputs = model(inputs)  # モデルの出力
            loss = criterion(outputs, targets)  # 損失の計算
            loss.backward()  # 勾配の計算
            optimizer.step()  # パラメータの更新

            if (epoch + 1) % 10 == 0:  # 10エポックごとに損失を出力
                print(f"Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}")


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
        x = torch.randn(1, 1, 5120 + (8 - 1) * 512)  # Batch size, Channels, Length
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
        show_signals(
            [
                ch1z[: converter.input_rate],
                converted_ch1z[: converter.output_rate],
            ],
            [
                "Original Signal",
                "Converted Signal",
            ],
        )


class TestLoader(unittest.TestCase):
    def test_240219_mat_loader(self):
        loader = MatLoader(
            "data/240219_Rawdata/Stop.mat",
            ["Time", "ECG", "ch1z", "ch2z", "ch3z", "ch4z", "ch5z", "ch6z"],
        )
        data = loader.load()
        print(data)

    def test_240517_mat_loader(self):
        loader = MatLoader(
            "data/240517_Rawdata/HS_data_serial.mat",
            ["ch1z"],
            data_key="HS_data",
        )
        data = loader.load()
        print(data)


class TestVisualize(unittest.TestCase):
    def test_sound_all(self):
        self.convert_to_wav("data/240219_Rawdata/Stop.mat", "stop_32000.wav")
        self.convert_to_wav("data/240219_Rawdata/Idling.mat", "idling_32000.wav")
        self.convert_to_wav("data/240219_Rawdata/100km.mat", "100km_32000.wav")

        self.convert_to_wav(
            "data/240219_Rawdata/Stop.mat", "stop_1024.wav", output_rate=1024
        )
        self.convert_to_wav(
            "data/240219_Rawdata/Idling.mat", "idling_1024.wav", output_rate=1024
        )
        self.convert_to_wav(
            "data/240219_Rawdata/100km.mat", "100km_1024.wav", output_rate=1024
        )

    def test_stft_32000(self):
        single_data = self.load("data/240219_Rawdata/100km.mat", "ch1z")
        show_spectrogram(single_data, 32000)

    def test_stft_1000(self):
        single_data = self.load("data/240219_Rawdata/Stop.mat", "ch1z")
        # single_data = self.load("data/100km.mat", "ch1z")
        single_data = self.convert_sample_rate(single_data, 32000, 1000)
        show_spectrogram(single_data[:10000], 1000, ylim=(0, 64))

    def test_show_stop(self):
        self.show("data/240219_Rawdata/Stop.mat")

    def test_show_idling(self):
        self.show("data/240219_Rawdata/Idling.mat")

    def test_show_100km(self):
        self.show("data/240219_Rawdata/100km.mat")

    def test_show_hs(self):
        data = self.load("data/240517_Rawdata/HS_data_serial.mat")
        show_signal(data[:5000], "HS_data")

    def test_show_hs_wavelet(self):
        self.show_wavelet(
            "data/240517_Rawdata/HS_data_serial.mat",
            modifier=lambda x: FIRBandpassFilter((25, 55), 1000).apply(x[:5000]),
        )

    def test_show_hs_spectrogram(self):
        self.show_spectrogram(
            "data/240517_Rawdata/HS_data_serial.mat",
            sample_rate=1000,
            modifier=lambda x: FIRBandpassFilter((25, 55), 1000).apply(x[:5000]),
        )

    def test_show_ecg(self):
        loader = MatLoader(
            "data/240517_Rawdata/HS_data.mat",
            [f"ch{i}z" for i in range(36)],
            data_key="HS_data",
        )
        data = loader.load()["ch1z"].to_numpy()
        show_signal(data[:10000], "ECG_data")

    def test_show_noise(self):
        self.show("data/240517_Rawdata/Noise_data_serial.mat")

    def test_show_noise_wavelet(self):
        self.show_wavelet(
            "data/240517_Rawdata/Noise_data_serial.mat",
            modifier=lambda x: FIRBandpassFilter((25, 55), 1000).apply(x[:5000]),
        )

    def test_show_all_randomize(self):
        single_data = self.load("data/240219_Rawdata/100km.mat")[: 32000 * 5]
        # single_data = self.convert_sample_rate(single_data, 32000, 1000)
        sample_shuffled_data = SampleShuffleRandomizer().shuffle(single_data)
        phase_shuffled_data = PhaseHalfShuffleRandomizer().shuffle(single_data)
        add_noise_data = AddUniformNoiseRandomizer().shuffle(single_data)

        show_signals(
            [
                single_data,
                sample_shuffled_data,
                phase_shuffled_data,
                add_noise_data,
            ],
            [
                "Original",
                "Sample Shuffled",
                "Phase Shuffled",
                "Add Noise",
            ],
        )

    def test_sound_240219_randomize(self):
        output_sample_rate = 3000
        single_data = self.load("data/240219_Rawdata/100km.mat")
        single_data = self.convert_sample_rate(
            single_data,
            32000,
            output_sample_rate,
        )[: output_sample_rate * 5]
        sample_shuffled_data = SampleShuffleRandomizer().shuffle(single_data)
        phase_shuffled_data = PhaseShuffleRandomizer().shuffle(single_data)
        add_uniform_noise_data = AddUniformNoiseRandomizer().shuffle(single_data)

        save_signal_to_wav_scipy(
            single_data,
            output_sample_rate,
            "100km_1000.wav",
        )
        save_signal_to_wav_scipy(
            sample_shuffled_data,
            output_sample_rate,
            "100km_1000_sample_randomized.wav",
        )
        save_signal_to_wav_scipy(
            phase_shuffled_data,
            output_sample_rate,
            "100km_1000_phase_shuffled.wav",
        )
        save_signal_to_wav_scipy(
            add_uniform_noise_data,
            output_sample_rate,
            "100km_1000_add_uniform_noise.wav",
        )

    def test_sound_240517(self):
        output_sample_rate = 3000

        def load(path: str, ch: str = "ch1z"):
            loader = DatasetFactory.build_loader(path)
            single_data = loader.load()["ch1z"].to_numpy()
            return self.convert_sample_rate(
                single_data,
                1000,
                output_sample_rate,
            )

        save_signal_to_wav_scipy(
            load("data/240517_Rawdata/HS_data_serial.mat"),
            output_sample_rate,
            "HS.wav",
        )

        save_signal_to_wav_scipy(
            load("data/240517_Rawdata/Noise_data_serial.mat"),
            output_sample_rate,
            "Noise.wav",
        )

    def test_filter_240517(self):
        filter = ButterworthLowpassFilter(20, 1000)
        data = self.load("data/240517_Rawdata/Noise_data_serial.mat")[:3000]
        show_signals(
            [
                data,
                filter.apply(data),
            ],
            ["Original", "Filtered"],
        )

    def test_bandpass_filter_240517_noise(self):
        filter = FIRBandpassFilter((25, 100), 1000)
        data = self.load("data/240517_Rawdata/Noise_data_serial.mat")[:3000]
        show_signals(
            [
                data,
                filter.apply(data),
            ],
            ["Original", "Filtered"],
        )

    def test_bandpass_filter_240517_hs(self):
        filter = FIRBandpassFilter((25, 55), 1000)
        data = self.load("data/240517_Rawdata/HS_data_serial.mat")[:3000]
        show_signals(
            [
                data,
                filter.apply(data),
            ],
            ["Original", "Filtered"],
        )

    def test_compare_noise_240219_240517(self):
        filter = ButterworthLowpassFilter(20, 1000)
        seconds = 10
        old_data = self.load("data/240219_Rawdata/100km.mat")[: 32000 * seconds]
        new_data = self.load("data/240517_Rawdata/Noise_data_serial.mat")[
            : 1000 * seconds
        ]
        show_signals(
            [
                old_data,
                new_data,
                filter.apply(new_data),
            ],
            ["100km 240219", "Noise 240517", "Noise 240517 Filtered"],
        )

    def test_compare_hs_240219_240517(self):
        filter = ButterworthLowpassFilter(80, 1000)
        seconds = 5
        old_data = self.load("data/240219_Rawdata/Stop.mat")[: 32000 * seconds]
        new_data = self.load("data/240517_Rawdata/HS_data_serial.mat")[: 1000 * seconds]
        show_signals(
            [
                old_data,
                new_data,
                filter.apply(new_data),
            ],
            ["Stop 240219", "HS 240517", "HS 240517 Filtered"],
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

    def show(self, path: str, ch: str = "ch1z"):
        single_data = self.load(path, ch)
        show_signal(single_data, ch)

    def show_wavelet(self, path: str, ch: str = "ch1z", modifier=None):
        waveform = self.load(path, ch)

        if modifier is not None:
            waveform = modifier(waveform)

        cwt_coeffs = wavelet_transform(torch.tensor(waveform), np.arange(1, 37))
        print(cwt_coeffs[0])
        print(cwt_coeffs[1])
        cwt_result = cwt_coeffs[0].numpy()
        show_wavelet(waveform, cwt_result, 36)

    def show_spectrogram(
        self, path: str, sample_rate: int, ch: str = "ch1z", modifier=None
    ):
        waveform = self.load(path, ch)

        if modifier is not None:
            waveform = modifier(waveform)

        show_spectrogram(waveform, sample_rate, ylim=(0, 512))

    def load(self, path: str, ch: str = "ch1z"):
        return DatasetFactory.build_loader(path).load()[ch].to_numpy()

    def convert_sample_rate(self, signal: np.ndarray, input_rate, output_rate):
        return ScipySamplingRateConverter(input_rate, output_rate).convert(signal)


class TestGainController(unittest.TestCase):
    def test_gain_controller(self):
        controller = ProgressiveGainController(epoch_from=0, epoch_to=4, max_gain=1.1)
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


class TestPyTorchFlow(unittest.TestCase):
    def test_pytorch_flow(self):
        from torch.utils.data import Dataset
        import matplotlib.pyplot as plt

        class NoisySignalDataset(Dataset):
            def __init__(self, clean_signals, noisy_signals):
                self.clean_signals = clean_signals
                self.noisy_signals = noisy_signals

            def __len__(self):
                return len(self.clean_signals)

            def __getitem__(self, idx):
                clean = self.clean_signals[idx]
                noisy = self.noisy_signals[idx]
                return torch.tensor(noisy, dtype=torch.float32).unsqueeze(
                    0
                ), torch.tensor(clean, dtype=torch.float32).unsqueeze(0)

        class SimpleAutoencoder(nn.Module):
            def __init__(self):
                super(SimpleAutoencoder, self).__init__()
                self.encoder = nn.Sequential(
                    nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                )
                self.decoder = nn.Sequential(
                    nn.ConvTranspose1d(
                        128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    nn.ReLU(),
                    nn.ConvTranspose1d(
                        64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    nn.ReLU(),
                    nn.ConvTranspose1d(
                        32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    nn.ReLU(),
                    nn.ConvTranspose1d(
                        16, 1, kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                )

            def forward(self, x):
                x = self.encoder(x)
                x = self.decoder(x)
                return x

        # データ生成（サイン波にノイズを加えたデータ）
        def generate_data(num_samples, length):
            np.random.seed(0)
            t = np.linspace(0, 1.0, length)
            clean_signals = [np.sin(2 * np.pi * 5 * t) for _ in range(num_samples)]
            noisy_signals = [
                clean + 0.5 * np.random.randn(length) for clean in clean_signals
            ]
            return np.array(clean_signals), np.array(noisy_signals)

        device = get_torch_device()

        signal_length = 5120
        batch_size = 16
        num_epochs = 10
        learning_rate = 0.001

        # データ生成
        clean_signals, noisy_signals = generate_data(1000, signal_length)
        dataset = NoisySignalDataset(clean_signals, noisy_signals)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # モデル、損失関数、最適化手法の設定
        model = SimpleAutoencoder()
        model.to(device)
        criterion = nn.MSELoss()
        # criterion = nn.L1Loss()
        # criterion = CombinedLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # トレーニングループ
        for epoch in range(num_epochs):
            for batch in dataloader:
                noisy, clean = map(lambda x: x.to(device), batch)

                optimizer.zero_grad()
                outputs = model(noisy)
                loss = criterion(outputs.to(device), clean)
                loss.backward()
                optimizer.step()

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")  # type: ignore

        # テストデータに対するモデルの適用
        test_clean_signals, test_noisy_signals = generate_data(10, signal_length)
        test_dataset = NoisySignalDataset(test_clean_signals, test_noisy_signals)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # 結果のプロット
        model.eval()

        with torch.no_grad():
            noisy, clean = map(lambda x: x.to(device), next(iter(test_dataloader)))

            denoised = model(noisy).cpu().squeeze().numpy()

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.plot(noisy.cpu().squeeze().numpy())
            plt.title("Noisy Signal")
            plt.subplot(1, 3, 2)
            plt.plot(clean.cpu().squeeze().numpy())
            plt.title("Clean Signal")
            plt.subplot(1, 3, 3)
            plt.plot(denoised)
            plt.title("Denoised Signal")
            plt.savefig("output/fig/sample.png")


if __name__ == "__main__":
    unittest.main()
