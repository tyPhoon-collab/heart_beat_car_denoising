"""
いろいろな組み合わせを確かめる必要がある。
いちいちコメント化するのも、書き換えるのも大変なので、テストケースとして用意する。
"""

import unittest

from dataset.dataset import NoisyHeartbeatDataset
from dataset.randomizer import NumpyRandomShuffleRandomizer
from dataset.sampling_rate_converter import ScipySamplingRateConverter
from eval import eval_model
from models.auto_encoder import Conv1DAutoencoder
from models.wave_u_net import WaveUNet
from train import build_logger, train_model
from utils.device import safe_load_dotenv
from utils.model_saver import WithDateModelSaver
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


def _build_dataset(
    clean_file_path="data/Stop.mat",
    noisy_file_path="data/100km.mat",
    input_rate=32000,
    output_rate=1024,
    randomizer=None,
    train=True,
):
    dataset = NoisyHeartbeatDataset(
        clean_file_path=clean_file_path,
        noisy_file_path=noisy_file_path,
        sampling_rate_converter=ScipySamplingRateConverter(
            input_rate=input_rate,
            output_rate=output_rate,
        ),
        randomizer=randomizer or NumpyRandomShuffleRandomizer(),
        train=train,
    )
    return dataset


def _build_loader(
    dataset,
    *,
    batch_size=1,
    shuffle=True,
):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return dataloader


class TestTrainSet(unittest.TestCase):
    def setUp(self):
        safe_load_dotenv()
        self._model_saver = WithDateModelSaver(base_directory="output/checkpoint")
        self._logger = build_logger()

    def test_l1_adam_wave_u_net(self):
        model = WaveUNet()
        train_dataloader = _build_loader(_build_dataset())

        train_model(
            model,
            train_dataloader,
            nn.L1Loss(),
            optim.Adam(model.parameters(), lr=0.001),
            model_saver=self._model_saver,
            logger=self._logger,
            epoch_size=5,
        )

    def test_smooth_l1_adam_wave_u_net(self):
        model = WaveUNet()
        train_dataloader = _build_loader(_build_dataset())

        train_model(
            model,
            train_dataloader,
            nn.SmoothL1Loss(),
            optim.Adam(model.parameters(), lr=0.001),
            model_saver=self._model_saver,
            logger=self._logger,
            epoch_size=5,
        )

    def test_smooth_l1_adam_autoencoder(self):
        model = Conv1DAutoencoder()
        train_dataloader = _build_loader(_build_dataset())

        train_model(
            model,
            train_dataloader,
            nn.SmoothL1Loss(),
            optim.Adam(model.parameters(), lr=0.001),
            model_saver=self._model_saver,
            logger=self._logger,
            epoch_size=5,
        )


class TestEvalSet(unittest.TestCase):
    def test_eval_l1_wave_u_net(self):
        model = WaveUNet()
        test_dataloader = _build_loader(_build_dataset(train=False))

        eval_model(
            model,
            "",
            test_dataloader,
            nn.L1Loss(),
        )

    def test_eval_smooth_l1_wave_u_net(self):
        model = WaveUNet()
        test_dataloader = _build_loader(_build_dataset(train=False))

        eval_model(
            model,
            "",
            test_dataloader,
            nn.SmoothL1Loss(),
        )

    def test_eval_smooth_l1_autoencoder(self):
        model = Conv1DAutoencoder()
        test_dataloader = _build_loader(_build_dataset(train=False))

        eval_model(
            model,
            "",
            test_dataloader,
            nn.SmoothL1Loss(),
        )
