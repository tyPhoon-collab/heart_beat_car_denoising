import torch.nn as nn

from models.modules.pixel_shuffle import PixelShuffle1d, PixelUnshuffle1d


class PixelShuffleConv1DAutoencoder(nn.Module):
    def __init__(self):
        super(PixelShuffleConv1DAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(
                1, 16, kernel_size=3, stride=1, padding=1
            ),  # output is (16, 5120)
            nn.ReLU(),
            PixelUnshuffle1d(2),  # output is (32, 2560)
            nn.Conv1d(
                32, 64, kernel_size=3, stride=1, padding=1
            ),  # output is (64, 2560)
            nn.ReLU(),
            PixelUnshuffle1d(2),  # output is (128, 1280)
            nn.Conv1d(
                128, 256, kernel_size=3, stride=1, padding=1
            ),  # output is (256, 1280)
            nn.ReLU(),
            PixelUnshuffle1d(2),  # output is (512, 640)
            nn.Conv1d(
                512, 1024, kernel_size=3, stride=1, padding=1
            ),  # output is (1024, 640)
            nn.ReLU(),
            PixelUnshuffle1d(2),  # output is (2048, 320)
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(
                2048, 1024, kernel_size=3, stride=1, padding=1
            ),  # output is (1024, 320)
            nn.ReLU(),
            PixelShuffle1d(2),  # output is (512, 640)
            nn.Conv1d(
                512, 256, kernel_size=3, stride=1, padding=1
            ),  # output is (256, 640)
            nn.ReLU(),
            PixelShuffle1d(2),  # output is (128, 1280)
            nn.Conv1d(
                128, 64, kernel_size=3, stride=1, padding=1
            ),  # output is (64, 1280)
            nn.ReLU(),
            PixelShuffle1d(2),  # output is (32, 2560)
            nn.Conv1d(
                32, 16, kernel_size=3, stride=1, padding=1
            ),  # output is (16, 2560)
            nn.ReLU(),
            PixelShuffle1d(2),  # output is (8, 5120)
            nn.Conv1d(8, 1, kernel_size=3, stride=1, padding=1),  # output is (1, 5120)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
