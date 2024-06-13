import torch.nn as nn

from models.wave_u_net_enhance_base import WaveUNetEnhanceBase


class WaveUNetEnhance(WaveUNetEnhanceBase):
    def __init__(self, n_layers=10, channels_interval=24):
        super(WaveUNetEnhance, self).__init__(n_layers, channels_interval)

        self.middle = nn.Sequential(
            nn.Conv1d(
                self.n_layers * self.channels_interval,
                self.n_layers * self.channels_interval,
                15,
                stride=1,
                padding=7,
            ),
            nn.BatchNorm1d(self.n_layers * self.channels_interval),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward_latent(self, middle):
        return self.middle(middle)
