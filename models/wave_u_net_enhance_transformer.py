import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class DownSamplingLayer(nn.Module):
    def __init__(
        self, channel_in, channel_out, dilation=1, kernel_size=15, stride=1, padding=7
    ):
        super(DownSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(
                channel_in,
                channel_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1),
        )

    def forward(self, ipt):
        return self.main(ipt)


class UpSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=5, stride=1, padding=2):
        super(UpSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(
                channel_in,
                channel_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, ipt):
        return self.main(ipt)


class WaveUNetEnhanceTransformer(nn.Module):
    def __init__(
        self,
        n_layers=10,
        channels_interval=24,
        nhead=8,
        num_encoder_layers=2,
        dim_feedforward=2048,
    ):
        super(WaveUNetEnhanceTransformer, self).__init__()

        self.n_layers = n_layers
        self.channels_interval = channels_interval
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward

        encoder_in_channels_list = [1] + [
            i * self.channels_interval for i in range(1, self.n_layers)
        ]
        encoder_out_channels_list = [
            i * self.channels_interval for i in range(1, self.n_layers + 1)
        ]

        self.encoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.encoder.append(
                DownSamplingLayer(
                    channel_in=encoder_in_channels_list[i],
                    channel_out=encoder_out_channels_list[i],
                )
            )

        decoder_in_channels_list = [
            (2 * i + 1) * self.channels_interval for i in range(1, self.n_layers)
        ] + [2 * self.n_layers * self.channels_interval]
        decoder_in_channels_list = decoder_in_channels_list[::-1]
        decoder_out_channels_list = encoder_out_channels_list[::-1]
        self.decoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.decoder.append(
                UpSamplingLayer(
                    channel_in=decoder_in_channels_list[i],
                    channel_out=decoder_out_channels_list[i],
                )
            )

        self.transformer_layer = TransformerEncoderLayer(
            d_model=self.n_layers * self.channels_interval,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
        )
        self.transformer_encoder = TransformerEncoder(
            self.transformer_layer, num_layers=self.num_encoder_layers
        )

        self.out = nn.Sequential(
            nn.Conv1d(1 + self.channels_interval, 1, kernel_size=1, stride=1), nn.Tanh()
        )

    def forward(self, input):
        tmp = []
        o = input

        # Up Sampling
        for i in range(self.n_layers):
            o = self.encoder[i](o)
            tmp.append(o)
            # [batch_size, T // 2, channels]
            o = o[:, :, ::2]

        # Reshape for Transformer (batch, sequence, feature)
        o = o.permute(2, 0, 1)
        o = self.transformer_encoder(o)
        o = o.permute(1, 2, 0)

        # Down Sampling
        for i in range(self.n_layers):
            # [batch_size, T * 2, channels]
            o = F.interpolate(o, scale_factor=2, mode="linear", align_corners=True)
            # Skip Connection
            o = torch.cat([o, tmp[self.n_layers - i - 1]], dim=1)
            o = self.decoder[i](o)

        o = torch.cat([o, input], dim=1)
        o = self.out(o)
        return o
