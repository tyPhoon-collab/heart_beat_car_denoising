import torch
import torch.nn as nn


class WaveUNetNorm(nn.Module):
    def __init__(self):
        super(WaveUNetNorm, self).__init__()
        self.encoder_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=1,
                        out_channels=16,
                        kernel_size=15,
                        stride=2,
                        padding=7,
                    ),
                    nn.BatchNorm1d(16),
                    nn.PReLU(),
                ),
                nn.Sequential(
                    nn.Conv1d(16, 32, 15, 2, 7), nn.BatchNorm1d(32), nn.PReLU()
                ),
                nn.Sequential(
                    nn.Conv1d(32, 64, 15, 2, 7), nn.BatchNorm1d(64), nn.PReLU()
                ),
                nn.Sequential(
                    nn.Conv1d(64, 128, 15, 2, 7), nn.BatchNorm1d(128), nn.PReLU()
                ),
                nn.Sequential(
                    nn.Conv1d(128, 256, 15, 2, 7), nn.BatchNorm1d(256), nn.PReLU()
                ),
                nn.Sequential(
                    nn.Conv1d(256, 512, 15, 2, 7), nn.BatchNorm1d(512), nn.PReLU()
                ),
                nn.Sequential(
                    nn.Conv1d(512, 1024, 15, 2, 7), nn.BatchNorm1d(1024), nn.PReLU()
                ),
                nn.Sequential(
                    nn.Conv1d(1024, 1024, 15, 2, 7), nn.BatchNorm1d(1024), nn.PReLU()
                ),
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose1d(1024, 1024, 15, 2, 7, output_padding=1),
                    nn.BatchNorm1d(1024),
                    nn.PReLU(),
                ),
                nn.Sequential(
                    nn.ConvTranspose1d(2048, 512, 15, 2, 7, output_padding=1),
                    nn.BatchNorm1d(512),
                    nn.PReLU(),
                ),
                nn.Sequential(
                    nn.ConvTranspose1d(1024, 256, 15, 2, 7, output_padding=1),
                    nn.BatchNorm1d(256),
                    nn.PReLU(),
                ),
                nn.Sequential(
                    nn.ConvTranspose1d(512, 128, 15, 2, 7, output_padding=1),
                    nn.BatchNorm1d(128),
                    nn.PReLU(),
                ),
                nn.Sequential(
                    nn.ConvTranspose1d(256, 64, 15, 2, 7, output_padding=1),
                    nn.BatchNorm1d(64),
                    nn.PReLU(),
                ),
                nn.Sequential(
                    nn.ConvTranspose1d(128, 32, 15, 2, 7, output_padding=1),
                    nn.BatchNorm1d(32),
                    nn.PReLU(),
                ),
                nn.Sequential(
                    nn.ConvTranspose1d(64, 16, 15, 2, 7, output_padding=1),
                    nn.BatchNorm1d(16),
                    nn.PReLU(),
                ),
                nn.Sequential(
                    nn.ConvTranspose1d(32, 1, 15, 2, 7, output_padding=1),
                ),
            ]
        )

    def forward(self, x):
        # Encoder
        skip_connections = []
        for layer in self.encoder_layers:
            x = layer(x)
            skip_connections.append(x)

        skip_connections = skip_connections[::-1]
        # Decoder
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x)
            if (
                i < len(self.decoder_layers) - 1
            ):  # Skip connection for all except last layer
                x = torch.cat(
                    [x, skip_connections[i + 1]], 1
                )  # Skip the first connection
        return x
