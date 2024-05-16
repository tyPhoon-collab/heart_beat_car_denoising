import torch.nn as nn


class Conv1DAutoencoder(nn.Module):
    def __init__(self):
        super(Conv1DAutoencoder, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv1d(
                1, 16, kernel_size=3, stride=2, padding=1
            ),  # input is (1, 5120), output is (16, 2560)
            nn.ReLU(),
            nn.Conv1d(
                16, 32, kernel_size=3, stride=2, padding=1
            ),  # output is (32, 1280)
            nn.ReLU(),
            nn.Conv1d(
                32, 64, kernel_size=3, stride=2, padding=1
            ),  # output is (64, 640)
            nn.ReLU(),
            nn.Conv1d(
                64, 128, kernel_size=3, stride=2, padding=1
            ),  # output is (128, 320)
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # output is (64, 640)
            nn.ReLU(),
            nn.ConvTranspose1d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # output is (32, 1280)
            nn.ReLU(),
            nn.ConvTranspose1d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # output is (16, 2560)
            nn.ReLU(),
            nn.ConvTranspose1d(
                16, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # output is (1, 5120)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
