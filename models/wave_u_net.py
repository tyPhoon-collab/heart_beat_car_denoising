import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveUNet(nn.Module):
    def __init__(self):
        super(WaveUNet, self).__init__()
        self.encoder_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=1, out_channels=16, kernel_size=15, stride=2, padding=7
                ),
                nn.Conv1d(16, 32, 15, 2, 7),
                nn.Conv1d(32, 64, 15, 2, 7),
                nn.Conv1d(64, 128, 15, 2, 7),
                nn.Conv1d(128, 256, 15, 2, 7),
                nn.Conv1d(256, 512, 15, 2, 7),
                nn.Conv1d(512, 1024, 15, 2, 7),
                nn.Conv1d(1024, 1024, 15, 2, 7),
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                nn.ConvTranspose1d(1024, 1024, 15, 2, 7, output_padding=1),
                nn.ConvTranspose1d(2048, 512, 15, 2, 7, output_padding=1),
                nn.ConvTranspose1d(1024, 256, 15, 2, 7, output_padding=1),
                nn.ConvTranspose1d(512, 128, 15, 2, 7, output_padding=1),
                nn.ConvTranspose1d(256, 64, 15, 2, 7, output_padding=1),
                nn.ConvTranspose1d(128, 32, 15, 2, 7, output_padding=1),
                nn.ConvTranspose1d(64, 16, 15, 2, 7, output_padding=1),
                nn.ConvTranspose1d(32, 1, 15, 2, 7, output_padding=1),
            ]
        )

    def forward(self, x):
        # Encoder
        skip_connections = []
        for layer in self.encoder_layers:
            x = F.leaky_relu(layer(x), 0.2)
            skip_connections.append(x)

        skip_connections = skip_connections[::-1]
        # Decoder
        for i, layer in enumerate(self.decoder_layers):
            x = F.leaky_relu(layer(x), 0.2)
            if (
                i < len(self.decoder_layers) - 1
            ):  # Skip connection for all except last layer
                x = torch.cat(
                    [x, skip_connections[i + 1]], 1
                )  # Skip the first connection
        return x


if __name__ == "__main__":
    # Create model and example input tensor
    model = WaveUNet()

    example_input = torch.rand(1, 1, 256 * 20)  # (batch_size, channels, length)
    # Get the model output
    output = model(example_input)
    print(output.shape)  # Should be torch.Size([1, 1, 5000])
