import torch
import torch.nn as nn


class PixelShuffle1d(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle1d, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        n, c, l = x.size()
        new_c = c // self.upscale_factor
        new_l = l * self.upscale_factor
        x = x.view(n, new_c, self.upscale_factor, l)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(n, new_c, new_l)
        return x


class PixelUnshuffle1d(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnshuffle1d, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x):
        n, c, l = x.size()
        new_c = c * self.downscale_factor
        new_l = l // self.downscale_factor
        x = x.view(n, c, new_l, self.downscale_factor)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(n, new_c, new_l)
        return x


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


if __name__ == "__main__":
    import torch.optim as optim

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
