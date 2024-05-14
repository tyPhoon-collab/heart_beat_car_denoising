import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Encoder
        for feature in features:
            self.encoder.append(self.conv_block(in_channels, feature))
            in_channels = feature

        # Decoder
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose1d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(self.conv_block(feature * 2, feature))

        self.bottleneck = self.conv_block(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv1d(features[0], out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        skip_connections = []

        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = torch.nn.functional.interpolate(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](concat_skip)

        return self.final_conv(x)


class DiffusionModel(nn.Module):
    def __init__(self, model, timesteps=1000):
        super(DiffusionModel, self).__init__()
        self.model = model
        self.timesteps = timesteps
        self.betas = torch.linspace(0.0001, 0.02, timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def forward(self, x, t):
        return self.model(x)

    def diffusion(self, x, t):
        sqrt_alpha_bar_t = torch.sqrt(self.alpha_bars[t])
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - self.alpha_bars[t])
        noise = torch.randn_like(x)
        return sqrt_alpha_bar_t * x + sqrt_one_minus_alpha_bar_t * noise

    def denoise(self, x, t):
        return self.model(x)


if __name__ == "__main__":
    import torch.optim as optim

    unet = UNet()
    diffusion_model = DiffusionModel(unet)

    # Model instantiation
    model = diffusion_model
    print(model)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Dummy data for demonstration
    x = torch.randn(1, 1, 5120)  # Batch size, Channels, Length
    t = torch.tensor([0])  # Dummy time step for demonstration

    # Forward pass
    noisy_signal = model.diffusion(x, t)
    outputs = model(noisy_signal, t)
    loss = criterion(outputs, x)
    print(f"Loss: {loss.item()}")

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
