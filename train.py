import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from dataset import NoisyHeartbeatDataset
from models.wave_u_net import WaveUNet
from randomizer import NumpyRandomShuffleRandomizer
from sampling_rate_converter import ScipySamplingRateConverter
from utils.notificator import (
    send_discord_notification,
    send_discord_notification_on_error,
)


send_discord_notification_on_error()

model = WaveUNet()

model.train()

train_dataset = NoisyHeartbeatDataset(
    clean_file_path="data/Stop.mat",
    noisy_file_path="data/100km.mat",
    sampling_rate_converter=ScipySamplingRateConverter(
        input_rate=32000, output_rate=1024
    ),
    randomizer=NumpyRandomShuffleRandomizer(),
    train=True,
)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5

for epoch in range(num_epochs):
    for noisy, clean in train_dataloader:
        optimizer.zero_grad()
        outputs = model(noisy)
        loss = criterion(outputs, clean)
        loss.backward()
        optimizer.step()

        # print(loss)

    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")  # type: ignore
    torch.save(model.state_dict(), f"checkpoints/model_weights_epoch_{epoch + 1}.pth")

send_discord_notification("Finished training.")
