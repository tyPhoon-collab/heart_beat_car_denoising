from dataset import NoisyHeartbeatDataset
from models.wave_u_net import WaveUNet
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from randomizer import NumpyRandomShuffleRandomizer
from sampling_rate_converter import ScipySamplingRateConverter

model = WaveUNet()

dataset = NoisyHeartbeatDataset(
    clean_file_path="data/Stop.mat",
    noisy_file_path="data/100km.mat",
    # noisy_file_path="data/Stop.mat",
    sampling_rate_converter=ScipySamplingRateConverter(
        input_rate=32000,
        output_rate=1024,
    ),
    randomizer=NumpyRandomShuffleRandomizer(),
)
dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
)

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5

for epoch in range(num_epochs):
    for noisy, clean in dataloader:
        optimizer.zero_grad()
        outputs = model(noisy)
        loss = criterion(outputs, clean)
        loss.backward()

        optimizer.step()

        # print(loss)

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")  # type: ignore
