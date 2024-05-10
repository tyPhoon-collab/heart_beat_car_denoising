from dotenv import load_dotenv
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from dataset import NoisyHeartbeatDataset
from models.wave_u_net import WaveUNet
from randomizer import NumpyRandomShuffleRandomizer
from sampling_rate_converter import ScipySamplingRateConverter
from utils.training_logger_impls.composite import CompositeLogger
from utils.training_logger_impls.discord import DiscordLogger
from utils.training_logger_impls.neptune import NeptuneLogger
from utils.training_logger import TrainingLogger


load_dotenv()

logger: TrainingLogger = CompositeLogger([NeptuneLogger(), DiscordLogger()])

logger.on_start()

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

        logger.on_batch_end(epoch, loss)
        # print(loss)

    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")  # type: ignore
    torch.save(model.state_dict(), f"checkpoints/model_weights_epoch_{epoch + 1}.pth")
    logger.on_epoch_end(epoch, loss)  # type: ignore

logger.on_finish()
