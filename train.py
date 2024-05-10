from logging import warn
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from dataset.dataset import NoisyHeartbeatDataset
from dataset.randomizer import NumpyRandomShuffleRandomizer
from dataset.sampling_rate_converter import ScipySamplingRateConverter
from logger.training_logger import NoopLogger, TrainingLogger
from logger.impls.composite import CompositeLogger
from logger.impls.discord import DiscordLogger
from logger.impls.neptune import NeptuneLogger
from utils.timeit import timeit
from models.wave_u_net import WaveUNet
from dotenv import load_dotenv


def __build_logger() -> TrainingLogger | None:
    enable_logging = os.getenv("LOGGING", "0")  # Default to logging disabled
    if enable_logging == "0":
        warn(
            "Logging is disabled. If you want to enable logging, set LOGGING=1 in .env"
        )
        return None
    return CompositeLogger([NeptuneLogger(), DiscordLogger()])


@timeit
def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    *,
    logger: TrainingLogger | None,
    num_epochs: int = 5,
):
    logger = logger or NoopLogger()
    logger.on_start()

    model.train()

    for epoch in range(num_epochs):
        for noisy, clean in dataloader:
            optimizer.zero_grad()
            outputs = model(noisy)
            loss = criterion(outputs, clean)
            loss.backward()
            optimizer.step()

            logger.on_batch_end(epoch, loss)

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")  # type: ignore
        torch.save(
            model.state_dict(),
            f"checkpoints/model_weights_epoch_{epoch + 1}.pth",
        )
        logger.on_epoch_end(epoch, loss)  # type: ignore

    logger.on_finish()


if __name__ == "__main__":
    load_dotenv()

    logger = __build_logger()

    model = WaveUNet()

    train_dataset = NoisyHeartbeatDataset(
        clean_file_path="data/Stop.mat",
        noisy_file_path="data/100km.mat",
        sampling_rate_converter=ScipySamplingRateConverter(
            input_rate=32000, output_rate=1024
        ),
        randomizer=NumpyRandomShuffleRandomizer(),
        train=True,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
    )

    train_model(
        model,
        train_dataloader,
        nn.L1Loss(),
        optim.Adam(model.parameters(), lr=0.001),
        logger=logger,
        num_epochs=5,
    )
