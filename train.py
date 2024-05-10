import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from dataset.dataset import NoisyHeartbeatDataset
from dataset.randomizer import NumpyRandomShuffleRandomizer
from dataset.sampling_rate_converter import ScipySamplingRateConverter
from logger.training_logger import TrainingLogger
from utils.timeit import timeit


@timeit
def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    *,
    logger: TrainingLogger | None = None,
    num_epochs: int = 5,
):
    logger = logger or CompositeLogger()
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

            break

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")  # type: ignore
        torch.save(
            model.state_dict(), f"checkpoints/model_weights_epoch_{epoch + 1}.pth"
        )
        logger.on_epoch_end(epoch, loss)  # type: ignore

    logger.on_finish()


if __name__ == "__main__":
    from dotenv import load_dotenv
    from models.wave_u_net import WaveUNet

    # from models.auto_encoder import Conv1DAutoencoder
    from logger.impls.composite import CompositeLogger
    from logger.impls.discord import DiscordLogger
    from logger.impls.neptune import NeptuneLogger

    load_dotenv()

    logger: TrainingLogger = CompositeLogger([NeptuneLogger(), DiscordLogger()])

    model = WaveUNet()
    # model = Conv1DAutoencoder()

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

    train_model(
        model,
        train_dataloader,
        nn.L1Loss(),
        optim.Adam(model.parameters(), lr=0.001),
        logger=logger,
        num_epochs=5,
    )
