import os
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from dataset.factory import DatasetFactory
from dataset.randomizer import SampleShuffleRandomizer
from logger.training_logger import Params, TrainingLogger
from logger.training_logger_factory import TrainingLoggerFactory
from utils.device import get_torch_device, load_local_dotenv
from utils.gain_controller import (
    EpochSensitive,
    GainController,
    ProgressiveGainController,
)
from utils.model_saver import ModelSaver, WithDateModelSaver
from utils.timeit import timeit
from models.wave_u_net import WaveUNet


@timeit
def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    *,
    model_saver: ModelSaver | None = None,
    logger: TrainingLogger | None = None,
    epoch_size: int = 5,
):
    logger = logger or TrainingLoggerFactory.noop()
    gain_controller: GainController | None = dataloader.dataset.gain_controller  # type: ignore

    device = get_torch_device()
    model.to(device)

    params = Params(
        learning_rate=optimizer.param_groups[0]["lr"],
        model_name=model.__class__.__name__,
        criterion_name=criterion.__class__.__name__,
        optimizer_name=optimizer.__class__.__name__,
        device_str=str(device),
        batch_size=dataloader.batch_size,
        epoch_size=epoch_size,
        gain=str(gain_controller) if gain_controller is not None else None,
    )

    print(params)

    logger.on_start(params)

    model.train()

    if os.getenv("ONLY_FIRST_BATCH") == "1":
        dataloader = [next(iter(dataloader))]  # type: ignore

    for epoch in range(epoch_size):
        if gain_controller is not None and isinstance(gain_controller, EpochSensitive):
            gain_controller.on_start_epoch(epoch)

        for noisy, clean in dataloader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()
            outputs = model(noisy)
            loss = criterion(outputs, clean)
            loss.backward()
            optimizer.step()

            logger.on_batch_end(epoch, loss)

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")  # type: ignore

        if model_saver is not None:
            model_saver.save(model, suffix=f"epoch_{epoch + 1}")

        logger.on_epoch_end(epoch, loss)  # type: ignore

    logger.on_finish()


if __name__ == "__main__":
    load_local_dotenv()

    model_saver = WithDateModelSaver(base_directory="output/checkpoint")
    logger = TrainingLoggerFactory.remote()

    model = WaveUNet()

    train_dataset = DatasetFactory.create_240219(
        randomizer=SampleShuffleRandomizer(),
        gain_controller=ProgressiveGainController(epoch_to=4, max_gain=1.1),
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
        # nn.L1Loss(),
        nn.SmoothL1Loss(),
        optim.Adam(model.parameters(), lr=0.001),
        model_saver=model_saver,
        logger=logger,
        epoch_size=5,
    )
