import os
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from dataset.dataset import NoisyHeartbeatDataset
from dataset.randomizer import SampleShuffleRandomizer
from dataset.sampling_rate_converter import ScipySamplingRateConverter
from logger.training_logger import Params, TrainingLogger
from logger.training_logger_factory import TrainingLoggerFactory
from utils.device import get_torch_device, load_local_dotenv
from utils.gain_controller import GainController
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

    params = Params(
        learning_rate=optimizer.param_groups[0]["lr"],
        model_name=model.__class__.__name__,
        criterion_name=criterion.__class__.__name__,
        optimizer_name=optimizer.__class__.__name__,
        batch_size=dataloader.batch_size,
        epoch_size=epoch_size,
    )

    logger.on_start(params)

    device = get_torch_device()
    model.to(device)

    model.train()

    gain_controller: GainController = dataloader.dataset.gain_controller  # type: ignore

    only_first_batch = os.getenv("ONLY_FIRST_BATCH") == "1"

    for epoch in range(epoch_size):
        gain_controller.set_gain_from_epoch(epoch)

        for noisy, clean in dataloader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()
            outputs = model(noisy)
            loss = criterion(outputs, clean)
            loss.backward()
            optimizer.step()

            logger.on_batch_end(epoch, loss)

            if only_first_batch:
                break

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

    train_dataset = NoisyHeartbeatDataset(
        clean_file_path="data/Stop.mat",
        noisy_file_path="data/100km.mat",
        sampling_rate_converter=ScipySamplingRateConverter(
            input_rate=32000, output_rate=1024
        ),
        randomizer=SampleShuffleRandomizer(),
        gain_controller=GainController(epoch_to=4, max_gain=1.1),
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
