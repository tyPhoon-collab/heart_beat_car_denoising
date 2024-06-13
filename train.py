from math import inf
import os
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from dataset.dataset import NoisyHeartbeatDataset
from dataset.factory import DatasetFactory
from dataset.randomizer import SampleShuffleRandomizer
from logger.training_logger import TrainingLogger
from logger.training_logger_factory import TrainingLoggerFactory
from utils.device import get_torch_device, load_local_dotenv
from utils.gain_controller import (
    GainController,
    ProgressiveGainController,
)
from utils.epoch_sensitive import EpochSensitive
from utils.model_saver import ModelSaver, WithDateModelSaver
from utils.plot import plot_signals
from utils.timeit import timeit
from models.legacy.wave_u_net import WaveUNet


def load_pretrained_model(model: nn.Module, path: str) -> nn.Module:
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"Loaded pretrained model from {path}")
    else:
        print(f"Pretrained model path {path} does not exist. Starting from scratch.")
    return model


@timeit
def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    *,
    val_dataloader: DataLoader | None = None,
    model_saver: ModelSaver | None = None,
    logger: TrainingLogger | None = None,
    epoch_size: int = 5,
    pretrained_weights_path: str | None = None,
):
    logger = logger or TrainingLoggerFactory.stdout()

    if isinstance(dataloader.dataset, NoisyHeartbeatDataset):
        dataset = dataloader.dataset
    else:
        raise TypeError("dataset is not an instance of NoisyHeartbeatDataset")

    gain_controller: GainController | None = dataset.gain_controller

    if pretrained_weights_path is not None:
        model = load_pretrained_model(model, pretrained_weights_path)

    device = get_torch_device()
    model.to(device)

    params = {
        "learning_rate": optimizer.param_groups[0]["lr"],
        "weight_decay": optimizer.param_groups[0]["weight_decay"],
        "model_name": model.__class__.__name__,
        "criterion_name": criterion.__class__.__name__,
        "optimizer_name": optimizer.__class__.__name__,
        "device": str(device),
        "batch_size": dataloader.batch_size,
        "epoch_size": epoch_size,
        "gain": str(gain_controller) if gain_controller is not None else None,
        "split_samples": dataset.split_samples,
        "stride_samples": dataset.stride_samples,
        "sample_rate": dataset.sample_rate,
        "init": (
            "pretrained_weights" if pretrained_weights_path is not None else "default"
        ),
    }

    logger.on_start(params)

    model.train()

    if os.getenv("ONLY_FIRST_BATCH") == "1":
        dataloader = [next(iter(dataloader))]  # type: ignore

    def save_model(model: nn.Module, suffix: str):
        if model_saver is not None:
            path = model_saver.save(model, suffix=suffix)
            logger.on_model_saved(path)

    lowest_loss = inf

    for epoch in range(epoch_size):
        if gain_controller is not None and isinstance(gain_controller, EpochSensitive):
            gain_controller.on_start_epoch(epoch)

        for batch in dataloader:
            noisy, clean = map(lambda x: x.to(device), batch)

            optimizer.zero_grad()
            outputs = model(noisy)
            loss = criterion(outputs, clean)
            loss.backward()
            optimizer.step()

            logger.on_batch_end(epoch, loss)

        loss_item = loss.item()  # type: ignore

        if lowest_loss > loss_item:
            lowest_loss = loss_item
            save_model(model, suffix="best")

        # 以下はエポックごとのモデルを保存するコードだが、ファイルサイズが大きくなるので、一旦コメント化
        # save_model(model, suffix=f"epoch_{epoch + 1}")

        if val_dataloader is not None:
            model.eval()

            with torch.no_grad():
                noisy, clean = map(lambda x: x.to(device), next(iter(val_dataloader)))

                outputs = model(noisy)

                cpu_noisy = noisy[0][0].cpu().numpy()
                cpu_clean = clean[0][0].cpu().numpy()
                cpu_outputs = outputs[0][0].cpu().numpy()

                plot_signals(
                    [cpu_noisy, cpu_clean, cpu_outputs],
                    ["Noisy", "Clean", "Output"],
                )

            model.train()

        logger.on_epoch_end(epoch, loss)  # type: ignore

        if val_dataloader is not None:
            plt.clf()

    logger.on_finish()


if __name__ == "__main__":
    load_local_dotenv()

    model_saver = WithDateModelSaver(base_directory="output/checkpoint")
    logger = TrainingLoggerFactory.env()

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
