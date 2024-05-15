import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset.dataset import NoisyHeartbeatDataset
from dataset.randomizer import NumpyRandomShuffleRandomizer
from dataset.sampling_rate_converter import ScipySamplingRateConverter
from models.wave_u_net import WaveUNet
from utils.device import get_device_name
from utils.plot import plot_three_signals
from utils.sound import save_signal_to_wav_scipy
from utils.timeit import timeit
from numpy.typing import ArrayLike


@timeit
def eval_model(
    model: nn.Module,
    state_dict_path: str,
    dataloader: DataLoader,
    criterion: nn.Module | None = None,
    *,
    figure_filename: str | None = None,
    clean_audio_filename: str | None = None,
    noisy_audio_filename: str | None = None,
    audio_filename: str | None = None,
):
    model.load_state_dict(torch.load(state_dict_path))
    device = get_device_name()
    model.to(device)

    model.eval()

    with torch.no_grad():
        for noisy, clean in dataloader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            outputs = model(noisy)

            if criterion is not None:
                loss = criterion(outputs, clean)
                print(f"Loss: {loss.item()}")

            cpu_noisy = noisy[0][0].cpu().numpy()
            cpu_clean = clean[0][0].cpu().numpy()
            cpu_outputs = outputs[0][0].cpu().numpy()

            save_figures(
                figure_filename,
                cpu_noisy,
                cpu_clean,
                cpu_outputs,
            )

            save_audio_files(
                clean_audio_filename,
                noisy_audio_filename,
                audio_filename,
                cpu_noisy,
                cpu_clean,
                cpu_outputs,
            )
            break


def save_figures(
    figure_filename: str | None,
    cpu_noisy: ArrayLike,
    cpu_clean: ArrayLike,
    cpu_outputs: ArrayLike,
):
    if figure_filename is not None:
        plot_three_signals(
            cpu_noisy,
            cpu_clean,
            cpu_outputs,
            "Noisy",
            "Clean",
            "Output",
            filename=figure_filename,
        )


def save_audio_files(
    clean_audio_filename: str | None,
    noisy_audio_filename: str | None,
    audio_filename: str | None,
    cpu_noisy: ArrayLike,
    cpu_clean: ArrayLike,
    cpu_outputs: ArrayLike,
):
    sample_rate = 1024

    if clean_audio_filename is not None:
        save_signal_to_wav_scipy(
            cpu_clean,
            filename=clean_audio_filename,
            sample_rate=sample_rate,
        )

    if noisy_audio_filename is not None:
        save_signal_to_wav_scipy(
            cpu_noisy,
            filename=noisy_audio_filename,
            sample_rate=sample_rate,
        )

    if audio_filename is not None:
        save_signal_to_wav_scipy(
            cpu_outputs,
            filename=audio_filename,
            sample_rate=sample_rate,
        )  # 最初のバッチのみ処理


if __name__ == "__main__":
    test_dataset = NoisyHeartbeatDataset(
        clean_file_path="data/Stop.mat",
        noisy_file_path="data/100km.mat",
        sampling_rate_converter=ScipySamplingRateConverter(
            input_rate=32000, output_rate=1024
        ),
        randomizer=NumpyRandomShuffleRandomizer(),
        train=False,
    )

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    eval_model(
        WaveUNet(),
        "output/checkpoint/2024-05-11/model_weights_epoch_5.pth",
        dataloader=test_dataloader,
        criterion=nn.L1Loss(),
    )
