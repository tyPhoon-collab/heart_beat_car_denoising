from matplotlib import pyplot as plt
import torch
from dataset import randomizer
from dataset.factory import DatasetFactory
from models.wave_u_net_enhance_transformer import WaveUNetEnhanceTransformer
from models.wave_u_net_enhance_two_stage_transformer import (
    WaveUNetEnhanceTwoStageTransformer,
)
from torch.utils.data import DataLoader

from utils import gain_controller
from utils.device import get_torch_device
from utils.plot import plot_signals

# from utils.sound import save_signal_to_wav_scipy


models = [
    WaveUNetEnhanceTransformer(),
    WaveUNetEnhanceTwoStageTransformer(),
]

models_state_dicts = [
    "output/checkpoint/single_stage_model_weights_best.pth",
    "output/checkpoint/two_stage_model_weights_best.pth",
]

gains = [0, 0.25, 0.5, 0.75, 1.0]

device = get_torch_device()


def select_batches(dataloader):
    data = list(dataloader)

    n = len(dataloader)
    divisions = 6
    step = n // divisions
    result = [i * step for i in range(divisions)]

    return [data[i] for i in result]


for gain in gains:
    dataset = DatasetFactory.create_240517_filtered(
        randomizer=randomizer.AddUniformNoiseRandomizer(),
        train=False,
        gain_controller=gain_controller.ConstantGainController(gain=gain),
        split_samples=5120,
        stride_samples=5120,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for i, batch in enumerate(select_batches(dataloader)):
        noisy, clean = map(lambda x: x.to(device), batch)
        for model, state in zip(models, models_state_dicts):
            basename = f"gain_{gain}_{model.__class__.__name__}_sample_{i}"

            model.to(device)
            model.eval()
            model.load_state_dict(torch.load(state, map_location=device))
            outputs = model(noisy)
            cpu_noisy = noisy[0][0].detach().cpu().numpy()
            cpu_clean = clean[0][0].detach().cpu().numpy()
            cpu_outputs = outputs[0][0].detach().cpu().numpy()
            plot_signals(
                [cpu_noisy, cpu_clean, cpu_outputs], ["Noisy", "Clean", "Output"]
            )
            plt.savefig(f"output/fig/{basename}.png")

            # save audio
            # sample_rate = 1000
            # save_signal_to_wav_scipy(
            #     cpu_clean,
            #     sample_rate,
            #     f"{basename}_clean.wav",
            # )
            # save_signal_to_wav_scipy(
            #     cpu_noisy,
            #     sample_rate,
            #     f"{basename}_noisy.wav",
            # )
            # save_signal_to_wav_scipy(
            #     cpu_outputs,
            #     sample_rate,
            #     f"{basename}_output.wav",
            # )
