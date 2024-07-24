import os
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

MODELS = [
    WaveUNetEnhanceTransformer(),
    WaveUNetEnhanceTwoStageTransformer(),
]

MODELS_STATE_DICTS = [
    "output/checkpoint/single_stage_model_weights_best.pth",
    "output/checkpoint/two_stage_model_weights_best.pth",
]

GAINS = [0, 0.25, 0.5, 0.75, 1.0]


def select_batches(dataloader):
    data = list(dataloader)

    n = len(dataloader)
    divisions = 6
    step = n // divisions
    result = [i * step for i in range(divisions)]

    return [data[i] for i in result]


def load(model, models_state_dict):
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(models_state_dict, map_location=device))
    return model


device = get_torch_device()

models = [load(model, state) for model, state in zip(MODELS, MODELS_STATE_DICTS)]


for gain in GAINS:
    dataset = DatasetFactory.create_240517_filtered(
        randomizer=randomizer.AddUniformNoiseRandomizer(),
        train=False,
        gain_controller=gain_controller.ConstantGainController(gain=gain),
        split_samples=5120,
        stride_samples=5120,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for i, batch in enumerate(select_batches(dataloader)):
        basename = f"gain_{gain}_sample_{i}"

        noisy, clean = map(lambda x: x.to(device), batch)
        outputs = []

        for model in models:
            output = model(noisy)
            outputs.append(output)

        cpu_noisy = noisy[0][0].detach().cpu().numpy()
        cpu_clean = clean[0][0].detach().cpu().numpy()
        cpu_outputs = map(lambda x: x[0][0].detach().cpu().numpy(), outputs)
        plot_signals(
            [
                cpu_noisy,
                cpu_clean,
                *cpu_outputs,
            ],
            [
                "Noisy",
                "Clean",
                *map(lambda x: f"Output {x.__class__.__name__}", MODELS),
            ],
        )
        directory = f"output/fig/{gain}"
        os.makedirs(directory, exist_ok=True)
        plt.savefig(f"{directory}/{basename}.png")
        plt.close()
