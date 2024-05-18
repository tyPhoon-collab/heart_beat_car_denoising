import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset.dataset import NoisyHeartbeatDataset
from dataset.randomizer import SampleShuffleRandomizer
from dataset.sampling_rate_converter import ScipySamplingRateConverter
from logger.evaluation_impls.noop import NoopEvaluationLogger
from logger.evaluation_logger import EvaluationLogger
from models.wave_u_net import WaveUNet
from utils.device import get_torch_device
from utils.timeit import timeit


@timeit
def eval_model(
    model: nn.Module,
    state_dict_path: str,
    dataloader: DataLoader,
    criterion: nn.Module | None = None,
    *,
    logger: EvaluationLogger | None = None,
):
    logger = logger or NoopEvaluationLogger()
    model.load_state_dict(torch.load(state_dict_path))
    device = get_torch_device()
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

            logger.on_data(cpu_noisy, cpu_clean, cpu_outputs)

            break


if __name__ == "__main__":
    test_dataset = NoisyHeartbeatDataset(
        clean_file_path="data/Stop.mat",
        noisy_file_path="data/100km.mat",
        sampling_rate_converter=ScipySamplingRateConverter(
            input_rate=32000, output_rate=1024
        ),
        randomizer=SampleShuffleRandomizer(),
        train=False,
    )

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    eval_model(
        WaveUNet(),
        "output/checkpoint/2024-05-11/model_weights_epoch_5.pth",
        dataloader=test_dataloader,
        criterion=nn.L1Loss(),
    )
