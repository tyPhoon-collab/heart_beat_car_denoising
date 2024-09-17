import numpy as np
import torch
from torch.nn import L1Loss
from torch.utils.data import DataLoader
from dataset.factory import DatasetFactory
from logger.evaluation_impls.audio import AudioEvaluationLogger
from logger.evaluation_impls.composite import CompositeEvaluationLogger
from logger.evaluation_impls.plotly import PlotlyEvaluationLogger
from logger.evaluation_impls.stdout import StdoutEvaluationLogger
from models.wave_u_net_enhance_transformer import WaveUNetEnhanceTransformer
from utils.device import get_torch_device

weights_path = "output/checkpoint/model_weights_best.pth"

device = get_torch_device()

model = WaveUNetEnhanceTransformer()
model.to(device)
model.load_state_dict(torch.load(weights_path))
model.eval()

logger = CompositeEvaluationLogger(
    [
        # FigureEvaluationLogger(filename="entire_eval.png"),
        PlotlyEvaluationLogger(filename="entire_eval.html"),
        AudioEvaluationLogger(
            sample_rate=1000,
            audio_filename="entire_eval_output.wav",
            clean_audio_filename="entire_eval_clean.wav",
            noisy_audio_filename="entire_eval_noisy.wav",
        ),
        StdoutEvaluationLogger(),
    ]
)

criterion = L1Loss()

total_loss = 0.0
count = 0

noisy_tensors = []
clean_tensors = []
outputs_tensors = []

dataset = DatasetFactory.create_240517_entire_noise()
dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
)

with torch.no_grad():
    for batch in dataloader:
        noisy, clean = map(lambda x: x.to(device), batch)
        outputs = model(noisy)

        loss = criterion(outputs, clean)
        total_loss += loss.item()
        count += 1

        noisy_tensors.append(noisy[0][0].cpu().numpy())
        clean_tensors.append(clean[0][0].cpu().numpy())
        outputs_tensors.append(outputs[0][0].cpu().numpy())

concat_noisy = np.concatenate(noisy_tensors)
concat_clean = np.concatenate(clean_tensors)
concat_outputs = np.concatenate(outputs_tensors)

logger.on_data(
    concat_noisy,
    concat_clean,
    concat_outputs,
)
logger.on_average_loss(total_loss / count)
