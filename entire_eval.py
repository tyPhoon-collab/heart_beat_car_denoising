from torch.nn import L1Loss
from torch.utils.data import DataLoader
from dataset.factory import DatasetFactory
from logger.evaluation_impls.audio import AudioEvaluationLogger
from logger.evaluation_impls.composite import CompositeEvaluationLogger
from logger.evaluation_impls.figure import FigureEvaluationLogger
from logger.evaluation_impls.plotly import PlotlyEvaluationLogger
from logger.evaluation_impls.stdout import StdoutEvaluationLogger
from loss.weighted import WeightedLoss
from models.wave_u_net_enhance_transformer import WaveUNetEnhanceTransformer
from solver import SimpleSolver
from utils.device import get_torch_device

weights_path = "output/checkpoint/model_weights_best.pth"

device = get_torch_device()

model = WaveUNetEnhanceTransformer()
solver = SimpleSolver(model, WeightedLoss())

model.to(device)

logger = CompositeEvaluationLogger(
    [
        FigureEvaluationLogger(filename="entire_eval.png"),
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

dataset = DatasetFactory.create_240517_entire_noise()
dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
)

solver.evaluate(
    dataloader=dataloader,
    criterion=criterion,
    state_dict_path=weights_path,
    logger=logger,
)
