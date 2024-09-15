from cli import prepare_train_data_loaders
from cli_options import CLIDataFolder
from dataset.randomizer import AddUniformNoiseRandomizer
from logger.training_logger_factory import TrainingLoggerFactory
from loss.weighted import WeightedLoss
from models.wave_u_net_enhance_transformer import WaveUNetEnhanceTransformer
import torch.optim as optim
from solver import SimpleSolver
from utils.gain_controller import ConstantGainController

model = WaveUNetEnhanceTransformer()
criterion = WeightedLoss()

solver = SimpleSolver(model, criterion)

randomizer = AddUniformNoiseRandomizer()
optimizer = optim.Adam(
    model.parameters(),
    lr=0.00025,
)
gain_controller = ConstantGainController(gain=1)
train_dataloader, val_dataloader = prepare_train_data_loaders(
    data_folder=CLIDataFolder.Raw240826,
    split_samples=5120,
    stride_samples=32,
    batch_size=64,
    randomizer=randomizer,
    gain_controller=gain_controller,
)

solver.train(
    train_dataloader,
    optimizer,
    logger=TrainingLoggerFactory.stdout(),
    val_dataloader=val_dataloader,
    epoch_size=100,
)
