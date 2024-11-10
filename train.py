from cli import prepare_train_data_loaders
from cli_options import DataFolder
from dataset.randomizer import AddUniformNoiseRandomizer
from logger.training_logger_factory import TrainingLoggerFactory
from loss.weighted import WeightedLoss
from models.wave_u_net_enhance_transformer import WaveUNetEnhanceTransformer
import torch.optim as optim
from solver import SimpleSolver
from utils.gain_controller import ConstantGainController
from utils.model_save_validator import (
    AnyCompositeModelSaveValidator,
    BestModelSaveValidator,
    SpecificEpochModelSaveValidator,
)
from utils.model_saver import SimpleModelSaver

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
    data_folder=DataFolder.Raw240826,
    split_samples=5120,
    stride_samples=32,
    batch_size=64,
    randomizer=randomizer,
    gain_controller=gain_controller,
)

epoch_size = 100

solver.train(
    train_dataloader,
    optimizer,
    logger=TrainingLoggerFactory.stdout(),
    model_saver=SimpleModelSaver(base_directory="output/checkpoint"),
    model_save_validator=AnyCompositeModelSaveValidator(
        validators=[
            BestModelSaveValidator(epoch_index_from=0),
            SpecificEpochModelSaveValidator.last(epoch_size),
        ]
    ),
    val_dataloader=val_dataloader,
    epoch_size=epoch_size,
)
