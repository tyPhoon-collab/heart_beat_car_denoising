from abc import abstractmethod
import os
import dotenv
from cli import prepare_train_data_loaders
from cli_options import CLIDataFolder
from dataset.randomizer import AddUniformNoiseRandomizer
from logger.training_impls.stdout import StdoutTrainingLogger
from loss.weighted import WeightedLoss
from parameter_tuning.tune import Loss, Tune
import torch.nn as nn
import torch.optim as optim

from solver import SimpleSolver
from utils.gain_controller import ConstantGainController, ProgressiveGainController


class TuneBase(Tune):
    @abstractmethod
    def build_model(self, config) -> nn.Module:
        pass

    def train(self, config) -> Loss:
        dotenv.load_dotenv()
        WORKING_DIR = os.getenv("RAYTUNE_WORKING_DIR") or ""

        data_folder = CLIDataFolder.Raw240517

        model = self.build_model(config)

        solver = SimpleSolver(model, training_criterion=WeightedLoss())

        randomizer = AddUniformNoiseRandomizer()
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["lr"],
        )
        gain_controller = (
            ProgressiveGainController(
                epoch_index_to=config["epoch_size"] // 2, max_gain=config["gain"]
            )
            if config["with_progressive_gain"]
            else ConstantGainController(gain=config["gain"])
        )
        train_dataloader, val_dataloader = prepare_train_data_loaders(
            data_folder=data_folder,
            split_samples=5120,
            stride_samples=32,
            batch_size=config["batch_size"],
            randomizer=randomizer,
            gain_controller=gain_controller,
            base_dir=WORKING_DIR,
        )

        data = solver.train(
            train_dataloader,
            optimizer,
            logger=StdoutTrainingLogger(),
            epoch_size=config["epoch_size"],
            val_dataloader=val_dataloader,
        )

        return data.final_loss or float("inf")
