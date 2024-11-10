import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate

from config import Config
from logger.evaluation_logger_factory import EvaluationLoggerFactory
from logger.training_logger_factory import TrainingLoggerFactory
from models.gaussian_diffusion import GaussianDiffusion
from solver import DiffusionSolver, SimpleSolver, Solver

import torch.optim as optim

from utils.dataloader_factory import DataLoaderFactory
from utils.gain_controller_factory import GainControllerFactory
from utils.saver_factory import ModelSaverFactory


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)


# TODO remove this. convert to PyTorch Lightning?
def create_solver(c: Config, model):
    solver: Solver

    criterion = instantiate(c.loss_fn)

    if isinstance(model, GaussianDiffusion):
        model.set_criterion(criterion)
        solver = DiffusionSolver(model, c.only_first_batch)
    else:
        solver = SimpleSolver(model, criterion, c.only_first_batch)
    return solver


def train(c: Config):
    model_saver, model_save_validator = ModelSaverFactory.config(c.train)

    model = instantiate(c.model)
    randomizer = instantiate(c.randomizer)
    gain_controller = GainControllerFactory.config(c)

    optimizer = optim.Adam(
        model.parameters(),
        lr=c.train.lr,
        weight_decay=c.train.weight_decay,
    )

    train_dataloader, val_dataloader = DataLoaderFactory.config(
        c,
        randomizer=randomizer,
        gain_controller=gain_controller,
    )

    solver = create_solver(c, model)

    solver.train(
        train_dataloader,
        optimizer,
        model_saver=model_saver,
        model_save_validator=model_save_validator,
        logger=TrainingLoggerFactory.config(c.logging, c.secret),
        epoch_size=c.train.epoch,
        val_dataloader=val_dataloader,
        pretrained_weights_path=c.train.pretrained_weight_path,
        additional_params={
            "data_folder": str(c.data),
        },
    )


def eval(c: Config):
    model = instantiate(c.model)
    randomizer = instantiate(c.randomizer)
    gain_controller = GainControllerFactory.config(c)

    if c.split != c.stride:
        print(
            "WARNING: --split-samples and --stride-samples are different. for evaluation, stride is dealt same as split"
        )

    _, test_dataloader = DataLoaderFactory.config(
        c,
        randomizer=randomizer,
        gain_controller=gain_controller,
    )

    solver = create_solver(c, model)

    solver.evaluate(
        test_dataloader,
        state_dict_path=c.eval.weight_path,
        logger=EvaluationLoggerFactory.config(c.logging, c.eval),
    )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Config):
    # from omegaconf import OmegaConf
    # print(OmegaConf.to_yaml(cfg))
    # return

    match cfg.mode:
        case "train":
            train(cfg)
        case "eval":
            eval(cfg)
        case mode:
            raise ValueError(f"Invalid mode: {mode}. Expected 'train' or 'eval'.")


if __name__ == "__main__":
    main()
