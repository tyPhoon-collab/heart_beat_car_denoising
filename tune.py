from dataclasses import asdict
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler

from cli import prepare_train_data_loaders
from dataset.randomizer import AddUniformNoiseRandomizer
from logger.training_impls.stdout import StdoutTrainingLogger
from loss.weighted import WeightedLoss
from models.wave_u_net_enhance_two_stage_transformer import (
    WaveUNetEnhanceTwoStageTransformer,
)
from solver import SimpleSolver

import torch.optim as optim

from utils.gain_controller import ConstantGainController


def train_various_model(config):
    model = WaveUNetEnhanceTwoStageTransformer(
        time_d_model=config["time_d_model"],
        time_nhead=config["time_nhead"],
    )

    solver = SimpleSolver(model, training_criterion=WeightedLoss())

    randomizer = AddUniformNoiseRandomizer()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["lr"],
    )
    gain_controller = ConstantGainController(gain=config["gain"])
    train_dataloader, val_dataloader = prepare_train_data_loaders(
        5120,
        32,
        config["batch_size"],
        randomizer,
        gain_controller,
    )

    data = solver.train(
        train_dataloader,
        optimizer,
        logger=StdoutTrainingLogger(),
        epoch_size=config["epoch_size"],
        val_dataloader=val_dataloader,
        # pretrained_weights_path=args.pretrained_weights_path,
    )

    return {
        "loss": data.final_loss,
    }


def fit_tuner(param_space: dict):
    tuner = tune.Tuner(
        train_various_model,
        param_space=param_space,
        run_config=train.RunConfig(
            name="test",
        ),
        tune_config=tune.TuneConfig(
            scheduler=ASHAScheduler(metric="loss", mode="min"),
            num_samples=5,
        ),
    )
    result = tuner.fit()
    print(result.get_best_result(metric="loss", mode="min").config)


def test_func(config):
    return {
        "score": config["a"] * 5 + config["b"],
    }


def test_tuner():
    tuner = tune.Tuner(
        test_func,
        param_space={
            "a": tune.grid_search([0, 1, 2, 3]),
            "b": tune.grid_search([0, 5, 10]),
        },
        tune_config=tune.TuneConfig(
            mode="min",
            max_concurrent_trials=1,
        ),
    )
    result = tuner.fit()
    print(result.get_best_result(metric="score", mode="min").config)


if __name__ == "__main__":
    # ray tuneの挙動テスト
    test_tuner()

    # fit_tuner(
    #     {
    #         "time_d_model": tune.grid_search([40]),
    #         "time_nhead": tune.grid_search([20, 40]),
    #         "lr": tune.grid_search([0.000025]),
    #         "batch_size": tune.grid_search([64]),
    #         "epoch_size": tune.grid_search([5]),
    #         "gain": tune.grid_search([0.5, 1]),
    #     }
    # )
