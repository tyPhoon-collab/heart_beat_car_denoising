import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from cli_options import (
    LOSS_FN,
    LOSS_FN_NAMES,
    MODEL,
    MODEL_NAMES,
    RANDOMIZER,
    RANDOMIZER_NAMES,
)
from dataset.factory import DatasetFactory
from dataset.randomizer import Randomizer
from torch.utils.data import DataLoader
from logger.evaluation_impls.audio import AudioEvaluationLogger
from logger.evaluation_impls.composite import CompositeEvaluationLogger
from logger.evaluation_impls.figure import FigureEvaluationLogger
from models.gaussian_diffusion import GaussianDiffusion
from solver import DiffusionSolver, SimpleSolver, Solver
from utils.load import load_local_dotenv
from utils.gain_controller import (
    ConstantGainController,
    GainController,
    ProgressiveGainController,
)
from utils.model_save_validator import (
    AnyCompositeModelSaveValidator,
    BestModelSaveValidator,
    SpecificEpochModelSaveValidator,
)
from utils.model_saver import WithDateModelSaver, WithIdModelSaver
from logger.training_logger_factory import TrainingLoggerFactory


def get_model(model_name: str) -> nn.Module:
    model_dict = {model.__name__: model for model in MODEL}
    if model_name in model_dict:
        return model_dict[model_name]()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_loss_function(loss_fn_name: str) -> nn.Module:
    loss_fn_dict = {loss_fn.__name__: loss_fn for loss_fn in LOSS_FN}
    if loss_fn_name in loss_fn_dict:
        return loss_fn_dict[loss_fn_name]()
    else:
        raise ValueError(f"Unknown loss function: {loss_fn_name}")


def get_randomizer(randomizer_name: str) -> Randomizer:
    randomizer_dict = {randomizer.__name__: randomizer for randomizer in RANDOMIZER}
    if randomizer_name in randomizer_dict:
        return randomizer_dict[randomizer_name]()
    else:
        raise ValueError(f"Unknown randomizer: {randomizer_name}")


def build_solver(args, model) -> Solver:
    criterion = get_loss_function(args.loss_fn)

    if isinstance(model, GaussianDiffusion):
        return DiffusionSolver(model)
    else:
        return SimpleSolver(model, criterion)


def prepare_saver(args):
    model_id = args.model_id
    model_saver = (
        WithDateModelSaver(base_directory=args.checkpoint_dir)
        if model_id is None
        else WithIdModelSaver(base_directory=args.checkpoint_dir, id=model_id)
    )

    best_considered_epoch_from = (
        args.progressive_epoch_to + 1 if args.with_progressive_gain else 1
    )

    model_save_validator = AnyCompositeModelSaveValidator(
        validators=[
            BestModelSaveValidator(epoch_index_from=(best_considered_epoch_from)),
            SpecificEpochModelSaveValidator.last(args.epoch_size),
        ]
    )
    return model_saver, model_save_validator


def prepare_data_loader(args, train, randomizer, gain_controller):
    dataset = DatasetFactory.create_240517_filtered(
        randomizer=randomizer,
        train=train,
        gain_controller=gain_controller,
        split_samples=args.split_samples,
        stride_samples=args.stride_samples,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train,
    )
    return dataloader


def prepare_train_data_loaders(args, randomizer, gain_controller):
    train_dataloader = prepare_data_loader(
        args,
        train=True,
        randomizer=randomizer,
        gain_controller=gain_controller,
    )
    val_dataloader = prepare_data_loader(
        args,
        train=False,
        randomizer=randomizer,
        gain_controller=gain_controller,
    )
    return train_dataloader, val_dataloader


def train(args):
    load_local_dotenv()
    logger = TrainingLoggerFactory.env()

    model_saver, model_save_validator = prepare_saver(args)
    model = get_model(args.model)
    randomizer = get_randomizer(args.randomizer)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    gain_controller = prepare_train_gain_controller(args)
    train_dataloader, val_dataloader = prepare_train_data_loaders(
        args,
        randomizer,
        gain_controller,
    )

    solver = build_solver(args, model)
    solver.train(
        train_dataloader,
        optimizer,
        model_saver=model_saver,
        model_save_validator=model_save_validator,
        logger=logger,
        epoch_size=args.epoch_size,
        val_dataloader=val_dataloader,
        pretrained_weights_path=args.pretrained_weights_path,
    )


def prepare_train_gain_controller(args) -> GainController:
    return (
        ProgressiveGainController(
            epoch_index_to=args.progressive_epoch_to - 1,
            min_gain=args.min_gain,
            max_gain=args.gain,
        )
        if args.with_progressive_gain
        else ConstantGainController(gain=args.gain)
    )


def evaluate(args):
    model = get_model(args.model)
    randomizer = get_randomizer(args.randomizer)
    gain_controller = ConstantGainController(gain=args.gain)

    model.load_state_dict(torch.load(args.weights_path))

    test_dataloader = prepare_data_loader(
        args,
        train=False,
        randomizer=randomizer,
        gain_controller=gain_controller,
    )

    solver = build_solver(args, model)
    solver.evaluate(
        test_dataloader,
        state_dict_path=args.weights_path,
        logger=CompositeEvaluationLogger(
            loggers=[
                FigureEvaluationLogger(filename=args.figure_filename),
                AudioEvaluationLogger(
                    sample_rate=test_dataloader.dataset.sample_rate,  # type: ignore
                    audio_filename=args.audio_filename,
                    clean_audio_filename=args.clean_audio_filename,
                    noisy_audio_filename=args.noisy_audio_filename,
                ),
            ]
        ),
    )


def add_common_arguments(parser):
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=MODEL_NAMES,
        help="Model name",
    )
    parser.add_argument(
        "--loss-fn",
        type=str,
        required=True,
        choices=LOSS_FN_NAMES,
        help="Loss function",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--gain",
        type=float,
        default=1.0,
        help="Gain. If you want to use progressive gain, use --with-progressive-gain",
    )
    parser.add_argument(
        "--stride-samples",
        type=int,
        default=32,
        help="Stride in samples",
    )
    parser.add_argument(
        "--split-samples",
        type=int,
        default=5120,
        help="Split in samples",
    )
    parser.add_argument(
        "--randomizer",
        type=str,
        default="AddUniformNoiseRandomizer",
        choices=RANDOMIZER_NAMES,
        help="Randomizer",
    )


def main():
    parser = argparse.ArgumentParser(description="Model Training and Evaluation CLI")
    subparsers = parser.add_subparsers(help="sub-command help")
    parser_train = subparsers.add_parser("train", help="Train the model")
    add_common_arguments(parser_train)
    parser_train.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser_train.add_argument(
        "--epoch-size",
        type=int,
        default=5,
        help="Number of epochs",
    )
    parser_train.add_argument(
        "--checkpoint-dir",
        type=str,
        default="output/checkpoint",
        help="Checkpoint directory",
    )
    parser_train.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="ID of the model",
    )
    parser_train.add_argument(
        "--with-progressive-gain",
        action="store_true",
        help="Enable progressive gain",
    )
    parser_train.add_argument(
        "--progressive-epoch-to",
        type=int,
        default=5,
        help="Number of epochs to progressive gain. not index. "
        "If --with-progressive-gain is not stored, this option is ignored.",
    )
    parser_train.add_argument(
        "--min-gain",
        type=float,
        default=0,
        help="Minimum gain. If --with-progressive-gain is not stored, this option is ignored.",
    )
    parser_train.add_argument(
        "--weight-decay",
        type=float,
        default=0,
        help="Weight decay",
    )
    parser_train.add_argument(
        "--pretrained-weights-path",
        type=str,
        default=None,
        help="Path to the pretrained weights",
    )
    parser_train.set_defaults(func=train)

    parser_eval = subparsers.add_parser("eval", help="Evaluate the model")
    add_common_arguments(parser_eval)
    parser_eval.add_argument(
        "--weights-path",
        type=str,
        required=True,
        help="Path to the model weights",
    )
    parser_eval.add_argument(
        "--figure-filename",
        type=str,
        default=None,
        help="Filename of the figure",
    )
    parser_eval.add_argument(
        "--clean-audio-filename",
        type=str,
        default=None,
        help="Filename of the clean audio",
    )
    parser_eval.add_argument(
        "--noisy-audio-filename",
        type=str,
        default=None,
        help="Filename of the noisy audio",
    )
    parser_eval.add_argument(
        "--audio-filename",
        type=str,
        default=None,
        help="Filename of the output audio",
    )
    parser_eval.set_defaults(func=evaluate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
