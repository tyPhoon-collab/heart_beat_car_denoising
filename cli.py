import argparse
import torch
import torch.optim as optim
from cli_options import (
    CLIDataFolder,
    CLILossFn,
    CLIModel,
    CLIRandomizer,
    build_cli_data_folder,
    build_cli_loss_fn,
    build_cli_model,
    build_cli_randomizer,
)
from torch.utils.data import DataLoader
from dataset.randomizer import Randomizer
from logger.evaluation_impls.audio import AudioEvaluationLogger
from logger.evaluation_impls.composite import CompositeEvaluationLogger
from logger.evaluation_impls.figure import FigureEvaluationLogger
from logger.evaluation_impls.stdout import StdoutEvaluationLogger
from models.gaussian_diffusion import GaussianDiffusion
from solver import DiffusionSolver, SimpleSolver, Solver
from utils.load_local_dotenv import load_local_dotenv
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


def build_solver(args, model) -> Solver:
    criterion = build_cli_loss_fn(args.loss_fn)

    if isinstance(model, GaussianDiffusion):
        model.set_criterion(criterion)
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


def prepare_data_loader(
    data_folder: CLIDataFolder,
    split_samples: int,
    stride_samples: int,
    batch_size: int,
    train: bool,
    randomizer: Randomizer,
    gain_controller: GainController,
    base_dir: str = "",
):
    dataset = build_cli_data_folder(
        data_folder,
        base_dir=base_dir,
        randomizer=randomizer,
        train=train,
        gain_controller=gain_controller,
        split_samples=split_samples,
        stride_samples=stride_samples,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
    )
    return dataloader


def prepare_train_data_loaders(**kwargs):
    """
    train_dataloaderとval_dataloaderを返す
    パラメータはprepare_data_loaderの引数を参照
    """
    train_dataloader = prepare_data_loader(
        train=True,
        **kwargs,
    )
    val_dataloader = prepare_data_loader(
        train=False,
        **kwargs,
    )
    return train_dataloader, val_dataloader


def train(args):
    load_local_dotenv()
    logger = TrainingLoggerFactory.env()

    model_saver, model_save_validator = prepare_saver(args)
    model = build_cli_model(args)
    randomizer = build_cli_randomizer(args.randomizer)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    gain_controller = prepare_train_gain_controller(args)
    train_dataloader, val_dataloader = prepare_train_data_loaders(
        data_folder=args.data_folder,
        split_samples=args.split_samples,
        stride_samples=args.stride_samples,
        batch_size=args.batch_size,
        randomizer=randomizer,
        gain_controller=gain_controller,
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
        additional_params={
            "data_folder": str(args.data_folder),
        },
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
    model = build_cli_model(args)
    randomizer = build_cli_randomizer(args.randomizer)
    gain_controller = ConstantGainController(gain=args.gain)

    model.load_state_dict(torch.load(args.weights_path))

    if args.split_samples != args.stride_samples:
        print(
            "WARNING: --split-samples and --stride-samples are different. for evaluation, stride is dealt same as split"
        )

    test_dataloader = prepare_data_loader(
        args.data_folder,
        args.split_samples,
        args.split_samples,  # for evaluation, stride same as split
        args.batch_size,
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
                StdoutEvaluationLogger(),
            ]
        ),
    )


def add_common_arguments(parser):
    parser.add_argument(
        "--model",
        type=lambda x: CLIModel[x],
        required=True,
        choices=list(CLIModel),
        help="Model name",
    )
    parser.add_argument(
        "--loss-fn",
        type=lambda x: CLILossFn[x],
        required=True,
        choices=list(CLILossFn),
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
        type=lambda x: CLIRandomizer[x],
        default="AddUniformNoiseRandomizer",
        choices=list(CLIRandomizer),
        help="Randomizer",
    )
    parser.add_argument(
        "--num-encoder-layers",
        type=int,
        default=2,
        help="Number of encoder layers. This only effects for using transformer model",
    )
    parser.add_argument(
        "--data-folder",
        type=lambda x: CLIDataFolder(x),
        default=CLIDataFolder.Raw240826,
        choices=list(CLIDataFolder),
        help="Data folder name. This is not path name. Please see choices in help",
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
