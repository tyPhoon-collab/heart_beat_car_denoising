"""
Model Training and Evaluation CLI

Usage:
    Training the model:
    python cli.py train --model Conv1DAutoencoder --loss-fn SmoothL1Loss --checkpoint-dir <path-to-checkpoint-dir>

    Evaluating the model:
    python cli.py eval --model Conv1DAutoencoder --loss-fn SmoothL1Loss --weights-path <path-to-weights-file> [--figure-filename <figure-path>] [--clean-audio-filename <clean-audio-path>] [--noisy-audio-filename <noisy-audio-path>] [--audio-filename <output-audio-path>]
"""

import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from dataset.factory import DataLoaderFactory, DatasetFactory
from dataset.randomizer import SampleShuffleRandomizer, PhaseShuffleRandomizer
from logger.evaluation_impls.audio import AudioEvaluationLogger
from logger.evaluation_impls.composite import CompositeEvaluationLogger
from logger.evaluation_impls.figure import FigureEvaluationLogger
from loss.combine import CombinedLoss
from models.pixel_shuffle_auto_encoder import PixelShuffleConv1DAutoencoder
from models.transformer_pixel_shuffle_auto_encoder import (
    PixelShuffleConv1DAutoencoderWithTransformer,
)
from train import train_model
from utils.device import load_local_dotenv
from utils.model_saver import WithDateModelSaver, WithIdModelSaver
from logger.training_logger_factory import TrainingLoggerFactory
from models.auto_encoder import Conv1DAutoencoder
from models.wave_u_net import WaveUNet
from eval import eval_model

MODEL = [
    WaveUNet,
    Conv1DAutoencoder,
    PixelShuffleConv1DAutoencoder,
    PixelShuffleConv1DAutoencoderWithTransformer,
]
LOSS_FN = [nn.L1Loss, nn.SmoothL1Loss, CombinedLoss]
RANDOMIZER = [SampleShuffleRandomizer, PhaseShuffleRandomizer]


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


def get_randomizer(randomizer_name: str) -> nn.Module:
    randomizer_dict = {randomizer.__name__: randomizer for randomizer in RANDOMIZER}
    if randomizer_name in randomizer_dict:
        return randomizer_dict[randomizer_name]()
    else:
        raise ValueError(f"Unknown randomizer: {randomizer_name}")


def get_model_names() -> list[str]:
    return [model.__name__ for model in MODEL]


def get_loss_function_names() -> list[str]:
    return [loss_fn.__name__ for loss_fn in LOSS_FN]


def get_randomizer_names() -> list[str]:
    return [randomizer.__name__ for randomizer in RANDOMIZER]


def add_common_arguments(parser):
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=get_model_names(),
        help="Model name",
    )
    parser.add_argument(
        "--loss-fn",
        type=str,
        required=True,
        choices=get_loss_function_names(),
        help="Loss function",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--randomizer",
        type=str,
        default="PhaseShuffleRandomizer",
        choices=get_randomizer_names(),
        help="Randomizer",
    )


def train(args):
    load_local_dotenv()

    # ロガーとモデルセーバーの準備
    logger = TrainingLoggerFactory.remote()

    model_id = args.model_id
    model_saver = (
        WithDateModelSaver(base_directory=args.checkpoint_dir)
        if model_id is None
        else WithIdModelSaver(base_directory=args.checkpoint_dir, id=model_id)
    )

    # モデルと損失関数の選択
    model = get_model(args.model)
    criterion = get_loss_function(args.loss_fn)
    randomizer = get_randomizer(args.randomizer)

    # データセットとデータローダーの準備
    train_dataset = DatasetFactory.create_train(randomizer=randomizer)
    train_dataloader = DataLoaderFactory.create_train(
        train_dataset, batch_size=args.batch_size
    )
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # モデルの訓練
    train_model(
        model,
        train_dataloader,
        criterion,
        optimizer,
        model_saver=model_saver,
        logger=logger,
        epoch_size=args.epoch_size,
    )


def evaluate(args):
    # モデルと損失関数の選択
    model = get_model(args.model)
    criterion = get_loss_function(args.loss_fn)
    randomizer = get_randomizer(args.randomizer)

    # モデルの重みのロード
    if args.weights_path:
        model.load_state_dict(torch.load(args.weights_path))

    # データセットとデータローダーの準備
    test_dataset = DatasetFactory.create_test(randomizer=randomizer)
    test_dataloader = DataLoaderFactory.create_test(
        test_dataset, batch_size=args.batch_size
    )

    # モデルの評価
    eval_model(
        model,
        args.weights_path,
        test_dataloader,
        criterion,
        logger=CompositeEvaluationLogger(
            loggers=[
                FigureEvaluationLogger(filename=args.figure_filename),
                AudioEvaluationLogger(
                    sample_rate=test_dataset.sample_rate(),
                    audio_filename=args.audio_filename,
                    clean_audio_filename=args.clean_audio_filename,
                    noisy_audio_filename=args.noisy_audio_filename,
                ),
            ]
        ),
    )


def main():
    parser = argparse.ArgumentParser(description="Model Training and Evaluation CLI")
    subparsers = parser.add_subparsers(help="sub-command help")

    # トレーニングサブコマンド
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
    parser_train.set_defaults(func=train)

    # 推論サブコマンド
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
