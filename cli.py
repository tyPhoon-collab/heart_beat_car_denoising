"""
Model Training and Evaluation CLI

Usage:
    Training the model:
    python cli.py train --model Conv1DAutoencoder --loss-fn SmoothL1Loss --checkpoint-dir <path-to-checkpoint-dir>

    Evaluating the model:
    python cli.py eval --model Conv1DAutoencoder --loss-fn SmoothL1Loss --weights-path <path-to-weights-file>
"""

import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from dataset.factory import DataLoaderFactory, DatasetFactory
from train import train_model
from utils.device import load_local_dotenv
from utils.model_saver import WithDateModelSaver
from logger.training_logger_factory import TrainingLoggerFactory
from models.auto_encoder import Conv1DAutoencoder
from models.wave_u_net import WaveUNet
from eval import eval_model


def get_model(model_name):
    match model_name:
        case "WaveUNet":
            return WaveUNet()
        case "Conv1DAutoencoder":
            return Conv1DAutoencoder()
        case _:
            raise ValueError(f"Unknown model: {model_name}")


def get_loss_function(loss_fn_name):
    match loss_fn_name:
        case "L1Loss":
            return nn.L1Loss()
        case "SmoothL1Loss":
            return nn.SmoothL1Loss()
        case _:
            raise ValueError(f"Unknown loss function: {loss_fn_name}")


def train(args):
    load_local_dotenv()

    # ロガーとモデルセーバーの準備
    logger = TrainingLoggerFactory.remote()
    model_saver = WithDateModelSaver(base_directory=args.checkpoint_dir)

    # モデルと損失関数の選択
    model = get_model(args.model)
    criterion = get_loss_function(args.loss_fn)

    # データセットとデータローダーの準備
    train_dataset = DatasetFactory.create_train()
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

    # モデルの重みのロード
    if args.weights_path:
        model.load_state_dict(torch.load(args.weights_path))

    # データセットとデータローダーの準備
    test_dataset = DatasetFactory.create_test()
    test_dataloader = DataLoaderFactory.create_test(
        test_dataset, batch_size=args.batch_size
    )

    # モデルの評価
    eval_model(
        model,
        args.weights_path,
        test_dataloader,
        criterion,
        figure_filename=args.figure_filename,
        clean_audio_filename=args.clean_audio_filename,
        noisy_audio_filename=args.noisy_audio_filename,
        audio_filename=args.audio_filename,
    )


def main():
    parser = argparse.ArgumentParser(description="Model Training and Evaluation CLI")
    subparsers = parser.add_subparsers(help="sub-command help")

    # トレーニングサブコマンド
    parser_train = subparsers.add_parser("train", help="Train the model")
    parser_train.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["WaveUNet", "Conv1DAutoencoder"],
        help="Model name",
    )
    parser_train.add_argument(
        "--loss-fn",
        type=str,
        required=True,
        choices=["L1Loss", "SmoothL1Loss"],
        help="Loss function",
    )
    parser_train.add_argument("--batch-size", type=int, default=1, help="Batch size")
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
    parser_train.set_defaults(func=train)

    # 推論サブコマンド
    parser_eval = subparsers.add_parser("eval", help="Evaluate the model")
    parser_eval.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["WaveUNet", "Conv1DAutoencoder"],
        help="Model name",
    )
    parser_eval.add_argument(
        "--loss-fn",
        type=str,
        required=True,
        choices=["L1Loss", "SmoothL1Loss"],
        help="Loss function",
    )
    parser_eval.add_argument("--batch-size", type=int, default=1, help="Batch size")
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
