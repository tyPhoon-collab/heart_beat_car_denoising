from contextlib import contextmanager
import os
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from dataset.dataset import NoisyHeartbeatDataset
from logger.evaluation_impls.noop import NoopEvaluationLogger
from logger.training_logger import TrainingLogger
from logger.training_logger_factory import TrainingLoggerFactory
from logger.evaluation_logger import EvaluationLogger
from utils.device import get_torch_device, load_local_dotenv
from utils.epoch_sensitive import EpochSensitive
from utils.gain_controller import GainController
from utils.model_save_validator import ModelSaveValidator
from utils.model_saver import ModelSaver
from utils.plot import plot_signals
from utils.timeit import timeit


class SimpleSolver:
    def __init__(self, model: nn.Module):
        self.model = model
        load_local_dotenv()
        self.device = get_torch_device()
        self.model.to(self.device)

    @timeit
    def train(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        *,
        val_dataloader: DataLoader | None = None,
        model_saver: ModelSaver | None = None,
        model_save_validator: ModelSaveValidator | None = None,
        logger: TrainingLogger | None = None,
        epoch_size: int = 5,
        pretrained_weights_path: str | None = None,
    ):
        self.model.train()

        if not isinstance(dataloader.dataset, NoisyHeartbeatDataset):
            raise TypeError("dataset is not an instance of NoisyHeartbeatDataset")

        dataset = dataloader.dataset

        # Load pretrained weights if provided
        if pretrained_weights_path:
            self.model = self._load_pretrained_model(pretrained_weights_path)

        # Set up logger
        logger = logger or TrainingLoggerFactory.stdout()

        # Set up gain controller
        gain_controller = dataset.gain_controller

        # Set up epoch sensitives
        epoch_sensitives = self._get_epoch_sensitives(
            gain_controller,
            model_save_validator,
        )

        params = self._get_params(
            optimizer,
            epoch_size,
            dataloader.batch_size or 0,
            gain_controller,
            criterion,
            dataset,
            pretrained_weights_path,
        )
        logger.on_start(params)

        # For debugging
        if os.getenv("ONLY_FIRST_BATCH") == "1":
            dataloader = [next(iter(dataloader))]  # type: ignore

        for epoch_index in range(epoch_size):
            for epoch_sensitive in epoch_sensitives:
                epoch_sensitive.on_start_epoch(epoch_index)

            for batch in dataloader:
                outputs, _, clean = self._process_batch(batch)

                optimizer.zero_grad()
                loss = criterion(outputs, clean)
                loss.backward()
                optimizer.step()

                logger.on_batch_end(epoch_index, loss)

            if model_saver:
                path = self._save_model_if_needed(
                    loss,  # type: ignore
                    model_saver,
                    model_save_validator,
                )
                if path:
                    logger.on_model_saved(path)

            if val_dataloader:
                self._validate_and_plot(val_dataloader)

            # 現在の設計上、プロットしたデータはon_epoch_endに対してグローバルに渡している
            # よって直前のコードでプロットし、後続の処理でplt.clfしている
            logger.on_epoch_end(epoch_index, loss)  # type: ignore

            if val_dataloader:
                plt.clf()

        logger.on_finish()

    @timeit
    def evaluate(
        self,
        dataloader: DataLoader,
        criterion: nn.Module | None = None,
        *,
        state_dict_path: str | None = None,
        logger: EvaluationLogger | None = None,
    ):
        self.model.eval()
        logger = logger or NoopEvaluationLogger()

        if state_dict_path:
            self.model.load_state_dict(torch.load(state_dict_path))

        if not isinstance(dataloader.dataset, NoisyHeartbeatDataset):
            raise TypeError("dataset is not an instance of NoisyHeartbeatDataset")

        with torch.no_grad():
            for batch in dataloader:
                outputs, noisy, clean = self._process_batch(batch)

                if criterion:
                    loss = criterion(outputs, clean)
                    print(f"Loss: {loss.item()}")

                logger.on_data(
                    noisy[0][0].cpu().numpy(),
                    clean[0][0].cpu().numpy(),
                    outputs[0][0].cpu().numpy(),
                )
                break  # remove this line if you want to process all batches

    def _load_pretrained_model(self, path: str) -> nn.Module:
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
            print(f"Loaded pretrained model from {path}")
        else:
            print(
                f"Pretrained model path {path} does not exist. Starting from scratch."
            )
        return self.model

    def _get_params(
        self,
        optimizer: optim.Optimizer,
        epoch_size: int,
        batch_size: int,
        gain_controller: GainController | None,
        criterion: nn.Module,
        dataset: NoisyHeartbeatDataset,
        pretrained_weights_path: str | None,
    ):
        optim_params = optimizer.param_groups[0]

        # Prepare training parameters for logging
        params = {
            "learning_rate": optim_params["lr"],
            "weight_decay": optim_params["weight_decay"],
            "model_name": self.model.__class__.__name__,
            "criterion_name": criterion.__class__.__name__,
            "optimizer_name": optimizer.__class__.__name__,
            "device": str(self.device),
            "batch_size": batch_size,
            "epoch_size": epoch_size,
            "gain": str(gain_controller) if gain_controller is not None else None,
            "split_samples": dataset.split_samples,
            "stride_samples": dataset.stride_samples,
            "sample_rate": dataset.sample_rate,
            "init": "pretrained_weights" if pretrained_weights_path else "default",
        }

        return params

    def _get_epoch_sensitives(self, *args):
        return [arg for arg in args if isinstance(arg, EpochSensitive)]

    def _process_batch(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ):
        noisy, clean = map(lambda x: x.to(self.device), batch)
        outputs = self.model(noisy)

        return outputs, noisy, clean

    def _plot(self, outputs, noisy, clean):
        cpu_noisy = noisy[0][0].cpu().numpy()
        cpu_clean = clean[0][0].cpu().numpy()
        cpu_outputs = outputs[0][0].cpu().numpy()
        plot_signals([cpu_noisy, cpu_clean, cpu_outputs], ["Noisy", "Clean", "Output"])

    def _save_model_if_needed(
        self,
        loss,
        model_saver: ModelSaver,
        model_save_validator: ModelSaveValidator | None = None,
    ):
        if model_save_validator is None:
            return model_saver.save(self.model)

        elif model_save_validator.validate(loss.item()):
            suffix = model_save_validator.suffix
            return model_saver.save(self.model, suffix=suffix)

    def _validate_and_plot(self, val_dataloader: DataLoader):
        with change_to_eval_mode_temporary(self.model):
            with torch.no_grad():
                self._plot(*self._process_batch(next(iter(val_dataloader))))


@contextmanager
def change_to_eval_mode_temporary(model: nn.Module):
    """Temporarily set the model to evaluation mode and then restore it."""
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        if was_training:
            model.train()
