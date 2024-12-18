from abc import ABC, abstractmethod
import os
from typing import Any
from attr import dataclass
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from config import Config
from dataset.dataset import NoisyHeartbeatDataset
from logger.evaluation_impls.noop import NoopEvaluationLogger
from logger.training_logger import TrainingLogger
from logger.training_logger_factory import TrainingLoggerFactory
from logger.evaluation_logger import EvaluationLogger
from models.gaussian_diffusion import GaussianDiffusion
from utils.context_manager import change_to_eval_mode_temporary
from utils.device import get_torch_device
from utils.epoch_sensitive import EpochSensitive
from utils.gain_controller import GainController
from utils.model_save_validator import ModelSaveValidator
from utils.model_saver import ModelSaver
from plot.plot_plt import plot_signals
from utils.timeit import timeit
from hydra.utils import instantiate


@dataclass
class TrainResult:
    model: nn.Module
    final_loss: float | None


class Solver(ABC):
    def __init__(self, model: nn.Module):
        self.model = model
        self.device = get_torch_device()
        self.model.to(self.device)

    @abstractmethod
    def train(
        self,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        *,
        val_dataloader: DataLoader | None = None,
        model_saver: ModelSaver | None = None,
        model_save_validator: ModelSaveValidator | None = None,
        logger: TrainingLogger | None = None,
        epoch_size: int = 5,
        pretrained_weights_path: str | None = None,
        additional_params: dict[str, Any] | None = None,
    ) -> TrainResult:
        pass

    @abstractmethod
    def evaluate(
        self,
        dataloader: DataLoader,
        criterion: nn.Module | None = None,
        *,
        state_dict_path: str | None = None,
        logger: EvaluationLogger | None = None,
    ):
        pass


# TODO remove this. convert to PyTorch Lightning?
class SolverFactory:
    @staticmethod
    def config(c: Config, model: nn.Module) -> Solver:
        solver: Solver

        criterion = instantiate(c.loss_fn)

        if isinstance(model, GaussianDiffusion):
            model.set_criterion(criterion)
            solver = DiffusionSolver(model, c.debug.only_first_batch)
        else:
            solver = SimpleSolver(model, criterion, c.debug.only_first_batch)
        return solver


class BaseSolver(Solver):
    def __init__(self, model: nn.Module, only_first_batch: bool = False):
        super().__init__(model)
        self.only_first_batch = only_first_batch

    def _load_pretrained_model(self, path: str) -> nn.Module:
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
            print(f"Loaded pretrained model from {path}")
        else:
            print(
                f"Pretrained model path {path} does not exist. Starting from scratch."
            )
        return self.model

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

    def _get_epoch_sensitives(self, *args):
        return [arg for arg in args if isinstance(arg, EpochSensitive)]

    def _get_params(
        self,
        optimizer: optim.Optimizer,
        epoch_size: int,
        batch_size: int,
        gain_controller: GainController | None,
        dataset: NoisyHeartbeatDataset,
        pretrained_weights_path: str | None,
        additional_params: dict[str, Any] | None = None,
    ):
        optim_params = optimizer.param_groups[0]

        # Prepare training parameters for logging
        params = {
            "learning_rate": optim_params["lr"],
            "weight_decay": optim_params["weight_decay"],
            "model_name": self.model.__class__.__name__,
            "model_description": str(self.model),
            "optimizer_name": optimizer.__class__.__name__,
            "device": str(self.device),
            "batch_size": batch_size,
            "epoch_size": epoch_size,
            "gain": str(gain_controller) if gain_controller is not None else None,
            "split_samples": dataset.split_samples,
            "stride_samples": dataset.stride_samples,
            "sample_rate": dataset.sample_rate,
            "init": "pretrained_weights" if pretrained_weights_path else "default",
            **(additional_params or {}),
        }

        return params

    @timeit
    def train(
        self,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        *,
        val_dataloader: DataLoader | None = None,
        model_saver: ModelSaver | None = None,
        model_save_validator: ModelSaveValidator | None = None,
        logger: TrainingLogger | None = None,
        epoch_size: int = 5,
        pretrained_weights_path: str | None = None,
        additional_params: dict[str, Any] | None = None,
    ) -> TrainResult:
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
            dataset,
            pretrained_weights_path,
            additional_params,
        )
        logger.on_start(params)

        # For debugging
        if self.only_first_batch:
            dataloader = [next(iter(dataloader))]  # type: ignore

        for epoch_index in range(epoch_size):
            for epoch_sensitive in epoch_sensitives:
                epoch_sensitive.on_start_epoch(epoch_index)

            for batch in dataloader:
                optimizer.zero_grad()
                loss = self.calculate_loss(batch)
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
                self.validate_and_plot(val_dataloader)

            # 現在の設計上、プロットしたデータはon_epoch_endに対してグローバルに渡している
            # よって直前のコードでプロットし、後続の処理でplt.clfしている
            logger.on_epoch_end(epoch_index, loss)  # type: ignore

            if val_dataloader:
                plt.clf()

        logger.on_finish()

        return TrainResult(self.model, loss.item())  # type: ignore

    @abstractmethod
    def calculate_loss(self, batch) -> Any:
        pass

    @abstractmethod
    def validate_and_plot(self, val_dataloader: DataLoader):
        pass


class SimpleSolver(BaseSolver):
    def __init__(
        self,
        model: nn.Module,
        training_criterion: nn.Module,
        only_first_batch: bool = False,
    ):
        super().__init__(model, only_first_batch)
        self.training_criterion = training_criterion

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
        criterion = criterion or self.training_criterion

        if state_dict_path:
            self.model.load_state_dict(torch.load(state_dict_path))

        if not isinstance(dataloader.dataset, NoisyHeartbeatDataset):
            raise TypeError("dataset is not an instance of NoisyHeartbeatDataset")

        total_loss = 0.0
        count = 0

        noisy_tensors = []
        clean_tensors = []
        outputs_tensors = []

        with torch.no_grad():
            for batch in dataloader:
                outputs, noisy, clean = self._process_batch(batch)

                loss = criterion(outputs, clean)
                total_loss += loss.item()
                count += 1

                noisy_tensors.append(noisy[0][0].cpu().numpy())
                clean_tensors.append(clean[0][0].cpu().numpy())
                outputs_tensors.append(outputs[0][0].cpu().numpy())

                if self.only_first_batch:
                    print(
                        "Only first batch is processed. Because only_first_batch is True."
                    )
                    break

        concat_noisy = np.concatenate(noisy_tensors)
        concat_clean = np.concatenate(clean_tensors)
        concat_outputs = np.concatenate(outputs_tensors)

        logger.on_data(
            concat_noisy,
            concat_clean,
            concat_outputs,
        )
        logger.on_average_loss(total_loss / count)

    def _process_batch(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ):
        noisy, clean = map(lambda x: x.to(self.device), batch)
        outputs = self.model(noisy)

        return outputs, noisy, clean

    def calculate_loss(self, batch) -> Any:
        outputs, _, clean = self._process_batch(batch)

        return self.training_criterion(outputs, clean)

    def validate_and_plot(self, val_dataloader: DataLoader):
        with change_to_eval_mode_temporary(self.model):
            with torch.no_grad():
                self._plot(*self._process_batch(next(iter(val_dataloader))))


class DiffusionSolver(BaseSolver):
    def __init__(self, model: GaussianDiffusion, only_first_batch: bool = False):
        super().__init__(model, only_first_batch)

    def _get_params(
        self,
        optimizer: optim.Optimizer,
        epoch_size: int,
        batch_size: int,
        gain_controller: GainController | None,
        dataset: NoisyHeartbeatDataset,
        pretrained_weights_path: str | None,
        additional_params: dict[str, Any] | None = None,
    ):
        return {
            **super()._get_params(
                optimizer,
                epoch_size,
                batch_size,
                gain_controller,
                dataset,
                pretrained_weights_path,
                additional_params,
            ),
            "criterion": self.model.criterion.__class__.__name__,
            "timesteps": self.model.num_timesteps,
        }

    def calculate_loss(self, batch) -> Any:
        _, clean = batch
        return self.model(clean.to(self.device))

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

        total_loss = 0.0
        count = 0

        noisy_tensors = []
        clean_tensors = []
        outputs_tensors = []

        with torch.no_grad():
            for batch in dataloader:
                noisy, clean = map(lambda x: x.to(self.device), batch)
                outputs = self.model.evaluate(noisy)

                if criterion:
                    loss = criterion(outputs, clean)
                    total_loss += loss.item()
                    count += 1

                noisy_tensors.append(noisy[0][0].cpu().numpy())
                clean_tensors.append(clean[0][0].cpu().numpy())
                outputs_tensors.append(outputs[0][0].cpu().numpy())

                if self.only_first_batch:
                    print(
                        "Only first batch is processed. Because only_first_batch is True."
                    )
                    break

        concat_noisy = np.concatenate(noisy_tensors)
        concat_clean = np.concatenate(clean_tensors)
        concat_outputs = np.concatenate(outputs_tensors)

        logger.on_data(
            concat_noisy,
            concat_clean,
            concat_outputs,
        )
        if count:
            logger.on_average_loss(total_loss / count)

    def validate_and_plot(self, val_dataloader: DataLoader):
        with change_to_eval_mode_temporary(self.model):
            with torch.no_grad():
                batch = next(iter(val_dataloader))
                noisy, clean = map(lambda x: x.to(self.device), batch)
                outputs = self.model.evaluate(noisy)
                self._plot(outputs, noisy, clean)
