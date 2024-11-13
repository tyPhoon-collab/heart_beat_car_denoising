from dataclasses import dataclass, field

from omegaconf import DictConfig


@dataclass(frozen=True)
class TrainConfig:
    id: str | None = None
    lr: float = 0.001
    batch: int = 16
    epoch: int = 5
    checkpoint_path: str = "output/checkpoint"

    pretrained_weight_path: str | None = None
    weight_decay: float = 0

    progressive_gain: bool = False
    progressive_end_epoch: int = 3
    progressive_min_gain: float = 0


@dataclass(frozen=True)
class EvalConfig:
    batch: int = 16
    weight_path: str | None = None
    figure_filename: str | None = None
    clean_audio_filename: str | None = None
    noisy_audio_filename: str | None = None
    audio_filename: str | None = None
    html_filename: str | None = None


@dataclass(frozen=True)
class LoggingConfig:
    remote: bool = False
    stdout: bool = True
    discord: bool = True
    neptune: bool = True
    neptune_save_model_state: bool = True


@dataclass(frozen=True)
class SecretConfig:
    discord_webhook_url: str | None = None
    neptune_project_name: str | None = None
    neptune_api_token: str | None = None
    raytune_working_dir: str | None = None


@dataclass(frozen=True)
class DataConfig:
    name: str = "Raw240517"
    gain: float = 1
    stride: int = 32
    split: int = 5120


@dataclass(frozen=True)
class DebugConfig:
    only_first_batch: bool = False


@dataclass(frozen=True)
class Config:
    mode: str = "train"

    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    model: DictConfig = field(default_factory=lambda: DictConfig({}))
    loss_fn: DictConfig = field(default_factory=lambda: DictConfig({}))
    randomizer: DictConfig = field(default_factory=lambda: DictConfig({}))
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    secret: SecretConfig = field(default_factory=SecretConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
