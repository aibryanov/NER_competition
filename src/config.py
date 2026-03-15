from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_REGEX_LABELS = (
    # "Номер телефона",
    "Сведения об ИНН",
    # "Паспортные данные",
    "Номер банковского счета",
    "Номер карты",
    # "Одноразовые коды",
    "Email",
)


class Paths(BaseModel):
    competition_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "competition")
    embeddings_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "word_embeddings")
    models_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "models")
    mappings_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "mappings")
    logs_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "logs")
    train_data_file: Path = Field(default_factory=lambda: PROJECT_ROOT / "competition" / "train_dataset.tsv")
    test_data_file: Path = Field(default_factory=lambda: PROJECT_ROOT / "competition" / "private_test_dataset.csv")
    mapping_file: Path = Field(default_factory=lambda: PROJECT_ROOT / "mappings" / "mapping.pkl")
    loss_plot_file: Path = Field(default_factory=lambda: PROJECT_ROOT / "models" / "loss_curve.png")

    def get_model_path(self, name: str) -> Path:
        return self.models_dir / name

    def get_fasttext_path(self, dim: int, lang: str) -> Path:
        return self.embeddings_dir / f"fasttext_{lang}_{dim}dim.bin"

    def get_training_log_path(self, name: str) -> Path:
        return self.logs_dir / f"{name}.log"

    def get_submission_path(self, name: str) -> Path:
        return self.models_dir / f"{name}_submission.csv"


class DataConfig(BaseModel):
    train_size: float = Field(0.9, gt=0, lt=1)
    seed: int = 42


class PreprocessingConfig(BaseModel):
    lower: bool = False
    zeros: bool = False


class ModelConfig(BaseModel):
    word_dim: int = Field(50, gt=0)
    word_lstm_dim: int = Field(300, gt=0)
    char_embedding_dim: int = Field(60, gt=0)
    char_cnn_channels: int = Field(30, gt=0)
    char_window_size: int = Field(5, gt=0)
    char_lstm_dim: int = Field(30, gt=0)
    char_mode: Literal["CNN", "LSTM"] = "CNN"
    crf: bool = True
    freeze_word_embeddings: bool = False
    lang: Literal["ru", "en", "de", "fr", "zh"] = "ru"


class TrainingConfig(BaseModel):
    epoch: int = Field(50, ge=1)
    dropout: float = Field(0.05, ge=0, le=1)
    gradient_clip: float = Field(5.0, gt=0)
    weights: str = ""
    name: str = "competition_bilstm_crf"
    optimizer_name: Literal["sgd", "adam", "adamw", "rmsprop"] = "adamw"
    learning_rate: float = Field(0.01, gt=0)
    embedding_learning_rate_scale: float = Field(0.2, gt=0)
    momentum: float = 0.9
    weight_decay: float = Field(1e-4, ge=0)
    scheduler_name: Literal["none", "reduce_on_plateau"] = "reduce_on_plateau"
    scheduler_factor: float = Field(0.5, gt=0, lt=1)
    scheduler_threshold: float = Field(1e-3, ge=0)
    scheduler_min_lr: float = Field(1e-5, gt=0)


class EvaluatingConfig(BaseModel):
    eval_every: int = 10_000
    plot_every: int = 200


class HardwareConfig(BaseModel):
    use_gpu: bool = Field(default_factory=torch.cuda.is_available)


class RegexConfig(BaseModel):
    enabled_labels: list[str] = Field(default_factory=lambda: list(SUPPORTED_REGEX_LABELS))

    @field_validator("enabled_labels")
    @classmethod
    def validate_enabled_labels(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - set(SUPPORTED_REGEX_LABELS))
        if unknown:
            raise ValueError(f"Unsupported regex labels: {unknown}. Supported labels: {list(SUPPORTED_REGEX_LABELS)}")
        return list(dict.fromkeys(value))


class SentenceEntityConfig(BaseModel):
    enabled: bool = False
    hidden_dim: int = Field(128, gt=0)
    pooling: Literal["mean", "max", "mean_max"] = "mean_max"
    loss_weight: float = Field(0.2, ge=0)
    threshold: float = Field(0.5, gt=0, lt=1)


class Config(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    START_TAG: str = "<START>"
    STOP_TAG: str = "<STOP>"

    paths: Paths = Field(default_factory=Paths)
    data: DataConfig = Field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    eval: EvaluatingConfig = Field(default_factory=EvaluatingConfig)
    regex: RegexConfig = Field(default_factory=RegexConfig)
    sentence_entities: SentenceEntityConfig = Field(default_factory=SentenceEntityConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)

    @model_validator(mode="after")
    def create_directories(self) -> "Config":
        self.paths.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.paths.models_dir.mkdir(parents=True, exist_ok=True)
        self.paths.mappings_dir.mkdir(parents=True, exist_ok=True)
        self.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def model_path(self) -> Path:
        return self.paths.get_model_path(self.training.name)

    @property
    def fasttext_path(self) -> Path:
        return self.paths.get_fasttext_path(self.model.word_dim, self.model.lang)

    @property
    def train_log_path(self) -> Path:
        return self.paths.get_training_log_path(self.training.name)

    @property
    def submission_path(self) -> Path:
        return self.paths.get_submission_path(self.training.name)

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if self.hardware.use_gpu else "cpu")


cfg = Config()
