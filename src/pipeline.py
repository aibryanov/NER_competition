from dataclasses import dataclass
import logging
from typing import Any

import numpy as np

from models.model import BiLSTM_CRF
from src.config import Config
from src.data.embeddings_utils import build_embedding_matrix
from src.data.loaders import load_competition_training_splits, load_fasttext
from src.data.mapping import apply_mapping_to_split, prepare_mappings, save_mappings
from src.data.processing import convert_split_to_bioes
from src.training.train import define_model

logger = logging.getLogger(__name__)


@dataclass
class TrainingArtifacts:
    mapping: dict[str, Any]
    train_data: list[dict[str, Any]]
    dev_data: list[dict[str, Any]]
    word_embeddings: np.ndarray
    model: BiLSTM_CRF


def build_training_artifacts(config: Config) -> TrainingArtifacts:
    logger.info("Loading competition dataset from %s", config.paths.train_data_file)
    raw_splits = load_competition_training_splits(
        train_path=config.paths.train_data_file,
        train_size=config.data.train_size,
        seed=config.data.seed,
    )
    processed_splits = convert_split_to_bioes(
        raw_splits,
        replace_digits=config.preprocessing.zeros,
    )
    logger.info(
        "Prepared dataset splits | train=%d dev=%d",
        len(processed_splits["train"]),
        len(processed_splits["dev"]),
    )

    mapping = prepare_mappings(processed_splits["train"], lower=config.preprocessing.lower)
    train_data, dev_data = apply_mapping_to_split(
        processed_splits,
        mapping,
        lower=config.preprocessing.lower,
    )

    fasttext_model = load_fasttext(config.fasttext_path)
    word_embeddings = build_embedding_matrix(
        mapping["word_to_id"],
        fasttext_model,
        embedding_dim=config.model.word_dim,
    )
    mapping = save_mappings(mapping, word_embeddings, config=config)

    model = define_model(mapping, config)
    if config.model.freeze_word_embeddings:
        for parameter in model.word_embeds.parameters():
            parameter.requires_grad = False
        logger.info("Word embeddings are frozen")

    return TrainingArtifacts(
        mapping=mapping,
        train_data=train_data,
        dev_data=dev_data,
        word_embeddings=word_embeddings,
        model=model,
    )


def print_run_summary(artifacts: TrainingArtifacts, config: Config) -> None:
    mapping = artifacts.mapping
    logger.info(
        "Run summary | device=%s model=%s fasttext=%s vocab=%d chars=%d tags=%d entity_types=%d "
        "train=%d dev=%d sentence_entity_head=%s",
        config.device,
        config.model_path,
        config.fasttext_path,
        len(mapping["word_to_id"]),
        len(mapping["char_to_id"]),
        len(mapping["tag_to_id"]),
        len(mapping.get("entity_to_id", {})),
        len(artifacts.train_data),
        len(artifacts.dev_data),
        config.sentence_entities.enabled,
    )
