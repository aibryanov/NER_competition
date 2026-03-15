from pathlib import Path
import logging

import fasttext

from src.data.competition import (
    load_competition_test_records,
    load_competition_train_records,
    split_records,
)

logger = logging.getLogger(__name__)


def load_competition_training_splits(
    train_path: str | Path,
    train_size: float = 0.8,
    seed: int = 42,
) -> dict[str, list[dict]]:
    train_records = load_competition_train_records(Path(train_path))
    return split_records(train_records, train_size=train_size, seed=seed)


def load_competition_test_set(test_path: str | Path) -> list[dict]:
    return load_competition_test_records(Path(test_path))


def load_fasttext(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"fastText model was not found: {path}. "
            "Place the embedding file into word_embeddings/ before training."
        )

    logger.info("Loading fastText model from %s", path)
    model = fasttext.load_model(str(path))
    logger.info("fastText loaded | dimension=%d words=%d", model.get_dimension(), len(model.words))
    return model
