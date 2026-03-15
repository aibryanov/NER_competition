import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


def build_embedding_matrix(
    word_to_id: Dict[str, int],
    fasttext_model,
    embedding_dim: int,
) -> np.ndarray:
    word_embeds = np.random.uniform(
        -np.sqrt(0.06),
        np.sqrt(0.06),
        (len(word_to_id), embedding_dim),
    ).astype(np.float32)

    loaded = 0
    for word, index in word_to_id.items():
        if word.startswith("<") and word.endswith(">"):
            continue
        word_embeds[index] = fasttext_model.get_word_vector(word)
        loaded += 1

    logger.info("Loaded %d pretrained embeddings", loaded)
    return word_embeds
