import _pickle as cPickle
import logging
from typing import Any

import numpy as np

from src.config import Config, cfg

logger = logging.getLogger(__name__)

WORD_UNK_TOKEN = "<UNK>"
CHAR_PAD_TOKEN = "<PAD_CHAR>"
CHAR_UNK_TOKEN = "<UNK_CHAR>"


def create_dico(item_list: list[list[str]]) -> dict[str, int]:
    dico: dict[str, int] = {}
    for items in item_list:
        for item in items:
            dico[item] = dico.get(item, 0) + 1
    return dico


def create_mapping(
    dico: dict[str, int],
    special_tokens: list[str] | None = None,
) -> tuple[dict[str, int], dict[int, str]]:
    sorted_items = sorted(dico.items(), key=lambda item: (-item[1], item[0]))
    special_tokens = special_tokens or []

    id_to_item = {index: token for index, token in enumerate(special_tokens)}
    next_index = len(id_to_item)
    seen_items = set(special_tokens)

    for item, _ in sorted_items:
        if item in seen_items:
            continue
        id_to_item[next_index] = item
        next_index += 1

    item_to_id = {value: index for index, value in id_to_item.items()}
    return item_to_id, id_to_item


def to_sentences(dataset_split: list[dict[str, Any]]) -> list[list[tuple[str, str]]]:
    return [list(zip(example["words"], example["labels"])) for example in dataset_split]


def word_mapping(
    sentences: list[list[tuple[str, str]]],
    lower: bool,
) -> tuple[dict[str, int], dict[str, int], dict[int, str]]:
    words = [[word.lower() if lower else word for word, _ in sentence] for sentence in sentences]
    dico = create_dico(words)
    dico[WORD_UNK_TOKEN] = 10_000_000
    word_to_id, id_to_word = create_mapping(dico)
    logger.info("Found %d unique words (%d tokens total)", len(dico), sum(len(sentence) for sentence in words))
    return dico, word_to_id, id_to_word


def char_mapping(
    sentences: list[list[tuple[str, str]]],
) -> tuple[dict[str, int], dict[str, int], dict[int, str]]:
    chars = [[char for word, _ in sentence for char in word] for sentence in sentences]
    dico = create_dico(chars)
    char_to_id, id_to_char = create_mapping(dico, special_tokens=[CHAR_PAD_TOKEN, CHAR_UNK_TOKEN])
    logger.info("Found %d unique characters", len(char_to_id))
    return dico, char_to_id, id_to_char


def tag_mapping(
    sentences: list[list[tuple[str, str]]],
    start_tag: str = cfg.START_TAG,
    stop_tag: str = cfg.STOP_TAG,
) -> tuple[dict[str, int], dict[str, int], dict[int, str]]:
    tags = [[tag for _, tag in sentence] for sentence in sentences]
    dico = create_dico(tags)
    dico[start_tag] = -1
    dico[stop_tag] = -2
    tag_to_id, id_to_tag = create_mapping(dico)
    logger.info("Found %d unique named entity tags", len(dico))
    return dico, tag_to_id, id_to_tag


def extract_entity_type(tag: str) -> str | None:
    if tag in {cfg.START_TAG, cfg.STOP_TAG, "O"} or "-" not in tag:
        return None
    _, entity_type = tag.split("-", maxsplit=1)
    return entity_type


def entity_mapping(
    sentences: list[list[tuple[str, str]]],
) -> tuple[dict[str, int], dict[str, int], dict[int, str]]:
    entity_sequences = [
        [entity_type for _, tag in sentence if (entity_type := extract_entity_type(tag)) is not None]
        for sentence in sentences
    ]
    dico = create_dico(entity_sequences)
    entity_to_id, id_to_entity = create_mapping(dico)
    logger.info("Found %d unique sentence-level entity types", len(entity_to_id))
    return dico, entity_to_id, id_to_entity


def prepare_mappings(dataset: list[dict[str, Any]], lower: bool = True) -> dict[str, dict[Any, Any]]:
    sentences = to_sentences(dataset)
    _, word_to_id, id_to_word = word_mapping(sentences, lower)
    _, char_to_id, id_to_char = char_mapping(sentences)
    _, tag_to_id, id_to_tag = tag_mapping(sentences)
    _, entity_to_id, id_to_entity = entity_mapping(sentences)

    return {
        "word_to_id": word_to_id,
        "id_to_word": id_to_word,
        "char_to_id": char_to_id,
        "id_to_char": id_to_char,
        "tag_to_id": tag_to_id,
        "id_to_tag": id_to_tag,
        "entity_to_id": entity_to_id,
        "id_to_entity": id_to_entity,
    }


def normalize_word(word: str, lower: bool = False) -> str:
    return word.lower() if lower else word


def prepare_dataset(
    records: list[dict[str, Any]],
    word_to_id: dict[str, int],
    char_to_id: dict[str, int],
    tag_to_id: dict[str, int],
    entity_to_id: dict[str, int] | None = None,
    lower: bool = False,
) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = []
    for record in records:
        str_words = list(record["raw_words"])
        words = [word_to_id.get(normalize_word(word, lower), word_to_id[WORD_UNK_TOKEN]) for word in record["words"]]
        chars = [
            [char_to_id.get(char, char_to_id[CHAR_UNK_TOKEN]) for char in word]
            for word in record["words"]
        ]
        tags = [tag_to_id[tag] for tag in record["labels"]]
        entity_targets = []
        if entity_to_id is not None:
            entity_targets = [0.0] * len(entity_to_id)
            for entity_type in {extract_entity_type(tag) for tag in record["labels"]}:
                if entity_type is None or entity_type not in entity_to_id:
                    continue
                entity_targets[entity_to_id[entity_type]] = 1.0

        data.append(
            {
                "record_id": record["record_id"],
                "text": record["text"],
                "offsets": list(record["offsets"]),
                "str_words": str_words,
                "words": words,
                "chars": chars,
                "tags": tags,
                "entity_targets": entity_targets,
            }
        )
    return data


def apply_mapping_to_split(
    dataset: dict[str, list[dict[str, Any]]],
    mappings: dict[str, dict[Any, Any]],
    lower: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_data = prepare_dataset(
        dataset["train"],
        mappings["word_to_id"],
        mappings["char_to_id"],
        mappings["tag_to_id"],
        mappings.get("entity_to_id"),
        lower=lower,
    )
    dev_data = prepare_dataset(
        dataset["dev"],
        mappings["word_to_id"],
        mappings["char_to_id"],
        mappings["tag_to_id"],
        mappings.get("entity_to_id"),
        lower=lower,
    )
    return train_data, dev_data


def save_mappings(
    mappings: dict[str, dict[Any, Any]],
    word_embeds: np.ndarray,
    config: Config = cfg,
) -> dict[str, Any]:
    serialized_mappings = {
        "word_to_id": mappings["word_to_id"],
        "tag_to_id": mappings["tag_to_id"],
        "char_to_id": mappings["char_to_id"],
        "id_to_tag": mappings["id_to_tag"],
        "entity_to_id": mappings["entity_to_id"],
        "id_to_entity": mappings["id_to_entity"],
        "parameters": config.model.model_dump(),
        "config": {
            "model": config.model.model_dump(),
            "preprocessing": config.preprocessing.model_dump(),
            "regex": config.regex.model_dump(),
            "sentence_entities": config.sentence_entities.model_dump(),
            "date_context": config.date_context.model_dump(),
        },
        "word_embs": word_embeds,
    }

    with config.paths.mapping_file.open("wb") as mapping_file:
        cPickle.dump(serialized_mappings, mapping_file)

    logger.info("Mappings saved to %s", config.paths.mapping_file)
    return {**mappings, "word_embs": word_embeds}
