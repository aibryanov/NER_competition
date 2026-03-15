import _pickle as cPickle
import logging
from pathlib import Path
from typing import Any

import torch

from src.config import Config

logger = logging.getLogger(__name__)

CHECKPOINT_FORMAT = "full_inference_checkpoint"
CHECKPOINT_VERSION = 1
MAPPING_KEYS = (
    "word_to_id",
    "id_to_word",
    "char_to_id",
    "id_to_char",
    "tag_to_id",
    "id_to_tag",
    "entity_to_id",
    "id_to_entity",
    "word_embs",
)
OPTIONAL_AUXILIARY_STATE_PREFIXES = (
    "sentence_entity_hidden.",
    "sentence_entity_output.",
)


def load_mapping_artifact(mapping_path: Path) -> dict[str, Any]:
    with mapping_path.open("rb") as mapping_file:
        return cPickle.load(mapping_file)


def load_checkpoint_artifact(checkpoint_path: Path, device: torch.device) -> Any:
    return torch.load(checkpoint_path, map_location=device)


def is_full_checkpoint(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("format") == CHECKPOINT_FORMAT and "model_state_dict" in payload


def extract_model_state_dict(payload: Any) -> dict[str, Any]:
    if is_full_checkpoint(payload):
        return payload["model_state_dict"]
    return payload


def extract_mapping(payload: Any) -> dict[str, Any] | None:
    if not is_full_checkpoint(payload):
        return None
    mapping = payload.get("mapping")
    return mapping if isinstance(mapping, dict) else None


def load_model_state_dict(model: torch.nn.Module, state_dict: dict[str, Any], context: str) -> None:
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing_keys = set(incompatible.missing_keys)
    unexpected_keys = set(incompatible.unexpected_keys)

    allowed_missing = {
        key
        for key in missing_keys
        if any(key.startswith(prefix) for prefix in OPTIONAL_AUXILIARY_STATE_PREFIXES)
    }
    allowed_unexpected = {
        key
        for key in unexpected_keys
        if any(key.startswith(prefix) for prefix in OPTIONAL_AUXILIARY_STATE_PREFIXES)
    }

    disallowed_missing = sorted(missing_keys - allowed_missing)
    disallowed_unexpected = sorted(unexpected_keys - allowed_unexpected)
    if disallowed_missing or disallowed_unexpected:
        raise RuntimeError(
            f"Incompatible checkpoint for {context}. "
            f"Missing keys: {disallowed_missing}. Unexpected keys: {disallowed_unexpected}."
        )

    if missing_keys or unexpected_keys:
        logger.info(
            "Loaded checkpoint with optional auxiliary head differences | context=%s missing=%s unexpected=%s",
            context,
            sorted(missing_keys),
            sorted(unexpected_keys),
        )


def apply_saved_runtime_settings(payload: dict[str, Any], config: Config) -> None:
    config_sections = payload.get("config", {})
    if not config_sections:
        legacy_parameters = payload.get("parameters", {})
        config_sections = {"model": legacy_parameters} if legacy_parameters else {}

    for section_name in ("model", "preprocessing", "regex", "sentence_entities"):
        saved_values = config_sections.get(section_name, {})
        if not saved_values:
            continue

        section = getattr(config, section_name)
        for field_name, field_value in saved_values.items():
            if hasattr(section, field_name):
                setattr(section, field_name, field_value)


def build_full_checkpoint(
    model: torch.nn.Module,
    mapping: dict[str, Any],
    config: Config,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "model_state_dict": model.state_dict(),
        "mapping": {key: mapping[key] for key in MAPPING_KEYS if key in mapping},
        "config": {
            "model": config.model.model_dump(),
            "preprocessing": config.preprocessing.model_dump(),
            "regex": config.regex.model_dump(),
            "sentence_entities": config.sentence_entities.model_dump(),
        },
    }

    if metadata:
        checkpoint["metadata"] = metadata

    return checkpoint


def save_full_checkpoint(
    model: torch.nn.Module,
    mapping: dict[str, Any],
    config: Config,
    metadata: dict[str, Any] | None = None,
) -> None:
    checkpoint = build_full_checkpoint(model, mapping, config, metadata=metadata)
    torch.save(checkpoint, config.model_path)
    logger.info("Saved full checkpoint to %s", config.model_path)
