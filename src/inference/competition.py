import csv
import logging
from pathlib import Path
from typing import Literal

import torch

from src.checkpoints import (
    apply_saved_runtime_settings,
    extract_mapping,
    extract_model_state_dict,
    is_full_checkpoint,
    load_checkpoint_artifact,
    load_model_state_dict,
    load_mapping_artifact,
)
from src.config import Config, cfg
from src.date_context import (
    infer_date_classifier_path,
    load_date_context_classifier,
    relabel_date_spans_with_classifier,
    train_and_save_date_context_classifier,
)
from src.data.competition import load_competition_train_records, split_records
from src.data.loaders import load_competition_test_set
from src.data.mapping import CHAR_UNK_TOKEN, WORD_UNK_TOKEN, prepare_dataset
from src.data.processing import convert_records_to_bioes, tokenize_with_offsets, zero_digits
from src.rules.regex_extractors import extract_regex_spans
from src.training.chunk_utils import get_chunks
from src.training.eval import evaluate_text_predictor_by_class, format_per_class_metrics_table, token_chunks_to_char_spans
from src.training.tensors import build_char_tensor
from src.training.train import define_model

logger = logging.getLogger(__name__)


def _load_or_train_date_context_classifier(
    checkpoint_path: Path,
    config: Config,
):
    if not config.date_context.enabled:
        return None

    classifier_path = infer_date_classifier_path(checkpoint_path)
    if classifier_path.exists():
        logger.info("Loading date context classifier from %s", classifier_path)
        return load_date_context_classifier(classifier_path)

    logger.info("Date context classifier not found for %s; training a new artifact", checkpoint_path)
    train_records = load_competition_train_records(config.paths.train_data_file)
    train_split = split_records(
        train_records,
        train_size=config.data.train_size,
        seed=config.data.seed,
    )["train"]
    return train_and_save_date_context_classifier(
        train_split,
        path=classifier_path,
        window_tokens=config.date_context.window_tokens,
        confidence_margin=config.date_context.confidence_margin,
    )


def load_model_and_mapping_from_files(
    checkpoint_path: str | Path,
    mapping_path: str | Path,
    config: Config = cfg,
):
    return load_inference_artifacts(
        config=config,
        checkpoint_path=checkpoint_path,
        mapping_path=mapping_path,
    )


def load_inference_artifacts(
    config: Config = cfg,
    checkpoint_path: str | Path | None = None,
    mapping_path: str | Path | None = None,
):
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else config.model_path
    mapping_path = Path(mapping_path) if mapping_path is not None else config.paths.mapping_file
    checkpoint = load_checkpoint_artifact(checkpoint_path, config.device)
    mapping = extract_mapping(checkpoint)

    if is_full_checkpoint(checkpoint):
        apply_saved_runtime_settings(checkpoint, config)
    else:
        logger.info("Legacy checkpoint detected in %s; loading mappings from %s", checkpoint_path, mapping_path)
        mapping = load_mapping_artifact(mapping_path)
        apply_saved_runtime_settings(mapping, config)

    if mapping is None:
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain mappings required for inference")

    model = define_model(mapping, config)
    load_model_state_dict(model, extract_model_state_dict(checkpoint), context=str(checkpoint_path))
    model.date_context_classifier = _load_or_train_date_context_classifier(checkpoint_path, config)
    model.to(config.device)
    return model, mapping


def _normalize_prediction_words(words: list[str], config: Config) -> list[str]:
    normalized_words = [zero_digits(word) if config.preprocessing.zeros else word for word in words]
    if config.preprocessing.lower:
        normalized_words = [word.lower() for word in normalized_words]
    return normalized_words


def build_model_inputs_for_text(model, mapping, text: str, config: Config = cfg):
    tokens = tokenize_with_offsets(text)
    if not tokens:
        return None

    raw_words = [token for token, _, _ in tokens]
    offsets = [(start, end) for _, start, end in tokens]
    normalized_words = _normalize_prediction_words(raw_words, config)

    word_ids = [mapping["word_to_id"].get(word, mapping["word_to_id"][WORD_UNK_TOKEN]) for word in normalized_words]
    chars = [
        [mapping["char_to_id"].get(char, mapping["char_to_id"][CHAR_UNK_TOKEN]) for char in word]
        for word in normalized_words
    ]

    char_tensor, char_lengths, restore_order = build_char_tensor(chars, config.model.char_mode)
    word_tensor = torch.as_tensor(word_ids, dtype=torch.long, device=config.device)
    char_tensor = char_tensor.to(config.device)
    return {
        "tokens": tokens,
        "offsets": offsets,
        "word_tensor": word_tensor,
        "char_tensor": char_tensor,
        "char_lengths": char_lengths,
        "restore_order": restore_order,
    }


def predict_model_spans_for_text(model, mapping, text: str, config: Config = cfg) -> list[tuple[int, int, str]]:
    model_inputs = build_model_inputs_for_text(model, mapping, text, config)
    if model_inputs is None:
        return []

    model.eval()
    with torch.no_grad():
        _, predicted_ids = model(
            model_inputs["word_tensor"],
            model_inputs["char_tensor"],
            model_inputs["char_lengths"],
            model_inputs["restore_order"],
        )

    chunks = get_chunks(predicted_ids, mapping["tag_to_id"])
    spans = token_chunks_to_char_spans(chunks, model_inputs["offsets"])
    unique_spans = sorted(set(spans), key=lambda item: (item[0], item[1], item[2]))
    return unique_spans


def predict_sentence_entity_presence_for_text(model, mapping, text: str, config: Config = cfg) -> dict[str, float]:
    model_inputs = build_model_inputs_for_text(model, mapping, text, config)
    if model_inputs is None or not getattr(model, "sentence_entity_enabled", False):
        return {}

    id_to_entity = mapping.get("id_to_entity")
    if not id_to_entity:
        return {}

    model.eval()
    with torch.no_grad():
        probabilities, _ = model.predict_sentence_entities_from_inputs(
            model_inputs["word_tensor"],
            model_inputs["char_tensor"],
            model_inputs["char_lengths"],
            model_inputs["restore_order"],
        )

    if probabilities is None:
        return {}

    return {
        id_to_entity[index]: float(probability)
        for index, probability in enumerate(probabilities.tolist())
    }


def predict_spans_for_text(model, mapping, text: str, config: Config = cfg) -> list[tuple[int, int, str]]:
    regex_labels = set(config.regex.enabled_labels)
    model_spans = relabel_date_spans_with_classifier(
        text,
        predict_model_spans_for_text(model, mapping, text, config),
        getattr(model, "date_context_classifier", None),
    )
    if not regex_labels:
        return model_spans

    regex_spans = relabel_date_spans_with_classifier(
        text,
        extract_regex_spans(text, config.regex.enabled_labels),
        getattr(model, "date_context_classifier", None),
    )
    filtered_model_spans = [span for span in model_spans if span[2] not in regex_labels]
    combined_spans = sorted(set(filtered_model_spans) | set(regex_spans), key=lambda item: (item[0], item[1], item[2]))
    return combined_spans


def generate_submission(
    config: Config = cfg,
    checkpoint_path: str | Path | None = None,
    mapping_path: str | Path | None = None,
) -> Path:
    model, mapping = load_inference_artifacts(
        config=config,
        checkpoint_path=checkpoint_path,
        mapping_path=mapping_path,
    )

    test_records = load_competition_test_set(config.paths.test_data_file)
    with config.submission_path.open("w", encoding="utf-8", newline="") as submission_file:
        writer = csv.writer(submission_file)
        writer.writerow(["id", "Prediction"])

        for record in test_records:
            prediction = predict_spans_for_text(model, mapping, record["text"], config)
            writer.writerow([record["record_id"], repr(prediction)])

    logger.info("Submission saved to %s", config.submission_path)
    return config.submission_path


def load_labeled_dataset_for_evaluation(
    mapping,
    config: Config = cfg,
    dataset_path: str | Path | None = None,
    split_name: Literal["train", "dev", "all"] = "dev",
):
    source_path = Path(dataset_path) if dataset_path is not None else config.paths.train_data_file
    raw_records = load_competition_train_records(source_path)
    if split_name != "all":
        raw_records = split_records(
            raw_records,
            train_size=config.data.train_size,
            seed=config.data.seed,
        )[split_name]

    processed_records = convert_records_to_bioes(raw_records, replace_digits=config.preprocessing.zeros)
    return prepare_dataset(
        processed_records,
        mapping["word_to_id"],
        mapping["char_to_id"],
        mapping["tag_to_id"],
        lower=config.preprocessing.lower,
    )


def evaluate_checkpoint_by_class(
    config: Config = cfg,
    dataset_path: str | Path | None = None,
    split_name: Literal["train", "dev", "all"] = "dev",
    checkpoint_path: str | Path | None = None,
    mapping_path: str | Path | None = None,
) -> str:
    model, mapping = load_inference_artifacts(
        config=config,
        checkpoint_path=checkpoint_path,
        mapping_path=mapping_path,
    )
    dataset = load_labeled_dataset_for_evaluation(
        mapping=mapping,
        config=config,
        dataset_path=dataset_path,
        split_name=split_name,
    )
    dataset_label = f"{split_name.upper()} evaluation"
    overall_metrics, metrics_by_class = evaluate_text_predictor_by_class(
        datas=dataset,
        mapping=mapping,
        predict_spans=lambda text: predict_spans_for_text(model, mapping, text, config),
        dataset=dataset_label,
    )
    return format_per_class_metrics_table(metrics_by_class, overall_metrics)
