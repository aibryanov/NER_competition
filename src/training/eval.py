from collections import defaultdict
from dataclasses import dataclass
import logging

import torch

from src.training.chunk_utils import get_chunks
from src.training.tensors import prepare_sequence_batch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationMetrics:
    precision: float
    recall: float
    f1: float
    token_accuracy: float
    correct_chunks: int
    predicted_chunks: int
    gold_chunks: int


@dataclass(frozen=True)
class PerClassMetrics:
    label: str
    precision: float
    recall: float
    f1: float
    true_positives: int
    predicted: int
    gold: int


def token_chunks_to_char_spans(
    chunks: list[tuple[str, int, int]],
    offsets: list[tuple[int, int]],
) -> list[tuple[int, int, str]]:
    spans = []
    for entity_type, token_start, token_end in chunks:
        if token_start >= token_end:
            continue
        char_start = offsets[token_start][0]
        char_end = offsets[token_end - 1][1]
        spans.append((char_start, char_end, entity_type))
    return spans


@torch.no_grad()
def iter_decoded_batches(model, datas, config):
    for data in datas:
        batch = prepare_sequence_batch(data, config.model.char_mode)
        words = batch.words.to(config.device)
        chars = batch.chars.to(config.device)

        _, predicted_id = model(words, chars, batch.char_lengths, batch.restore_order)
        yield data, predicted_id


@torch.no_grad()
def evaluating(model, datas, best_F, mapping, config, dataset="Train"):
    save = False
    correct_preds, total_correct, total_preds = 0.0, 0.0, 0.0
    correct_tokens, total_tokens = 0.0, 0.0

    for data, predicted_id in iter_decoded_batches(model, datas, config):
        gold_chunks = get_chunks(data["tags"], mapping["tag_to_id"])
        predicted_chunks = get_chunks(predicted_id, mapping["tag_to_id"])

        gold_spans = set(token_chunks_to_char_spans(gold_chunks, data["offsets"]))
        predicted_spans = set(token_chunks_to_char_spans(predicted_chunks, data["offsets"]))

        correct_preds += len(gold_spans & predicted_spans)
        total_preds += len(predicted_spans)
        total_correct += len(gold_spans)
        correct_tokens += sum(int(predicted == gold) for predicted, gold in zip(predicted_id, data["tags"]))
        total_tokens += len(data["tags"])

    precision = correct_preds / total_preds if correct_preds > 0 else 0.0
    recall = correct_preds / total_correct if correct_preds > 0 else 0.0
    new_F = 2 * precision * recall / (precision + recall) if correct_preds > 0 else 0.0
    token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0

    metrics = EvaluationMetrics(
        precision=precision,
        recall=recall,
        f1=new_F,
        token_accuracy=token_accuracy,
        correct_chunks=int(correct_preds),
        predicted_chunks=int(total_preds),
        gold_chunks=int(total_correct),
    )

    logger.info(
        "%s | precision=%.4f recall=%.4f f1=%.4f token_acc=%.4f best_f1=%.4f",
        dataset,
        metrics.precision,
        metrics.recall,
        metrics.f1,
        metrics.token_accuracy,
        best_F,
    )

    if new_F > best_F:
        best_F = new_F
        save = True

    return best_F, metrics, save


@torch.no_grad()
def evaluate_f1_by_class(model, datas, mapping, config, dataset="Evaluation"):
    was_training = model.training
    model.eval()

    correct_preds, total_correct, total_preds = 0.0, 0.0, 0.0
    correct_tokens, total_tokens = 0.0, 0.0
    per_class_counts = defaultdict(lambda: {"tp": 0, "pred": 0, "gold": 0})
    for data, predicted_id in iter_decoded_batches(model, datas, config):
        gold_chunks = get_chunks(data["tags"], mapping["tag_to_id"])
        predicted_chunks = get_chunks(predicted_id, mapping["tag_to_id"])

        gold_spans = set(token_chunks_to_char_spans(gold_chunks, data["offsets"]))
        predicted_spans = set(token_chunks_to_char_spans(predicted_chunks, data["offsets"]))

        correct_preds += len(gold_spans & predicted_spans)
        total_preds += len(predicted_spans)
        total_correct += len(gold_spans)
        correct_tokens += sum(int(predicted == gold) for predicted, gold in zip(predicted_id, data["tags"]))
        total_tokens += len(data["tags"])

        for _, _, label in gold_spans:
            per_class_counts[label]["gold"] += 1
        for _, _, label in predicted_spans:
            per_class_counts[label]["pred"] += 1
        for _, _, label in gold_spans & predicted_spans:
            per_class_counts[label]["tp"] += 1

    precision = correct_preds / total_preds if correct_preds > 0 else 0.0
    recall = correct_preds / total_correct if correct_preds > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if correct_preds > 0 else 0.0
    token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0

    overall_metrics = EvaluationMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        token_accuracy=token_accuracy,
        correct_chunks=int(correct_preds),
        predicted_chunks=int(total_preds),
        gold_chunks=int(total_correct),
    )

    metrics_by_class: list[PerClassMetrics] = []
    for label in sorted(per_class_counts):
        counts = per_class_counts[label]
        precision = counts["tp"] / counts["pred"] if counts["pred"] > 0 else 0.0
        recall = counts["tp"] / counts["gold"] if counts["gold"] > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        metrics_by_class.append(
            PerClassMetrics(
                label=label,
                precision=precision,
                recall=recall,
                f1=f1,
                true_positives=counts["tp"],
                predicted=counts["pred"],
                gold=counts["gold"],
            )
        )

    if was_training:
        model.train()

    logger.info("%s per-class report:\n%s", dataset, format_per_class_metrics_table(metrics_by_class, overall_metrics))
    return overall_metrics, metrics_by_class


def evaluate_text_predictor_by_class(datas, mapping, predict_spans, dataset="Evaluation"):
    correct_preds, total_correct, total_preds = 0.0, 0.0, 0.0
    per_class_counts = defaultdict(lambda: {"tp": 0, "pred": 0, "gold": 0})

    for data in datas:
        gold_chunks = get_chunks(data["tags"], mapping["tag_to_id"])
        gold_spans = set(token_chunks_to_char_spans(gold_chunks, data["offsets"]))
        predicted_spans = set(predict_spans(data["text"]))

        correct_preds += len(gold_spans & predicted_spans)
        total_preds += len(predicted_spans)
        total_correct += len(gold_spans)

        for _, _, label in gold_spans:
            per_class_counts[label]["gold"] += 1
        for _, _, label in predicted_spans:
            per_class_counts[label]["pred"] += 1
        for _, _, label in gold_spans & predicted_spans:
            per_class_counts[label]["tp"] += 1

    precision = correct_preds / total_preds if correct_preds > 0 else 0.0
    recall = correct_preds / total_correct if correct_preds > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if correct_preds > 0 else 0.0

    overall_metrics = EvaluationMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        token_accuracy=0.0,
        correct_chunks=int(correct_preds),
        predicted_chunks=int(total_preds),
        gold_chunks=int(total_correct),
    )

    metrics_by_class: list[PerClassMetrics] = []
    for label in sorted(per_class_counts):
        counts = per_class_counts[label]
        label_precision = counts["tp"] / counts["pred"] if counts["pred"] > 0 else 0.0
        label_recall = counts["tp"] / counts["gold"] if counts["gold"] > 0 else 0.0
        label_f1 = (
            2 * label_precision * label_recall / (label_precision + label_recall)
            if label_precision + label_recall > 0
            else 0.0
        )
        metrics_by_class.append(
            PerClassMetrics(
                label=label,
                precision=label_precision,
                recall=label_recall,
                f1=label_f1,
                true_positives=counts["tp"],
                predicted=counts["pred"],
                gold=counts["gold"],
            )
        )

    logger.info("%s per-class report:\n%s", dataset, format_per_class_metrics_table(metrics_by_class, overall_metrics))
    return overall_metrics, metrics_by_class


def format_per_class_metrics_table(
    metrics_by_class: list[PerClassMetrics],
    overall_metrics: EvaluationMetrics | None = None,
) -> str:
    headers = ("Class", "Precision", "Recall", "F1", "TP", "Pred", "Gold")
    rows = [
        (
            metrics.label,
            f"{metrics.precision:.4f}",
            f"{metrics.recall:.4f}",
            f"{metrics.f1:.4f}",
            str(metrics.true_positives),
            str(metrics.predicted),
            str(metrics.gold),
        )
        for metrics in metrics_by_class
    ]

    if overall_metrics is not None:
        rows.append(
            (
                "TOTAL",
                f"{overall_metrics.precision:.4f}",
                f"{overall_metrics.recall:.4f}",
                f"{overall_metrics.f1:.4f}",
                str(overall_metrics.correct_chunks),
                str(overall_metrics.predicted_chunks),
                str(overall_metrics.gold_chunks),
            )
        )

    if not rows:
        rows.append(("TOTAL", "0.0000", "0.0000", "0.0000", "0", "0", "0"))

    widths = [max(len(header), max(len(row[index]) for row in rows)) for index, header in enumerate(headers)]
    header_row = " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    separator = "-+-".join("-" * widths[index] for index in range(len(headers)))
    body = [" | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) for row in rows]
    return "\n".join([header_row, separator, *body])
