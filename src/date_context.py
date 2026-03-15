import _pickle as cPickle
import logging
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.data.processing import tokenize_with_offsets, zero_digits

logger = logging.getLogger(__name__)

DATE_TYPE_LABELS = (
    "Дата рождения",
    "Дата регистрации по месту жительства или пребывания",
    "Дата окончания срока действия карты",
)


def infer_date_classifier_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.name}_date_classifier.pkl")


def _normalize_feature_token(token: str) -> str:
    return zero_digits(token.lower()) if any(char.isdigit() for char in token) else token.lower()


def _find_span_token_bounds(
    tokens: list[tuple[str, int, int]],
    start: int,
    end: int,
) -> tuple[int | None, int | None]:
    first_index = None
    last_index = None
    for index, (_, token_start, token_end) in enumerate(tokens):
        if token_start < end and token_end > start:
            if first_index is None:
                first_index = index
            last_index = index
    return first_index, last_index


def _extract_features(text: str, start: int, end: int, window_tokens: int) -> Counter[str]:
    tokens = tokenize_with_offsets(text)
    first_index, last_index = _find_span_token_bounds(tokens, start, end)
    if first_index is None or last_index is None:
        return Counter({"BIAS": 1})

    features = Counter({"BIAS": 1})
    span_text = text[start:end]
    span_text_lower = span_text.lower()

    is_card_expiry_dot = "." in span_text and len(span_text) <= 7
    if "/" in span_text:
        features["SPAN_FORMAT=card_expiry_slash"] += 1
    if is_card_expiry_dot:
        features["SPAN_FORMAT=card_expiry_dot"] += 1
    if ":" in span_text:
        features["SPAN_HAS_TIME"] += 1
    if any(
        weekday in span_text_lower
        for weekday in ("понедель", "вторник", "сред", "четверг", "пятниц", "суббот", "воскрес")
    ):
        features["SPAN_HAS_WEEKDAY"] += 1
    if any(month in span_text_lower for month in ("январ", "феврал", "март", "апрел", "мая", "июн", "июл", "август", "сентябр", "октябр", "ноябр", "декабр")):
        features["SPAN_HAS_MONTH"] += 1
    if span_text_lower.startswith("до "):
        features["SPAN_PREFIX=до"] += 1
    if span_text_lower.startswith("с "):
        features["SPAN_PREFIX=с"] += 1
    if any(token in span_text_lower for token in ("следующ", "через неделю", "в конце", "в начале")):
        features["SPAN_IS_RELATIVE"] += 1

    left_tokens = tokens[max(0, first_index - window_tokens):first_index]
    right_tokens = tokens[last_index + 1:last_index + 1 + window_tokens]

    for distance, (token, _, _) in enumerate(reversed(left_tokens), start=1):
        normalized = _normalize_feature_token(token)
        features[f"L{distance}={normalized}"] += 1
        features[f"L*={normalized}"] += 1

    for distance, (token, _, _) in enumerate(right_tokens, start=1):
        normalized = _normalize_feature_token(token)
        features[f"R{distance}={normalized}"] += 1
        features[f"R*={normalized}"] += 1

    context_tokens = left_tokens + right_tokens
    for token, _, _ in context_tokens:
        normalized = _normalize_feature_token(token)
        features[f"CTX={normalized}"] += 1

    return features


@dataclass
class DateContextClassifier:
    labels: tuple[str, ...]
    log_priors: dict[str, float]
    log_probs: dict[str, dict[str, float]]
    default_log_probs: dict[str, float]
    window_tokens: int
    confidence_margin: float

    def score(self, text: str, start: int, end: int) -> dict[str, float]:
        features = _extract_features(text, start, end, self.window_tokens)
        scores = {}
        for label in self.labels:
            score = self.log_priors[label]
            label_log_probs = self.log_probs[label]
            default_log_prob = self.default_log_probs[label]
            for feature, count in features.items():
                score += count * label_log_probs.get(feature, default_log_prob)
            scores[label] = score
        return scores

    def predict(self, text: str, start: int, end: int, current_label: str | None = None) -> str:
        scores = self.score(text, start, end)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_label, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else -math.inf
        if current_label is not None and current_label in self.labels and best_score - second_score < self.confidence_margin:
            return current_label
        return best_label


def build_date_context_classifier(
    records: list[dict],
    window_tokens: int = 8,
    confidence_margin: float = 1.0,
) -> DateContextClassifier:
    label_counts = Counter()
    feature_counts = {label: Counter() for label in DATE_TYPE_LABELS}
    vocabulary = set()

    for record in records:
        for entity in record.get("entities", []):
            label = entity["label"]
            if label not in DATE_TYPE_LABELS:
                continue
            label_counts[label] += 1
            features = _extract_features(record["text"], entity["start"], entity["end"], window_tokens)
            feature_counts[label].update(features)
            vocabulary.update(features)

    if not sum(label_counts.values()):
        raise ValueError("No date-labeled entities were found to train the date context classifier")

    total_examples = sum(label_counts.values())
    vocabulary_size = max(1, len(vocabulary))
    log_priors = {
        label: math.log((label_counts[label] + 1.0) / (total_examples + len(DATE_TYPE_LABELS)))
        for label in DATE_TYPE_LABELS
    }
    log_probs: dict[str, dict[str, float]] = {}
    default_log_probs: dict[str, float] = {}
    for label in DATE_TYPE_LABELS:
        total_feature_count = sum(feature_counts[label].values())
        denominator = total_feature_count + vocabulary_size
        default_log_probs[label] = math.log(1.0 / denominator)
        log_probs[label] = {
            feature: math.log((count + 1.0) / denominator)
            for feature, count in feature_counts[label].items()
        }

    return DateContextClassifier(
        labels=DATE_TYPE_LABELS,
        log_priors=log_priors,
        log_probs=log_probs,
        default_log_probs=default_log_probs,
        window_tokens=window_tokens,
        confidence_margin=confidence_margin,
    )


def save_date_context_classifier(classifier: DateContextClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        cPickle.dump(classifier, output)
    logger.info("Saved date context classifier to %s", path)


def load_date_context_classifier(path: Path) -> DateContextClassifier:
    with path.open("rb") as source:
        classifier = cPickle.load(source)
    if not isinstance(classifier, DateContextClassifier):
        raise TypeError(f"Unsupported date context classifier payload in {path}")
    return classifier


def train_and_save_date_context_classifier(
    records: list[dict],
    path: Path,
    window_tokens: int,
    confidence_margin: float,
) -> DateContextClassifier:
    classifier = build_date_context_classifier(
        records,
        window_tokens=window_tokens,
        confidence_margin=confidence_margin,
    )
    save_date_context_classifier(classifier, path)
    return classifier


def relabel_date_spans_with_classifier(
    text: str,
    spans: list[tuple[int, int, str]],
    classifier: DateContextClassifier | None,
) -> list[tuple[int, int, str]]:
    if classifier is None:
        return sorted(set(spans), key=lambda item: (item[0], item[1], item[2]))

    relabeled = []
    for start, end, label in spans:
        if label in DATE_TYPE_LABELS:
            label = classifier.predict(text, start, end, current_label=label)
        relabeled.append((start, end, label))
    return sorted(set(relabeled), key=lambda item: (item[0], item[1], item[2]))
