import re
from typing import Any

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def tokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    return [(match.group(), match.start(), match.end()) for match in TOKEN_PATTERN.finditer(text)]


def zero_digits(text: str) -> str:
    return re.sub(r"\d", "0", text)


def collapse_nested_same_label_spans(
    spans: list[tuple[int, int, str]],
) -> list[tuple[int, int, str]]:
    collapsed: list[tuple[int, int, str]] = []
    for span in sorted(spans, key=lambda item: (item[0], -(item[1] - item[0]), item[2])):
        start, end, label = span
        if collapsed:
            previous_start, previous_end, previous_label = collapsed[-1]
            if label == previous_label and previous_start <= start and end <= previous_end:
                continue
        collapsed.append(span)
    return collapsed


def convert_to_bio(
    text: str,
    annotations: list[dict[str, Any]],
) -> tuple[list[str], list[tuple[int, int]], list[str]]:
    tokens = tokenize_with_offsets(text)

    spans = collapse_nested_same_label_spans(
        [(entity["start"], entity["end"], entity["label"]) for entity in annotations]
    )
    spans.sort(key=lambda item: (item[0], item[1], item[2]))

    words: list[str] = []
    offsets: list[tuple[int, int]] = []
    labels: list[str] = []
    for word, word_start, word_end in tokens:
        words.append(word)
        offsets.append((word_start, word_end))

        label = "O"
        for span_start, span_end, entity in spans:
            if word_start < span_end and word_end > span_start:
                label = f"B-{entity}" if word_start == span_start else f"I-{entity}"
                break
        labels.append(label)

    return words, offsets, labels


def normalize_iob2(tags: list[str]) -> list[str]:
    normalized_tags = list(tags)
    for index, tag in enumerate(normalized_tags):
        if tag == "O":
            continue

        split_tag = tag.split("-", maxsplit=1)
        if len(split_tag) != 2 or split_tag[0] not in {"I", "B"}:
            raise ValueError(f"Invalid IOB format: {tags}")

        if split_tag[0] == "B":
            continue

        if index == 0 or normalized_tags[index - 1] == "O" or normalized_tags[index - 1][1:] != tag[1:]:
            normalized_tags[index] = "B" + tag[1:]

    return normalized_tags


def iob_to_iobes(tags: list[str]) -> list[str]:
    converted_tags: list[str] = []
    for index, tag in enumerate(tags):
        if tag == "O":
            converted_tags.append(tag)
        elif tag.startswith("B-"):
            next_is_inside = index + 1 < len(tags) and tags[index + 1].startswith("I-")
            converted_tags.append(tag if next_is_inside else tag.replace("B-", "S-"))
        elif tag.startswith("I-"):
            next_is_inside = index + 1 < len(tags) and tags[index + 1].startswith("I-")
            converted_tags.append(tag if next_is_inside else tag.replace("I-", "E-"))
        else:
            raise ValueError(f"Invalid IOB format: {tag}")

    return converted_tags


def process_record(record: dict[str, Any], replace_digits: bool = True) -> dict[str, Any]:
    raw_words, offsets, labels = convert_to_bio(record["text"], record["entities"])
    normalized_labels = normalize_iob2(labels)
    words = [zero_digits(word) for word in raw_words] if replace_digits else list(raw_words)

    return {
        "record_id": record["record_id"],
        "text": record["text"],
        "raw_words": raw_words,
        "words": words,
        "offsets": offsets,
        "labels": iob_to_iobes(normalized_labels),
    }


def convert_records_to_bioes(
    records: list[dict[str, Any]],
    replace_digits: bool = True,
) -> list[dict[str, Any]]:
    return [process_record(record, replace_digits=replace_digits) for record in records]


def convert_split_to_bioes(
    split_records: dict[str, list[dict[str, Any]]],
    replace_digits: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    return {
        split_name: convert_records_to_bioes(records, replace_digits=replace_digits)
        for split_name, records in split_records.items()
    }
