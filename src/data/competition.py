import ast
import csv
import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _normalize_entity_item(item: Any) -> dict[str, Any]:
    if isinstance(item, (list, tuple)) and len(item) == 3:
        start, end, label = item
        return {"start": int(start), "end": int(end), "label": str(label)}

    if isinstance(item, dict):
        start = item.get("start")
        end = item.get("end")
        label = item.get("label")
        if start is None or end is None or label is None:
            raise ValueError(f"Unsupported entity dict format: {item!r}")
        return {"start": int(start), "end": int(end), "label": str(label)}

    raise ValueError(f"Unsupported entity format: {item!r}")


def parse_entities(raw_value: Any) -> list[dict[str, Any]]:
    if raw_value is None:
        return []

    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if not stripped or stripped == "empty" or stripped == "[]":
            return []
        parsed = ast.literal_eval(stripped)
    else:
        parsed = raw_value

    if parsed is None:
        return []
    if isinstance(parsed, dict):
        parsed = [parsed]

    return [_normalize_entity_item(item) for item in parsed]


def load_competition_train_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Competition train file was not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for index, row in enumerate(reader):
            records.append(
                {
                    "record_id": index,
                    "text": row["text"],
                    "entities": parse_entities(row["target"]),
                }
            )

    logger.info("Loaded %d labeled competition rows from %s", len(records), path)
    return records


def load_competition_test_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Competition test file was not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            records.append(
                {
                    "record_id": row["id"],
                    "text": row["text"],
                }
            )

    logger.info("Loaded %d unlabeled competition rows from %s", len(records), path)
    return records


def split_records(
    records: list[dict[str, Any]],
    train_size: float = 0.9,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    train_end = int(len(shuffled) * train_size)

    return {
        "train": shuffled[:train_end],
        "dev": shuffled[train_end:],
    }
