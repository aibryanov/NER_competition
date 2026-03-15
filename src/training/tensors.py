from dataclasses import dataclass

import numpy as np
import torch

CHAR_PAD_ID = 0


@dataclass
class SequenceBatch:
    words: torch.Tensor
    tags: torch.Tensor
    chars: torch.Tensor
    char_lengths: list[int]
    restore_order: dict[int, int]
    entity_targets: torch.Tensor | None


def _normalize_char_sequences(char_sequences: list[list[int]]) -> list[list[int]]:
    return [sequence if sequence else [CHAR_PAD_ID] for sequence in char_sequences]


def _pad_char_sequences(char_sequences: list[list[int]]) -> torch.Tensor:
    max_length = max((len(sequence) for sequence in char_sequences), default=1)
    padded = np.zeros((len(char_sequences), max_length), dtype=np.int64)

    for index, sequence in enumerate(char_sequences):
        padded[index, : len(sequence)] = sequence

    return torch.as_tensor(padded, dtype=torch.long)


def build_char_tensor(
    chars: list[list[int]],
    char_mode: str,
) -> tuple[torch.Tensor, list[int], dict[int, int]]:
    normalized_chars = _normalize_char_sequences(chars)

    if char_mode == "LSTM":
        sorted_pairs = sorted(
            enumerate(normalized_chars),
            key=lambda item: len(item[1]),
            reverse=True,
        )
        sorted_chars = [char_ids for _, char_ids in sorted_pairs]
        restore_order = {
            sorted_index: original_index
            for sorted_index, (original_index, _) in enumerate(sorted_pairs)
        }
        char_lengths = [len(char_ids) for char_ids in sorted_chars]
        return _pad_char_sequences(sorted_chars), char_lengths, restore_order

    if char_mode == "CNN":
        char_lengths = [len(char_ids) for char_ids in normalized_chars]
        return _pad_char_sequences(normalized_chars), char_lengths, {}

    raise ValueError(f"Unsupported char mode: {char_mode}")


def prepare_sequence_batch(data: dict[str, list], char_mode: str) -> SequenceBatch:
    chars, char_lengths, restore_order = build_char_tensor(data["chars"], char_mode)
    entity_targets = None
    if data.get("entity_targets"):
        entity_targets = torch.as_tensor(data["entity_targets"], dtype=torch.float32)
    return SequenceBatch(
        words=torch.as_tensor(data["words"], dtype=torch.long),
        tags=torch.as_tensor(data["tags"], dtype=torch.long),
        chars=chars,
        char_lengths=char_lengths,
        restore_order=restore_order,
        entity_targets=entity_targets,
    )
