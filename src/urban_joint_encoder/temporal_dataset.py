"""Leakage-safe next-step temporal windows with original row indices."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class TrafficWindowDataset(Dataset):
    def __init__(self, normalized_traffic: np.ndarray, target_start: int, target_end: int, sequence_length: int = 12):
        if target_start < sequence_length or target_end > len(normalized_traffic) or target_start >= target_end:
            raise ValueError("Invalid target boundaries for requested history length")
        self.data = torch.as_tensor(normalized_traffic, dtype=torch.float32)
        self.sequence_length = int(sequence_length)
        self.target_indices = np.arange(target_start, target_end, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.target_indices)

    def __getitem__(self, index: int):
        target_row = int(self.target_indices[index])
        history_start = target_row - self.sequence_length
        x = self.data[history_start:target_row].unsqueeze(-1)
        y = self.data[target_row]
        return x, y, torch.tensor(target_row, dtype=torch.long)

    def audit(self) -> dict[str, object]:
        first, last = int(self.target_indices[0]), int(self.target_indices[-1])
        return {
            "samples": len(self),
            "target_row_range_inclusive": [first, last],
            "first_history_row_range_inclusive": [first - self.sequence_length, first - 1],
            "last_history_row_range_inclusive": [last - self.sequence_length, last - 1],
            "future_information_in_input": False,
        }

