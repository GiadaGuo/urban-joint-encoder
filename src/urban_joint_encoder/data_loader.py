"""Header-safe loading and train-only normalization for traffic graph data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    sequence_length: int = 12
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class Standardizer:
    mean: float
    std: float
    fitted_row_start: int
    fitted_row_end_inclusive: int

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def load_traffic_graph(root: Path) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    base = root / "data" / "raw" / "Main data" / "prediction_data"
    traffic_path, adjacency_path = base / "feature_matrix_X.csv", base / "adj_matrix.csv"
    traffic = pd.read_csv(traffic_path, header=None, encoding="utf-8-sig").to_numpy(np.float32)
    adjacency = pd.read_csv(adjacency_path, header=None, encoding="utf-8-sig").to_numpy(np.float32)
    if traffic.ndim != 2 or adjacency.shape != (traffic.shape[1], traffic.shape[1]):
        raise ValueError(f"Incompatible shapes: traffic={traffic.shape}, adjacency={adjacency.shape}")
    if not np.isfinite(traffic).all() or not np.isfinite(adjacency).all():
        raise ValueError("Traffic and adjacency must contain only finite values")
    return traffic, adjacency, {"traffic": str(traffic_path.relative_to(root)), "adjacency": str(adjacency_path.relative_to(root))}


def split_boundaries(num_rows: int, config: SplitConfig) -> dict[str, tuple[int, int]]:
    if not np.isclose(config.train_ratio + config.validation_ratio + config.test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1")
    train_end = int(num_rows * config.train_ratio)
    validation_end = train_end + int(num_rows * config.validation_ratio)
    return {"train": (0, train_end), "validation": (train_end, validation_end), "test": (validation_end, num_rows)}


def fit_train_standardizer(traffic: np.ndarray, train_end_exclusive: int) -> Standardizer:
    train = traffic[:train_end_exclusive]
    mean, std = float(train.mean()), float(train.std())
    if std <= 0:
        raise ValueError("Training standard deviation must be positive")
    return Standardizer(mean, std, 0, train_end_exclusive - 1)

