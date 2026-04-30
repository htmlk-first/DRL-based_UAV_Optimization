"""Small shared helpers for the per-algorithm experiment scripts.

The folders in this repository intentionally stay runnable as standalone
experiments.  This module keeps the repeated path, logging, and reporting
plumbing consistent without changing each algorithm's learning logic.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class ExperimentPaths:
    """Standard paths for one algorithm folder."""

    script_dir: Path
    results_dir: Path

    @classmethod
    def from_file(
        cls,
        file: str | Path,
        results_name: str = "results",
        create: bool = True,
    ) -> "ExperimentPaths":
        script_dir = Path(file).resolve().parent
        results_dir = script_dir / results_name
        if create:
            results_dir.mkdir(parents=True, exist_ok=True)
        return cls(script_dir=script_dir, results_dir=results_dir)

    def result(self, *parts: str) -> str:
        return str(self.results_dir.joinpath(*parts))


class EpisodeCsvLogger:
    """Context-managed CSV writer for episode-level training logs."""

    def __init__(self, path: str | Path | None, columns: Iterable[str]):
        self.path = Path(path) if path is not None else None
        self.columns = list(columns)
        self._file = None
        self._writer = None

    def __enter__(self) -> "EpisodeCsvLogger":
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("w", newline="", encoding="utf-8")
            self._writer = csv.writer(self._file)
            self._writer.writerow(self.columns)
        return self

    def writerow(self, values: Iterable[Any]) -> None:
        if self._writer is not None:
            self._writer.writerow(list(values))

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._file is not None:
            self._file.close()


def success_from_info(info: dict[str, Any]) -> int:
    return 1 if info.get("event") == "mission_complete" else 0


def mean_or_zero(values: list[float], window: int | None = None) -> float:
    if not values:
        return 0.0
    recent = values[-window:] if window else values
    return float(np.mean(recent))


def nanmean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def print_eval_summary(
    rewards: list[float],
    successes: int,
    n_episodes: int,
    reward_label: str = "Avg Reward",
    success_label: str = "Success Rate",
) -> None:
    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    print(f"\n=== Evaluation ({n_episodes} episodes) ===")
    print(f"{reward_label}: {avg_reward:.1f}")
    print(f"{success_label}: {successes}/{n_episodes} ({successes/n_episodes*100:.0f}%)")


def read_training_log(results_dir: str | Path, filename: str = "training_log.csv") -> list[dict[str, str]]:
    path = Path(results_dir) / filename
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def optional_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if not value or value.lower() == "nan":
        return None
    return float(value)


def select_existing_result(results_dir: str | Path, *filenames: str) -> str:
    results_path = Path(results_dir)
    for filename in filenames:
        candidate = results_path / filename
        if candidate.exists():
            return str(candidate)
    return str(results_path / filenames[-1])

