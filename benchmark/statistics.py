"""Statistical aggregation utilities for benchmark results."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float | bool | int],
    *,
    n_resamples: int = 10000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) using percentile bootstrap."""
    arr = np.array([float(v) for v in values], dtype=np.float64)
    observed_mean = float(arr.mean())
    rng = np.random.default_rng(seed)
    resampled_means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        sample = rng.choice(arr, size=len(arr), replace=True)
        resampled_means[i] = sample.mean()
    lo = float(np.percentile(resampled_means, 2.5))
    hi = float(np.percentile(resampled_means, 97.5))
    return observed_mean, lo, hi


def aggregate_episode_results(
    rows: list[dict[str, Any]],
    *,
    group_by: list[str],
    metric: str,
) -> dict[tuple, dict[str, Any]]:
    """Group rows and compute count/mean for a given metric field."""
    groups: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in group_by)
        groups[key].append(float(row[metric]))

    result: dict[tuple, dict[str, Any]] = {}
    for key, vals in groups.items():
        result[key] = {
            "count": len(vals),
            "mean": float(np.mean(vals)) if vals else 0.0,
        }
    return result