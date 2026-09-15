from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def trim_observation_steps(
    samples: Mapping[str, Any],
    *,
    n_obs_steps: int | None,
    obs_camera_keys: Iterable[str],
) -> dict[str, Any]:
    """Trim observation tensors to n_obs_steps while keeping action horizon intact."""
    if n_obs_steps is None:
        return dict(samples)
    n_obs_steps = int(n_obs_steps)
    if n_obs_steps <= 0:
        raise ValueError(f"n_obs_steps must be positive or None, got {n_obs_steps}")

    result = dict(samples)
    for key in obs_camera_keys:
        if key in result:
            result[key] = result[key][:n_obs_steps]
    if "state" in result:
        result["state"] = result["state"][:n_obs_steps]
    return result
