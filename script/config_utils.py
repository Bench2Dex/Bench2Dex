"""Config loading helpers for script entrypoints."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def _parse_scalar(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except Exception:
        return value


def load_config_with_overrides(
    config_path: str | Path,
    overrides: list[str] | None = None,
    *,
    extra_updates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")

    override_items = list(overrides or [])
    if override_items:
        if len(override_items) % 2 != 0:
            raise ValueError("Overrides must be provided as '--key value' pairs.")
        iterator = iter(override_items)
        for key in iterator:
            raw_value = next(iterator)
            normalized_key = key.lstrip("-").replace("-", "_")
            data[normalized_key] = _parse_scalar(raw_value)

    if extra_updates:
        data.update(extra_updates)

    return data


def build_config_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--host", type=str, default=None, help="Server host override.")
    parser.add_argument("--port", type=int, default=None, help="Server port override.")
    parser.add_argument(
        "--overrides",
        nargs=argparse.REMAINDER,
        help="Config override pairs, e.g. --overrides --task_name demo --seed 0",
    )
    return parser

