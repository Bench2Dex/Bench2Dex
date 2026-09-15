"""Shared robot placement helpers."""

from __future__ import annotations

from typing import Tuple

from build.table_geometry import ROBOT_BACK_REFERENCE_Y


def robot_back_edge_y(_table_size: Tuple[float, float, float]) -> float:
    """Return the fixed rear table reference used for global robot placement."""
    return ROBOT_BACK_REFERENCE_Y
