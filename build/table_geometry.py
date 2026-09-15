"""Helpers for table bounds and height generalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from .constants import Aabb


ROBOT_BACK_REFERENCE_Y = -0.55
NEGATIVE_Y_SHORTENING_M = 0.25
ROBOT_SUPPORT_TABLE_HEIGHT = 0.75
ROBOT_SUPPORT_TABLE_X_BOUNDS = (-0.85, 0.85)
ROBOT_SUPPORT_TABLE_Y_BOUNDS = (ROBOT_BACK_REFERENCE_Y, -0.35)
ROBOT_SUPPORT_TABLE_COLOR = (0.45, 0.30, 0.20)


@dataclass(frozen=True)
class TableGeometry:
    """Resolved table geometry in world coordinates."""

    original_size: Tuple[float, float, float]
    size: Tuple[float, float, float]
    spawn_size: Tuple[float, float, float]
    center: Tuple[float, float, float]
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    table_z: float

    def tabletop_aabb(self, height: float) -> Aabb:
        return (
            (self.x_min, self.y_min, self.table_z),
            (self.x_max, self.y_max, self.table_z + float(height)),
        )

    def bounds_dict(self) -> dict[str, float]:
        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
        }


@dataclass(frozen=True)
class RobotSupportTableGeometry:
    """Independent rear support table for robot bases."""

    size: Tuple[float, float, float]
    center: Tuple[float, float, float]
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    table_z: float

    def bounds_dict(self) -> dict[str, float]:
        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
        }


@dataclass(frozen=True)
class TableHeights:
    nominal_table_z: float
    table_z: float
    robot_mount_height: float
    height_offset_m: float


def resolve_table_geometry(
    table_size: Tuple[float, float, float],
    *,
    table_z: float,
    negative_y_shortening_m: float = NEGATIVE_Y_SHORTENING_M,
) -> TableGeometry:
    sx, sy, sz = (float(v) for v in table_size)
    x_min = -sx / 2.0
    x_max = sx / 2.0
    y_max = sy / 2.0
    y_min = -sy / 2.0 + float(negative_y_shortening_m)
    if y_min >= y_max:
        raise ValueError(
            f"Resolved table y bounds are invalid: y_min={y_min:.4f}, y_max={y_max:.4f}. "
            f"Reduce negative_y_shortening_m or increase table size_y."
        )

    resolved_sy = y_max - y_min
    return TableGeometry(
        original_size=(sx, sy, sz),
        size=(sx, resolved_sy, sz),
        spawn_size=(sx, resolved_sy, float(table_z)),
        center=(0.0, (y_min + y_max) / 2.0, float(table_z) / 2.0),
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        table_z=float(table_z),
    )


def resolve_robot_support_table_geometry() -> RobotSupportTableGeometry | None:
    """Return the fixed rear support table for robot base mounting.

    Dimensions are independent of the main table geometry so that the
    robot mount stays at a known, stable height regardless of table
    height generalization.
    """
    table_z = ROBOT_SUPPORT_TABLE_HEIGHT
    x_min, x_max = ROBOT_SUPPORT_TABLE_X_BOUNDS
    y_min, y_max = ROBOT_SUPPORT_TABLE_Y_BOUNDS
    support_length_x = float(x_max - x_min)
    support_width_y = float(y_max - y_min)
    if support_length_x <= 0.0 or support_width_y <= 0.0:
        return None

    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    return RobotSupportTableGeometry(
        size=(support_length_x, support_width_y, table_z),
        center=(center_x, center_y, table_z / 2.0),
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        table_z=table_z,
    )


def resolve_table_heights(
    nominal_table_z: float,
    sample_like: Mapping[str, Any] | None,
    *,
    generalization_enabled: bool,
) -> TableHeights:
    table_height_sample = _nested_mapping_get(sample_like or {}, ("spatial", "table_height"))
    table_height_enabled = bool(table_height_sample.get("enabled", False)) if table_height_sample else False
    height_offset_m = (
        float(table_height_sample.get("height_offset_m", 0.0) or 0.0)
        if generalization_enabled and table_height_enabled
        else 0.0
    )
    table_z = float(nominal_table_z) + height_offset_m
    robot_mount_height = float(nominal_table_z) if height_offset_m != 0.0 else table_z
    return TableHeights(
        nominal_table_z=float(nominal_table_z),
        table_z=table_z,
        robot_mount_height=robot_mount_height,
        height_offset_m=height_offset_m,
    )


def _nested_mapping_get(data: Mapping[str, Any], keys: tuple[str, ...]) -> Mapping[str, Any]:
    current: Any = data
    for key in keys:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key, {})
    return current if isinstance(current, Mapping) else {}
