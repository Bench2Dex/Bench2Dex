"""Shared constants and common types for scene construction."""

from typing import Tuple, TypeAlias

SETTLE_STEPS = 60
MIN_SEPARATION = 0.01
MAX_PLACE_ATTEMPTS = 80
DEFAULT_BOUNDS_HALF = 0.02

# Default table policy (used when scene YAML omits table fields)
FORCE_TABLE_SIZE = (2.2, 1.1, 0.04)  # meters: 220cm x 110cm x 4cm
FORCE_TABLE_HEIGHT = 0.75
TABLETOP_ZONE_HEIGHT = 0.10

ENABLE_GLOBAL_ROBOT = True

Aabb: TypeAlias = Tuple[Tuple[float, float, float], Tuple[float, float, float]]
