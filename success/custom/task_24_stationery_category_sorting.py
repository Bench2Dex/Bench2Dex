"""Task-specific success evaluator for 24_stationery_category_sorting.

Phases:
  - terminal (default): all 4 objects in containers + containers upright + all static
  - stage + object: single object in zone + container upright + object static
  - placed + object: single object in zone + container upright (for drop safety)

Pens (obj_221_pen_3, obj_166_markpen_4) → pen cup (obj_167_pencup_1):
  - Upright: pen's long axis (local_Y) within 60° of world Z
  - Zone: generous bounds — if pen is roughly upright it must be in the cup

Box objects (obj_199_glue_5, obj_168_battery_6) → plastic box (obj_169_plasticbox_2):
  - Zone: precise bounds from USD mesh inspection (box interior cavity)
"""

from __future__ import annotations

from typing import Any, Dict

from success.condition_evaluator import evaluate_conditions

# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------
_PEN_CUP = "obj_167_pencup_1"
_PLASTIC_BOX = "obj_169_plasticbox_2"

# ---------------------------------------------------------------------------
# Container upright: local_Y = world Z
#   - pen cup: rpy=[90,0,0]  → mesh Y → world Z
#   - plastic box: rpy=[90,0,90] → mesh Y → world Z
# Both kinematic-enabled, always perfectly upright.
# ---------------------------------------------------------------------------
_CUP_UPRIGHT = {
    "type": "object_upright",
    "object": _PEN_CUP,
    "tolerance_deg": 30,
    "local_axis": "y",
}
_BOX_UPRIGHT = {
    "type": "object_upright",
    "object": _PLASTIC_BOX,
    "tolerance_deg": 30,
    "local_axis": "y",
}

# ---------------------------------------------------------------------------
# Pen cup zone (generous)
# Mesh (scaled 0.13): local X [-0.057, 0.057], Y [0, 0.245], Z [-0.080, 0.082]
# Container centre offset = mid-height of cup interior.
# Zone: ±8 cm XZ covers full cup opening, full cup height in Y.
# ---------------------------------------------------------------------------
_PEN_CUP_CENTER = [0.0, 0.12, 0.0]
_PEN_CUP_ZONE_LO = [-0.08, -0.12, -0.08]
_PEN_CUP_ZONE_HI = [0.08, 0.12, 0.08]

# ---------------------------------------------------------------------------
# Plastic box zone (precise from USD mesh)
# rpy=[90,0,90]: mesh X→box local X (long), mesh Y→box local Y (height), mesh Z→box local Z (short)
# Mesh (scaled 0.14): local X [-0.133, 0.133], Y [0, 0.115], Z [-0.099, 0.099]
# Interior cavity (~10% wall reduction): X [-0.12, 0.12], Z [-0.08, 0.08]
# Y zone extended upward to allow objects that protrude above the rim.
# Centre at mid-height of box cavity.
# ---------------------------------------------------------------------------
_BOX_CENTER = [0.0, 0.055, 0.0]
_BOX_ZONE_LO = [-0.12, -0.06, -0.08]
_BOX_ZONE_HI = [0.12, 0.07, 0.08]

# ---------------------------------------------------------------------------
# Object definitions
# ---------------------------------------------------------------------------
_OBJECTS: Dict[str, dict] = {
    "obj_221_pen_3": {
        "container": _PEN_CUP,
        "container_center": _PEN_CUP_CENTER,
        "zone_lo": _PEN_CUP_ZONE_LO,
        "zone_hi": _PEN_CUP_ZONE_HI,
        "container_upright": _CUP_UPRIGHT,
        # Pen's long axis (local Y) points upward when standing in the cup.
        "upright": {
            "type": "object_upright",
            "object": "obj_221_pen_3",
            "tolerance_deg": 60,
            "local_axis": "y",
        },
    },
    "obj_166_markpen_4": {
        "container": _PEN_CUP,
        "container_center": _PEN_CUP_CENTER,
        "zone_lo": _PEN_CUP_ZONE_LO,
        "zone_hi": _PEN_CUP_ZONE_HI,
        "container_upright": _CUP_UPRIGHT,
        # Pen's long axis (local Y) points upward when standing in the cup.
        "upright": {
            "type": "object_upright",
            "object": "obj_166_markpen_4",
            "tolerance_deg": 60,
            "local_axis": "y",
        },
    },
    "obj_199_glue_5": {
        "container": _PLASTIC_BOX,
        "container_center": _BOX_CENTER,
        "zone_lo": _BOX_ZONE_LO,
        "zone_hi": _BOX_ZONE_HI,
        "container_upright": _BOX_UPRIGHT,
        # No upright check — glue may lie on its side in the box
    },
    "obj_168_battery_6": {
        "container": _PLASTIC_BOX,
        "container_center": _BOX_CENTER,
        "zone_lo": _BOX_ZONE_LO,
        "zone_hi": _BOX_ZONE_HI,
        "container_upright": _BOX_UPRIGHT,
        # No upright check — battery orientation is unconstrained
    },
}

_ALL_OBJ_IDS = list(_OBJECTS.keys())


def _in_zone_cond(obj_id: str) -> dict:
    spec = _OBJECTS[obj_id]
    return {
        "type": "object_in_container_zone",
        "object": obj_id,
        "container": spec["container"],
        "container_center_offset": spec["container_center"],
        "zone_lo": spec["zone_lo"],
        "zone_hi": spec["zone_hi"],
        "container_frame": True,
    }


def _static_cond(obj_id: str) -> dict:
    return {
        "type": "object_static",
        "object": obj_id,
        "threshold": 0.05,
        "check_angular": False,
    }


def check_success(states: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> bool:
    phase = str(params.get("phase", "terminal"))

    if phase == "terminal":
        conditions: list[dict] = []
        for obj_id in _ALL_OBJ_IDS:
            spec = _OBJECTS[obj_id]
            conditions.append(spec["container_upright"])
            conditions.append(_in_zone_cond(obj_id))
            if "upright" in spec:
                conditions.append(spec["upright"])
            conditions.append(_static_cond(obj_id))
        return evaluate_conditions(conditions, states, ctx)

    if phase == "stage":
        obj_id = str(params.get("object", ""))
        if obj_id not in _OBJECTS:
            raise ValueError(f"Unknown object for stage phase: {obj_id!r}")
        spec = _OBJECTS[obj_id]
        conditions = [
            spec["container_upright"],
            _in_zone_cond(obj_id),
        ]
        if "upright" in spec:
            conditions.append(spec["upright"])
        conditions.append(_static_cond(obj_id))
        return evaluate_conditions(conditions, states, ctx)

    if phase == "placed":
        obj_id = str(params.get("object", ""))
        if obj_id not in _OBJECTS:
            raise ValueError(f"Unknown object for placed phase: {obj_id!r}")
        spec = _OBJECTS[obj_id]
        conditions = [
            spec["container_upright"],
            _in_zone_cond(obj_id),
        ]
        return evaluate_conditions(conditions, states, ctx)

    raise ValueError(f"Unknown phase: {phase!r}")
