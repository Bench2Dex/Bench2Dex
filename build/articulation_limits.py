"""Utilities for applying YAML articulation joint limit overrides."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def apply_articulation_joint_limits(
    prim_path: str,
    joint_limits: Mapping[str, Mapping[str, Any]] | None,
    *,
    log_prefix: str,
) -> None:
    """Apply per-joint lower/upper limit overrides to spawned USD joint prims."""
    if not joint_limits:
        return

    try:
        import omni.usd  # type: ignore
        from pxr import Sdf  # type: ignore
    except ImportError:
        print(f"[{log_prefix}] WARNING: omni.usd not available, skipping joint limit overrides")
        return

    stage = omni.usd.get_context().get_stage()
    pending = {str(name): dict(spec) for name, spec in joint_limits.items()}
    applied: list[str] = []

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(prim_path):
            continue
        joint_name = prim.GetPath().name
        if joint_name not in pending:
            continue

        type_name = prim.GetTypeName()
        if type_name not in {"PhysicsRevoluteJoint", "PhysicsPrismaticJoint"}:
            continue

        spec = pending.pop(joint_name)
        lower = float(spec["lower"])
        upper = float(spec["upper"])
        unit = str(spec.get("unit", "rad")).lower()
        if type_name == "PhysicsRevoluteJoint" and unit == "rad":
            lower = math.degrees(lower)
            upper = math.degrees(upper)

        prim.CreateAttribute("physics:lowerLimit", Sdf.ValueTypeNames.Float).Set(lower)
        prim.CreateAttribute("physics:upperLimit", Sdf.ValueTypeNames.Float).Set(upper)
        applied.append(f"{joint_name}=[{lower:.6g}, {upper:.6g}]({unit})")

    if applied:
        print(f"[{log_prefix}] joint limit overrides applied: {applied}")
    if pending:
        print(f"[{log_prefix}] WARNING: joint limit overrides did not match joints: {sorted(pending)}")