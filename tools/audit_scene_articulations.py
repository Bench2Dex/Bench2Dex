#!/usr/bin/env python3
"""Audit scene articulation defaults and actuator coverage against USD joints.

This is a fast, non-simulation check for the two failures that Isaac Lab only
reports during ``sim.reset()``: movable joints without actuators and initial
joint positions outside the authored (or scene-overridden) limits.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from build.object_initial_state import resolve_articulation_spawn_config


def _usd_joints(path: Path) -> dict[str, tuple[float, float]]:
    try:
        from pxr import Usd
    except ImportError as exc:
        raise RuntimeError("pxr is required to audit scene articulation assets") from exc

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"could not open USD: {path}")

    joints: dict[str, tuple[float, float]] = {}
    for prim in stage.Traverse():
        type_name = prim.GetTypeName()
        if type_name not in {"PhysicsRevoluteJoint", "PhysicsPrismaticJoint"}:
            continue
        lower = prim.GetAttribute("physics:lowerLimit").Get()
        upper = prim.GetAttribute("physics:upperLimit").Get()
        if lower is None or upper is None:
            continue
        lower, upper = float(lower), float(upper)
        if type_name == "PhysicsRevoluteJoint":
            lower, upper = math.radians(lower), math.radians(upper)
        joints[str(prim.GetName())] = (lower, upper)
    return joints


def _matches(patterns: Mapping[str, Any], joint_name: str) -> list[tuple[str, Any]]:
    return [(expr, value) for expr, value in patterns.items() if re.fullmatch(expr, joint_name)]


def _task_number(path: Path) -> int | None:
    match = re.match(r"(\d+)_", path.name)
    return int(match.group(1)) if match else None


def audit_scene(path: Path) -> tuple[int, int, list[str]]:
    scene = yaml.safe_load(path.read_text()) or {}
    assets = scene.get("assets") or {}
    failures: list[str] = []
    object_count = 0
    joint_count = 0

    for obj in scene.get("objects") or []:
        asset_key = str(obj.get("asset", ""))
        asset = assets.get(asset_key) or {}
        if asset.get("body_type") != "articulation":
            continue
        object_count += 1
        obj_id = str(obj.get("id", "<unknown>"))
        usd_path = (path.parent / str(asset.get("path", ""))).resolve()
        if not usd_path.is_file():
            failures.append(f"{obj_id}: USD does not exist: {usd_path}")
            continue

        try:
            resolved = resolve_articulation_spawn_config(asset, obj)
            joints = _usd_joints(usd_path)
        except Exception as exc:
            failures.append(f"{obj_id}: {exc}")
            continue

        joint_count += len(joints)
        overrides = resolved.get("joint_limits") or {}
        initial_patterns = resolved.get("joint_pos") or {".*": 0.0}
        actuator_patterns = {
            expr: group
            for group, spec in resolved["actuators"].items()
            for expr in spec.get("joint_names_expr", [])
        }

        # Isaac Lab rejects an actuator configuration if even one expression
        # matches no joint, independently of whether other expressions cover
        # every movable joint.
        for expression, group in actuator_patterns.items():
            if not any(re.fullmatch(expression, joint_name) for joint_name in joints):
                failures.append(
                    f"{obj_id}: actuator '{group}' expression '{expression}' matches no USD joint"
                )

        for joint_name, usd_limits in joints.items():
            lower, upper = usd_limits
            if joint_name in overrides:
                override = overrides[joint_name]
                lower, upper = float(override["lower"]), float(override["upper"])
                if str(override.get("unit", "rad")) == "deg":
                    lower, upper = math.radians(lower), math.radians(upper)

            actuator_matches = _matches(actuator_patterns, joint_name)
            if not actuator_matches:
                failures.append(f"{obj_id}/{joint_name}: not covered by any actuator")

            initial_matches = _matches(initial_patterns, joint_name)
            if not initial_matches:
                initial = 0.0
            elif len(initial_matches) == 1:
                initial = float(initial_matches[0][1])
            else:
                expressions = [expr for expr, _ in initial_matches]
                failures.append(f"{obj_id}/{joint_name}: ambiguous joint_pos patterns {expressions}")
                continue
            if not lower <= initial <= upper:
                failures.append(
                    f"{obj_id}/{joint_name}: initial {initial:.9g} outside [{lower:.9g}, {upper:.9g}]"
                )

    return object_count, joint_count, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", nargs="*", type=Path, help="scene YAML paths; defaults to scenes/*.yaml")
    parser.add_argument("--min-task", type=int, default=None)
    parser.add_argument("--max-task", type=int, default=None)
    args = parser.parse_args()

    paths = args.scenes or sorted((REPO_ROOT / "scenes").glob("*.yaml"))
    paths = [path.resolve() for path in paths]
    if args.min_task is not None:
        paths = [
            path
            for path in paths
            if (number := _task_number(path)) is not None and number >= args.min_task
        ]
    if args.max_task is not None:
        paths = [
            path
            for path in paths
            if (number := _task_number(path)) is not None and number <= args.max_task
        ]

    total_objects = total_joints = 0
    all_failures: list[str] = []
    for path in paths:
        object_count, joint_count, failures = audit_scene(path)
        total_objects += object_count
        total_joints += joint_count
        state = "FAIL" if failures else "OK"
        print(f"{state:4} {path.name:44} articulations={object_count:2} joints={joint_count:3}")
        all_failures.extend(f"{path.name}: {failure}" for failure in failures)

    if all_failures:
        print("\n".join(all_failures), file=sys.stderr)
        return 1
    print(f"Audited {len(paths)} scenes, {total_objects} articulations, {total_joints} movable joints.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
