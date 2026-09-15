#!/usr/bin/env python3
"""Audit registered robot joint manifests against shipped URDF and USD assets.

Run this after converting or replacing robot assets.  The check is deliberately
outside rollout: an asset mismatch invalidates the safety metric rather than
being something a policy episode can repair.
"""

from __future__ import annotations

import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = REPO_ROOT / "robots" / "active_dof_maps.yml"
DATASET_ROOT = Path(os.environ.get("DEX2BENCH_DATASET_ROOT", REPO_ROOT.parent / "dex2bench_dataset"))


def _urdf_limits(path: Path, name_sanitize: dict[str, str]) -> dict[str, tuple[float, float]]:
    root = ET.parse(path).getroot()
    output = {}
    for joint in root.findall(".//joint"):
        if joint.get("type") == "fixed":
            continue
        limit = joint.find("limit")
        if limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
            continue
        raw_name = str(joint.get("name"))
        output[str(name_sanitize.get(raw_name, raw_name))] = (
            float(limit.attrib["lower"]), float(limit.attrib["upper"])
        )
    return output


def _usd_limits(path: Path) -> dict[str, tuple[float, float]]:
    try:
        from pxr import Usd
    except ImportError as exc:
        raise RuntimeError("pxr is required to audit USD joint limits") from exc
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"could not open USD: {path}")
    output = {}
    for prim in stage.Traverse():
        if prim.GetTypeName() not in {"PhysicsRevoluteJoint", "PhysicsPrismaticJoint"}:
            continue
        lower = prim.GetAttribute("physics:lowerLimit").Get()
        upper = prim.GetAttribute("physics:upperLimit").Get()
        if lower is None or upper is None:
            continue
        # USD angular limits are degrees; prismatic limits remain metres.
        if prim.GetTypeName() == "PhysicsRevoluteJoint":
            lower, upper = math.radians(float(lower)), math.radians(float(upper))
        output[prim.GetName()] = (float(lower), float(upper))
    return output


def main() -> int:
    entries = (yaml.safe_load(MAP_PATH.read_text()) or {}).get("robots", {})
    failures = []
    for robot_key, cfg in sorted(entries.items()):
        urdf = DATASET_ROOT / str(cfg["urdf_path"]).replace("Bench2Dex/", "")
        usd = (REPO_ROOT / str(cfg["usd_path"])).resolve()
        expected = set(map(str, cfg["full_joint_names"]))
        urdf_data = _urdf_limits(urdf, dict(cfg.get("name_sanitize") or {}))
        usd_data = _usd_limits(usd)
        names_ok = expected == set(urdf_data) == set(usd_data)
        limit_ok = all(
            abs(urdf_data[name][0] - usd_data[name][0]) <= 1e-4
            and abs(urdf_data[name][1] - usd_data[name][1]) <= 1e-4
            for name in expected
        ) if names_ok else False
        state = "OK" if names_ok and limit_ok else "FAIL"
        print(f"{state:4} {robot_key:46} joints={len(expected)}")
        if state == "FAIL":
            failures.append(robot_key)
    if failures:
        print(f"Failed assets: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
