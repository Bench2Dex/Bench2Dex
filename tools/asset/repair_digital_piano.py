#!/usr/bin/env python3
"""Repair the converted digital-piano USD in place.

The source articulation labels keys 16 and 29 with the opposite prismatic
direction from every other piano key.  The converted asset also marks the
per-link visual containers instanceable.  Isaac Sim's Fabric scene delegate
does not reliably expose the nested ``World/Looks`` material paths below those
instance proxies, which can make an otherwise valid material look broken in
the viewport.

This repair authors a small, strong override into the asset's root layer:

* make keys 16 and 29 use the same local joint frame as adjacent keys;
* de-instance composed visual containers so their mesh/material paths are
  ordinary scene prims;
* verify every authored texture resolves to an existing file before saving.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Gf, Sdf, Usd


BAD_KEY_IDS = (16, 29)
EXPECTED_KEY_FRAME = Gf.Quatf(0.0, Gf.Vec3f(0.0, 1.0, 0.0))


def _quat_is_close(actual: Gf.Quatf, expected: Gf.Quatf, tolerance: float = 1.0e-6) -> bool:
    return abs(actual.GetReal() - expected.GetReal()) <= tolerance and Gf.IsClose(
        actual.GetImaginary(), expected.GetImaginary(), tolerance
    )


def _resolved_textures(stage: Usd.Stage) -> tuple[set[Path], list[str]]:
    resolved: set[Path] = set()
    unresolved: list[str] = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        for attr in prim.GetAttributes():
            if "texture" not in attr.GetName().lower():
                continue
            value = attr.Get()
            if not isinstance(value, Sdf.AssetPath) or not value.path:
                continue
            if value.resolvedPath:
                path = Path(value.resolvedPath)
                if path.is_file():
                    resolved.add(path)
                    continue
            unresolved.append(f"{prim.GetPath()}.{attr.GetName()}: {value.path}")
    return resolved, unresolved


def repair(asset_path: Path) -> None:
    asset_path = asset_path.expanduser().resolve()
    if not asset_path.is_file():
        raise FileNotFoundError(asset_path)

    stage = Usd.Stage.Open(str(asset_path), load=Usd.Stage.LoadAll)
    if stage is None or not stage.GetDefaultPrim():
        raise RuntimeError(f"Could not open a default prim from {asset_path}")

    textures, unresolved = _resolved_textures(stage)
    if unresolved:
        details = "\n  ".join(unresolved)
        raise RuntimeError(f"Refusing to save with unresolved textures:\n  {details}")

    stage.SetEditTarget(stage.GetRootLayer())
    root_path = str(stage.GetDefaultPrim().GetPath())

    repaired_joints: list[str] = []
    for key_id in BAD_KEY_IDS:
        joint_path = f"{root_path}/joints/joint_{key_id}_to_1"
        joint = stage.GetPrimAtPath(joint_path)
        if not joint:
            raise RuntimeError(f"Missing expected piano-key joint: {joint_path}")
        for attr_name in ("physics:localRot0", "physics:localRot1"):
            attr = joint.GetAttribute(attr_name)
            if not attr:
                raise RuntimeError(f"Missing {attr_name} on {joint_path}")
            attr.Set(EXPECTED_KEY_FRAME)
        repaired_joints.append(joint_path)

    instance_paths = [prim.GetPath() for prim in stage.TraverseAll() if prim.IsInstance()]
    for prim_path in instance_paths:
        stage.GetPrimAtPath(prim_path).SetInstanceable(False)

    stage.GetRootLayer().Save()

    check = Usd.Stage.Open(str(asset_path), load=Usd.Stage.LoadAll)
    if check is None:
        raise RuntimeError(f"Could not reopen repaired asset: {asset_path}")
    remaining_instances = [str(prim.GetPath()) for prim in check.TraverseAll() if prim.IsInstance()]
    if remaining_instances:
        raise RuntimeError(f"Instance overrides did not compose: {remaining_instances[:5]}")
    for joint_path in repaired_joints:
        joint = check.GetPrimAtPath(joint_path)
        for attr_name in ("physics:localRot0", "physics:localRot1"):
            if not _quat_is_close(joint.GetAttribute(attr_name).Get(), EXPECTED_KEY_FRAME):
                raise RuntimeError(f"Joint-frame repair did not compose: {joint_path}.{attr_name}")
    checked_textures, checked_unresolved = _resolved_textures(check)
    if checked_unresolved or checked_textures != textures:
        raise RuntimeError("Texture dependencies changed while repairing the USD")

    print(f"Repaired: {asset_path}")
    print(f"Joint frames corrected: {', '.join(map(str, BAD_KEY_IDS))}")
    print(f"Instance proxies removed: {len(instance_paths)}")
    print(f"Resolved textures preserved: {len(checked_textures)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path, help="Path to digital_piano.usd")
    args = parser.parse_args()
    repair(args.asset)


if __name__ == "__main__":
    main()
