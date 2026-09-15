from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CollisionApproximationStrategy:
    importer_collider_type: str
    postprocess_approximation: str | None = None


def resolve_collision_approximation_strategy(name: str) -> CollisionApproximationStrategy:
    if name == "convex_hull":
        return CollisionApproximationStrategy(importer_collider_type="convex_hull")
    if name == "convex_decomposition":
        return CollisionApproximationStrategy(importer_collider_type="convex_decomposition")
    if name == "sdf":
        return CollisionApproximationStrategy(importer_collider_type="convex_hull", postprocess_approximation="sdf")
    raise ValueError(f"Unsupported collision approximation: {name}")


def add_convert_urdf_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=str, help="The path to the input URDF file.")
    parser.add_argument("output", type=str, help="The path to store the USD file.")
    parser.add_argument(
        "--merge-joints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Consolidate links connected by fixed joints. Use --no-merge-joints to disable.",
    )
    parser.add_argument("--fix-base", action="store_true", default=False, help="Fix the base at import pose.")
    parser.add_argument(
        "--collision-approximation",
        type=str,
        choices=["convex_hull", "convex_decomposition", "sdf"],
        default="convex_hull",
        help="Collision approximation for URDF link meshes. SDF is applied as a postprocess.",
    )
    parser.add_argument("--make-instanceable", action="store_true", default=False)
    parser.add_argument("--joint-stiffness", type=float, default=100.0)
    parser.add_argument("--joint-damping", type=float, default=1.0)
    parser.add_argument(
        "--joint-target-type",
        type=str,
        default="position",
        choices=["position", "velocity", "none"],
    )
    parser.add_argument("--disable-sanitize-invalid-names", action="store_true", default=False)
    parser.add_argument("--sanitize-inplace", action="store_true", default=False)
    parser.add_argument("--keep-sanitize-workdir", action="store_true", default=False)
    parser.add_argument("--preview-stage", action="store_true", default=False)
    parser.add_argument("--skip-selection-cleanup", action="store_true", default=False)


def build_arg_parser(
    add_app_launcher_args: Callable[[argparse.ArgumentParser], None] | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Utility to convert a URDF into USD format with optional sanitization.")
    add_convert_urdf_args(parser)
    if add_app_launcher_args is not None:
        add_app_launcher_args(parser)
    return parser


def build_urdf_converter_cfg(
    args: argparse.Namespace,
    *,
    asset_path: str,
    dest_path: str,
    cfg_cls,
    joint_drive_cls=None,
    pd_gains_cls=None,
):
    if joint_drive_cls is None:
        joint_drive_cls = cfg_cls.JointDriveCfg
    if pd_gains_cls is None:
        pd_gains_cls = joint_drive_cls.PDGainsCfg
    collision_strategy = resolve_collision_approximation_strategy(args.collision_approximation)

    return cfg_cls(
        asset_path=asset_path,
        usd_dir=os.path.dirname(dest_path),
        usd_file_name=os.path.basename(dest_path),
        fix_base=args.fix_base,
        merge_fixed_joints=args.merge_joints,
        force_usd_conversion=True,
        make_instanceable=args.make_instanceable,
        collider_type=collision_strategy.importer_collider_type,
        joint_drive=joint_drive_cls(
            gains=pd_gains_cls(
                stiffness=args.joint_stiffness,
                damping=args.joint_damping,
            ),
            target_type=args.joint_target_type,
        ),
    )
