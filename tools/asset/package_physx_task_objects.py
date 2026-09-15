#!/usr/bin/env python3
"""Package and repair the PhysX assets used by tasks 100-124."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from pxr import Sdf


@dataclass(frozen=True)
class AssetSpec:
    number: int
    source_id: str
    slug: str
    category: str
    source_usd_stem: str | None = None

    @property
    def usd_stem(self) -> str:
        return self.source_usd_stem or self.source_id


ASSETS = (
    AssetSpec(300, "101408", "mouse", "Computer Mouse"),
    AssetSpec(301, "102839", "dip_switch", "DIP Switch"),
    AssetSpec(302, "101517", "soap_dispenser", "Liquid Soap Dispenser"),
    AssetSpec(303, "102527", "camera", "Camera"),
    AssetSpec(304, "100248", "suitcase", "Suitcase"),
    AssetSpec(305, "100426", "box", "Box"),
    AssetSpec(306, "102970", "pen", "Pen"),
    AssetSpec(307, "102945", "click_pen", "Pen"),
    AssetSpec(308, "103723", "folding_knife", "Knife"),
    AssetSpec(309, "101253", "swiss_army_knife", "Swiss Army Knife"),
    AssetSpec(310, "100491", "shopping_cart", "Shopping Cart"),
    AssetSpec(311, "102074", "pliers", "Pliers"),
    AssetSpec(312, "103619", "spray_bottle", "Dispenser"),
    AssetSpec(313, "885", "dual_faucet", "Faucet"),
    AssetSpec(314, "47185", "cabinet", "Cabinet"),
    AssetSpec(315, "101284", "eyeglasses", "Eyeglasses"),
    AssetSpec(316, "100309", "lighter", "Lighter"),
    AssetSpec(317, "10040", "laptop", "Laptop"),
    AssetSpec(318, "101668", "locking_suitcase", "Suitcase"),
    AssetSpec(319, "10557", "scissors", "Scissors"),
    AssetSpec(320, "100061", "usb_drive", "USB Drive"),
    AssetSpec(321, "47089", "drawer_cabinet", "Storage Cabinet with Drawers"),
    AssetSpec(322, "102645", "toilet", "Toilet"),
    AssetSpec(323, "101612", "safe", "Safe"),
    AssetSpec(324, "102990", "stapler", "Stapler"),
    AssetSpec(325, "148", "faucet", "Faucet", "148_no_collision"),
)


def rewrite_urdf(source: Path, destination: Path, source_id: str) -> None:
    text = source.read_text()
    pattern = rf"(?:\./)?\.\./partseg/{re.escape(source_id)}/objs/"
    text, replacements = re.subn(pattern, "./textured_objs/", text)
    if replacements == 0:
        raise RuntimeError(f"No mesh paths rewritten in {source}")
    destination.write_text(text)


def copy_visual_textures(source_root: Path, asset_root: Path, source_id: str) -> int:
    """Copy source images and the converter's USD-relative texture payload."""
    source_images = source_root / "partseg" / source_id / "images"
    if not source_images.is_dir():
        return 0

    asset_images = asset_root / "images"
    usd_textures = asset_root / "usd_decomposition_linux" / "configuration" / "materials" / "textures"
    shutil.copytree(source_images, asset_images, dirs_exist_ok=True)
    shutil.copytree(source_images, usd_textures, dirs_exist_ok=True)
    return sum(path.is_file() for path in source_images.iterdir())


def rewrite_usd_layer(source: Path, destination: Path, source_stem: str) -> None:
    layer = Sdf.Layer.FindOrOpen(str(source))
    if layer is None:
        raise RuntimeError(f"Cannot open USD layer: {source}")
    text = layer.ExportToString()
    text = text.replace(f"configuration/{source_stem}_", "configuration/mobility_")
    text = text.replace(f"{source_stem}_base.usd", "mobility_base.usd")
    text = text.replace(f"{source_stem}_physics.usd", "mobility_physics.usd")
    text = text.replace(f"{source_stem}_robot.usd", "mobility_robot.usd")
    text = text.replace(f"{source_stem}_sensor.usd", "mobility_sensor.usd")

    destination.parent.mkdir(parents=True, exist_ok=True)
    output = Sdf.Layer.CreateAnonymous(".usda")
    if output is None or not output.ImportFromString(text):
        raise RuntimeError(f"Cannot write USD layer: {destination}")
    if not output.Export(str(destination)):
        raise RuntimeError(f"Cannot export USD layer: {destination}")


def sanitize_articulation_mass_properties(physics_layer_path: Path) -> tuple[int, int]:
    """Let collider-bearing links derive mass/inertia from the scene density.

    The converter authors 1 kg and unit inertia on every rigid body, including
    topology-only links with no collision geometry. An authored positive mass
    takes precedence over ``MassPropertiesCfg.density``, so small articulated
    objects otherwise become multi-kilogram assemblies.

    Collider-bearing links have the conflicting authored properties removed.
    Empty helper links retain a negligible positive mass and inertia because
    PhysX cannot derive either quantity from an empty collision shape.
    """
    layer = Sdf.Layer.FindOrOpen(str(physics_layer_path))
    if layer is None:
        raise RuntimeError(f"Cannot open USD layer: {physics_layer_path}")
    scene = layer.GetPrimAtPath("/scene")
    if scene is None:
        raise RuntimeError(f"Missing /scene in USD layer: {physics_layer_path}")

    computed_links = 0
    helper_links = 0
    inertia_names = (
        "physics:centerOfMass",
        "physics:diagonalInertia",
        "physics:principalAxes",
    )
    for body in scene.nameChildren:
        mass_attr = body.properties.get("physics:mass")
        if mass_attr is None:
            continue
        collisions = body.nameChildren.get("collisions")
        collision_refs = tuple(collisions.referenceList.prependedItems) if collisions is not None else ()
        if collision_refs:
            body.RemoveProperty(mass_attr)
            for name in inertia_names:
                prop = body.properties.get(name)
                if prop is not None:
                    body.RemoveProperty(prop)
            computed_links += 1
            continue

        # Empty links only carry articulation topology. Keep their contribution
        # six orders of magnitude below a gram-scale manipulated object.
        mass_attr.default = 1.0e-6
        diagonal_inertia = body.properties.get("physics:diagonalInertia")
        if diagonal_inertia is not None:
            diagonal_inertia.default = (1.0e-9, 1.0e-9, 1.0e-9)
        helper_links += 1

    if computed_links == 0:
        raise RuntimeError(f"No collider-bearing rigid links found in {physics_layer_path}")
    if not layer.Save():
        raise RuntimeError(f"Cannot save USD layer: {physics_layer_path}")
    return computed_links, helper_links


def package_asset(source_root: Path, objects_root: Path, spec: AssetSpec) -> Path:
    object_root = objects_root / f"{spec.number:03d}_{spec.slug}"
    asset_root = object_root / f"01_{spec.slug}" / "01"
    usd_root = asset_root / "usd_decomposition_linux"
    configuration_root = usd_root / "configuration"

    if object_root.exists():
        raise FileExistsError(f"Destination already exists: {object_root}")

    asset_root.mkdir(parents=True)
    shutil.copytree(source_root / "partseg" / spec.source_id / "objs", asset_root / "textured_objs")
    rewrite_urdf(source_root / "urdf" / f"{spec.source_id}.urdf", asset_root / "mobility.urdf", spec.source_id)
    shutil.copy2(source_root / "finaljson" / f"{spec.source_id}.json", asset_root / "mobility_v2.json")

    source_usd_root = source_root / "usd_sdf_linux"
    rewrite_usd_layer(source_usd_root / f"{spec.usd_stem}.usd", usd_root / "mobility.usd", spec.usd_stem)
    for suffix in ("base", "physics", "robot", "sensor"):
        rewrite_usd_layer(
            source_usd_root / "configuration" / f"{spec.usd_stem}_{suffix}.usd",
            configuration_root / f"mobility_{suffix}.usd",
            spec.usd_stem,
        )
    copy_visual_textures(source_root, asset_root, spec.source_id)
    sanitize_articulation_mass_properties(configuration_root / "mobility_physics.usd")

    for name in ("config.yaml", ".asset_hash"):
        source = source_usd_root / name
        if source.exists():
            shutil.copy2(source, usd_root / name)

    metadata = {
        "category": spec.category,
        "source": "PhysX_mobility",
        "source_id": spec.source_id,
        "task_object_number": spec.number,
        "usd_collision_approximation": "none" if spec.source_id == "148" else "convex_decomposition",
    }
    (asset_root / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return asset_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("../PhysX_mobility"))
    parser.add_argument(
        "--objects-root",
        type=Path,
        default=Path("../dex2bench_dataset/Objects"),
    )
    parser.add_argument(
        "--sanitize-existing",
        action="store_true",
        help="Repair mass properties in already packaged Objects directories.",
    )
    parser.add_argument(
        "--copy-existing-textures",
        action="store_true",
        help="Copy source images into already packaged Objects and USD texture directories.",
    )
    args = parser.parse_args()

    if args.copy_existing_textures:
        for spec in ASSETS:
            candidates = list(
                args.objects_root.glob(
                    f"{spec.number:03d}_{spec.slug}/01_{spec.slug}/01"
                )
            )
            if not candidates:
                print(f"{spec.number}: skipped (not packaged)")
                continue
            count = copy_visual_textures(args.source_root, candidates[0], spec.source_id)
            print(f"{spec.number}: copied {count} texture file(s)")
        return

    if args.sanitize_existing:
        for spec in ASSETS:
            physics_layer = (
                args.objects_root
                / f"{spec.number:03d}_{spec.slug}"
                / f"01_{spec.slug}"
                / "01"
                / "usd_decomposition_linux"
                / "configuration"
                / "mobility_physics.usd"
            )
            if not physics_layer.exists():
                print(f"{spec.number}: skipped (not packaged)")
                continue
            computed, helpers = sanitize_articulation_mass_properties(physics_layer)
            print(f"{spec.number}: repaired {computed} collider links, {helpers} helper links")
        return

    for spec in ASSETS:
        result = package_asset(args.source_root, args.objects_root, spec)
        print(f"{spec.number}: {spec.source_id} -> {result}")


if __name__ == "__main__":
    main()
