"""Validate that every released task scene can resolve its external assets."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT.parent / "dex2bench_dataset"


def remap_dataset_path(scene_path: Path, raw_path: str, dataset_root: Path) -> Path:
    """Resolve a scene asset, replacing the legacy dataset-root segment."""

    source = Path(raw_path).expanduser()
    parts = source.parts
    if "dex2bench_dataset" in parts:
        suffix = parts[parts.index("dex2bench_dataset") + 1 :]
        return dataset_root.joinpath(*suffix).resolve()
    if source.is_absolute():
        return source.resolve()
    return (scene_path.parent / source).resolve()


def collect_scene_assets(
    scene_paths: Iterable[Path],
    dataset_root: Path,
) -> dict[Path, list[Path]]:
    """Return the file-backed asset paths declared by each task scene."""

    result: dict[Path, list[Path]] = {}
    for scene_path in sorted(scene_paths):
        scene = yaml.safe_load(scene_path.read_text(encoding="utf-8-sig")) or {}
        paths: list[Path] = []
        for spec in (scene.get("assets") or {}).values():
            if isinstance(spec, dict) and isinstance(spec.get("path"), str):
                paths.append(remap_dataset_path(scene_path, spec["path"], dataset_root))
        result[scene_path] = paths
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"External dex2bench_dataset directory (default: {DEFAULT_DATASET_ROOT}).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    scene_paths = sorted((REPO_ROOT / "scenes").glob("[0-9][0-9]_*.yaml"))
    assets_by_scene = collect_scene_assets(scene_paths, args.dataset_root.resolve())
    missing = [
        (scene, asset)
        for scene, assets in assets_by_scene.items()
        for asset in assets
        if not asset.is_file()
    ]
    total = sum(len(assets) for assets in assets_by_scene.values())
    if missing:
        print(f"Missing {len(missing)}/{total} required assets under {args.dataset_root}:")
        for scene, asset in missing:
            print(f"  {scene.name}: {asset}")
        return 1
    print(f"Validated {total} assets for {len(scene_paths)} release tasks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
