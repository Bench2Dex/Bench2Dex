from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


_USD_SUFFIXES = (".usd", ".usda", ".usdc")
_ASSET_PATH_PATTERN = re.compile(r"@([^@\r\n]+)@")
_QUOTED_PATH_PATTERN = re.compile(r'"((?:file://)?(?:/|[A-Za-z]:[\\/])[^"\r\n]+)"')
_WINDOWS_ABS_PATTERN = re.compile(r"^[A-Za-z]:[\\/]")


def _is_absolute_asset_path(path_str: str) -> bool:
    if not path_str:
        return False
    if path_str.startswith("/"):
        return True
    return _WINDOWS_ABS_PATTERN.match(path_str) is not None


def _to_posix(path_str: str) -> str:
    return path_str.replace("\\", "/")


def _iter_usd_layers(root_path: Path) -> list[Path]:
    if root_path.is_file():
        base_dir = root_path.parent
    else:
        base_dir = root_path
    layers: list[Path] = []
    for suffix in _USD_SUFFIXES:
        layers.extend(base_dir.rglob(f"*{suffix}"))
    return sorted({path.resolve() for path in layers if path.is_file()})


def _make_relative_path(
    raw: str,
    *,
    layer_path: Path,
    package_root: Path,
    external_targets: set[str],
) -> str | None:
    normalized = raw[7:] if raw.startswith("file://") else raw
    if not _is_absolute_asset_path(normalized):
        return None

    target_path = Path(normalized)
    try:
        relative = os.path.relpath(target_path, start=layer_path.parent)
    except ValueError:
        return None

    relative_posix = _to_posix(relative)
    if target_path.is_absolute():
        try:
            target_path.relative_to(package_root)
        except ValueError:
            external_targets.add(normalized)
    if relative_posix == raw:
        return None
    return relative_posix


def _rewrite_stage_asset_paths(layer_path: Path, package_root: Path) -> tuple[int, list[str]]:
    from pxr import Sdf, Usd

    stage = Usd.Stage.Open(layer_path.as_posix())
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {layer_path}")

    external_targets: set[str] = set()
    replacements = 0

    for prim in stage.TraverseAll():
        for attr in prim.GetAttributes():
            type_name = attr.GetTypeName()
            if type_name == Sdf.ValueTypeNames.Asset:
                value = attr.Get()
                if not isinstance(value, Sdf.AssetPath):
                    continue
                rewritten = _make_relative_path(
                    value.path,
                    layer_path=layer_path,
                    package_root=package_root,
                    external_targets=external_targets,
                )
                if rewritten is None:
                    continue
                attr.Set(Sdf.AssetPath(rewritten))
                replacements += 1
            elif type_name == Sdf.ValueTypeNames.AssetArray:
                values = attr.Get()
                if not values:
                    continue
                updated_values = []
                changed = False
                for value in values:
                    if not isinstance(value, Sdf.AssetPath):
                        updated_values.append(value)
                        continue
                    rewritten = _make_relative_path(
                        value.path,
                        layer_path=layer_path,
                        package_root=package_root,
                        external_targets=external_targets,
                    )
                    if rewritten is None:
                        updated_values.append(value)
                        continue
                    updated_values.append(Sdf.AssetPath(rewritten))
                    replacements += 1
                    changed = True
                if changed:
                    attr.Set(updated_values)

    if replacements:
        stage.GetRootLayer().Save()

    return replacements, sorted(external_targets)


def _relativize_text_paths(text: str, *, layer_path: Path, package_root: Path) -> tuple[str, int, list[str]]:
    external_targets: set[str] = set()
    replacements = 0

    def _to_relative(raw: str) -> str | None:
        nonlocal replacements
        relative_posix = _make_relative_path(
            raw,
            layer_path=layer_path,
            package_root=package_root,
            external_targets=external_targets,
        )
        if relative_posix is None:
            return None
        replacements += 1
        return relative_posix

    def _replace_asset(match: re.Match[str]) -> str:
        raw = match.group(1)
        rewritten = _to_relative(raw)
        if rewritten is None:
            return match.group(0)
        return f"@{rewritten}@"

    def _replace_quoted(match: re.Match[str]) -> str:
        raw = match.group(1)
        rewritten = _to_relative(raw)
        if rewritten is None:
            return match.group(0)
        return f'"{rewritten}"'

    updated = _ASSET_PATH_PATTERN.sub(_replace_asset, text)
    updated = _QUOTED_PATH_PATTERN.sub(_replace_quoted, updated)
    return updated, replacements, sorted(external_targets)


def _rewrite_layer_asset_paths(layer_path: Path, package_root: Path) -> tuple[int, list[str]]:
    from pxr import Sdf

    stage_replacements, stage_external_targets = _rewrite_stage_asset_paths(layer_path, package_root)

    layer = Sdf.Layer.FindOrOpen(layer_path.as_posix())
    if layer is None:
        raise RuntimeError(f"Failed to open USD layer: {layer_path}")

    original = layer.ExportToString()
    updated, replacements, external_targets = _relativize_text_paths(
        original,
        layer_path=layer_path,
        package_root=package_root,
    )
    if updated != original:
        result = layer.ImportFromString(updated)
        if result is False:
            raise RuntimeError(f"Failed to import rewritten USD text for layer: {layer_path}")
        layer.Save()

    total_replacements = stage_replacements + replacements
    all_external_targets = sorted(set(stage_external_targets).union(external_targets))
    return total_replacements, all_external_targets


def relativize_usd_asset_tree(root_path: str | Path) -> dict[str, object]:
    root = Path(root_path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"USD root path does not exist: {root}")
    package_root = root.parent if root.is_file() else root

    total_replacements = 0
    changed_layers: list[str] = []
    external_targets: set[str] = set()

    for layer_path in _iter_usd_layers(root):
        replacements, layer_external_targets = _rewrite_layer_asset_paths(layer_path, package_root)
        if replacements:
            total_replacements += replacements
            changed_layers.append(str(layer_path))
        external_targets.update(layer_external_targets)

    return {
        "root": str(root),
        "layers_changed": changed_layers,
        "replacement_count": total_replacements,
        "external_absolute_targets": sorted(external_targets),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite absolute USD asset paths to relative paths.")
    parser.add_argument("path", type=str, help="Root USD file or directory to rewrite.")
    args = parser.parse_args()

    try:
        from pxr import Sdf  # noqa: F401
    except ModuleNotFoundError:
        from isaaclab.app import AppLauncher

        bootstrap_parser = argparse.ArgumentParser(add_help=False)
        AppLauncher.add_app_launcher_args(bootstrap_parser)
        bootstrap_args = bootstrap_parser.parse_args(["--headless"])
        simulation_app = AppLauncher(bootstrap_args).app
    else:
        simulation_app = None

    try:
        summary = relativize_usd_asset_tree(args.path)
        print(
            f"[relativize-usd] updated {summary['replacement_count']} asset path(s) "
            f"across {len(summary['layers_changed'])} layer(s)"
        )
        if summary["external_absolute_targets"]:
            print("[relativize-usd] warning: some targets are outside the asset package:")
            for target in summary["external_absolute_targets"]:
                print(f"  - {target}")
    finally:
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    main()
