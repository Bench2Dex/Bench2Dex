"""Audit scene generalization visual asset splits.

This script is intentionally lightweight so it can run outside Isaac Sim.  It
checks the two discrete visual asset pools that must be split for seen/unseen
evaluation: table-surface textures and iTHOR scene backgrounds.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_import_stubs() -> None:
    try:
        import isaaclab.utils.io  # type: ignore  # noqa: F401
    except Exception:
        import yaml

        isaaclab_mod = types.ModuleType("isaaclab")
        isaaclab_utils_mod = types.ModuleType("isaaclab.utils")
        isaaclab_utils_io_mod = types.ModuleType("isaaclab.utils.io")

        def _load_yaml(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)

        isaaclab_utils_io_mod.load_yaml = _load_yaml
        sys.modules.setdefault("isaaclab", isaaclab_mod)
        sys.modules.setdefault("isaaclab.utils", isaaclab_utils_mod)
        sys.modules.setdefault("isaaclab.utils.io", isaaclab_utils_io_mod)

    try:
        import pxr  # type: ignore  # noqa: F401
    except Exception:
        pxr_mod = types.ModuleType("pxr")
        pxr_mod.Usd = types.SimpleNamespace()
        pxr_mod.UsdGeom = types.SimpleNamespace()
        sys.modules.setdefault("pxr", pxr_mod)


def _asset_key(asset) -> str:
    return str(getattr(asset, "asset_id", "") or getattr(asset, "uri", ""))


def _split_category_counts(assets) -> dict:
    counts: dict[str, dict[str, int]] = {}
    for asset in assets:
        split = str(getattr(asset, "split", "seen") or "seen")
        category = str(getattr(asset, "category", "") or "uncategorized")
        counts.setdefault(split, {})
        counts[split][category] = counts[split].get(category, 0) + 1
    return {split: dict(sorted(categories.items())) for split, categories in sorted(counts.items())}


def _split_intersections(assets) -> dict:
    by_key: dict[str, set[str]] = {}
    for asset in assets:
        by_key.setdefault(_asset_key(asset), set()).add(str(getattr(asset, "split", "train") or "train"))
    return {key: sorted(splits) for key, splits in sorted(by_key.items()) if len(splits) > 1}


def _enabled_category_report(enabled_categories) -> list[str]:
    if isinstance(enabled_categories, dict):
        return [
            f"{kind}:{category}"
            for kind, categories in sorted(enabled_categories.items())
            for category in categories
        ] or ["<all>"]
    values = list(enabled_categories or ())
    return sorted(str(value) for value in values) if values else ["<all>"]


def _asset_enabled(asset, enabled_categories) -> bool:
    if not enabled_categories:
        return True
    asset_kind = str(getattr(asset, "kind", "") or "")
    asset_category = str(getattr(asset, "category", "") or "")
    if isinstance(enabled_categories, dict):
        categories = enabled_categories.get(asset_kind) or enabled_categories.get("*", ())
        return asset_category in set(categories)
    return asset_category in set(enabled_categories)


def audit(config_path: str) -> dict:
    _install_import_stubs()
    from build.generalization import load_scene_generalization_config

    cfg = load_scene_generalization_config(config_path)
    background = cfg.appearance.background
    table_surface = cfg.appearance.table_surface

    enabled_categories = getattr(background, "enabled_categories", ()) or ()
    active_background_assets = [
        asset for asset in background.assets
        if _asset_enabled(asset, enabled_categories)
    ]
    active_hdr_background_assets = [
        getattr(asset, "uri", "")
        for asset in active_background_assets
        if str(getattr(asset, "kind", "")) == "image" or str(getattr(asset, "uri", "")).lower().endswith(".hdr")
    ]

    report = {
        "config": str(Path(config_path).resolve()),
        "asset_split": cfg.asset_split,
        "table_surface": {
            "total": len(table_surface.assets),
            "split_category_counts": _split_category_counts(table_surface.assets),
            "split_intersections": _split_intersections(table_surface.assets),
        },
        "background": {
            "configured_total": len(background.assets),
            "active_total": len(active_background_assets),
            "active_categories": _enabled_category_report(enabled_categories),
            "active_hdr_count": len(active_hdr_background_assets),
            "active_hdr_assets": active_hdr_background_assets,
            "split_category_counts": _split_category_counts(active_background_assets),
            "split_intersections": _split_intersections(active_background_assets),
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit scene generalization visual asset splits.")
    parser.add_argument("config", nargs="?", default="configs/scene/generalization.yaml")
    args = parser.parse_args()

    report = audit(args.config)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    has_errors = bool(
        report["table_surface"]["split_intersections"]
        or report["background"]["split_intersections"]
        or report["background"]["active_hdr_count"]
    )
    return 1 if has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
