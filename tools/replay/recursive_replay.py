"""Recursively replay all HDF5 episodes under a root directory.

For each ``episode_*.hdf5`` file found (recursively, at any depth) that does
NOT end with ``_replay.hdf5``, run replay.py and save the output next to the
original as ``<stem>_replay.hdf5``. Files that already have a corresponding
``_replay.hdf5`` sibling are skipped, which lets you incrementally collect new
data and only replay the missing ones.

Usage:
    python tools/replay/recursive_replay.py
    python tools/replay/recursive_replay.py --root outputs/test --enable-rgb
    python tools/replay/recursive_replay.py --root outputs/test --enable-depth
S
    # New-style resample generalization (mirrors batch_replay.py):
    python tools/replay/recursive_replay.py \
        --root ../teleopdata_ckpt/dataset/06_fruit_bowl_loading/origin-generalization-narrow-h-double \
        --enable-rgb \
        --resample-groups background,table_surface,light \
        --generalization-split seen

Default ``--root`` is ``outputs/test``.

Capture flags mirror batch_replay.py:
    --enable-rgb / --enable-depth / --enable-tactile / --tactile-only
    (no flag => --enable-rgb for backward compat)

Generalization options:
    Default behavior restores the exact generalization stored in each HDF5
    (``--restore-generalization``).
    ``--resample-groups`` switches replay to current config selective resampling.
    Safe visual groups are: background, table_surface, light, camera.

Environment variables:
    OVERWRITE=1   Re-replay even if ``<stem>_replay.hdf5`` already exists.
                  Default 0 (skip existing).


python tools/replay/batch_replay.py \
    --origin-dir ../teleopdata_ckpt/dataset/06_fruit_bowl_loading/origin-generalization \
    --replay-dir ../teleopdata_ckpt/dataset/06_fruit_bowl_loading/replay-generalization \
    --enable-rgb \
    --resample-groups background,table_surface,light \
    --generalization-split seen
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

DEFAULT_ROOT = "outputs/test"
REPO_DIR = Path(__file__).resolve().parents[2]


# ── Args ─────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively replay all episode_*.hdf5 files under a root directory."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=DEFAULT_ROOT,
        help=f"Root directory to scan recursively. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument("--enable-rgb", action="store_true")
    parser.add_argument("--enable-depth", action="store_true")
    parser.add_argument("--enable-tactile", action="store_true")
    parser.add_argument("--tactile-only", action="store_true")
    parser.add_argument(
        "--resample-groups",
        type=str,
        default=None,
        help="Comma-separated scene generalization groups to resample from current config "
             "(e.g. background,table_surface,light). When omitted, restores from HDF5.",
    )
    parser.add_argument(
        "--generalization-config",
        type=str,
        default=None,
        help="Path to scene generalization YAML. Defaults to configs/scene/generalization.yaml.",
    )
    parser.add_argument(
        "--generalization-split",
        type=str,
        default=None,
        choices=["seen", "unseen", "all"],
        help="Discrete visual asset split for resampled scene generalization.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be replayed without executing.",
    )
    return parser.parse_args()


def _resolve_capture_flags(ns: argparse.Namespace) -> tuple[bool, bool, bool, bool]:
    enable_rgb = bool(ns.enable_rgb)
    enable_depth = bool(ns.enable_depth)
    enable_tactile = bool(ns.enable_tactile)
    tactile_only = bool(ns.tactile_only)
    if tactile_only:
        return False, False, True, True
    if not (enable_rgb or enable_depth or enable_tactile):
        enable_rgb = True
    return enable_rgb, enable_depth, enable_tactile, False


# ── Discovery ────────────────────────────────────────────────────────────────

def _is_replay_file(path: Path) -> bool:
    return path.stem.endswith("_replay")


def _replay_sibling(path: Path) -> Path:
    """Return the expected replay sibling path: ``<stem>_replay.hdf5``."""
    return path.with_name(f"{path.stem}_replay{path.suffix}")


def _discover_targets(root: Path) -> list[Path]:
    targets: list[Path] = []
    for hdf5_path in root.rglob("episode_*.hdf5"):
        if _is_replay_file(hdf5_path):
            continue
        targets.append(hdf5_path)
    return sorted(targets)


# ── Command ──────────────────────────────────────────────────────────────────

def _resamples_background(resample_groups: str | None) -> bool:
    groups = {g.strip() for g in str(resample_groups or "").split(",") if g.strip()}
    return "background" in groups


def _build_replay_command(
    hdf5_path: Path,
    output_path: Path,
    *,
    enable_rgb: bool,
    enable_depth: bool,
    enable_tactile: bool,
    tactile_only: bool,
    resample_groups: str | None,
    generalization_config: str | None,
    generalization_split: str | None,
    background_asset_index: int | None,
) -> str:
    parts = [
        sys.executable,
        "replay.py",
        "--hdf5",
        str(hdf5_path),
        "--output",
        str(output_path),
        "--headless",
    ]
    if resample_groups:
        parts.extend(["--enable-generalization", "--resample-groups", resample_groups])
        if generalization_config:
            parts.extend(["--generalization-config", generalization_config])
        if generalization_split:
            parts.extend(["--generalization-split", generalization_split])
        if background_asset_index is not None:
            parts.extend(["--background-asset-index", str(background_asset_index)])
    else:
        parts.append("--restore-generalization")
    if tactile_only:
        parts.append("--tactile-only")
    else:
        if enable_rgb:
            parts.append("--enable-rgb")
        if enable_depth:
            parts.append("--enable-depth")
        if enable_tactile:
            parts.append("--enable-tactile")
    return " ".join(shlex.quote(p) for p in parts)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    args = _parse_args()
    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"[ERROR] root is not a directory: {root}")
        return 1

    enable_rgb, enable_depth, enable_tactile, tactile_only = _resolve_capture_flags(args)
    overwrite = os.environ.get("OVERWRITE", "0") == "1"
    resample_groups = args.resample_groups
    generalization_config = args.generalization_config
    generalization_split = args.generalization_split

    targets = _discover_targets(root)
    if not targets:
        print(f"[recursive_replay] No episode_*.hdf5 files found under {root}")
        return 0

    pending: list[tuple[Path, Path]] = []
    skipped: list[Path] = []
    for src in targets:
        dst = _replay_sibling(src)
        if dst.exists() and not overwrite:
            skipped.append(src)
        else:
            pending.append((src, dst))

    print(f"[recursive_replay] Root      : {root}")
    print(f"[recursive_replay] Found     : {len(targets)} non-replay episodes")
    print(f"[recursive_replay] Skipped   : {len(skipped)} (replay sibling exists)")
    print(f"[recursive_replay] Pending   : {len(pending)}")
    print(f"[recursive_replay] OVERWRITE : {1 if overwrite else 0}")
    print(
        "[recursive_replay] Capture   : "
        f"rgb={enable_rgb}, depth={enable_depth}, tactile={enable_tactile}, "
        f"tactile_only={tactile_only}"
    )
    if resample_groups:
        print(
            "[recursive_replay] Generalization: "
            f"resample_groups={resample_groups}, "
            f"split={generalization_split or '<config>'}, "
            f"config={generalization_config or '<default>'}"
        )
        if _resamples_background(resample_groups):
            print("[recursive_replay] Background sequence: episode order maps to filtered background index, cycling if needed")
    else:
        print("[recursive_replay] Generalization: restore from HDF5")

    if args.dry_run or not pending:
        for src, dst in pending:
            print(f"[DRY] {src} -> {dst}")
        if args.dry_run:
            return 0

    # Replay must run from repo root so that ``replay.py`` is on cwd path.
    os.chdir(REPO_DIR)

    ok_count = 0
    fail_count = 0
    failed: list[str] = []

    for idx, (src, dst) in enumerate(pending, start=1):
        cmd = _build_replay_command(
            src,
            dst,
            enable_rgb=enable_rgb,
            enable_depth=enable_depth,
            enable_tactile=enable_tactile,
            tactile_only=tactile_only,
            resample_groups=resample_groups,
            generalization_config=generalization_config,
            generalization_split=generalization_split,
            background_asset_index=(idx - 1) if _resamples_background(resample_groups) else None,
        )
        rel = src.relative_to(root) if src.is_relative_to(root) else src
        print(f"\n{'=' * 80}")
        print(f"[episode {idx}/{len(pending)}] {rel}")
        print(f"CMD: {cmd}")
        print(f"{'=' * 80}\n")

        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"\n[ERROR] Replay failed: {src}, exit={exc.returncode}")
            fail_count += 1
            failed.append(str(src))
            continue

        if dst.exists():
            print(f"[OK] {src.name} -> {dst}")
            ok_count += 1
        else:
            print(f"[ERROR] Output file missing: {dst}")
            fail_count += 1
            failed.append(str(src))

    print(f"\n{'=' * 80}")
    print("[recursive_replay] Done")
    print(f"  OK:      {ok_count}")
    print(f"  Skipped: {len(skipped)}")
    print(f"  Failed:  {fail_count}")
    if failed:
        print("[recursive_replay] Failed episodes:")
        for name in failed:
            print(f"  - {name}")
    return 0 if fail_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
