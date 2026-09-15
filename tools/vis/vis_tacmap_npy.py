#!/usr/bin/env python3
"""Visualize TacMap .npy point clouds and normals.

Examples:
    # Single tactile_sensor directory, save one image and do not open a GUI.
    python tools/vis/vis_tacmap_npy.py \
        ../dex2bench_dataset/Robots_p/kuka+sharpa/tactile_sensor \
        --output /tmp/tacmap_kuka_sharpa.png
        --apply-runtime-flip
        
    # Single directory, only render selected TacMap groups.
    python tools/vis/vis_tacmap_npy.py \
        ../dex2bench_dataset/Robots_p/panda+allegro/tactile_sensor \
        --groups RTH R4F \
        --output /tmp/tacmap_panda_allegro_right.png

    # Batch mode: scan every Robots_p/*/tactile_sensor and save one image per robot.
    python tools/vis/vis_tacmap_npy.py \
        --dataset-root ../dex2bench_dataset \
        --output-dir /tmp/tacmap_all

    # Batch mode, only render selected groups.
    python tools/vis/vis_tacmap_npy.py \
        --dataset-root ../dex2bench_dataset \
        --output-dir /tmp/tacmap_all_right \
        --groups RTH R4F

Notes:
    Passing --output or --output-dir disables GUI display and uses the Agg backend.
    Without --output / --output-dir, the script opens a matplotlib GUI window.
    A thick red arrow shows the macro raycast direction computed from the mean
    normal of each group. Use --apply-runtime-flip to include TacMap runtime
    flip_normals settings when judging the actual replay ray direction.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from collector.tacmap_configs import ROBOT_KEY_TO_TACMAP_CFG


COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red",
    "tab:purple", "tab:brown", "tab:pink", "tab:olive",
]


def _discover_groups(directory: Path) -> list[dict[str, str]]:
    groups = []
    for pt_path in sorted(directory.glob("*_point.npy")):
        stem = pt_path.name.removesuffix("_point.npy")
        nr_path = directory / f"{stem}_normal.npy"
        if nr_path.is_file():
            key = stem.removeprefix("tactileSensor_map_")
            groups.append({"key": key, "points": pt_path.name, "normals": nr_path.name})
    return groups


def _filter_groups(groups: list[dict[str, str]], selected_names: list[str] | None) -> list[dict[str, str]]:
    if not selected_names:
        return groups
    selected = {name.upper() for name in selected_names}
    return [group for group in groups if group["key"].upper() in selected]


def _figure_title(directory: Path) -> str:
    return directory.parent.name if directory.name == "tactile_sensor" else directory.name


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.+-]+", "_", value.strip())
    return safe.strip("._") or "tacmap"


def _find_tactile_sensor_dirs(dataset_root: Path) -> list[Path]:
    root = dataset_root.resolve()
    robots_root = root / "Robots_p" if (root / "Robots_p").is_dir() else root
    return sorted(path for path in robots_root.glob("*/tactile_sensor") if path.is_dir())


def _robot_flip_normals(directory: Path) -> bool:
    robot_dir = directory.parent.name if directory.name == "tactile_sensor" else directory.name
    for cfg in ROBOT_KEY_TO_TACMAP_CFG.values():
        if cfg.dataset_key == robot_dir:
            return bool(cfg.flip_normals)
    return False


def _normalize_vector(vector: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        return None
    return vector / norm


def _macro_direction(normals: np.ndarray) -> np.ndarray | None:
    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    lengths = np.linalg.norm(normals, axis=1)
    valid = np.isfinite(lengths) & (lengths > 1e-12)
    if not np.any(valid):
        return None
    unit_normals = normals[valid] / lengths[valid, None]
    return _normalize_vector(np.mean(unit_normals, axis=0))


def _plot_macro_arrow(
    ax,
    points: np.ndarray,
    direction: np.ndarray | None,
    *,
    length: float,
    label: str,
) -> None:
    if direction is None:
        return
    center = np.mean(points, axis=0)
    ax.quiver(
        center[0], center[1], center[2],
        direction[0], direction[1], direction[2],
        color="red", linewidth=4.0, length=length, normalize=True,
        arrow_length_ratio=0.28,
    )
    ax.text(
        center[0] + direction[0] * length,
        center[1] + direction[1] * length,
        center[2] + direction[2] * length,
        label,
        color="red",
        fontsize=9,
        weight="bold",
    )


def _plot_group(
    ax,
    points: np.ndarray,
    normals: np.ndarray,
    title: str,
    color: str,
    quiver_length: float,
    *,
    macro_direction: np.ndarray | None,
    macro_length: float,
    macro_label: str,
):
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.3, color=color, alpha=0.6)
    ax.quiver(
        points[:, 0], points[:, 1], points[:, 2],
        normals[:, 0], normals[:, 1], normals[:, 2],
        color=color, alpha=0.5, linewidth=0.5, length=quiver_length, normalize=True,
    )
    _plot_macro_arrow(ax, points, macro_direction, length=macro_length, label=macro_label)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_aspect("equal")
    if macro_direction is None:
        ax.set_title(f"{title}\nmacro: unavailable")
    else:
        ax.set_title(
            f"{title}\nmacro=({macro_direction[0]:+.2f}, {macro_direction[1]:+.2f}, {macro_direction[2]:+.2f})"
        )


def _render_directory(
    *,
    plt,
    directory: Path,
    output_path: Path | None,
    selected_groups: list[str] | None,
    down_sample: int,
    quiver_length: float,
    macro_arrow: bool,
    macro_arrow_length: float,
    apply_runtime_flip: bool,
) -> bool:
    groups = _discover_groups(directory)
    if not groups:
        print(f"[vis_tacmap_npy] skip {directory}: no *_point.npy / *_normal.npy pairs")
        return False

    groups = _filter_groups(groups, selected_groups)
    if not groups:
        print(f"[vis_tacmap_npy] skip {directory}: no matching groups for {selected_groups}")
        return False

    ds = max(1, down_sample)
    flip_normals = _robot_flip_normals(directory) if apply_runtime_flip else False
    ncols = min(len(groups), 4)
    nrows = (len(groups) + ncols - 1) // ncols
    fig = plt.figure(figsize=(6 * ncols, 6 * nrows))

    for i, group in enumerate(groups):
        points_path = directory / group["points"]
        pts_full = np.load(points_path)
        pts = pts_full[::ds, ::ds].reshape(-1, 3)
        nrm = np.load(directory / group["normals"])[::ds, ::ds].reshape(-1, 3)

        span = float(np.max(pts.max(axis=0) - pts.min(axis=0)))
        ql = quiver_length if quiver_length > 0 else span * 0.08
        macro_length = macro_arrow_length if macro_arrow_length > 0 else span * 0.35
        macro_dir = _macro_direction(nrm)
        if flip_normals and macro_dir is not None:
            macro_dir = -macro_dir
        macro_label = "macro ray"
        if apply_runtime_flip and flip_normals:
            macro_label += "\n(runtime flipped)"

        color = COLORS[i % len(COLORS)]
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        _plot_group(
            ax,
            pts,
            nrm,
            group["key"],
            color,
            ql,
            macro_direction=macro_dir if macro_arrow else None,
            macro_length=macro_length,
            macro_label=macro_label,
        )
        macro_text = "unavailable" if macro_dir is None else f"({macro_dir[0]:+.3f}, {macro_dir[1]:+.3f}, {macro_dir[2]:+.3f})"
        print(
            f"{directory.parent.name}/{group['key']}: {pts.shape[0]} points (ds={ds}), "
            f"shape={pts_full.shape}, macro_ray={macro_text}"
        )

    fig.suptitle(_figure_title(directory), fontsize=14)
    plt.tight_layout()
    if output_path is not None:
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[vis_tacmap_npy] wrote {output_path}")
    else:
        plt.show()
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, help="Directory containing *_point.npy / *_normal.npy pairs")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Batch mode: dataset root containing Robots_p/*/tactile_sensor.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Batch mode output directory; one PNG is written per robot tactile_sensor directory.")
    parser.add_argument("--down-sample", type=int, default=4, help="Keep every N-th point along each axis")
    parser.add_argument("--quiver-length", type=float, default=0.0, help="Normal arrow length (0 = auto)")
    parser.add_argument("--no-macro-arrow", action="store_true", help="Do not draw the thick red macro raycast direction arrow.")
    parser.add_argument("--macro-arrow-length", type=float, default=0.0, help="Macro arrow length (0 = auto)")
    parser.add_argument(
        "--apply-runtime-flip",
        action="store_true",
        help="Apply TacMap runtime flip_normals when drawing the macro ray direction.",
    )
    parser.add_argument("--groups", nargs="*", default=None, help="Only show these groups (e.g. RTH R4F)")
    parser.add_argument("--output", type=Path, default=None, help="Save figure to this image path instead of opening a GUI window.")
    args = parser.parse_args()

    batch_mode = args.dataset_root is not None
    if batch_mode and args.output_dir is None:
        raise SystemExit("--output-dir is required with --dataset-root")
    if batch_mode and args.output is not None:
        raise SystemExit("--output cannot be used with --dataset-root; use --output-dir")
    if not batch_mode and args.path is None:
        raise SystemExit("path is required unless --dataset-root is used")

    if args.output is not None or args.output_dir is not None:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if batch_mode:
        tactile_dirs = _find_tactile_sensor_dirs(args.dataset_root)
        if not tactile_dirs:
            raise SystemExit(f"No tactile_sensor directories found under {args.dataset_root}")

        output_dir = args.output_dir.resolve()
        written = 0
        for directory in tactile_dirs:
            robot_name = directory.parent.name
            output_path = output_dir / f"{_safe_filename(robot_name)}_tacmap.png"
            if _render_directory(
                plt=plt,
                directory=directory,
                output_path=output_path,
                selected_groups=args.groups,
                down_sample=args.down_sample,
                quiver_length=args.quiver_length,
                macro_arrow=not args.no_macro_arrow,
                macro_arrow_length=args.macro_arrow_length,
                apply_runtime_flip=args.apply_runtime_flip,
            ):
                written += 1
        print(f"[vis_tacmap_npy] batch complete: wrote {written}/{len(tactile_dirs)} figures to {output_dir.resolve()}")
        return

    directory = args.path.resolve()
    if not directory.is_dir():
        raise SystemExit(f"Not a directory: {directory}")
    output_path = args.output.resolve() if args.output is not None else None
    ok = _render_directory(
        plt=plt,
        directory=directory,
        output_path=output_path,
        selected_groups=args.groups,
        down_sample=args.down_sample,
        quiver_length=args.quiver_length,
        macro_arrow=not args.no_macro_arrow,
        macro_arrow_length=args.macro_arrow_length,
        apply_runtime_flip=args.apply_runtime_flip,
    )
    if not ok:
        raise SystemExit(f"No renderable *_point.npy / *_normal.npy pairs found in {directory}")


if __name__ == "__main__":
    main()
