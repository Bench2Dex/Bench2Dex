from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_YCB_ROOT = Path("../../datasets/ycb_spaien")
DEFAULT_MAP_FILE = DEFAULT_YCB_ROOT / "robotwin_semantic_import_map.txt"
DEFAULT_CONVERT_MESH = Path("../../code/IsaacLab/scripts/tools/convert_mesh.py")
DEFAULT_CONVERT_URDF = Path("../../code/IsaacLab/scripts/tools/convert_urdf_.py")
DEFAULT_ISAACLAB_SOURCE = Path("../../code/IsaacLab/source")

DEFAULT_MASS_KG = 0.5

OBJ_SKIP_NAMES = {"coacd_collision.obj"}
URDF_SKIP_SUFFIXES = ("_sanitized.urdf",)


MASS_BY_SEMANTIC = {
    "alarm-clock": 0.35,
    "baguette": 0.25,
    "basket": 0.30,
    "battery": 0.05,
    "bell": 0.30,
    "block": 0.20,
    "board": 0.60,
    "book": 0.60,
    "bookcase": 4.00,
    "boxdrink": 0.40,
    "bread": 0.40,
    "breadbasket": 0.35,
    "brush": 0.08,
    "brush-pen": 0.03,
    "cabinet": 15.00,
    "calculator": 0.20,
    "callbell": 0.20,
    "can": 0.35,
    "candlestick": 0.50,
    "chips-tub": 0.12,
    "cleaner": 0.80,
    "coaster": 0.03,
    "coffee-box": 0.40,
    "cube": 0.20,
    "cup": 0.25,
    "cup-with-liquid": 0.35,
    "displaystand": 0.60,
    "drill": 1.80,
    "dumbbell": 2.00,
    "dumbbell-rack": 2.50,
    "dustbin": 0.60,
    "electronicscale": 0.70,
    "fluted-block": 0.12,
    "french-fries": 0.15,
    "fruit": 0.20,
    "glue": 0.10,
    "gong": 1.20,
    "hamburg": 0.25,
    "hydrating-oil": 0.25,
    "jam-jar": 0.45,
    "markpen": 0.03,
    "microphone": 0.30,
    "milk-box": 1.00,
    "milk-tea": 0.45,
    "mini-chalkboard": 0.35,
    "msg": 0.40,
    "notebook": 0.35,
    "olive-oil": 0.90,
    "paymentsign": 0.15,
    "pencup": 0.20,
    "perfume": 0.25,
    "pet-collar": 0.08,
    "phonestand": 0.18,
    "pillbottle": 0.10,
    "plant": 0.70,
    "plant-pot": 1.20,
    "plasticbox": 0.30,
    "playingcards": 0.10,
    "rack": 1.20,
    "remotecontrol": 0.20,
    "rest": 0.30,
    "roll-paper": 0.18,
    "roller": 0.35,
    "rubikscube": 0.12,
    "sand-clock": 0.40,
    "sauce-can": 0.40,
    "scanner": 2.00,
    "screen": 2.50,
    "screwdriver": 0.12,
    "seal": 0.15,
    "shampoo": 0.45,
    "shoe": 0.35,
    "shoe-box": 0.35,
    "small-speaker": 0.80,
    "smallshovel": 0.25,
    "soap": 0.15,
    "soy-sauce": 0.70,
    "speaker": 1.20,
    "steamer": 1.00,
    "table-tennis": 0.20,
    "tabletrashbin": 0.50,
    "tea-box": 0.30,
    "teanet": 0.60,
    "tissue-box": 0.25,
    "tooth-paste": 0.12,
    "toycar": 0.30,
    "tray": 0.45,
    "trophy": 0.45,
    "vagetable": 0.20,
    "vinegar": 0.70,
    "vis-box": 0.30,
    "waterer": 0.50,
    "whiteboard-eraser": 0.06,
    "wineglass": 0.22,
    "wooden-box": 0.80,
    "woodenblock": 0.25,
    "woodenmallet": 0.50,
}


MASS_BY_OBJAVERSE_CLASS = {
    "bottle": 0.50,
    "bowl": 0.35,
    "brush": 0.08,
    "can": 0.35,
    "chip_can": 0.20,
    "clock": 0.50,
    "drinkbox": 0.40,
    "hammer": 0.90,
    "marker": 0.03,
    "notebook": 0.35,
    "plate": 0.35,
    "pot": 1.00,
    "ramen_box": 0.35,
    "remote": 0.20,
    "slipper": 0.25,
    "snack_box": 0.30,
    "snack_package": 0.10,
    "sneaker": 0.45,
    "spoon": 0.05,
    "steel_tape": 0.30,
    "tape": 0.09,
    "thermos": 0.60,
    "tissue": 0.20,
    "toothbrush": 0.03,
    "toy_car": 0.25,
    "wallet": 0.12,
}


def _norm(text: str) -> str:
    return text.strip().lower().replace("_", "-")


def _extract_asset_dirs(map_file: Path, ycb_root: Path) -> list[Path]:
    pattern = re.compile(r"->\s*(\d+_.+)$")
    dirs: list[Path] = []
    for raw in map_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pattern.search(raw)
        if not m:
            continue
        asset_dir = ycb_root / m.group(1).strip()
        if asset_dir.exists():
            dirs.append(asset_dir)
    return sorted(set(dirs))


def _semantic_from_asset_dir(asset_dir: Path) -> str:
    return _norm(re.sub(r"^\d+[_-]?", "", asset_dir.name))


def _estimate_mass(mesh_path: Path, asset_dir: Path, semantic: str) -> float:
    if semantic == "objaverse":
        rel_parts = mesh_path.relative_to(asset_dir).parts
        if rel_parts:
            cls = rel_parts[0].strip().lower()
            return MASS_BY_OBJAVERSE_CLASS.get(cls, DEFAULT_MASS_KG)

    mass = MASS_BY_SEMANTIC.get(semantic, DEFAULT_MASS_KG)

    # "original-*.obj" is usually a split-part mesh from an articulated asset.
    # Distribute total estimated mass to avoid assigning the full object mass to each piece.
    if mesh_path.suffix.lower() == ".obj" and mesh_path.name.lower().startswith("original-"):
        part_count = len(list(mesh_path.parent.glob("original-*.obj")))
        if part_count > 0:
            mass = max(0.02, mass / part_count)
    return mass


@dataclass(frozen=True)
class MeshTask:
    asset_dir: Path
    semantic: str
    src: Path
    dst: Path


@dataclass(frozen=True)
class UrdfTask:
    src: Path
    dst: Path


def _collect_tasks(asset_dirs: list[Path]) -> tuple[list[MeshTask], list[UrdfTask]]:
    mesh_tasks: list[MeshTask] = []
    urdf_tasks: list[UrdfTask] = []
    for asset_dir in asset_dirs:
        semantic = _semantic_from_asset_dir(asset_dir)

        for src in asset_dir.rglob("*.glb"):
            mesh_tasks.append(MeshTask(asset_dir=asset_dir, semantic=semantic, src=src, dst=src.with_suffix(".usd")))

        for src in asset_dir.rglob("*.obj"):
            if src.name.lower() in OBJ_SKIP_NAMES:
                continue
            mesh_tasks.append(MeshTask(asset_dir=asset_dir, semantic=semantic, src=src, dst=src.with_suffix(".usd")))

        for src in asset_dir.rglob("*.urdf"):
            low = src.name.lower()
            if any(low.endswith(suf) for suf in URDF_SKIP_SUFFIXES):
                continue
            urdf_tasks.append(UrdfTask(src=src, dst=src.with_suffix(".usd")))

    mesh_tasks.sort(key=lambda t: str(t.src))
    urdf_tasks.sort(key=lambda t: str(t.src))
    return mesh_tasks, urdf_tasks


def _run(cmd: list[str], env: dict[str, str]) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    return proc.returncode, proc.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch convert imported RobotWin assets to USD.")
    parser.add_argument("--ycb-root", type=Path, default=DEFAULT_YCB_ROOT)
    parser.add_argument("--map-file", type=Path, default=DEFAULT_MAP_FILE)
    parser.add_argument("--convert-mesh", type=Path, default=DEFAULT_CONVERT_MESH)
    parser.add_argument("--convert-urdf", type=Path, default=DEFAULT_CONVERT_URDF)
    parser.add_argument("--python-exe", type=Path, default=Path(sys.executable))
    parser.add_argument("--isaaclab-source", type=Path, default=DEFAULT_ISAACLAB_SOURCE)
    parser.add_argument("--collision-approximation", default="convexDecomposition")
    parser.add_argument("--max-mesh", type=int, default=None, help="Only process first N mesh tasks.")
    parser.add_argument("--max-urdf", type=int, default=None, help="Only process first N urdf tasks.")
    parser.add_argument("--force", action="store_true", help="Convert even if destination USD already exists.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-path", type=Path, default=None)
    args = parser.parse_args()

    if args.report_path is None:
        args.report_path = args.ycb_root / "robotwin_usd_conversion_report.txt"

    asset_dirs = _extract_asset_dirs(args.map_file, args.ycb_root)
    mesh_tasks, urdf_tasks = _collect_tasks(asset_dirs)
    if args.max_mesh is not None:
        mesh_tasks = mesh_tasks[: max(0, args.max_mesh)]
    if args.max_urdf is not None:
        urdf_tasks = urdf_tasks[: max(0, args.max_urdf)]

    lines: list[str] = []
    lines.append(f"asset_dirs={len(asset_dirs)}")
    lines.append(f"mesh_tasks={len(mesh_tasks)}")
    lines.append(f"urdf_tasks={len(urdf_tasks)}")

    print(f"[INFO] asset_dirs={len(asset_dirs)}")
    print(f"[INFO] mesh_tasks={len(mesh_tasks)} urdf_tasks={len(urdf_tasks)}")

    mesh_done = mesh_skipped = mesh_failed = 0
    urdf_done = urdf_skipped = urdf_failed = 0
    t0 = time.time()
    env = os.environ.copy()
    source_path = str(args.isaaclab_source)
    current_pythonpath = env.get("PYTHONPATH", "")
    if current_pythonpath:
        if source_path not in current_pythonpath.split(";"):
            env["PYTHONPATH"] = f"{source_path};{current_pythonpath}"
    else:
        env["PYTHONPATH"] = source_path

    for idx, task in enumerate(mesh_tasks, start=1):
        if task.dst.exists() and not args.force:
            mesh_skipped += 1
            continue
        mass = _estimate_mass(task.src, task.asset_dir, task.semantic)
        cmd = [
            str(args.python_exe),
            str(args.convert_mesh),
            str(task.src),
            str(task.dst),
            "--make-instanceable",
            "--collision-approximation",
            args.collision_approximation,
            "--mass",
            f"{mass:.4f}",
            "--headless",
        ]
        if args.dry_run:
            print("[DRY-RUN][MESH]", " ".join(cmd))
            continue
        code, out = _run(cmd, env=env)
        if code == 0:
            mesh_done += 1
        else:
            mesh_failed += 1
            tail = "\n".join(out.splitlines()[-20:])
            lines.append(f"[MESH][FAIL] {task.src} -> {task.dst}\n{tail}")
        if idx % 20 == 0:
            print(f"[PROGRESS][MESH] {idx}/{len(mesh_tasks)} done={mesh_done} skipped={mesh_skipped} failed={mesh_failed}")

    for idx, task in enumerate(urdf_tasks, start=1):
        if task.dst.exists() and not args.force:
            urdf_skipped += 1
            continue
        cmd = [
            str(args.python_exe),
            str(args.convert_urdf),
            str(task.src),
            str(task.dst),
            "--merge-joints",
            "--headless",
        ]
        if args.dry_run:
            print("[DRY-RUN][URDF]", " ".join(cmd))
            continue
        code, out = _run(cmd, env=env)
        if code == 0:
            urdf_done += 1
        else:
            urdf_failed += 1
            tail = "\n".join(out.splitlines()[-20:])
            lines.append(f"[URDF][FAIL] {task.src} -> {task.dst}\n{tail}")
        if idx % 10 == 0:
            print(f"[PROGRESS][URDF] {idx}/{len(urdf_tasks)} done={urdf_done} skipped={urdf_skipped} failed={urdf_failed}")

    elapsed = time.time() - t0
    summary = (
        f"mesh_done={mesh_done} mesh_skipped={mesh_skipped} mesh_failed={mesh_failed} "
        f"urdf_done={urdf_done} urdf_skipped={urdf_skipped} urdf_failed={urdf_failed} "
        f"elapsed_sec={elapsed:.2f}"
    )
    lines.append(summary)
    args.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[INFO]", summary)
    print(f"[INFO] report={args.report_path}")
    return 1 if (mesh_failed or urdf_failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
