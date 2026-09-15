r"""Capture one screenshot per task (episode_000003) with task-76 scene look.

Batch tool built on top of ``screenshot_6d.py``'s render pipeline.  It captures a
single frame-0 PNG for every task that has a ``replay-generalization`` directory,
using:

  * each task's own 03-episode object positions + distractor (clutter) sample,
  * task-76's background (usd_scene FloorPlan9) and lighting (usd_scene_light),
  * the default solid table color (table-surface generalization disabled),
  * its own table height (NO table-height sync, as requested),
  * camera --camera-pos 0.0,1.0,2.0,50.0,0.0,180.0.

Output PNGs are written flat into ``<pic_dir>/all/<tasknum>.png``.

This script is fully self-contained: it does NOT modify ``screenshot_6d.py``,
``batch_screenshot_6d.py`` or any HDF5/episode data.  Original files are never
touched; only new PNGs are written.

Usage
-----
Batch (26 tasks -> pic/all/<tasknum>.png)::

    python tools/replay/screenshot_6d_all.py --batch

Single task (worker mode, used by --batch internally)::

    python tools/replay/screenshot_6d_all.py \\
        --hdf5 .../replay-generalization/episode_000003.hdf5 \\
        --output-png .../pic/all/06.png \\
        --camera-pos 0.0,1.0,2.0,50.0,0.0,180.0 \\
        --ref-hdf5 .../76_soup_serving/replay-generalization/episode_000003.hdf5

Environment variables:
    OVERWRITE=0   Skip tasks whose <tasknum>.png already exists (default 1).
"""

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys

# ── Default paths (batch mode) ───────────────────────────────────────────────
DEFAULT_DATASET_DIR = "../teleopdata/dataset"
DEFAULT_PIC_DIR = "../jcy/pic"
DEFAULT_REF_TASK = "76_soup_serving"
DEFAULT_CAMERA_POS = "0.0,1.0,2.0,50.0,0.0,180.0"
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080

# Ensure the dex2bench repo root is on sys.path before any project imports.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Task dir -> episode_000003 path suffix (relative to the task dir) override.
# Task 08's top-level replay-generalization was recorded against an older scene
# YAML (obj_181_displaystand_1 vs the current obj_333_portable_stove_6), so it
# fails object-ID validation.  The multi_ur5_schunk_hand_with_flange copy
# matches the current object set and is the one used here.
TASK_EP03_OVERRIDES = {
    "08_frypan_stand_pour": "multi_ur5_schunk_hand_with_flange/replay-generalization/episode_000003.hdf5",
}

# Task numbers whose robot should be placed at its HOME joint pose instead of
# the recorded frame-0 qpos.  multi_rm_65_with_revo2 (tasks 24/67) records
# frame-0 with the arm pointing straight up; HOME gives a natural hand-down pose.
ROBOT_HOME_TASKS = {"24", "67"}

parser = argparse.ArgumentParser(
    description="Single or batch frame-0 screenshots with task-76 background/lighting "
                "and default solid table color (no table-height sync)."
)
# Worker args (also accepted in batch mode but only used when spawning worker).
parser.add_argument("--hdf5", type=str, default=None, help="Path to source episode HDF5.")
parser.add_argument("--output-png", type=str, default=None,
                    help="Output PNG path for the screenshot.")
parser.add_argument("--camera-pos", type=str, default=DEFAULT_CAMERA_POS,
                    help="6D camera pose: Tx,Ty,Tz,Rx,Ry,Rz (m, deg).")
parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Screenshot width.")
parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Screenshot height.")
parser.add_argument("--focal-length", type=float, default=10.5,
                    help="Camera focal length in mm.")
parser.add_argument("--scene", type=str, default=None, help="Override scene YAML path.")
parser.add_argument("--ref-hdf5", type=str, default=None,
                    help="Reference HDF5 whose background + usd_scene_light are forced "
                         "(defaults to task-76 episode_000003).")
parser.add_argument("--no-clutter", action="store_true",
                    help="Disable distractor/clutter generalization (not spawned).")
parser.add_argument("--frame", type=int, default=0,
                    help="Frame index to render (default 0); uses the actual recorded pose at that frame.")
parser.add_argument("--robot-home", action="store_true",
                    help="Use the robot's HOME joint pose instead of the recorded frame-0 qpos "
                         "(fixes robots whose recorded frame-0 arm points up, e.g. multi_rm_65_with_revo2).")
parser.add_argument("--robot-home-zero-wrist", action="store_true",
                    help="With --robot-home, additionally zero joint5/l_joint5 so the wrist does "
                         "not point the hand up (multi_rm_65_with_revo2 HOME has joint5=±1.57).")
# Batch args
parser.add_argument("--batch", action="store_true",
                    help="Batch mode: loop all tasks with replay-generalization, one "
                         "episode_000003 screenshot each, into pic/all/<tasknum>.png.")
parser.add_argument("--dataset-dir", type=str, default=None,
                    help="Override dataset dir (batch mode).")
parser.add_argument("--pic-dir", type=str, default=None,
                    help="Override pic root dir (batch mode); outputs go to <pic>/all/.")
parser.add_argument("--ref-task", type=str, default=DEFAULT_REF_TASK,
                    help="Task dir name used as the lighting/background reference.")
parser.add_argument("--no-validate", action="store_true",
                    help="Skip post-capture PNG validation in batch mode.")
parser.add_argument("--subdir", type=str, default="all",
                    help="Output subdirectory under --pic-dir (batch mode).")

# Isaac launch args (worker mode)
from isaaclab.app import AppLauncher  # noqa: E402
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


def _task_num(task_dir_name: str) -> str:
    """Extract leading numeric id from a task dir name, e.g. '06' from '06_fruit_bowl_loading'."""
    return str(task_dir_name.split("_")[0])


def _kill_process_group(proc: subprocess.Popen, grace: float = 3.0) -> None:
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except OSError:
        pass
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        print("[batch] Graceful shutdown timed out - SIGKILL", flush=True)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            pass
        proc.wait()


def _validate_png(png_path: str) -> tuple[bool, str]:
    if not os.path.isfile(png_path):
        return False, "PNG file missing"
    if os.path.getsize(png_path) < 100:
        return False, f"PNG too small ({os.path.getsize(png_path)} bytes)"
    try:
        from PIL import Image
        with Image.open(png_path) as img:
            img.verify()
        return True, "ok"
    except Exception as exc:
        return False, f"corrupt: {exc}"


def run_batch() -> int:
    """Loop every task with a replay-generalization dir and capture episode_000003."""
    dataset_dir = args_cli.dataset_dir or os.environ.get("DEX2BENCH_DATASET_DIR", DEFAULT_DATASET_DIR)
    pic_dir = args_cli.pic_dir or os.environ.get("DEX2BENCH_PIC_DIR", DEFAULT_PIC_DIR)
    output_dir = os.path.join(pic_dir, args_cli.subdir)
    os.makedirs(output_dir, exist_ok=True)

    ref_task_dir = os.path.join(dataset_dir, args_cli.ref_task)
    ref_hdf5 = os.path.join(ref_task_dir, "replay-generalization", "episode_000003.hdf5")
    if not os.path.isfile(ref_hdf5):
        print(f"[batch][ERROR] Reference HDF5 not found: {ref_hdf5}")
        return 1

    # Discover tasks: dirs under dataset_dir that contain replay-generalization/episode_000003.
    items: list[tuple[str, str, str]] = []  # (task_dir_name, tasknum, hdf5_path)
    if os.path.isdir(dataset_dir):
        for name in sorted(os.listdir(dataset_dir)):
            ep03_suffix = TASK_EP03_OVERRIDES.get(
                name, os.path.join("replay-generalization", "episode_000003.hdf5")
            )
            candidate = os.path.join(dataset_dir, name, ep03_suffix)
            if os.path.isfile(candidate):
                items.append((name, _task_num(name), candidate))
    if not items:
        print(f"[batch][ERROR] No replay-generalization/episode_000003 found under {dataset_dir}")
        return 1

    overwrite = os.environ.get("OVERWRITE", "1") == "1"
    os.chdir(_REPO_ROOT)

    worker_cmd = [sys.executable, os.path.abspath(__file__)]
    ok_count = 0
    fail_count = 0
    skip_count = 0
    failed: list[str] = []
    total = len(items)

    print(f"[batch] Dataset      : {dataset_dir}")
    print(f"[batch] Output dir   : {output_dir}")
    print(f"[batch] Ref HDF5     : {ref_hdf5}")
    print(f"[batch] Tasks        : {total}")
    print(f"[batch] OVERWRITE    : {1 if overwrite else 0}")
    print(f"[batch] Camera       : {args_cli.camera_pos}")
    print(f"[batch] Resolution   : {args_cli.width}x{args_cli.height}")
    print(f"[batch] Table height : KEEP each task's own (no sync)")
    print(f"[batch] Table surface: default solid color (texture generalization OFF)")
    print(f"[batch] Clutter      : {'DISABLED (no distractors)' if args_cli.no_clutter else 'RESTORED from HDF5'}")

    interrupted = False
    _active_proc: subprocess.Popen | None = None

    for idx, (task_name, tasknum, hdf5_path) in enumerate(items, start=1):
        output_png = os.path.join(output_dir, f"{tasknum}.png")
        if os.path.isfile(output_png) and not overwrite:
            print(f"[SKIP] {idx}/{total} {tasknum}.png (already exists)")
            skip_count += 1
            continue

        cmd_list = worker_cmd + [
            "--hdf5", hdf5_path,
            "--output-png", output_png,
            "--camera-pos", args_cli.camera_pos,
            "--width", str(args_cli.width),
            "--height", str(args_cli.height),
            "--focal-length", str(args_cli.focal_length),
            "--ref-hdf5", ref_hdf5,
            "--headless",
        ]
        # All tasks: disable distractor/clutter generalization.
        cmd_list.append("--no-clutter")
        if tasknum in ROBOT_HOME_TASKS:
            cmd_list.append("--robot-home")
            # multi_rm_65_with_revo2 HOME has joint5=±1.57 (hands up); zero the wrist.
            cmd_list.append("--robot-home-zero-wrist")
        cmd_str = " ".join(shlex.quote(p) for p in cmd_list)
        print(f"\n{'=' * 80}")
        print(f"[task {idx}/{total}] {task_name} (ep03) -> {tasknum}.png")
        print(f"CMD: {cmd_str}")
        print(f"{'=' * 80}\n")

        _active_proc = subprocess.Popen(cmd_list, start_new_session=True)
        try:
            ret = _active_proc.wait()
        except KeyboardInterrupt:
            print(f"\n[batch] Interrupted - stopping {task_name} ...", flush=True)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            _kill_process_group(_active_proc)
            interrupted = True
            break
        finally:
            _active_proc = None

        if ret != 0:
            print(f"[ERROR] Screenshot failed: {task_name}, exit={ret}", flush=True)
            fail_count += 1
            failed.append(task_name)
            if os.path.isfile(output_png):
                os.unlink(output_png)
            continue

        if os.path.isfile(output_png):
            size_kb = os.path.getsize(output_png) / 1024
            print(f"[OK] {task_name} -> {tasknum}.png ({size_kb:.0f} KB)", flush=True)
            ok_count += 1
        else:
            print(f"[ERROR] Output PNG missing: {output_png}", flush=True)
            fail_count += 1
            failed.append(task_name)

    # ── Post-capture validation ────────────────────────────────────────────
    if not args_cli.no_validate and not interrupted:
        print(f"\n{'=' * 80}")
        print("[validate] Checking output PNGs ...")
        bad_count = 0
        for png in sorted(os.listdir(output_dir)):
            png_path = os.path.join(output_dir, png)
            valid, reason = _validate_png(png_path)
            if valid:
                continue
            print(f"[validate] BAD: {png} - {reason}")
            bad_count += 1
            if png not in failed:
                failed.append(png)
        if bad_count:
            fail_count += bad_count
            print(f"[validate] {bad_count} PNG(s) missing or corrupt")
        else:
            print("[validate] All PNGs look valid.")

    print(f"\n{'=' * 80}")
    if interrupted:
        print("[batch] Interrupted by user")
    print("[batch] Done")
    print(f"  Output:  {output_dir}")
    print(f"  OK:      {ok_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Failed:  {fail_count}")
    if failed:
        print("[batch] Failed tasks:")
        for name in failed:
            print(f"  {name}")
        return 1
    return 0


if args_cli.batch:
    # Batch mode only needs stdlib + the args above.  Exit before any Isaac imports.
    raise SystemExit(run_batch())


# ── Worker mode: heavy Isaac imports + render pipeline ──────────────────────
# (Copied from tools/replay/screenshot_6d.py with two additions:
#   1. --ref-hdf5: force the reference task's background + usd_scene_light.
#   2. table_surface disabled -> default solid table color.
#  Table height is intentionally NOT synced.)

from utils.isaac_rendering import configure_headless_camera_parity_experience  # noqa: E402

args_cli.enable_cameras = True
configure_headless_camera_parity_experience(args_cli, log_prefix="screenshot_6d_all")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-launch imports ────────────────────────────────────────────────────
import h5py  # noqa: E402
import math  # noqa: E402

try:
    from scipy.spatial.transform import Rotation  # noqa: E402
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors.camera import Camera, CameraCfg  # noqa: E402
from isaaclab.utils.io import load_yaml  # noqa: E402

from build import (  # noqa: E402
    build_scene,
    dict_to_generalization_sample,
    parse_scene_generalization_config,
    resolve_sample_paths,
    resolve_table_spec,
    scene_generalization_sample_debug_lines,
)
from utils.replay_support import resolve_replay_physics_dt, resolve_replay_sim_device  # noqa: E402
from utils.scene_paths import resolve_scene_yaml_path  # noqa: E402

try:
    from utils.runtime_helpers import classify_interactive_objects, initialize_articulation_state  # noqa: E402
except ModuleNotFoundError:
    from runtime_helpers import classify_interactive_objects, initialize_articulation_state  # noqa: E402


# ── Helpers (identical to screenshot_6d.py) ────────────────────────────────

def _rpy_deg_to_quat_wxyz(rx: float, ry: float, rz: float) -> list:
    import numpy as np  # noqa: E402
    r = math.radians(rx) * 0.5
    p = math.radians(ry) * 0.5
    y = math.radians(rz) * 0.5
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    quat = np.array([
        cr * cp * cy + sr * sp * sy,   # w
        sr * cp * cy - cr * sp * sy,   # x
        cr * sp * cy + sr * cp * sy,   # y
        cr * cp * sy - sr * sp * cy,   # z
    ], dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    return quat / norm if norm > 1.0e-8 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def parse_6d_pose(camera_pos_str: str) -> tuple:
    import numpy as np  # noqa: E402
    parts = [float(x.strip()) for x in camera_pos_str.split(",")]
    if len(parts) != 6:
        raise ValueError(
            f"--camera-pos expects 6 comma-separated values (Tx,Ty,Tz,Rx,Ry,Rz), "
            f"got {len(parts)}: {camera_pos_str!r}"
        )
    tx, ty, tz, rx, ry, rz = parts
    position = np.array([tx, ty, tz], dtype=np.float32)
    if _HAS_SCIPY:
        rot = Rotation.from_euler("xyz", [rx, ry, rz], degrees=True)
        q_xyzw = rot.as_quat()
        quat_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)
    else:
        quat_wxyz = _rpy_deg_to_quat_wxyz(rx, ry, rz)
    return position, quat_wxyz


def _make_rigid_bodies_kinematic(object_prim_paths: dict, object_body_types: dict) -> int:
    try:
        import omni.usd
        from pxr import Usd, UsdPhysics
    except ImportError:
        print("[WARN] omni.usd/pxr unavailable - cannot set kinematic mode")
        return 0
    stage = omni.usd.get_context().get_stage()
    count = 0
    for obj_id, prim_path in object_prim_paths.items():
        if object_body_types.get(obj_id) != "dynamic":
            continue
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        for p in Usd.PrimRange(prim):
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(p).GetKinematicEnabledAttr().Set(True)
                count += 1
    return count


def _build_joint_map(hdf5_names: list, robot_names: list) -> dict:
    lookup = {name: idx for idx, name in enumerate(robot_names)}
    return {h: lookup[n] for h, n in enumerate(hdf5_names) if n in lookup}


def _write_frame0_state(
    *,
    sim,
    robot_art,
    qpos_t,
    interactive_objects: dict,
    object_trajectories: dict,
    obj_joint_maps: dict,
    frame_idx: int = 0,
) -> None:
    import torch  # noqa: E402
    qvel_t = torch.zeros_like(qpos_t)
    robot_art.write_joint_state_to_sim(qpos_t, qvel_t)
    robot_art.set_joint_position_target(qpos_t)
    robot_art.write_data_to_sim()
    for obj_id, traj in object_trajectories.items():
        obj = interactive_objects.get(obj_id)
        if obj is None:
            continue
        p = traj["pose_world"][frame_idx]
        if hasattr(obj, "write_root_pose_to_sim"):
            pose_wxyz = torch.tensor(
                [[p[0], p[1], p[2], p[6], p[3], p[4], p[5]]],
                dtype=torch.float32, device=sim.device,
            )
            obj.write_root_pose_to_sim(pose_wxyz)
        if obj_id in obj_joint_maps and "qpos" in traj:
            omap = obj_joint_maps[obj_id]
            oq = obj.data.default_joint_pos.clone()
            for h_i, r_i in omap.items():
                oq[0, r_i] = float(traj["qpos"][frame_idx, h_i])
            ov = torch.zeros_like(oq)
            obj.write_joint_state_to_sim(oq, ov)


def _create_screenshot_camera(
    sim, width: int, height: int, focal_length: float,
    horizontal_aperture: float, clipping_range: tuple,
):
    import omni.usd  # noqa: E402
    base_path = "/World/ScreenshotCamera"
    sensor_path = f"{base_path}/Sensor"
    stage = omni.usd.get_context().get_stage()
    old_prim = stage.GetPrimAtPath(base_path)
    if old_prim.IsValid():
        stage.RemovePrim(base_path)
    stage.DefinePrim(base_path, "Xform")
    spawn_cfg = sim_utils.PinholeCameraCfg(
        focal_length=focal_length,
        horizontal_aperture=horizontal_aperture,
        clipping_range=clipping_range,
    )
    camera_cfg = CameraCfg(
        prim_path=sensor_path,
        update_period=0,
        width=width,
        height=height,
        data_types=["rgb"],
        spawn=spawn_cfg,
    )
    cam = Camera(cfg=camera_cfg)
    if not hasattr(cam, "_rep_registry"):
        cam._rep_registry = {}
    print(f"[screenshot_6d_all] Camera created at {base_path}")
    return cam


def _load_generalization_sample(hdf5_path: str):
    """Return (sample_json, config_json) read from the HDF5, paths resolved."""
    with h5py.File(hdf5_path, "r") as f:
        sample_json = None
        if "meta/scene_generalization_sample" in f:
            raw = f["meta/scene_generalization_sample"][()]
            try:
                sample_json = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
            except (json.JSONDecodeError, UnicodeDecodeError):
                sample_json = None
        config_json = None
        if "meta/scene_generalization_config" in f:
            raw = f["meta/scene_generalization_config"][()]
            try:
                config_json = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
            except (json.JSONDecodeError, UnicodeDecodeError):
                config_json = None
    if sample_json is not None:
        from build.generalization import _find_dataset_root
        _dataset_root = _find_dataset_root(_REPO_ROOT)
        sample_json = resolve_sample_paths(sample_json, _dataset_root)
    return sample_json, config_json


def _apply_reference_environment(sample_json: dict, ref_hdf5: str) -> None:
    """Force the reference task's background + usd_scene_light onto the task sample,
    and disable the table-surface texture generalization (default solid color).

    Table height is intentionally left untouched (no sync).
    """
    ref_sample_json, _ = _load_generalization_sample(ref_hdf5)
    if ref_sample_json is None:
        print(f"[screenshot_6d_all][WARN] No generalization sample in ref HDF5: {ref_hdf5}")
        return
    ref_appearance = ref_sample_json.get("appearance") or {}
    appearance = sample_json.setdefault("appearance", {})

    if ref_appearance.get("background"):
        appearance["background"] = dict(ref_appearance["background"])
        print("[screenshot_6d_all] Background  <- reference (usd_scene background)")

    if ref_appearance.get("usd_scene_light"):
        appearance["usd_scene_light"] = dict(ref_appearance["usd_scene_light"])
        print("[screenshot_6d_all] Lighting     <- reference (usd_scene_light)")

    ts = appearance.setdefault("table_surface", {})
    ts["enabled"] = False
    ts["clean_surface"] = True
    ts.pop("asset_uri", None)
    ts.pop("asset_id", None)
    print("[screenshot_6d_all] Table surface: disabled -> default solid color")

    if getattr(args_cli, "no_clutter", False):
        clutter = sample_json.setdefault("clutter", {}).setdefault("tabletop", {})
        clutter["enabled"] = False
        clutter["count"] = 0
        print("[screenshot_6d_all] Clutter: DISABLED (no distractors)")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    import numpy as np  # noqa: E402
    import torch  # noqa: E402

    hdf5_path = os.path.abspath(args_cli.hdf5)
    if not os.path.isfile(hdf5_path):
        raise FileNotFoundError(f"HDF5 not found: {hdf5_path}")

    cam_position, cam_quat_wxyz = parse_6d_pose(args_cli.camera_pos)

    # ── 1. Load trajectory metadata from HDF5 ─────────────────────────────
    f = h5py.File(hdf5_path, "r")
    recorded_scene_file = f["meta/scene_file"][()].decode()
    scene_name = f["meta/scene_name"][()].decode() if "meta/scene_name" in f else None
    scene_file = resolve_scene_yaml_path(
        cli_scene_path=args_cli.scene,
        recorded_scene_path=recorded_scene_file,
        recorded_scene_name=scene_name,
        script_dir=_REPO_ROOT,
    )
    robot_key = f["meta/robot_key"][()].decode()
    frame_count = int(f["meta/frame_count"][()])

    qpos_all = f["robot/qpos"][:]
    hdf5_joint_names = [
        n.decode() if isinstance(n, bytes) else str(n)
        for n in f["robot/joint_names"][:]
    ]
    replay_physics_dt = resolve_replay_physics_dt(f["meta"])

    object_trajectories: dict = {}
    for obj_id in f["objects"].keys():
        grp = f[f"objects/{obj_id}"]
        traj: dict = {"pose_world": grp["pose_world"][:]}
        if "joint_names" in grp:
            traj["joint_names"] = [
                n.decode() if isinstance(n, bytes) else str(n)
                for n in grp["joint_names"][:]
            ]
            traj["qpos"] = grp["qpos"][:]
        object_trajectories[obj_id] = traj
    f.close()

    # ── 2. Generalization sample + reference environment merge ────────────
    recorded_generalization_sample_json, recorded_generalization_config_json = \
        _load_generalization_sample(hdf5_path)

    ref_hdf5 = args_cli.ref_hdf5
    if ref_hdf5 is None:
        ref_hdf5 = os.path.join(
            DEFAULT_DATASET_DIR, DEFAULT_REF_TASK, "replay-generalization", "episode_000003.hdf5",
        )
    if recorded_generalization_sample_json is not None and os.path.isfile(ref_hdf5):
        _apply_reference_environment(recorded_generalization_sample_json, ref_hdf5)

    print(f"[screenshot_6d_all] HDF5 : {hdf5_path}")
    print(f"[screenshot_6d_all] Scene: {scene_file}")
    print(f"[screenshot_6d_all] Robot: {robot_key}")
    print(f"[screenshot_6d_all] Frames: {frame_count}")
    print(f"[screenshot_6d_all] Physics dt: {replay_physics_dt:.7f}")
    print(f"[screenshot_6d_all] Output: {args_cli.output_png}")

    # ── 3. Setup simulation context ───────────────────────────────────────
    sim_device = resolve_replay_sim_device(
        requested_device=getattr(args_cli, "device", None),
        enable_tactile=False,
    )
    sim_cfg = sim_utils.SimulationCfg(dt=replay_physics_dt, device=sim_device)
    sim = sim_utils.SimulationContext(sim_cfg)
    print(f"[screenshot_6d_all] Sim device: {sim_device}")

    task = load_yaml(scene_file)
    task_dir = os.path.dirname(scene_file)
    _, table_h, _ = resolve_table_spec(task)
    sim.set_camera_view([0.0, 1.0, table_h + 0.45], [0.0, 0.0, table_h])

    # ── 4. Scene generalization (restore from HDF5) ───────────────────────
    generalization_enabled = recorded_generalization_sample_json is not None
    generalization_cfg = None
    generalization_sample = None
    if generalization_enabled:
        from build.generalization import ObjectGeneralizationCfg  # noqa: E402

        if recorded_generalization_config_json is not None:
            generalization_cfg = parse_scene_generalization_config(
                recorded_generalization_config_json
            )
            gen_cfg_path = "<from HDF5>"
        else:
            gen_cfg_path = os.path.join(_REPO_ROOT, "configs", "scene", "generalization.yaml")
            gen_raw = load_yaml(gen_cfg_path) or {}
            generalization_cfg = parse_scene_generalization_config(
                gen_raw, config_path=gen_cfg_path,
            )
        generalization_cfg.spatial.object_pose = ObjectGeneralizationCfg(enabled=False)

        generalization_sample = dict_to_generalization_sample(
            recorded_generalization_sample_json
        )
        generalization_sample.robot_key = robot_key
        print(f"[screenshot_6d_all] Generalization RESTORED from HDF5 (cfg: {gen_cfg_path})")
        for line in scene_generalization_sample_debug_lines(generalization_sample):
            print(f"[screenshot_6d_all]   {line}")

    # ── 5. Build scene ────────────────────────────────────────────────────
    runtime = build_scene(
        task, task_dir,
        robot_key=robot_key,
        generalization_enabled=generalization_enabled,
        generalization_cfg=generalization_cfg,
        generalization_sample=generalization_sample,
    )
    interactive_objects = runtime["interactive_objects"]
    object_prim_paths = runtime["object_prim_paths"]
    object_body_types = runtime["object_body_types"]

    robot_art = interactive_objects.get("global_robot")
    if robot_art is None:
        raise RuntimeError("No global_robot in scene - cannot replay.")

    n_kin = _make_rigid_bodies_kinematic(object_prim_paths, object_body_types)
    print(f"[screenshot_6d_all] Kinematic rigid prims: {n_kin}")

    # ── 6. Create screenshot camera ───────────────────────────────────────
    screenshot_cam = _create_screenshot_camera(
        sim,
        width=args_cli.width,
        height=args_cli.height,
        focal_length=args_cli.focal_length,
        horizontal_aperture=20.955,
        clipping_range=tuple((0.01, 100.0)),
    )

    # ── 7. sim.reset() + initialize ───────────────────────────────────────
    sim.reset()

    groups = classify_interactive_objects(interactive_objects)
    for art in groups.controlled_articulations:
        initialize_articulation_state(art)
    for art in groups.task_articulations:
        initialize_articulation_state(art, set_hold_target=False)
    for obj in groups.other_objects:
        obj.reset()

    initialize_impl = getattr(screenshot_cam, "_initialize_impl", None)
    if callable(initialize_impl):
        initialize_impl()
    if hasattr(screenshot_cam, "_is_initialized"):
        screenshot_cam._is_initialized = True
    if not hasattr(screenshot_cam, "_rep_registry"):
        screenshot_cam._rep_registry = {}
    if not hasattr(screenshot_cam, "_ALL_INDICES"):
        view = getattr(screenshot_cam, "_view", None)
        count = int(getattr(view, "count", 1) or 1)
        device = getattr(screenshot_cam, "_device", "cpu")
        screenshot_cam._ALL_INDICES = torch.arange(count, device=device, dtype=torch.long)

    screenshot_cam.set_world_poses(
        positions=torch.tensor(cam_position[None, :], dtype=torch.float32, device=sim.device),
        orientations=torch.tensor(cam_quat_wxyz[None, :], dtype=torch.float32, device=sim.device),
        convention="opengl",
    )
    print(f"[screenshot_6d_all] Camera pose set (OpenGL convention):")
    print(f"  position (world): {cam_position.tolist()}")
    print(f"  quat_wxyz:        {cam_quat_wxyz.tolist()}")

    # ── 8. Joint index mapping ────────────────────────────────────────────
    robot_joint_names = list(robot_art.joint_names)
    n_robot_joints = len(robot_joint_names)
    joint_map = _build_joint_map(hdf5_joint_names, robot_joint_names)
    print(f"[screenshot_6d_all] Joints: {len(joint_map)}/{len(hdf5_joint_names)} mapped -> {n_robot_joints} robot joints")

    obj_joint_maps: dict = {}
    for obj_id, traj in object_trajectories.items():
        if "joint_names" not in traj:
            continue
        obj = interactive_objects.get(obj_id)
        if obj is None or not hasattr(obj, "joint_names"):
            continue
        obj_joint_maps[obj_id] = _build_joint_map(traj["joint_names"], list(obj.joint_names))

    # ── 9. Set frame-<frame_idx> state ────────────────────────────────────
    frame_idx = int(getattr(args_cli, "frame", 0))
    if frame_idx < 0 or frame_idx >= frame_count:
        raise ValueError(f"--frame {frame_idx} out of range (frame_count={frame_count})")
    qpos_t = robot_art.data.default_joint_pos.clone()
    if getattr(args_cli, "robot_home", False):
        # Keep the robot at its HOME joint pose (default_joint_pos) instead of
        # the recorded qpos.  Objects still use their recorded poses.
        print("[screenshot_6d_all] Robot qpos: using HOME pose (recorded qpos ignored)", flush=True)
    else:
        for h_idx, r_idx in joint_map.items():
            qpos_t[0, r_idx] = float(qpos_all[frame_idx, h_idx])

    if getattr(args_cli, "robot_home_zero_wrist", False):
        # Zero the wrist-rotation joints so the hands do not point up.
        for jn in ("joint5", "l_joint5"):
            if jn in robot_joint_names:
                qpos_t[0, robot_joint_names.index(jn)] = 0.0
        print("[screenshot_6d_all] Robot qpos: zeroed wrist joints (joint5/l_joint5)", flush=True)

    _write_frame0_state(
        sim=sim,
        robot_art=robot_art,
        qpos_t=qpos_t,
        interactive_objects=interactive_objects,
        object_trajectories=object_trajectories,
        obj_joint_maps=obj_joint_maps,
        frame_idx=frame_idx,
    )
    print(f"[screenshot_6d_all] Rendering frame {frame_idx} (robot={'HOME' if getattr(args_cli, 'robot_home', False) else 'recorded'})", flush=True)

    # ── 10. Warm-up renders ───────────────────────────────────────────────
    for _ in range(5):
        sim.render()

    # ── 11. Render and capture ────────────────────────────────────────────
    sim.render()
    screenshot_cam.update(dt=0.0, force_recompute=True)

    cam_data = screenshot_cam.data
    rgb_output = cam_data.output.get("rgb")
    if rgb_output is None or rgb_output.numel() == 0:
        raise RuntimeError("Screenshot camera returned no RGB data")

    rgb_gpu = rgb_output[0]
    if rgb_gpu.shape[-1] > 3:
        rgb_gpu = rgb_gpu[..., :3]
    rgb = rgb_gpu.to(dtype=torch.uint8).contiguous().cpu().numpy()

    print(f"[screenshot_6d_all] Captured RGB: {rgb.shape}")

    # ── 12. Save PNG ──────────────────────────────────────────────────────
    output_dir = os.path.dirname(os.path.abspath(args_cli.output_png))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    from PIL import Image  # noqa: E402
    Image.fromarray(rgb).save(args_cli.output_png)
    print(f"[screenshot_6d_all] Saved: {args_cli.output_png}", flush=True)

    try:
        screenshot_cam._rep_registry = {}
    except Exception:
        pass

    print("[screenshot_6d_all] Done")
    os._exit(0)


if __name__ == "__main__":
    main()
