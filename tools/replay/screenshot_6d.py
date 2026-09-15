r"""Capture a single screenshot from a custom 6D camera pose at frame 0 of an HDF5 episode.

Loads the scene and trajectory from an HDF5 episode, sets the robot and
objects to their recorded frame-0 state, places a world-fixed camera at
a user-specified 6D pose (Tx,Ty,Tz,Rx,Ry,Rz — translation in meters,
Euler rotation in degrees), renders one frame, and saves it as a PNG.

Usage (batch wrapper):
    bash tools/replay/batch_screenshot_6d.sh

Or directly:
    python tools/replay/screenshot_6d.py \
        --hdf5 outputs/.../episode_000000.hdf5 \
        --output-png outputs/screenshots/episode_000000.png \
        --camera-pos 0.0,-1.36,2.48,50.0,0.0,0.0

    # Custom resolution:
    python tools/replay/screenshot_6d.py \
        --hdf5 outputs/.../episode_000000.hdf5 \
        --output-png screenshot.png \
        --camera-pos 0.0,-1.36,2.48,50.0,0.0,0.0 \
        --width 1920 --height 1080

Format of --camera-pos:
    Tx,Ty,Tz,Rx,Ry,Rz
    Tx,Ty,Tz: camera world position in meters
    Rx,Ry,Rz: camera Euler angles in degrees (extrinsic XYZ / RPY convention)
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

# Ensure the dex2bench repo root is on sys.path before any project imports.
# This script lives in tools/replay/, two levels below the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ── Isaac Sim launch (must happen before other Isaac imports) ──────────
from isaaclab.app import AppLauncher
from utils.isaac_rendering import configure_headless_camera_parity_experience

parser = argparse.ArgumentParser(
    description="Capture a single PNG screenshot from a custom 6D camera pose."
)
parser.add_argument("--hdf5", type=str, required=True, help="Path to source episode HDF5.")
parser.add_argument("--output-png", type=str, required=True,
                    help="Output PNG path for the screenshot.")
parser.add_argument("--camera-pos", type=str, default="0.0,-1.36,2.48,50.0,0.0,0.0",
                    help="6D camera pose: Tx,Ty,Tz,Rx,Ry,Rz (m, deg).")
parser.add_argument("--width", type=int, default=1280, help="Screenshot width.")
parser.add_argument("--height", type=int, default=720, help="Screenshot height.")
parser.add_argument("--scene", type=str, default=None, help="Override scene YAML path.")
parser.add_argument("--focal-length", type=float, default=10.5,
                    help="Camera focal length in mm.")
parser.add_argument("--horizontal-aperture", type=float, default=20.955,
                    help="Camera horizontal aperture in mm.")
parser.add_argument("--clipping-range", type=float, nargs=2, default=(0.01, 100.0),
                    help="Camera near/far clipping range in m.")
parser.add_argument("--clean-background", action="store_true",
                    help="Skip the recorded background (USD room / env texture); use Isaac's "
                         "built-in solid-color dome + default ground for angle testing.")
parser.add_argument("--zero-wrist", action="store_true",
                    help="After setting the frame-0 qpos, zero joint5/l_joint5 so the wrists do "
                         "not point the hands up (e.g. multi_rm_65_with_revo2).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
configure_headless_camera_parity_experience(args_cli, log_prefix="screenshot_6d")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-launch imports ────────────────────────────────────────────────
import h5py
import math

try:
    from scipy.spatial.transform import Rotation
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.utils.io import load_yaml

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


# ── Helpers ────────────────────────────────────────────────────────────

def _rpy_deg_to_quat_wxyz(rx: float, ry: float, rz: float) -> np.ndarray:
    """Convert RPY Euler angles (degrees, extrinsic XYZ) to wxyz quaternion.

    Equivalent to scipy Rotation.from_euler('xyz', ..., degrees=True).
    """
    # Convert to radians and half-angles
    r = math.radians(rx) * 0.5
    p = math.radians(ry) * 0.5
    y = math.radians(rz) * 0.5
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)  (extrinsic XYZ)
    quat = np.array([
        cr * cp * cy + sr * sp * sy,   # w
        sr * cp * cy - cr * sp * sy,   # x
        cr * sp * cy + sr * cp * sy,   # y
        cr * cp * sy - sr * sp * cy,   # z
    ], dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    return quat / norm if norm > 1.0e-8 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def parse_6d_pose(camera_pos_str: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse "Tx,Ty,Tz,Rx,Ry,Rz" string into position (m) and quaternion (wxyz).

    Rotation angles are in degrees, interpreted as extrinsic XYZ (RPY).
    """
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
        q_xyzw = rot.as_quat()  # scipy returns [x, y, z, w]
        quat_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)
    else:
        quat_wxyz = _rpy_deg_to_quat_wxyz(rx, ry, rz)

    return position, quat_wxyz


def _make_rigid_bodies_kinematic(object_prim_paths: dict, object_body_types: dict) -> int:
    """Set dynamic rigid bodies to kinematic so physics won't move them."""
    try:
        import omni.usd
        from pxr import Usd, UsdPhysics
    except ImportError:
        print("[WARN] omni.usd/pxr unavailable — cannot set kinematic mode")
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


def _build_joint_map(hdf5_names: list[str], robot_names: list[str]) -> dict[int, int]:
    """Map HDF5 joint index → robot articulation joint index."""
    lookup = {name: idx for idx, name in enumerate(robot_names)}
    return {h: lookup[n] for h, n in enumerate(hdf5_names) if n in lookup}


def _write_frame0_state(
    *,
    sim,
    robot_art,
    qpos_t: torch.Tensor,
    interactive_objects: dict,
    object_trajectories: dict[str, dict],
    obj_joint_maps: dict[str, dict[int, int]],
) -> None:
    """Write frame-0 recorded state to the simulation (kinematic)."""
    qvel_t = torch.zeros_like(qpos_t)
    robot_art.write_joint_state_to_sim(qpos_t, qvel_t)
    robot_art.set_joint_position_target(qpos_t)
    robot_art.write_data_to_sim()

    for obj_id, traj in object_trajectories.items():
        obj = interactive_objects.get(obj_id)
        if obj is None:
            continue
        p = traj["pose_world"][0]  # frame 0
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
                oq[0, r_i] = float(traj["qpos"][0, h_i])
            ov = torch.zeros_like(oq)
            obj.write_joint_state_to_sim(oq, ov)


def _create_screenshot_camera(
    sim,
    width: int,
    height: int,
    focal_length: float,
    horizontal_aperture: float,
    clipping_range: tuple[float, float],
) -> Camera:
    """Create a world-fixed Isaac Lab Camera at the given pose.

    The camera is placed at ``/World/ScreenshotCamera/Sensor``.
    """
    import omni.usd

    base_path = "/World/ScreenshotCamera"
    sensor_path = f"{base_path}/Sensor"

    # Remove stale prims from a previous run
    stage = omni.usd.get_context().get_stage()
    old_prim = stage.GetPrimAtPath(base_path)
    if old_prim.IsValid():
        stage.RemovePrim(base_path)

    # Create parent Xform
    stage.DefinePrim(base_path, "Xform")

    # Build camera config
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

    print(f"[screenshot_6d] Camera created at {base_path}")
    return cam


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    hdf5_path = os.path.abspath(args_cli.hdf5)
    if not os.path.isfile(hdf5_path):
        raise FileNotFoundError(f"HDF5 not found: {hdf5_path}")

    # Parse camera 6D pose
    cam_position, cam_quat_wxyz = parse_6d_pose(args_cli.camera_pos)

    # ── 1. Load trajectory metadata from HDF5 ─────────────────────────
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

    qpos_all = f["robot/qpos"][:]  # (N, n_joints) float32
    hdf5_joint_names = [
        n.decode() if isinstance(n, bytes) else str(n)
        for n in f["robot/joint_names"][:]
    ]
    replay_physics_dt = resolve_replay_physics_dt(f["meta"])

    # Object trajectories
    object_trajectories: dict[str, dict] = {}
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

    # Generalization sample (restore for scene building)
    recorded_generalization_sample_json: dict | None = None
    if "meta/scene_generalization_sample" in f:
        raw = f["meta/scene_generalization_sample"][()]
        try:
            recorded_generalization_sample_json = json.loads(
                raw.decode() if isinstance(raw, bytes) else str(raw)
            )
        except (json.JSONDecodeError, UnicodeDecodeError):
            print("[screenshot_6d][WARN] Failed to parse scene_generalization_sample from HDF5")
    f.close()

    if recorded_generalization_sample_json is not None:
        from build.generalization import _find_dataset_root
        _dataset_root = _find_dataset_root(_REPO_ROOT)
        recorded_generalization_sample_json = resolve_sample_paths(
            recorded_generalization_sample_json, _dataset_root,
        )

    print(f"[screenshot_6d] HDF5 : {hdf5_path}")
    print(f"[screenshot_6d] Scene: {scene_file}")
    print(f"[screenshot_6d] Robot: {robot_key}")
    print(f"[screenshot_6d] Frames: {frame_count}")
    print(f"[screenshot_6d] Physics dt: {replay_physics_dt:.7f}")
    print(f"[screenshot_6d] Output: {args_cli.output_png}")

    # ── 2. Setup simulation context ───────────────────────────────────
    sim_device = resolve_replay_sim_device(
        requested_device=getattr(args_cli, "device", None),
        enable_tactile=False,
    )
    sim_cfg = sim_utils.SimulationCfg(dt=replay_physics_dt, device=sim_device)
    sim = sim_utils.SimulationContext(sim_cfg)
    print(f"[screenshot_6d] Sim device: {sim_device}")

    task = load_yaml(scene_file)
    task_dir = os.path.dirname(scene_file)
    _, table_h, _ = resolve_table_spec(task)
    sim.set_camera_view([0.0, 1.0, table_h + 0.45], [0.0, 0.0, table_h])

    # ── 3. Scene generalization (restore from HDF5) ───────────────────
    generalization_enabled = recorded_generalization_sample_json is not None
    generalization_cfg = None
    generalization_sample = None
    if generalization_enabled:
        from build.generalization import ObjectGeneralizationCfg

        # Parse config from the original episode's config if available,
        # otherwise from the default config file.
        recorded_generalization_config_json = None
        # Re-open to read config
        with h5py.File(hdf5_path, "r") as f2:
            if "meta/scene_generalization_config" in f2:
                raw = f2["meta/scene_generalization_config"][()]
                try:
                    recorded_generalization_config_json = json.loads(
                        raw.decode() if isinstance(raw, bytes) else str(raw)
                    )
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass

        if recorded_generalization_config_json is not None:
            generalization_cfg = parse_scene_generalization_config(
                recorded_generalization_config_json
            )
            gen_cfg_path = "<from HDF5>"
        else:
            gen_cfg_path = os.path.join(
                _REPO_ROOT, "configs", "scene", "generalization.yaml",
            )
            gen_raw = load_yaml(gen_cfg_path) or {}
            generalization_cfg = parse_scene_generalization_config(
                gen_raw, config_path=gen_cfg_path,
            )
        generalization_cfg.spatial.object_pose = ObjectGeneralizationCfg(enabled=False)

        if args_cli.clean_background:
            # Override the recorded background (USD room / env texture) with Isaac's
            # built-in solid-color dome background. asset_kind="" (not "usd_scene")
            # makes build_scene skip the room and use the flat gray dome + ground.
            _bg = recorded_generalization_sample_json.setdefault("appearance", {}).setdefault("background", {})
            _bg.update({
                "enabled": True,
                "clean_background": True,
                "asset_kind": "",
                "asset_uri": "",
                "asset_id": "",
                "asset_category": "",
                "asset_split": "",
                "scene_offset": [0.0, 0.0, 0.0],
                "physics_enabled": False,
            })
            print("[screenshot_6d] Clean background override: solid-color dome (no USD room / env texture)")

        generalization_sample = dict_to_generalization_sample(
            recorded_generalization_sample_json
        )
        generalization_sample.robot_key = robot_key
        print(f"[screenshot_6d] Generalization RESTORED from HDF5 (cfg: {gen_cfg_path})")
        for line in scene_generalization_sample_debug_lines(generalization_sample):
            print(f"[screenshot_6d]   {line}")

    # ── 4. Build scene ────────────────────────────────────────────────
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
        raise RuntimeError("No global_robot in scene — cannot replay.")

    n_kin = _make_rigid_bodies_kinematic(object_prim_paths, object_body_types)
    print(f"[screenshot_6d] Kinematic rigid prims: {n_kin}")

    # ── 5. Create screenshot camera ───────────────────────────────────
    screenshot_cam = _create_screenshot_camera(
        sim,
        width=args_cli.width,
        height=args_cli.height,
        focal_length=args_cli.focal_length,
        horizontal_aperture=args_cli.horizontal_aperture,
        clipping_range=tuple(args_cli.clipping_range),
    )

    # ── 6. sim.reset() + initialize ───────────────────────────────────
    sim.reset()

    groups = classify_interactive_objects(interactive_objects)
    for art in groups.controlled_articulations:
        initialize_articulation_state(art)
    for art in groups.task_articulations:
        initialize_articulation_state(art, set_hold_target=False)
    for obj in groups.other_objects:
        obj.reset()

    # Initialize screenshot camera after reset (mirrors CameraRig._ensure_camera_initialized)
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

    # Set world pose now that the camera is initialized.
    # Use "opengl" convention: the Euler→quat we computed represents the
    # orientation of an OpenGL/USD camera (looks along -Z, +Y up), which
    # matches what you see when manually setting a Camera prim transform
    # in the Isaac Sim GUI.
    screenshot_cam.set_world_poses(
        positions=torch.tensor(cam_position[None, :], dtype=torch.float32, device=sim.device),
        orientations=torch.tensor(cam_quat_wxyz[None, :], dtype=torch.float32, device=sim.device),
        convention="opengl",
    )
    print(f"[screenshot_6d] Camera pose set (OpenGL convention):")
    print(f"  position (world): {cam_position.tolist()}")
    print(f"  quat_wxyz:        {cam_quat_wxyz.tolist()}")

    # ── 7. Joint index mapping ────────────────────────────────────────
    robot_joint_names = list(robot_art.joint_names)
    n_robot_joints = len(robot_joint_names)
    joint_map = _build_joint_map(hdf5_joint_names, robot_joint_names)
    print(f"[screenshot_6d] Joints: {len(joint_map)}/{len(hdf5_joint_names)} mapped → {n_robot_joints} robot joints")

    obj_joint_maps: dict[str, dict[int, int]] = {}
    for obj_id, traj in object_trajectories.items():
        if "joint_names" not in traj:
            continue
        obj = interactive_objects.get(obj_id)
        if obj is None or not hasattr(obj, "joint_names"):
            continue
        obj_joint_maps[obj_id] = _build_joint_map(traj["joint_names"], list(obj.joint_names))

    # ── 8. Set frame-0 state ──────────────────────────────────────────
    qpos_t = robot_art.data.default_joint_pos.clone()
    for h_idx, r_idx in joint_map.items():
        qpos_t[0, r_idx] = float(qpos_all[0, h_idx])

    if args_cli.zero_wrist:
        for jn in ("joint5", "l_joint5"):
            if jn in robot_joint_names:
                qpos_t[0, robot_joint_names.index(jn)] = 0.0
        print("[screenshot_6d] Zeroed wrist joints (joint5/l_joint5)", flush=True)

    _write_frame0_state(
        sim=sim,
        robot_art=robot_art,
        qpos_t=qpos_t,
        interactive_objects=interactive_objects,
        object_trajectories=object_trajectories,
        obj_joint_maps=obj_joint_maps,
    )

    # ── 9. Warm-up renders ────────────────────────────────────────────
    for _ in range(5):
        sim.render()

    # ── 10. Render and capture ────────────────────────────────────────
    sim.render()
    screenshot_cam.update(dt=0.0, force_recompute=True)

    # Extract RGB from the camera
    cam_data = screenshot_cam.data
    rgb_output = cam_data.output.get("rgb")
    if rgb_output is None or rgb_output.numel() == 0:
        raise RuntimeError("Screenshot camera returned no RGB data")

    rgb_gpu = rgb_output[0]  # (H, W, 4) RGBA
    if rgb_gpu.shape[-1] > 3:
        rgb_gpu = rgb_gpu[..., :3]
    rgb = rgb_gpu.to(dtype=torch.uint8).contiguous().cpu().numpy()

    print(f"[screenshot_6d] Captured RGB: {rgb.shape}")

    # ── 11. Save PNG ──────────────────────────────────────────────────
    output_dir = os.path.dirname(os.path.abspath(args_cli.output_png))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    from PIL import Image
    Image.fromarray(rgb).save(args_cli.output_png)
    print(f"[screenshot_6d] Saved: {args_cli.output_png}")

    # Clean up camera to avoid shutdown issues
    try:
        screenshot_cam._rep_registry = {}
    except Exception:
        pass

    print("[screenshot_6d] Done")
    os._exit(0)


if __name__ == "__main__":
    main()
