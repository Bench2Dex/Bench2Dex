r"""Replay an HDF5 episode kinematically in Isaac Sim and optionally capture data.

Loads trajectory data (joint states + object poses) from an HDF5 episode,
re-builds the matching scene, replays frame-by-frame with kinematic objects.
Optionally captures RGB, depth, and/or tactile data and writes them into a
copy of the HDF5.

Without any --enable-* flags the trajectory is replayed without saving data.

Usage:
    # Replay only (no data capture):
    python replay.py --hdf5 outputs/.../episode_000025.hdf5

    # Capture RGB:
    python replay.py --hdf5 outputs/.../episode_000025.hdf5 --enable-rgb

    # Capture RGB + depth:
    python replay.py --hdf5 outputs/.../episode_000025.hdf5 --enable-rgb --enable-depth

    # Capture RGB + depth + tactile:
    python replay.py --hdf5 outputs/.../episode_000025.hdf5 \
        --enable-rgb --enable-depth --enable-tactile

    # Tactile only:
    python replay.py --hdf5 outputs/.../episode_000025.hdf5 --tactile-only
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time

import numpy as np

from isaaclab.app import AppLauncher
from utils.isaac_rendering import configure_headless_camera_parity_experience
from utils.logging_config import add_logging_arguments, configure_logging


logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Replay HDF5 episode and inject RGB/tactile data.")
parser.add_argument("--hdf5", type=str, required=True, help="Path to source episode HDF5.")
parser.add_argument("--collect-config", type=str, default=None,
                    help="Collect config YAML (camera defs + modalities). "
                         "Defaults to configs/collect/default.yaml.")
parser.add_argument("--scene", type=str, default=None, help="Override scene YAML path.")
parser.add_argument("--cameras", type=str, nargs="*", default=None, help="Camera IDs (default: all from config).")
parser.add_argument("--output", type=str, default=None,
                    help="Output HDF5 path. Default: <original>_replay.hdf5 in same dir.")
parser.add_argument("--warmup-steps", type=int, default=5, help="Render warm-up steps (default: 5).")
parser.add_argument("--realtime", action="store_true",
                    help="Throttle replay to original recording FPS (no-op when --enable-rgb is set).")
parser.add_argument("--save-sample-frames", type=str, default=None,
                    help="Dir to save PNG frames at key indices (for debugging).")
parser.add_argument("--enable-rgb", action="store_true", help="Enable RGB image capture during replay.")
parser.add_argument("--enable-depth", action="store_true", help="Enable depth image capture during replay.")
parser.add_argument("--enable-tactile", action="store_true", help="Enable tactile capture during replay.")
parser.add_argument("--tactile-only", action="store_true",
                    help="Capture only tactile data; implies --enable-tactile and skips RGB camera output.")
parser.add_argument("--enable-generalization", action="store_true",
                    help="Enable scene generalization (background, lighting, table texture, camera perturbation, clutter).")
parser.add_argument("--restore-generalization", action="store_true",
                    help="Restore the exact generalization from the source HDF5 episode "
                         "(background, lighting, table texture, camera offsets). "
                         "Mutually exclusive with --enable-generalization.")
parser.add_argument("--resample-groups", type=str, default=None,
                    help="Comma-separated groups to re-randomize while keeping the rest from "
                         "the original episode.  Requires --enable-generalization.  "
                         "Safe groups: background, table_surface, light, camera.  "
                         "Example: --resample-groups background,light")
parser.add_argument("--generalization-config", type=str, default=None,
                    help="Path to scene generalization YAML. Defaults to configs/scene/generalization.yaml.")
parser.add_argument("--generalization-split", type=str, default=None, choices=["seen", "unseen", "all"],
                    help="Discrete visual asset split for scene generalization. Defaults to the YAML asset_split.")
parser.add_argument("--background-asset-index", type=int, default=None,
                    help="0-based index into the filtered background asset pool. "
                         "Used for deterministic batch replay background cycling.")
parser.add_argument("--seed", type=int, default=None,
                    help="Random seed for scene generalization sampling during replay. "
                         "When unset, generalization is unseeded (non-deterministic).")
AppLauncher.add_app_launcher_args(parser)
add_logging_arguments(parser)
args_cli = parser.parse_args()
configure_logging(args_cli.log_level)
args_cli.enable_cameras = True  # Isaac camera stack is needed for render + optional sensors.
configure_headless_camera_parity_experience(args_cli, log_prefix="replay")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import h5py
import torch

import isaaclab.sim as sim_utils
from isaaclab.utils.io import load_yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from build import (  # noqa: E402
    build_scene,
    collect_task_asset_codes,
    dict_to_generalization_sample,
    merge_generalization_samples,
    parse_scene_generalization_config,
    relativize_sample_paths,
    resample_light,
    resample_usd_scene_light,
    resolve_sample_paths,
    resolve_table_spec,
    sample_scene_generalization,
    scene_generalization_config_to_dict,
    scene_generalization_sample_debug_lines,
    scene_generalization_sample_to_dict,
    SAFE_RESAMPLE_GROUPS,
)
from utils.seed_policy import seed_everything  # noqa: E402
from collector.cameras import CameraRig  # noqa: E402
from collector.config import load_collect_config  # noqa: E402
from collector.tacmap_rig import TacMapRig, build_tacmap_raycast_targets, tacmap_supported  # noqa: E402
from utils.scene_paths import resolve_scene_yaml_path  # noqa: E402
from utils.replay_support import (  # noqa: E402
    IncrementalTacMapWriter,
    ensure_file_writable,
    inject_camera_buffers_into_hdf5,
    resolve_replay_physics_dt,
    resolve_replay_sim_device,
    should_enable_tactile,
)
try:
    from utils.runtime_helpers import classify_interactive_objects, initialize_articulation_state  # noqa: E402
except ModuleNotFoundError:
    from runtime_helpers import classify_interactive_objects, initialize_articulation_state  # noqa: E402



def _make_rigid_bodies_kinematic(object_prim_paths: dict, object_body_types: dict) -> int:
    """Set dynamic rigid bodies to kinematic so physics won't move them.

    Must be called AFTER build_scene() but BEFORE sim.reset().
    """
    try:
        import omni.usd
        from pxr import Usd, UsdPhysics
    except ImportError:
        print("[WARN] omni.usd/pxr unavailable 鈥?cannot set kinematic mode")
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
    """Map HDF5 joint index 鈫?robot articulation joint index."""
    lookup = {name: idx for idx, name in enumerate(robot_names)}
    return {h: lookup[n] for h, n in enumerate(hdf5_names) if n in lookup}


def _write_replay_state_to_sim(
    *,
    sim,
    robot_art,
    qpos_t: torch.Tensor,
    qvel_t: torch.Tensor,
    interactive_objects: dict,
    object_trajectories: dict[str, dict],
    obj_joint_maps: dict[str, dict[int, int]],
    frame_idx: int,
) -> None:
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


def _refresh_recorded_state_through_physx(
    *,
    sim,
    pre_step_hooks: list,
    write_recorded_state,
) -> None:
    for hook in pre_step_hooks:
        hook()
    sim.step(render=False)
    write_recorded_state()


def _inject_rgb_into_hdf5(
    output_path: str,
    cam_rgb_data: dict[str, object],
    cam_depth_data: dict[str, object],
    cam_intrinsics: dict[str, np.ndarray],
    cam_extrinsics: dict[str, object],
) -> None:
    """Compatibility wrapper for older callers."""
    inject_camera_buffers_into_hdf5(
        output_path,
        cam_rgb_data,
        cam_depth_data,
        cam_intrinsics,
        cam_extrinsics,
        clear_after_write=True,
    )
    return



def main() -> None:
    if args_cli.restore_generalization and args_cli.enable_generalization:
        parser.error("--restore-generalization and --enable-generalization are mutually exclusive.")
    if args_cli.resample_groups and not args_cli.enable_generalization:
        parser.error("--resample-groups requires --enable-generalization.")

    hdf5_path = os.path.abspath(args_cli.hdf5)
    if not os.path.isfile(hdf5_path):
        raise FileNotFoundError(f"HDF5 not found: {hdf5_path}")

    f = h5py.File(hdf5_path, "r")
    recorded_scene_file = f["meta/scene_file"][()].decode()
    scene_name = f["meta/scene_name"][()].decode() if "meta/scene_name" in f else None
    scene_file = resolve_scene_yaml_path(
        cli_scene_path=args_cli.scene,
        recorded_scene_path=recorded_scene_file,
        recorded_scene_name=scene_name,
        script_dir=SCRIPT_DIR,
    )
    robot_key = f["meta/robot_key"][()].decode()
    data_fps = int(f["meta/fps"][()])
    frame_count = int(f["meta/frame_count"][()])

    qpos_all = f["robot/qpos"][:]  # (N, n_joints) float32
    hdf5_joint_names = [
        n.decode() if isinstance(n, bytes) else str(n)
        for n in f["robot/joint_names"][:]
    ]
    # Read original physics dt & step_stride for accurate replay
    original_step_stride = int(f["meta/step_stride"][()])
    replay_physics_dt = resolve_replay_physics_dt(f["meta"])

    # Object trajectories: pose_world = [x, y, z, qx, qy, qz, qw] (xyzw)
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

    recorded_generalization_sample_json: dict | None = None
    if "meta/scene_generalization_sample" in f:
        raw = f["meta/scene_generalization_sample"][()]
        try:
            recorded_generalization_sample_json = json.loads(
                raw.decode() if isinstance(raw, bytes) else str(raw)
            )
        except (json.JSONDecodeError, UnicodeDecodeError):
            print("[replay][WARN] Failed to parse scene_generalization_sample from HDF5")

    if recorded_generalization_sample_json is not None:
        from build.generalization import _find_dataset_root
        _dataset_root = _find_dataset_root(SCRIPT_DIR)
        recorded_generalization_sample_json = resolve_sample_paths(
            recorded_generalization_sample_json, _dataset_root,
        )

    recorded_generalization_config_json: dict | None = None
    if "meta/scene_generalization_config" in f:
        raw = f["meta/scene_generalization_config"][()]
        try:
            recorded_generalization_config_json = json.loads(
                raw.decode() if isinstance(raw, bytes) else str(raw)
            )
        except (json.JSONDecodeError, UnicodeDecodeError):
            print("[replay][WARN] Failed to parse scene_generalization_config from HDF5")
    f.close()

    print(f"[replay] HDF5 : {hdf5_path}")
    print(f"[replay] Scene : {scene_file}")
    print(f"[replay] Robot : {robot_key}")
    print(f"[replay] Frames: {frame_count} @ {data_fps} fps")
    print(f"[replay] Physics dt: {replay_physics_dt:.7f} (step_stride={original_step_stride})")
    print(f"[replay] Objects: {list(object_trajectories.keys())}")

    collect_cfg_path = args_cli.collect_config or os.path.join(SCRIPT_DIR, "configs", "collect", "default.yaml")
    collect_cfg = load_collect_config(collect_cfg_path, robot_key=robot_key)
    tactile_only = bool(args_cli.tactile_only)
    tactile_enabled = should_enable_tactile(
        cli_enable_tactile=bool(args_cli.enable_tactile or tactile_only),
        collect_tactile_enabled=collect_cfg.enabled("tactile"),
    )
    if tactile_enabled and not tacmap_supported(robot_key):
        raise ValueError(
            f"TacMap assets are missing or TacMap is not configured for robot_key={robot_key!r}."
        )
    rgb_enabled = bool(args_cli.enable_rgb) and not tactile_only
    depth_enabled = bool(args_cli.enable_depth) and not tactile_only
    camera_needed = rgb_enabled or depth_enabled
    camera_cfgs = collect_cfg.cameras if camera_needed else []

    if args_cli.cameras and camera_needed:
        selected = set(args_cli.cameras)
        camera_cfgs = [c for c in camera_cfgs if c.camera_id in selected]
    elif args_cli.cameras and not camera_needed:
        print("[replay] --cameras ignored because neither --enable-rgb nor --enable-depth is set")
    if args_cli.save_sample_frames and not rgb_enabled:
        print("[replay] --save-sample-frames ignored because --enable-rgb is not set")

    cam_ids_to_capture = [c.camera_id for c in camera_cfgs]
    any_capture = camera_needed or tactile_enabled
    if not any_capture:
        print("[replay] No capture enabled (no --enable-rgb / --enable-depth / --enable-tactile / --tactile-only)")
        print("[replay] Will replay trajectory without saving data")
    print(f"[replay] RGB: {rgb_enabled}")
    print(f"[replay] Depth: {depth_enabled}")
    print(f"[replay] Cameras: {cam_ids_to_capture}")
    print(f"[replay] Tactile: {tactile_enabled}")
    if tactile_enabled:
        print("[replay] Tactile backend: TacMap")

    sim_device = resolve_replay_sim_device(
        requested_device=getattr(args_cli, "device", None),
        enable_tactile=tactile_enabled,
    )
    sim_cfg = sim_utils.SimulationCfg(
        dt=replay_physics_dt,
        device=sim_device,
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    print(f"[replay] Sim device: {sim_device}")

    task = load_yaml(scene_file)
    task_dir = os.path.dirname(scene_file)
    _, table_h, _ = resolve_table_spec(task)
    sim.set_camera_view([0.0, 1.0, table_h + 0.45], [0.0, 0.0, table_h])

    # ── 3b. Scene generalization (optional) ──────────────────────────
    generalization_enabled = bool(args_cli.enable_generalization or args_cli.restore_generalization)
    generalization_cfg = None
    generalization_sample = None
    _replay_generalization_mode = "none"

    def _load_generalization_cfg(raw_config=None):
        """Load and parse generalization config, force-disabling unsafe groups.

        When *raw_config* is a dict (e.g. from HDF5), it is used directly;
        otherwise the config is loaded from disk (CLI arg or default path).
        """
        from build.generalization import ObjectGeneralizationCfg
        if raw_config is not None:
            cfg = parse_scene_generalization_config(raw_config)
            gen_cfg_path = "<from HDF5>"
        else:
            gen_cfg_path = args_cli.generalization_config or os.path.join(
                SCRIPT_DIR, "configs", "scene", "generalization.yaml",
            )
            gen_raw = load_yaml(gen_cfg_path) or {}
            cfg = parse_scene_generalization_config(gen_raw, config_path=gen_cfg_path)
        print("[replay] Force-disabled unsafe groups: object_pose")
        cfg.spatial.object_pose = ObjectGeneralizationCfg(enabled=False)
        return cfg, gen_cfg_path

    if args_cli.restore_generalization:
        # ── Branch A: restore exact generalization from HDF5 ──
        if recorded_generalization_sample_json is None:
            raise RuntimeError(
                "--restore-generalization: no scene_generalization_sample in source HDF5."
            )
        generalization_cfg, gen_cfg_path = _load_generalization_cfg(
            raw_config=recorded_generalization_config_json,
        )
        generalization_sample = dict_to_generalization_sample(
            recorded_generalization_sample_json
        )
        generalization_sample.robot_key = robot_key
        _replay_generalization_mode = "restored"
        print(f"[replay] Generalization RESTORED from HDF5 (cfg source: {gen_cfg_path})")
        for line in scene_generalization_sample_debug_lines(generalization_sample):
            print(f"[replay]   {line}")

    elif args_cli.enable_generalization:
        generalization_cfg, gen_cfg_path = _load_generalization_cfg()
        resample_group_names: set[str] | None = None

        if args_cli.seed is not None:
            seed_everything(args_cli.seed)
            print(f"[replay] seed_everything({args_cli.seed})")

        if args_cli.resample_groups:
            # ── Branch B: selective re-randomization ──
            resample_group_names = {
                g.strip() for g in args_cli.resample_groups.split(",") if g.strip()
            }
            if not resample_group_names:
                parser.error("--resample-groups must specify at least one group name.")
            unsafe = resample_group_names - SAFE_RESAMPLE_GROUPS
            if unsafe:
                raise ValueError(
                    f"Unsafe resample groups for kinematic replay: {unsafe}. "
                    f"Safe groups: {sorted(SAFE_RESAMPLE_GROUPS)}"
                )
            if recorded_generalization_sample_json is None:
                raise RuntimeError(
                    "--resample-groups: no scene_generalization_sample in source HDF5 "
                    "to use as base."
                )
            base_sample = dict_to_generalization_sample(
                recorded_generalization_sample_json
            )
            new_sample = sample_scene_generalization(
                generalization_cfg,
                enabled=True,
                task_asset_codes=collect_task_asset_codes(task),
                asset_split=args_cli.generalization_split,
                background_asset_index=args_cli.background_asset_index,
            )

            # ── Cross-group dependency: light ↔ background ──
            follow_bg_yaw = getattr(generalization_cfg.appearance.light, "follow_background_yaw", False)
            resample_light_only = "light" in resample_group_names and "background" not in resample_group_names
            resample_bg_only = "background" in resample_group_names and "light" not in resample_group_names
            if resample_light_only:
                base_bg = getattr(getattr(base_sample, "appearance", None), "background", None)
                base_bg_kind = getattr(base_bg, "asset_kind", None)
                base_bg_yaw = getattr(base_bg, "yaw_deg", 0.0)
                base_is_usd_scene = str(base_bg_kind or "").strip().lower() == "usd_scene"
                new_sample.appearance.usd_scene_light = resample_usd_scene_light(
                    generalization_cfg,
                    enabled=True,
                    background_asset_kind=base_bg_kind,
                    asset_split=args_cli.generalization_split,
                )
                if base_is_usd_scene:
                    new_sample.appearance.light = resample_light(
                        generalization_cfg,
                        enabled=False,
                        background_yaw_deg=base_bg_yaw,
                        asset_split=args_cli.generalization_split,
                    )
                    print("[replay] Corrected light sample: re-sampled USD scene lights for base usd_scene background")
                elif follow_bg_yaw:
                    new_sample.appearance.light = resample_light(
                        generalization_cfg,
                        enabled=True,
                        background_yaw_deg=base_bg_yaw,
                        asset_split=args_cli.generalization_split,
                    )
                    print(
                        f"[replay] Corrected light sample: re-sampled with base "
                        f"background yaw_deg={base_bg_yaw:.1f} (follow_background_yaw=True)"
                    )
            if resample_bg_only and follow_bg_yaw:
                print(
                    "[replay][WARN] Resampling background without light while "
                    "follow_background_yaw=True: restored light direction may "
                    "not track the new background yaw."
                )

            generalization_sample = merge_generalization_samples(
                base_sample, new_sample, resample_group_names,
            )
            generalization_sample.robot_key = robot_key
            _replay_generalization_mode = f"selective:{','.join(sorted(resample_group_names))}"
            print(
                f"[replay] Generalization SELECTIVE resample: "
                f"{sorted(resample_group_names)} (cfg: {gen_cfg_path})"
            )
            for line in scene_generalization_sample_debug_lines(generalization_sample):
                print(f"[replay]   {line}")
        else:
            # ── Branch C: full resample (existing behaviour) ──
            generalization_sample = sample_scene_generalization(
                generalization_cfg,
                enabled=True,
                task_asset_codes=collect_task_asset_codes(task),
                asset_split=args_cli.generalization_split,
                background_asset_index=args_cli.background_asset_index,
            )
            _replay_generalization_mode = "resampled"
            print(f"[replay] Generalization FULL resample (cfg: {gen_cfg_path})")
            for line in scene_generalization_sample_debug_lines(generalization_sample):
                print(f"[replay]   {line}")

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
    robot_runtime = runtime.get("robot_runtime", {})
    pre_step_hooks = list(robot_runtime.get("pre_step_hooks", []))
    if pre_step_hooks:
        print(f"[replay] Gravity compensation hooks: {len(pre_step_hooks)}")
    tactile_rig = None
    if tactile_enabled:
        robot_prim_path = str(runtime.get("robot_pose", {}).get("prim_path", "/World/Objects/GlobalRobot"))
        tacmap_object_prim_paths, tacmap_object_body_types = build_tacmap_raycast_targets(
            object_prim_paths,
            object_body_types,
        )
        tactile_rig = TacMapRig(
            robot_key=robot_key,
            object_prim_paths=tacmap_object_prim_paths,
            object_body_types=tacmap_object_body_types,
            robot_prim_path=robot_prim_path,
        )

    n_kin = _make_rigid_bodies_kinematic(object_prim_paths, object_body_types)
    print(f"[replay] Kinematic rigid prims: {n_kin}")

    robot_art = interactive_objects.get("global_robot")
    camera_rig = None
    camera_gen_sample = None
    if camera_needed:
        camera_gen_sample = (
            scene_generalization_sample_to_dict(generalization_sample)
            if generalization_sample is not None else None
        )
        camera_rig = CameraRig(
            sim, camera_cfgs,
            enable_rgb=rgb_enabled,
            enable_depth=depth_enabled,
            robot_articulation=robot_art,
            camera_generalization_sample=camera_gen_sample,
        )
        print(f"[replay] Camera rig IDs: {camera_rig.camera_ids}")
    else:
        print("[replay] Skipping camera rig (no RGB/depth enabled)")

    sim.reset()

    groups = classify_interactive_objects(interactive_objects)
    for art in groups.controlled_articulations:
        initialize_articulation_state(art)
    for art in groups.task_articulations:
        initialize_articulation_state(art, set_hold_target=False)
    for obj in groups.other_objects:
        obj.reset()

    if camera_rig is not None:
        camera_rig.set_robot_articulation(robot_art)
        camera_rig.initialize_after_reset()
    if tactile_rig is not None:
        tactile_rig.initialize_after_reset(robot_art)

    if robot_art is None:
        raise RuntimeError("No global_robot in scene 鈥?cannot replay.")
    robot_joint_names = list(robot_art.joint_names)
    n_robot_joints = len(robot_joint_names)
    joint_map = _build_joint_map(hdf5_joint_names, robot_joint_names)
    print(f"[replay] Joints: {len(joint_map)}/{len(hdf5_joint_names)} mapped 鈫?{n_robot_joints} robot joints")

    obj_joint_maps: dict[str, dict[int, int]] = {}
    for obj_id, traj in object_trajectories.items():
        if "joint_names" not in traj:
            continue
        obj = interactive_objects.get(obj_id)
        if obj is None or not hasattr(obj, "joint_names"):
            continue
        obj_joint_maps[obj_id] = _build_joint_map(traj["joint_names"], list(obj.joint_names))

    output_path = None
    if any_capture:
        if args_cli.output:
            output_path = os.path.abspath(args_cli.output)
            if os.path.isdir(output_path):
                output_path = os.path.join(output_path, os.path.basename(hdf5_path))
        else:
            stem = os.path.splitext(os.path.basename(hdf5_path))[0]
            output_path = os.path.join(os.path.dirname(hdf5_path), f"{stem}_replay.hdf5")

        # Copy original HDF5 as starting point (skip when in-place)
        if os.path.exists(output_path) and os.path.samefile(hdf5_path, output_path):
            print(f"[replay] In-place update: {output_path}")
        else:
            print(f"[replay] Copying {hdf5_path} -> {output_path}")
            shutil.copy2(hdf5_path, output_path)
        ensure_file_writable(output_path)

    cam_rgb_bufs: dict[str, list[np.ndarray]] = {cid: [] for cid in cam_ids_to_capture}
    cam_depth_bufs: dict[str, list[np.ndarray]] = {cid: [] for cid in cam_ids_to_capture}
    cam_intrinsics: dict[str, np.ndarray] = {}
    cam_extrinsics: dict[str, list[np.ndarray]] = {cid: [] for cid in cam_ids_to_capture}
    cam_metadata: dict[str, dict[str, object]] = {}
    tactile_writer: IncrementalTacMapWriter | None = None
    if tactile_rig is not None:
        tactile_writer = IncrementalTacMapWriter(output_path, frame_count)
        tactile_writer.open()

    # Kinematic replay is the default: every frame is an explicit HDF5 state.
    # PhysX stepping is reserved exclusively for tactile scene queries;
    # RGB and depth replay are purely kinematic (no sim.step()).
    _preflight_dir = os.environ.get("DEX2BENCH_PREFLIGHT_DIR", "")
    print(
        "[replay] Step mode: kinematic (tactile=%s)" % (tactile_rig is not None),
        flush=True,
    )

    for _ in range(args_cli.warmup_steps):
        sim.render()

    # ── 11b. Camera preflight at HOME (for comparison with run_policy) ─
    if camera_rig is not None:
        home_qpos = robot_art.data.default_joint_pos.clone()
        robot_art.write_joint_state_to_sim(home_qpos, torch.zeros_like(home_qpos))
        robot_art.set_joint_position_target(home_qpos)
        robot_art.write_data_to_sim()
        camera_rig.sync_mounted_camera_poses()
        sim.render()
        pf_frames = camera_rig.capture(0.0)
        # Save preflight images if DEX2BENCH_PREFLIGHT_DIR is set
        if _preflight_dir and camera_rig is not None:
            os.makedirs(_preflight_dir, exist_ok=True)
            for cid, frm in pf_frames.items():
                if frm is not None and frm.rgb is not None:
                    import cv2 as _cv2
                    img_bgr = _cv2.cvtColor(frm.rgb, _cv2.COLOR_RGB2BGR)
                    _cv2.imwrite(os.path.join(_preflight_dir, f"replay_preflight_{cid}.png"), img_bgr)
            _np = __import__('numpy')
            with open(os.path.join(_preflight_dir, "replay_preflight_extrinsics.txt"), "w") as _ef:
                for cid in camera_rig.camera_ids:
                    frm = pf_frames.get(cid)
                    if frm is not None:
                        p = _np.asarray(frm.cam_pos_w).round(6).tolist()
                        q = _np.asarray(frm.cam_quat_wxyz).round(6).tolist()
                        _ef.write(f"{cid} pos_w={p} quat_wxyz={q}\n")
        print("[replay] Camera preflight at HOME:", flush=True)
        for cid in camera_rig.camera_ids:
            frm = pf_frames.get(cid)
            if frm is not None:
                p = np.asarray(frm.cam_pos_w).round(4).tolist()
                q = np.asarray(frm.cam_quat_wxyz).round(4).tolist()
                _rgb_info = f"rgb={tuple(frm.rgb.shape)}" if frm.rgb is not None else "rgb=off"
                _depth_info = f"depth={tuple(frm.depth_m.shape)}" if frm.depth_m is not None else "depth=off"
                print(f"  [camera] {cid}: {_rgb_info} {_depth_info} pos_w={p} quat_wxyz={q}", flush=True)
            else:
                print(f"  [camera] {cid}: MISSING", flush=True)

    # ── 11c. Robot HOME alignment (DISABLED) ──────────────────────────
    # NOTE: An older version applied a per-joint offset = (USD_HOME - qpos[0])
    # to "align" the trajectory to the current USD HOME pose.  That is wrong:
    # HDF5 frame 0 is the MEASURED qpos at episode start (which may already be
    # mid-motion), not a HOME snapshot.  Applying the offset shifts every link
    # by several mm-cm in world space, producing the visual "grabbing air"
    # symptom in kinematic replay.  We now apply the raw HDF5 qpos directly
    # and only warn if a joint's start state looks far from the current USD
    # HOME (which would indicate the robot USD was changed since collection
    # and the data may no longer be replayable on this asset).
    default_qpos = robot_art.data.default_joint_pos[0].clone()
    joint_offset = torch.zeros(n_robot_joints, device=default_qpos.device)
    diffs = []
    for h_idx, r_idx in joint_map.items():
        d = float(qpos_all[0, h_idx]) - float(default_qpos[r_idx])
        if abs(d) > 0.3:  # > ~17 deg — normal if collection started mid-motion
            diffs.append((robot_joint_names[r_idx], d))
    if diffs:
        print(f"[replay] NOTE: {len(diffs)} joints start >0.3 rad from current USD HOME (OK if collection started mid-motion):")
        for name, d in diffs[:5]:
            print(f"  {name}: qpos[0]-HOME = {d:+.3f} rad")
        if len(diffs) > 5:
            print(f"  ... ({len(diffs) - 5} more)")

    t0 = time.monotonic()
    _frame_dt = 1.0 / data_fps if (args_cli.realtime and not any_capture) else None
    _next_frame_wall = time.monotonic()

    for frame_idx in range(frame_count):
        # A) Set robot joint state AND position-drive targets so the PD
        # controller holds the commanded pose if a fallback/tactile PhysX
        # refresh is needed. In the normal RGB/depth path, this is pure
        # kinematic state assignment followed by render.
        qpos_t = robot_art.data.default_joint_pos.clone()
        for h_idx, r_idx in joint_map.items():
            qpos_t[0, r_idx] = float(qpos_all[frame_idx, h_idx]) + joint_offset[r_idx]
        qvel_t = torch.zeros_like(qpos_t)

        def _write_current_recorded_state() -> None:
            _write_replay_state_to_sim(
                sim=sim,
                robot_art=robot_art,
                qpos_t=qpos_t,
                qvel_t=qvel_t,
                interactive_objects=interactive_objects,
                object_trajectories=object_trajectories,
                obj_joint_maps=obj_joint_maps,
                frame_idx=frame_idx,
            )

        _write_current_recorded_state()

        # B) PhysX refresh for tactile sensing only.  RGB and depth replay
        # are purely kinematic — no sim.step() is needed for those modalities.
        if tactile_rig is not None:
            _refresh_recorded_state_through_physx(
                sim=sim,
                pre_step_hooks=pre_step_hooks,
                write_recorded_state=_write_current_recorded_state,
            )

        if camera_rig is not None:
            camera_rig.sync_mounted_camera_poses()

        sim.render()

        # Opt-in frame-0 profile dump: report world positions of robot root,
        # all body links (so you can pick out fingertips), and each object's
        # set/actual root pose.  Compare these numbers to the same dump from
        # tools/replay/replay_commanded.py (which grabs correctly) to localise
        # the misalignment (robot side vs object side).
        #
        # NOTE: body_state_w is refreshed lazily by IsaacLab when body poses are
        # read after write_joint_state_to_sim() invalidates their timestamp.
        # Do not add a diagnostic sim.step() here; replay should stay kinematic
        # unless the explicit refresh path above is active.
        if args_cli.profile and frame_idx == 0:
            try:
                root_pos = robot_art.data.root_pos_w[0].detach().cpu().numpy()
                root_quat = robot_art.data.root_quat_w[0].detach().cpu().numpy()
                logger.info("replay robot root_pos_w=%s", root_pos.tolist())
                logger.info("replay robot root_quat_w(wxyz)=%s", root_quat.tolist())
            except Exception as e:
                logger.info("replay robot root pose unavailable: %s", e)
            try:
                body_names = list(robot_art.body_names)
                body_pos = robot_art.data.body_pos_w[0].detach().cpu().numpy()
                # Print only the hand / tip / tcp / wrist / tool / palm links
                # plus the first 5 body links so users can scan.
                keywords = ("thumb", "index", "middle", "ring", "little",
                            "tip", "tcp", "tool", "wrist", "palm", "ee", "end_effector")
                for i, n in enumerate(body_names):
                    lname = n.lower()
                    if any(k in lname for k in keywords) or i < 3:
                        logger.info("replay link[%02d] %s world_xyz=%s", i, n, body_pos[i].round(4).tolist())
            except Exception as e:
                logger.info("replay body link dump failed: %s", e)
            for obj_id, traj in object_trajectories.items():
                p_rec = traj["pose_world"][0]
                obj = interactive_objects.get(obj_id)
                actual = None
                try:
                    if obj is not None and hasattr(obj, "data"):
                        actual = obj.data.root_pos_w[0].detach().cpu().numpy()
                except Exception:
                    actual = None
                rec_xyz = [float(p_rec[0]), float(p_rec[1]), float(p_rec[2])]
                act_xyz = actual.round(4).tolist() if actual is not None else "N/A"
                if actual is not None:
                    derr = float(np.linalg.norm(actual - p_rec[:3])) * 1000.0
                    logger.info(
                        "replay object %s recorded_xyz=%s actual_xyz=%s delta=%.2fmm",
                        obj_id,
                        rec_xyz,
                        act_xyz,
                        derr,
                    )
                else:
                    logger.info("replay object %s recorded_xyz=%s actual=N/A", obj_id, rec_xyz)

        # Real-time throttle (visualization only, disabled when capturing)
        if _frame_dt is not None:
            _next_frame_wall += _frame_dt
            _sleep = _next_frame_wall - time.monotonic()
            if _sleep > 0:
                time.sleep(_sleep)

        # E) Capture all RGB cameras via CameraRig.capture()
        if camera_rig is not None:
            _capture_dt = sim.get_physics_dt() if tactile_rig is not None else 0.0
            cam_frames = camera_rig.capture(_capture_dt)

            # Save first-frame raw capture as PNG for ground-truth comparison
            if frame_idx == 0 and _preflight_dir:
                import cv2 as _cv2
                for _cid, _frm in cam_frames.items():
                    if _frm is not None and _frm.rgb is not None:
                        _bgr = _cv2.cvtColor(_frm.rgb, _cv2.COLOR_RGB2BGR)
                        _cv2.imwrite(os.path.join(_preflight_dir, f"replay_frame0_{_cid}.png"), _bgr)
                with open(os.path.join(_preflight_dir, "replay_frame0_extrinsics.txt"), "w") as _ef:
                    for _cid in camera_rig.camera_ids:
                        _frm = cam_frames.get(_cid)
                        if _frm is not None:
                            _p = np.asarray(_frm.cam_pos_w).round(6).tolist()
                            _q = np.asarray(_frm.cam_quat_wxyz).round(6).tolist()
                            _ef.write(f"{_cid} pos_w={_p} quat_wxyz={_q}\n")
                print(f"[replay] Frame 0 raw capture saved to {_preflight_dir}", flush=True)

            for cam_id in cam_ids_to_capture:
                cf = cam_frames.get(cam_id)
                if cf is None:
                    # Pad with zeros if capture failed
                    if cam_rgb_bufs[cam_id]:
                        cam_rgb_bufs[cam_id].append(np.zeros_like(cam_rgb_bufs[cam_id][0]))
                    if cam_depth_bufs[cam_id]:
                        cam_depth_bufs[cam_id].append(np.zeros_like(cam_depth_bufs[cam_id][0]))
                    if cam_rgb_bufs[cam_id] or cam_depth_bufs[cam_id]:
                        cam_extrinsics[cam_id].append(np.eye(4, dtype=np.float32))
                    continue

                rgb = cf.rgb
                if rgb is not None:
                    if rgb.shape[-1] > 3:
                        rgb = rgb[..., :3]
                    cam_rgb_bufs[cam_id].append(rgb.astype(np.uint8))
                else:
                    if cam_rgb_bufs[cam_id]:
                        cam_rgb_bufs[cam_id].append(np.zeros_like(cam_rgb_bufs[cam_id][0]))

                depth = cf.depth_m
                if depth is not None:
                    cam_depth_bufs[cam_id].append(depth.astype(np.float32))
                else:
                    if cam_depth_bufs[cam_id]:
                        cam_depth_bufs[cam_id].append(np.zeros_like(cam_depth_bufs[cam_id][0]))

                cam_extrinsics[cam_id].append(cf.extrinsic_world_from_cam.astype(np.float32))
                if cam_id not in cam_intrinsics:
                    cam_intrinsics[cam_id] = cf.intrinsic.astype(np.float32)
                if cam_id not in cam_metadata:
                    cam_metadata[cam_id] = {
                        "camera_model": getattr(cf, "camera_model", "pinhole"),
                        "clipping_range": getattr(cf, "clipping_range", None),
                        "fisheye_camera_matrix": getattr(cf, "fisheye_camera_matrix", None),
                        "distortion_coefficients": getattr(cf, "distortion_coefficients", None),
                        "fisheye_valid_mask": getattr(cf, "fisheye_valid_mask", None),
                    }

        if tactile_rig is not None and tactile_writer is not None:
            tactile_writer.write_frame(frame_idx, tactile_rig.capture(dt=sim.get_physics_dt()))

        # F) Progress
        done = frame_idx + 1
        if done % 50 == 0 or done == frame_count:
            elapsed = time.monotonic() - t0
            actual_fps = done / max(elapsed, 1e-6)
            eta = (frame_count - done) / max(actual_fps, 0.01)
            print(f"[replay] {done}/{frame_count} ({actual_fps:.1f} fps, ETA {eta:.0f}s)")

        # G) Save sample frames as PNG if requested
        if camera_rig is not None and rgb_enabled and args_cli.save_sample_frames and frame_idx in (0, 114, 228, 342, 456, 570, 684, 799):
            sample_dir = os.path.abspath(args_cli.save_sample_frames)
            os.makedirs(sample_dir, exist_ok=True)
            for cam_id in cam_ids_to_capture:
                if cam_id in cam_rgb_bufs and cam_rgb_bufs[cam_id]:
                    try:
                        from PIL import Image
                        rgb = cam_rgb_bufs[cam_id][-1]
                        Image.fromarray(rgb).save(os.path.join(sample_dir, f"{cam_id}_f{frame_idx:04d}.png"))
                    except Exception:
                        pass

    for cam_id in cam_ids_to_capture:
        n_rgb = len(cam_rgb_bufs[cam_id])
        n_depth = len(cam_depth_bufs[cam_id])
        n_extr = len(cam_extrinsics[cam_id])
        print(f"[replay] Camera '{cam_id}': {n_rgb} RGB frames, {n_depth} depth frames, {n_extr} extrinsics")

    if tactile_writer is not None:
        try:
            print(f"[replay] Finalizing tactile data in {output_path}")
            tactile_writer.close()
        finally:
            if tactile_rig is not None:
                tactile_rig.close()
            tactile_writer = None
            tactile_rig = None

    if camera_needed and output_path:
        print(f"[replay] Writing camera data to {output_path}")
        inject_camera_buffers_into_hdf5(
            output_path,
            cam_rgb_bufs,
            cam_depth_bufs,
            cam_intrinsics,
            cam_extrinsics,
            cam_metadata,
            clear_after_write=True,
        )
    elif not any_capture:
        print("[replay] Replay-only mode, no data written")

    # ── 13b. Store generalization metadata in output HDF5 ─────────
    if generalization_enabled and output_path and generalization_sample is not None:
        sample_dict = scene_generalization_sample_to_dict(generalization_sample)
        sample_dict["_replay_generalization_mode"] = _replay_generalization_mode
        sample_dict["_generalization_split"] = args_cli.generalization_split or getattr(generalization_cfg, "asset_split", None)
        sample_dict = relativize_sample_paths(sample_dict)
        config_dict = scene_generalization_config_to_dict(generalization_cfg)
        config_dict = relativize_sample_paths(config_dict)
        with h5py.File(output_path, "a") as f_out:
            meta = f_out.require_group("meta")
            str_dtype = h5py.string_dtype(encoding="utf-8")
            if "scene_generalization_sample" in meta:
                del meta["scene_generalization_sample"]
            meta.create_dataset(
                "scene_generalization_sample",
                data=np.asarray(json.dumps(sample_dict), dtype=str_dtype),
            )
            if "scene_generalization_config" in meta:
                del meta["scene_generalization_config"]
            meta.create_dataset(
                "scene_generalization_config",
                data=np.asarray(json.dumps(config_dict), dtype=str_dtype),
            )
        print(f"[replay] Generalization sample written to {output_path} (mode={_replay_generalization_mode})")

    elapsed_total = time.monotonic() - t0
    print(f"[replay] Done: {frame_count} frames in {elapsed_total:.1f}s")
    if output_path:
        print(f"[replay] Output: {output_path}")
    else:
        print("[replay] No output file (replay-only mode)")
    # simulation_app.close() may hang due to replicator wait; force exit
    os._exit(0)


if __name__ == "__main__":
    main()
