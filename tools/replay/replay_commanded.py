r"""Replay an HDF5 episode by SENDING COMMANDED ACTIONS through the same PD
control path used at inference time, instead of teleporting joints to qpos.

Diagnostic purpose
------------------
This script answers the question: "Is the bottleneck for ACT eval the data,
or the model?"  It re-plays the *commanded* actions stored in
``action/commanded`` (the same targets the teleop / scripted controller wrote
to the PD controller during data collection) at the original 20 Hz beat,
upsampled by linear interpolation to 60 Hz physics.

Differences from replay.py
--------------------------
- replay.py: kinematic — writes *qpos* to ``write_joint_state_to_sim`` every
  recorded frame.  Objects are kinematic too.  Result: the trajectory is
  guaranteed to look like the source (no physics interaction).
- replay_commanded.py: dynamic — writes *commanded actions* to
  ``set_joint_position_target`` and lets the PD controller + physics decide
  the actual motion.  Objects are NOT kinematic; they react to contact.
  Initial state of robot + objects is loaded from HDF5 frame 0 (no
  generalization, no random object poses).

If commanded replay reliably reproduces the source episode, the data is
fine and the ACT model is the weak link.  If it diverges, then either the
data quality is poor or the control method needs to be improved (and the
fix should be mirrored in run_policy.py).

Usage
-----
    cd .
    python ./tools/replay/replay_commanded.py --hdf5 ../teleopdata/dataset/21_condiment_box_loading/replay-generalization/episode_000000.hdf5

    # Headless:
    python ./tools/replay/replay_commanded.py --hdf5 ... --headless
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import pinocchio  # noqa: F401  (mirrors run_policy.py – pre-Isaac import)
except ImportError:
    pass

from isaaclab.app import AppLauncher
from utils.logging_config import add_logging_arguments, configure_logging


logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description="Replay an HDF5 episode by sending commanded actions (PD) at 20 Hz.",
)
parser.add_argument("--hdf5", type=str, required=True, help="Path to source HDF5 episode.")
parser.add_argument("--scene", type=str, default=None, help="Override scene YAML path.")
parser.add_argument(
    "--warmup-steps", type=int, default=60,
    help="Physics steps to settle robot at frame-0 commanded target before replay.",
)
parser.add_argument(
    "--realtime", action="store_true",
    help="Throttle to real-time pace (sleep). Default: max speed.",
)
parser.add_argument(
    "--no-interp", action="store_true",
    help="Disable 20Hz->60Hz linear interpolation (apply staircase commands).",
)
parser.add_argument(
    "--max-frames", type=int, default=0,
    help="Limit replay to first N commanded frames (0 = all).",
)
parser.add_argument(
    "--restore-generalization", action="store_true",
    help="Restore the exact scene generalization (table_height, lighting, background, "
         "camera offsets, resolved_object_placements, clutter) from the source HDF5. "
         "Required for physical-replay diagnostics to match the collection-time scene.",
)
parser.add_argument(
    "--generalization-config", type=str, default=None,
    help="Optional path to scene generalization YAML. Defaults to the config stored in "
         "the HDF5 episode (or configs/scene/generalization.yaml as fallback).",
)
AppLauncher.add_app_launcher_args(parser)
add_logging_arguments(parser)
args_cli = parser.parse_args()
configure_logging(args_cli.log_level)
args_cli.enable_cameras = False  # commanded replay does not need RGB inference

logger.info("Launching AppLauncher")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
logger.info("AppLauncher ready")

# ── Post-launch imports ───────────────────────────────────────────────
import h5py  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.utils.io import load_yaml  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from build import (  # noqa: E402
    build_scene,
    dict_to_generalization_sample,
    parse_scene_generalization_config,
    resolve_sample_paths,
    scene_generalization_sample_debug_lines,
)
from utils.episode_runtime import (  # noqa: E402
    initialize_scene_runtime_state,
    move_controlled_articulations_home,
)
from robots.gravity_compensation import GravityCompensator  # noqa: E402
from utils.runtime_helpers import (  # noqa: E402
    is_kinematic_rigid_object,
    write_articulation_targets,
    update_sim_objects,
)
from utils.scene_paths import resolve_scene_yaml_path  # noqa: E402


def _build_joint_map(hdf5_names: list[str], robot_names: list[str]) -> dict[int, int]:
    """Map HDF5 joint index -> robot articulation joint index, by name."""
    lookup = {name: idx for idx, name in enumerate(robot_names)}
    return {h: lookup[n] for h, n in enumerate(hdf5_names) if n in lookup}


def _decode_str_array(arr) -> list[str]:
    return [n.decode() if isinstance(n, bytes) else str(n) for n in arr]


def main() -> None:
    hdf5_path = os.path.abspath(args_cli.hdf5)
    if not os.path.isfile(hdf5_path):
        raise FileNotFoundError(f"HDF5 not found: {hdf5_path}")

    # ── 1. Load HDF5 trajectory ──────────────────────────────────────
    with h5py.File(hdf5_path, "r") as f:
        recorded_scene_file = f["meta/scene_file"][()].decode()
        scene_name = (
            f["meta/scene_name"][()].decode()
            if "meta/scene_name" in f else None
        )
        robot_key = f["meta/robot_key"][()].decode()
        data_fps = int(f["meta/fps"][()])
        frame_count = int(f["meta/frame_count"][()])
        step_stride = int(f["meta/step_stride"][()])

        # Robot trajectory
        qpos_all = f["robot/qpos"][:]  # (N, n_joints) for HOME offset
        hdf5_joint_names = _decode_str_array(f["robot/joint_names"][:])

        # Commanded actions (the whole point of this script)
        if "action/commanded" not in f:
            raise RuntimeError(
                f"HDF5 has no 'action/commanded' dataset: {hdf5_path}\n"
                "This episode was recorded without action capture.",
            )
        commanded_all = f["action/commanded"][:]  # (N, action_dim) float32
        if "action/action_names" in f:
            action_names = _decode_str_array(f["action/action_names"][:])
        else:
            action_names = list(hdf5_joint_names)
        action_valid = (
            f["action/action_valid"][:].astype(bool)
            if "action/action_valid" in f
            else np.ones(frame_count, dtype=bool)
        )

        # Object initial poses + per-frame qpos for articulated objects
        # We snapshot frame 0 only — objects are dynamic afterwards.
        object_initial: dict[str, dict] = {}
        for obj_id in f["objects"].keys():
            grp = f[f"objects/{obj_id}"]
            entry: dict = {"pose0_xyzw": grp["pose_world"][0]}  # (7,) [xyz, qx,qy,qz,qw]
            if "qpos" in grp and "joint_names" in grp:
                entry["joint_names"] = _decode_str_array(grp["joint_names"][:])
                entry["qpos0"] = grp["qpos"][0]
            object_initial[obj_id] = entry
        # Scene generalization (sample + config) — required for diagnostic parity
        # with the kinematic replay so robot/table/object placements line up.
        recorded_sample_json: dict | None = None
        if "meta/scene_generalization_sample" in f:
            raw = f["meta/scene_generalization_sample"][()]
            try:
                recorded_sample_json = json.loads(
                    raw.decode() if isinstance(raw, bytes) else str(raw)
                )
            except (json.JSONDecodeError, UnicodeDecodeError):
                print("[replay_cmd][WARN] Failed to parse scene_generalization_sample.")
        recorded_config_json: dict | None = None
        if "meta/scene_generalization_config" in f:
            raw = f["meta/scene_generalization_config"][()]
            try:
                recorded_config_json = json.loads(
                    raw.decode() if isinstance(raw, bytes) else str(raw)
                )
            except (json.JSONDecodeError, UnicodeDecodeError):
                print("[replay_cmd][WARN] Failed to parse scene_generalization_config.")
    scene_file = resolve_scene_yaml_path(
        cli_scene_path=args_cli.scene,
        recorded_scene_path=recorded_scene_file,
        recorded_scene_name=scene_name,
        script_dir=SCRIPT_DIR,
    )

    if args_cli.max_frames > 0:
        frame_count = min(frame_count, int(args_cli.max_frames))

    print(f"[replay_cmd] HDF5      : {hdf5_path}")
    print(f"[replay_cmd] Scene     : {scene_file}")
    print(f"[replay_cmd] Robot     : {robot_key}")
    print(f"[replay_cmd] Frames    : {frame_count} @ {data_fps} fps  (stride={step_stride})")
    print(f"[replay_cmd] Action dim: {commanded_all.shape[1]} ({len(action_names)} names)")
    print(f"[replay_cmd] Objects   : {list(object_initial.keys())}")

    # ── 2. Sim context ───────────────────────────────────────────────
    physics_dt = 1.0 / (data_fps * step_stride)  # match data collection (60Hz)
    sim_cfg = sim_utils.SimulationCfg(
        dt=physics_dt,
        device=args_cli.device,
        physx=sim_utils.PhysxCfg(
            enable_ccd=True,
            enable_stabilization=True,
            bounce_threshold_velocity=0.01,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**22,
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    print(f"[replay_cmd] Sim dt={physics_dt:.5f}s ({1.0/physics_dt:.1f}Hz)  device={args_cli.device}")

    task = load_yaml(scene_file)
    task_dir = os.path.dirname(scene_file)
    sim.set_camera_view([0.0, 1.0, 1.2], [0.0, 0.0, 0.8])

    # ── 3. Resolve scene generalization (optional --restore-generalization) ──
    generalization_enabled = bool(args_cli.restore_generalization)
    generalization_cfg = None
    generalization_sample = None
    if generalization_enabled:
        if recorded_sample_json is None:
            raise RuntimeError(
                "--restore-generalization: no scene_generalization_sample in source HDF5.",
            )
        # Resolve any dataset-relative paths in the saved sample
        try:
            from build.generalization import _find_dataset_root
            _dataset_root = _find_dataset_root(SCRIPT_DIR)
            recorded_sample_json = resolve_sample_paths(
                recorded_sample_json, _dataset_root,
            )
        except Exception as e:
            print(f"[replay_cmd][WARN] resolve_sample_paths failed: {e}")

        # Build a cfg: prefer HDF5-recorded cfg, fall back to disk yaml.
        if recorded_config_json is not None:
            generalization_cfg = parse_scene_generalization_config(recorded_config_json)
            gen_cfg_source = "<from HDF5>"
        else:
            gen_cfg_path = args_cli.generalization_config or os.path.join(
                SCRIPT_DIR, "configs", "scene", "generalization.yaml",
            )
            gen_raw = load_yaml(gen_cfg_path) or {}
            generalization_cfg = parse_scene_generalization_config(
                gen_raw, config_path=gen_cfg_path,
            )
            gen_cfg_source = gen_cfg_path

        # Force-disable groups that would re-randomize on top of the restored sample.
        try:
            from build.generalization import ObjectGeneralizationCfg
            generalization_cfg.spatial.object_pose = ObjectGeneralizationCfg(enabled=False)
        except Exception as e:
            print(f"[replay_cmd][WARN] Could not lock generalization cfg overrides: {e}")

        generalization_sample = dict_to_generalization_sample(recorded_sample_json)
        generalization_sample.robot_key = robot_key
        print(f"[replay_cmd] Generalization RESTORED from HDF5  (cfg source: {gen_cfg_source})")
        for line in scene_generalization_sample_debug_lines(generalization_sample):
            print(f"[replay_cmd]   {line}")
    else:
        print("[replay_cmd] Generalization DISABLED (pass --restore-generalization to match collect-time scene).")

    # ── 3b. Build scene ──────────────────────────────────────────────
    runtime = build_scene(
        task, task_dir,
        robot_key=robot_key,
        generalization_enabled=generalization_enabled,
        generalization_cfg=generalization_cfg,
        generalization_sample=generalization_sample,
    )
    interactive_objects = runtime["interactive_objects"]

    sim.reset()

    groups, controlled_articulations, _, _, articulation_hold_targets = (
        initialize_scene_runtime_state(
            sim=sim,
            physics_dt=physics_dt,
            interactive_objects=interactive_objects,
            object_prim_paths=runtime.get("object_prim_paths", {}),
            object_display_colors=runtime.get("object_display_colors", {}),
            collector=None,
            app_running_state_fn=lambda: simulation_app.is_running(),
        )
    )

    robot_art = groups.robot_articulation
    if robot_art is None:
        raise RuntimeError("No global_robot in scene — cannot replay.")
    robot_joint_names = list(robot_art.joint_names)
    n_robot_joints = len(robot_joint_names)

    # ── Gravity compensation ────────────────────────────────────────────
    gravity_comp = GravityCompensator(robot_art)

    # ── 4. Joint maps ────────────────────────────────────────────────
    qpos_map = _build_joint_map(hdf5_joint_names, robot_joint_names)
    action_map = _build_joint_map(action_names, robot_joint_names)
    print(f"[replay_cmd] qpos joints mapped : {len(qpos_map)}/{len(hdf5_joint_names)} -> {n_robot_joints}")
    print(f"[replay_cmd] action joints mapped: {len(action_map)}/{len(action_names)} -> {n_robot_joints}")

    # HOME offset removed: an older version applied `offset = current_HOME - qpos[0]`
    # to "align" the trajectory to the current USD HOME.  That is wrong because
    # HDF5 frame 0 is the measured qpos at episode start, not a HOME snapshot,
    # and the offset injects a cm-scale world-space error into every link.
    # See replay.py for the matching change.
    joint_offset = torch.zeros(n_robot_joints, device=robot_art.data.default_joint_pos.device)

    # ── 5. Load initial object + robot state from HDF5 frame 0 ───────
    # 5a. Robot qpos (with HOME offset)
    qpos_init = robot_art.data.default_joint_pos.clone()
    for h_idx, r_idx in qpos_map.items():
        qpos_init[0, r_idx] = float(qpos_all[0, h_idx]) + joint_offset[r_idx]
    qvel_init = torch.zeros_like(qpos_init)
    robot_art.write_joint_state_to_sim(qpos_init, qvel_init)

    # 5b. Object root poses (HDF5 [x,y,z, qx,qy,qz,qw] -> Isaac [x,y,z, qw,qx,qy,qz])
    for obj_id, entry in object_initial.items():
        obj = interactive_objects.get(obj_id)
        if obj is None:
            continue
        p = entry["pose0_xyzw"]
        if hasattr(obj, "write_root_pose_to_sim"):
            pose_wxyz = torch.tensor(
                [[p[0], p[1], p[2], p[6], p[3], p[4], p[5]]],
                dtype=torch.float32, device=sim.device,
            )
            obj.write_root_pose_to_sim(pose_wxyz)
            if hasattr(obj, "write_root_velocity_to_sim") and not is_kinematic_rigid_object(obj):
                obj.write_root_velocity_to_sim(
                    torch.zeros((1, 6), dtype=torch.float32, device=sim.device),
                )
        if "qpos0" in entry and hasattr(obj, "joint_names"):
            obj_map = _build_joint_map(entry["joint_names"], list(obj.joint_names))
            if obj_map:
                oq = obj.data.default_joint_pos.clone()
                for h_i, r_i in obj_map.items():
                    oq[0, r_i] = float(entry["qpos0"][h_i])
                ov = torch.zeros_like(oq)
                obj.write_joint_state_to_sim(oq, ov)

    # ── 6. Seed hold target ──────────────────────────────────────────
    # IMPORTANT: seed with the recorded MEASURED qpos[0], not commanded[0].
    # commanded[0] can be a far-away teleop spike that would cause the robot
    # to sweep across the workspace during warmup and knock objects.
    def _commanded_to_target(cmd_row: np.ndarray, fallback_target) -> torch.Tensor:
        """Map an HDF5 commanded action row into a robot-joint target tensor."""
        target = fallback_target.clone()
        for a_idx, r_idx in action_map.items():
            v = float(cmd_row[a_idx])
            if not np.isnan(v):
                target[0, r_idx] = v + joint_offset[r_idx]
        return target

    robot_target = articulation_hold_targets.get(id(robot_art))
    if robot_target is None:
        robot_target = robot_art.data.joint_pos.clone()
        articulation_hold_targets[id(robot_art)] = robot_target
    # Use measured qpos[0] (with HOME offset) as the warmup seed.
    robot_target.copy_(qpos_init)

    # Print initial object poses for diagnostics.
    print("[replay_cmd] Initial object poses (after override):")
    for obj_id in object_initial.keys():
        obj = interactive_objects.get(obj_id)
        if obj is None:
            continue
        try:
            pos = obj.data.root_pos_w[0].detach().cpu().numpy()
            recorded = object_initial[obj_id]["pose0_xyzw"][:3]
            err = np.linalg.norm(pos - recorded)
            print(f"  {obj_id}: actual={pos.round(4).tolist()}  "
                  f"recorded={np.round(recorded, 4).tolist()}  "
                  f"|d|={err*1000:.1f}mm")
        except Exception as e:
            print(f"  {obj_id}: <read failed: {e}>")

    # ── 7. Warmup: settle robot at recorded qpos[0] ──────────────────
    print(f"[replay_cmd] Warmup ({args_cli.warmup_steps} phys steps) holding measured qpos[0]...")
    for _ in range(args_cli.warmup_steps):
        gravity_comp.apply()
        write_articulation_targets(controlled_articulations, articulation_hold_targets)
        sim.step()
        update_sim_objects(interactive_objects.values(), physics_dt)
        if not simulation_app.is_running():
            return

    if args_cli.profile:
        try:
            root_pos = robot_art.data.root_pos_w[0].detach().cpu().numpy()
            root_quat = robot_art.data.root_quat_w[0].detach().cpu().numpy()
            logger.info("commanded replay robot root_pos_w=%s", root_pos.tolist())
            logger.info("commanded replay robot root_quat_w(wxyz)=%s", root_quat.tolist())
        except Exception as exc:
            logger.info("commanded replay robot root pose unavailable: %s", exc)
        try:
            body_names = list(robot_art.body_names)
            body_pos = robot_art.data.body_pos_w[0].detach().cpu().numpy()
            keywords = ("thumb", "index", "middle", "ring", "little",
                        "tip", "tcp", "tool", "wrist", "palm", "ee", "end_effector")
            for i, name in enumerate(body_names):
                if any(key in name.lower() for key in keywords) or i < 3:
                    logger.info(
                        "commanded replay link[%02d] %s world_xyz=%s",
                        i,
                        name,
                        body_pos[i].round(4).tolist(),
                    )
        except Exception as exc:
            logger.info("commanded replay body link dump failed: %s", exc)
        for obj_id, entry in object_initial.items():
            recorded = entry["pose0_xyzw"]
            obj = interactive_objects.get(obj_id)
            actual = None
            try:
                if obj is not None and hasattr(obj, "data"):
                    actual = obj.data.root_pos_w[0].detach().cpu().numpy()
            except Exception:
                actual = None
            recorded_xyz = [float(recorded[0]), float(recorded[1]), float(recorded[2])]
            if actual is not None:
                delta_mm = float(np.linalg.norm(actual - recorded[:3])) * 1000.0
                logger.info(
                    "commanded replay object %s recorded_xyz=%s actual_xyz=%s delta=%.2fmm",
                    obj_id,
                    recorded_xyz,
                    actual.round(4).tolist(),
                    delta_mm,
                )
            else:
                logger.info("commanded replay object %s recorded_xyz=%s actual=N/A", obj_id, recorded_xyz)

    # ── 8. Commanded replay loop ─────────────────────────────────────
    POLICY_STRIDE = step_stride  # phys steps per commanded frame (60Hz/20Hz=3)
    total_phys_steps = frame_count * POLICY_STRIDE
    print(
        f"[replay_cmd] Running commanded replay: "
        f"{frame_count} commanded frames x {POLICY_STRIDE} phys steps "
        f"= {total_phys_steps} phys steps  (interp={'off' if args_cli.no_interp else 'on'})"
    )

    prev_target_np: np.ndarray | None = None
    cur_target_np: np.ndarray | None = robot_target[0].detach().cpu().numpy().copy()
    substep_idx = 0
    # On the first 20Hz beat we transition from "warmup hold (=measured qpos[0])"
    # to commanded[0]; ramp this transition smoothly via interpolation too.
    prev_target_np = cur_target_np.copy()
    t0 = time.monotonic()
    next_wall = time.monotonic() if args_cli.realtime else None
    max_qpos_err = 0.0
    qpos_err_at_max_step = 0

    for step in range(total_phys_steps):
        if not simulation_app.is_running():
            break

        # 20Hz beat: pull next commanded action.
        if step % POLICY_STRIDE == 0:
            frame_idx = step // POLICY_STRIDE
            if not action_valid[frame_idx]:
                # Hold previous target if this row is NaN/invalid.
                pass
            else:
                new_target_t = _commanded_to_target(
                    commanded_all[frame_idx], robot_target,
                )
                prev_target_np = cur_target_np
                cur_target_np = new_target_t[0].detach().cpu().numpy().copy()
                substep_idx = 0

        # Write target into hold tensor (with optional 60Hz interp).
        if cur_target_np is not None:
            if (
                not args_cli.no_interp
                and prev_target_np is not None
                and prev_target_np.shape == cur_target_np.shape
            ):
                alpha = float(substep_idx + 1) / POLICY_STRIDE
                blended_np = (1.0 - alpha) * prev_target_np + alpha * cur_target_np
            else:
                blended_np = cur_target_np
            robot_target[0, :] = torch.from_numpy(blended_np).to(
                dtype=robot_target.dtype, device=robot_target.device,
            )
            substep_idx = min(substep_idx + 1, POLICY_STRIDE - 1)

        gravity_comp.apply()
        write_articulation_targets(controlled_articulations, articulation_hold_targets)
        sim.step()
        update_sim_objects(interactive_objects.values(), physics_dt)

        if next_wall is not None:
            next_wall += physics_dt
            sleep = next_wall - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)

        # Track tracking error vs recorded qpos (sampled every 60 steps = 1s).
        if step % 60 == 0:
            frame_idx = min(step // POLICY_STRIDE, frame_count - 1)
            qpos_now = robot_art.data.joint_pos[0].detach().cpu().numpy()
            recorded = np.zeros_like(qpos_now)
            for h_idx, r_idx in qpos_map.items():
                recorded[r_idx] = float(qpos_all[frame_idx, h_idx]) + float(joint_offset[r_idx])
            err = float(np.max(np.abs(qpos_now - recorded)))
            if err > max_qpos_err:
                max_qpos_err = err
                qpos_err_at_max_step = step

        # Progress every ~5s of sim time.
        if (step + 1) % (POLICY_STRIDE * 100) == 0:
            elapsed = time.monotonic() - t0
            phys_done = step + 1
            sim_t = phys_done * physics_dt
            print(
                f"[replay_cmd] phys={phys_done}/{total_phys_steps} "
                f"sim={sim_t:.1f}s  wall={elapsed:.1f}s  "
                f"max|q-q_rec|={max_qpos_err:.3f}rad",
                flush=True,
            )

    elapsed_total = time.monotonic() - t0
    print(
        f"[replay_cmd] Done in {elapsed_total:.1f}s  "
        f"(max|q-q_rec|={max_qpos_err:.3f}rad at phys step {qpos_err_at_max_step})"
    )
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass
