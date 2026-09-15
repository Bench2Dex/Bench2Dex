r"""Replay HDF5 episode through active-DOF path (same as ACT active inference).

Mirrors replay_commanded.py but inserts ``select_active`` / ``expand_to_full``
between the HDF5 commanded actions and the PD controller, exactly as
``run_policy.py`` does for active-DOF ACT inference.

Usage:
    cd .
    python tools/replay/replay_commanded_active_dof.py \
        --hdf5 dataset/44_microwave_bowl_loading/replay-generalization/episode_000003.hdf5
"""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np

try:
    import pinocchio  # noqa: F401
except ImportError:
    pass

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Replay HDF5 episode through active-DOF pipeline (select_active + expand_to_full).",
)
parser.add_argument("--hdf5", type=str, required=True, help="Path to source HDF5 episode.")
parser.add_argument("--scene", type=str, default=None, help="Override scene YAML path.")
parser.add_argument("--warmup-steps", type=int, default=60)
parser.add_argument("--realtime", action="store_true")
parser.add_argument("--no-interp", action="store_true")
parser.add_argument("--max-frames", type=int, default=0)
parser.add_argument("--restore-generalization", action="store_true")
parser.add_argument("--generalization-config", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False

print("[replay_active] Launching AppLauncher...", flush=True)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[replay_active] AppLauncher ready.", flush=True)

import h5py, torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.utils.io import load_yaml  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from build import (  # noqa: E402
    build_scene, dict_to_generalization_sample,
    parse_scene_generalization_config, resolve_sample_paths,
    scene_generalization_sample_debug_lines,
)
from utils.episode_runtime import (  # noqa: E402
    initialize_scene_runtime_state, move_controlled_articulations_home,
)
from utils.runtime_helpers import (  # noqa: E402
    is_kinematic_rigid_object, write_articulation_targets, update_sim_objects,
)
from utils.scene_paths import resolve_scene_yaml_path  # noqa: E402
from robots.active_dof_utils import (  # noqa: E402
    get_active_dof_info, get_active_dof_info_for_runtime,
    select_active, expand_to_full,
)


def _build_joint_map(hdf5_names: list[str], robot_names: list[str]) -> dict[int, int]:
    lookup = {name: idx for idx, name in enumerate(robot_names)}
    return {h: lookup[n] for h, n in enumerate(hdf5_names) if n in lookup}


def _decode_str_array(arr) -> list[str]:
    return [n.decode() if isinstance(n, bytes) else str(n) for n in arr]


def main() -> None:
    hdf5_path = os.path.abspath(args_cli.hdf5)
    if not os.path.isfile(hdf5_path):
        raise FileNotFoundError(f"HDF5 not found: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as f:
        recorded_scene_file = f["meta/scene_file"][()].decode()
        scene_name = f["meta/scene_name"][()].decode() if "meta/scene_name" in f else None
        robot_key = f["meta/robot_key"][()].decode()
        data_fps = int(f["meta/fps"][()])
        frame_count = int(f["meta/frame_count"][()])
        step_stride = int(f["meta/step_stride"][()])

        qpos_all = f["robot/qpos"][:]
        hdf5_joint_names = _decode_str_array(f["robot/joint_names"][:])

        if "action/commanded" not in f:
            raise RuntimeError("HDF5 has no 'action/commanded' dataset.")
        commanded_all = f["action/commanded"][:]
        if "action/action_names" in f:
            action_names = _decode_str_array(f["action/action_names"][:])
        else:
            action_names = list(hdf5_joint_names)
        action_valid = (
            f["action/action_valid"][:].astype(bool)
            if "action/action_valid" in f
            else np.ones(frame_count, dtype=bool)
        )

        object_initial: dict[str, dict] = {}
        for obj_id in f["objects"].keys():
            grp = f[f"objects/{obj_id}"]
            entry = {"pose0_xyzw": grp["pose_world"][0]}
            if "qpos" in grp and "joint_names" in grp:
                entry["joint_names"] = _decode_str_array(grp["joint_names"][:])
                entry["qpos0"] = grp["qpos"][0]
            object_initial[obj_id] = entry

        recorded_sample_json = None
        if "meta/scene_generalization_sample" in f:
            raw = f["meta/scene_generalization_sample"][()]
            try:
                recorded_sample_json = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
            except Exception:
                pass
        recorded_config_json = None
        if "meta/scene_generalization_config" in f:
            raw = f["meta/scene_generalization_config"][()]
            try:
                recorded_config_json = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
            except Exception:
                pass

    scene_file = resolve_scene_yaml_path(
        cli_scene_path=args_cli.scene, recorded_scene_path=recorded_scene_file,
        recorded_scene_name=scene_name, script_dir=SCRIPT_DIR,
    )
    if args_cli.max_frames > 0:
        frame_count = min(frame_count, int(args_cli.max_frames))

    # ── Active DOF setup ─────────────────────────────────────────────
    full_dof_info = get_active_dof_info(robot_key)
    print(f"[replay_active] Robot: {robot_key}  full_dof={full_dof_info.full_dof}  "
          f"active_dof={full_dof_info.active_dof}  mimic={full_dof_info.mimic_dof}")
    if full_dof_info.active_dof == full_dof_info.full_dof:
        print("[replay_active] active_dof == full_dof — active path is identity. "
              "Use replay_commanded.py instead.")

    # Build full→active index map for commanded actions
    full_to_active = {f: a for a, f in enumerate(full_dof_info.active_indices)}

    print(f"[replay_active] HDF5      : {hdf5_path}")
    print(f"[replay_active] Scene     : {scene_file}")
    print(f"[replay_active] Robot     : {robot_key}")
    print(f"[replay_active] Frames    : {frame_count} @ {data_fps}fps (stride={step_stride})")

    # ── Sim context ─────────────────────────────────────────────────
    physics_dt = 1.0 / (data_fps * step_stride)
    sim_cfg = sim_utils.SimulationCfg(
        dt=physics_dt, device=args_cli.device,
        physx=sim_utils.PhysxCfg(
            enable_ccd=True, enable_stabilization=True,
            bounce_threshold_velocity=0.01,
            gpu_max_rigid_contact_count=2**23, gpu_max_rigid_patch_count=2**22,
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    task = load_yaml(scene_file)
    task_dir = os.path.dirname(scene_file)
    sim.set_camera_view([0.0, 1.0, 1.2], [0.0, 0.0, 0.8])

    # ── Generalization ──────────────────────────────────────────────
    generalization_enabled = bool(args_cli.restore_generalization)
    generalization_cfg, generalization_sample = None, None
    if generalization_enabled:
        if recorded_sample_json is None:
            raise RuntimeError("--restore-generalization: no sample in HDF5.")
        try:
            from build.generalization import _find_dataset_root
            recorded_sample_json = resolve_sample_paths(recorded_sample_json, _find_dataset_root(SCRIPT_DIR))
        except Exception as e:
            print(f"[replay_active][WARN] resolve_sample_paths: {e}")
        if recorded_config_json is not None:
            generalization_cfg = parse_scene_generalization_config(recorded_config_json)
        else:
            gen_cfg_path = args_cli.generalization_config or os.path.join(
                SCRIPT_DIR, "configs", "scene", "generalization.yaml")
            gen_raw = load_yaml(gen_cfg_path) or {}
            generalization_cfg = parse_scene_generalization_config(gen_raw, config_path=gen_cfg_path)
        try:
            from build.generalization import ObjectGeneralizationCfg
            generalization_cfg.spatial.object_pose = ObjectGeneralizationCfg(enabled=False)
        except Exception:
            pass
        generalization_sample = dict_to_generalization_sample(recorded_sample_json)
        generalization_sample.robot_key = robot_key
    else:
        print("[replay_active] Generalization DISABLED")

    # ── Build scene ─────────────────────────────────────────────────
    runtime = build_scene(
        task, task_dir, robot_key=robot_key,
        generalization_enabled=generalization_enabled,
        generalization_cfg=generalization_cfg,
        generalization_sample=generalization_sample,
    )
    interactive_objects = runtime["interactive_objects"]
    sim.reset()

    groups, controlled_articulations, _, _, articulation_hold_targets = (
        initialize_scene_runtime_state(
            sim=sim, physics_dt=physics_dt,
            interactive_objects=interactive_objects,
            object_prim_paths=runtime.get("object_prim_paths", {}),
            object_display_colors=runtime.get("object_display_colors", {}),
            collector=None,
            app_running_state_fn=lambda: simulation_app.is_running(),
        )
    )
    robot_art = groups.robot_articulation
    if robot_art is None:
        raise RuntimeError("No robot in scene.")
    robot_joint_names = list(robot_art.joint_names)
    n_robot_joints = len(robot_joint_names)

    # Re-index active DOF to runtime joint order (same as run_policy.py)
    active_dof_info = get_active_dof_info_for_runtime(robot_key, robot_joint_names)
    print(f"[replay_active] Runtime active indices: {active_dof_info.active_indices}")

    qpos_map = _build_joint_map(hdf5_joint_names, robot_joint_names)
    action_map = _build_joint_map(action_names, robot_joint_names)
    print(f"[replay_active] qpos map: {len(qpos_map)}/{len(hdf5_joint_names)}, "
          f"action map: {len(action_map)}/{len(action_names)}")

    joint_offset = torch.zeros(n_robot_joints, device=robot_art.data.default_joint_pos.device)

    # ── Load initial state ──────────────────────────────────────────
    qpos_init = robot_art.data.default_joint_pos.clone()
    for h_idx, r_idx in qpos_map.items():
        qpos_init[0, r_idx] = float(qpos_all[0, h_idx]) + joint_offset[r_idx]
    robot_art.write_joint_state_to_sim(qpos_init, torch.zeros_like(qpos_init))

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
                obj.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=sim.device))
        if "qpos0" in entry and hasattr(obj, "joint_names"):
            obj_map = _build_joint_map(entry["joint_names"], list(obj.joint_names))
            if obj_map:
                oq = obj.data.default_joint_pos.clone()
                for h_i, r_i in obj_map.items():
                    oq[0, r_i] = float(entry["qpos0"][h_i])
                obj.write_joint_state_to_sim(oq, torch.zeros_like(oq))

    robot_target = articulation_hold_targets.get(id(robot_art))
    if robot_target is None:
        robot_target = robot_art.data.joint_pos.clone()
        articulation_hold_targets[id(robot_art)] = robot_target
    robot_target.copy_(qpos_init)

    # ── Warmup ──────────────────────────────────────────────────────
    print(f"[replay_active] Warmup ({args_cli.warmup_steps} steps)...")
    for _ in range(args_cli.warmup_steps):
        write_articulation_targets(controlled_articulations, articulation_hold_targets)
        sim.step()
        update_sim_objects(interactive_objects.values(), physics_dt)
        if not simulation_app.is_running():
            return

    # ── Active-DOF commanded replay ─────────────────────────────────
    POLICY_STRIDE = step_stride
    total_phys_steps = frame_count * POLICY_STRIDE
    print(f"[replay_active] Replay: {frame_count} frames x {POLICY_STRIDE} = {total_phys_steps} phys steps "
          f"(through active-DOF pipeline: select_active → expand_to_full)")

    prev_target_full = robot_target[0].detach().cpu().numpy().copy()
    cur_target_full = prev_target_full.copy()
    substep_idx = 0
    max_qpos_err = 0.0

    for step in range(total_phys_steps):
        if not simulation_app.is_running():
            break

        if step % POLICY_STRIDE == 0:
            frame_idx = step // POLICY_STRIDE
            if action_valid[frame_idx]:
                # Build full-DOF target from commanded actions (same as replay_commanded)
                full_target = robot_target.clone()
                for a_idx, r_idx in action_map.items():
                    v = float(commanded_all[frame_idx, a_idx])
                    if not np.isnan(v):
                        full_target[0, r_idx] = v + joint_offset[r_idx]

                # === Active-DOF path (mirrors run_policy.py inference) ===
                full_np = full_target[0].detach().cpu().numpy().astype(np.float32)
                active_np = select_active(full_np, active_dof_info)
                expanded_np = expand_to_full(active_np, active_dof_info)
                # === End active-DOF path ===

                prev_target_full = cur_target_full.copy()
                cur_target_full = expanded_np.copy()
                substep_idx = 0

        if not args_cli.no_interp and prev_target_full.shape == cur_target_full.shape:
            alpha = float(substep_idx + 1) / POLICY_STRIDE
            blended = (1.0 - alpha) * prev_target_full + alpha * cur_target_full
        else:
            blended = cur_target_full

        robot_target[0, :] = torch.from_numpy(blended).to(
            dtype=robot_target.dtype, device=robot_target.device,
        )
        substep_idx = min(substep_idx + 1, POLICY_STRIDE - 1)

        write_articulation_targets(controlled_articulations, articulation_hold_targets)
        sim.step()
        update_sim_objects(interactive_objects.values(), physics_dt)

        if step % 60 == 0:
            frame_idx = min(step // POLICY_STRIDE, frame_count - 1)
            qpos_now = robot_art.data.joint_pos[0].detach().cpu().numpy()
            recorded = np.zeros_like(qpos_now)
            for h_idx, r_idx in qpos_map.items():
                recorded[r_idx] = float(qpos_all[frame_idx, h_idx]) + float(joint_offset[r_idx])
            err = float(np.max(np.abs(qpos_now - recorded)))
            if err > max_qpos_err:
                max_qpos_err = err

        if (step + 1) % (POLICY_STRIDE * 100) == 0:
            elapsed = time.monotonic() - time.monotonic()  # roughly 0 for now
            print(f"[replay_active] phys={step+1}/{total_phys_steps}  "
                  f"max|q-q_rec|={max_qpos_err:.3f}rad", flush=True)

    print(f"[replay_active] Done. max|q-q_rec|={max_qpos_err:.3f}rad")
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass
