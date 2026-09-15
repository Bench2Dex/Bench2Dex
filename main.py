"""Build and run an Isaac Sim scene from a YAML task file."""

import argparse
import logging
import os
import sys

# ── Pre-import pinocchio before Isaac Sim ──────────────────────────────────
# pinocchio uses boost::python for C++ bindings (eigenpy).  If Isaac Sim's
# C++ runtime (pybind11-based) loads first, it corrupts the boost::python
# std::vector<> converters, causing TypeError for model.nqs / model.names.
# Importing pinocchio here registers the converters before Isaac Sim loads.
try:
    import pinocchio  # noqa: F401
except ImportError:
    pass  # pinocchio not installed — teleop won't work but main.py should still run

from isaaclab.app import AppLauncher
from utils.isaac_rendering import configure_headless_camera_parity_experience, is_env_flag_enabled
from utils.logging_config import add_logging_arguments, configure_logging


logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description="Build a scene from a YAML task file.")
parser.add_argument(
    "--task",
    type=str,
    default=None,
    help="Path to task YAML.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed namespace base. Collection defaults to 10000; unset non-collection runs are unseeded.")
parser.add_argument("--task-seed-id", type=int, default=None, help="Numeric task id for seed partitioning when the task filename has no numeric prefix.")
parser.add_argument(
    "--enable-generalization",
    action="store_true",
    help="Enable scene generalization from configs/scene/generalization.yaml.",
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
    help="Discrete visual asset split for scene generalization. Defaults to the YAML asset_split.",
)
parser.add_argument(
    "--ghost-hdf5",
    type=str,
    default=None,
    help="Path to an existing HDF5 episode for ghost replay guidance during teleop collection.",
)
parser.add_argument("--collect", action="store_true", help="Enable dataset collection.")
parser.add_argument("--teleop", action="store_true", help="Enable teleoperation.")
parser.add_argument(
    "--collect-config",
    type=str,
    default=None,
    help="Path to collector YAML config. Defaults to configs/collect/default.yaml.",
)
parser.add_argument(
    "--episode-steps",
    type=int,
    default=None,
    help="Maximum simulation steps for one episode. If omitted, runs until app is closed.",
)
parser.add_argument(
    "--output-root",
    type=str,
    default=None,
    help="Override dataset output root from collect config.",
)
AppLauncher.add_app_launcher_args(parser)
add_logging_arguments(parser)
args_cli = parser.parse_args()
configure_logging(args_cli.log_level)
enable_cameras_requested = bool(getattr(args_cli, "enable_cameras", False) or is_env_flag_enabled("ENABLE_CAMERAS"))
configure_headless_camera_parity_experience(args_cli, enable_cameras=enable_cameras_requested, log_prefix="main")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
# AppLauncher consumes some launcher-specific fields from the Namespace.
# Keep the camera request available for the collector guard below.
args_cli.enable_cameras = enable_cameras_requested

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.utils.io import load_yaml  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from build import (  # noqa: E402
    build_scene,
    collect_task_asset_codes,
    ghost_replay,
    load_scene_generalization_config,
    merge_scene_generalization_overrides,
    parse_scene_generalization_config,
    sample_scene_generalization,
    scene_generalization_sample_debug_lines,
)
from collector import DataCollector, load_collect_config, load_collect_scene_background  # noqa: E402
from utils.episode_session import EpisodeSessionController, EpisodeSessionState, EpisodeCommand  # noqa: E402
from utils.episode_runtime import (  # noqa: E402
    initialize_scene_runtime_state,
    move_controlled_articulations_home,
    check_articulations_at_home,
    summarize_articulations_home_errors,
    resample_scene_for_next_episode,
    setup_episode_keyboard_controls,
    teardown_episode_keyboard_controls,
    ramp_homing_targets,
)
from robots import DEFAULT_ROBOT_KEY  # noqa: E402
from utils.seed_policy import (  # noqa: E402
    COLLECT_BASE_SEED_DEFAULT,
    CollectionSeedAllocator,
    collection_output_dir,
    resolve_task_seed_id,
    seed_everything,
)
from utils.runtime_helpers import (  # noqa: E402
    classify_interactive_objects,
    log_runtime_exception,
    shutdown_with_diagnostics,
    update_sim_objects,
    write_articulation_targets,
)


def _resolve_hand_type(robot_key: str) -> str | None:
    """Auto-detect hand type from robot key."""
    from teleop.teleop_controller import TeleopController
    return TeleopController.auto_detect_hand_type(robot_key)


def _seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    seed_everything(int(seed))



def _safe_app_running_state() -> bool | None:
    try:
        return bool(simulation_app.is_running())
    except Exception:
        return None


def _build_fixed_background_sample(scene_background: dict | None) -> dict | None:
    config = dict(scene_background or {})
    if not bool(config.get("enabled", False)):
        return None
    uri = str(config.get("uri", "") or "")
    if not uri:
        return None
    bg_type = str(config.get("type", "hdr") or "hdr")
    yaw_deg = float(config.get("yaw_deg", 0.0) or 0.0) % 360.0
    asset_kind = "usd_scene" if bg_type == "usd_scene" else "image"
    bg = {
        "enabled": True,
        "clean_background": False,
        "asset_kind": asset_kind,
        "asset_category": "collect_config",
        "asset_uri": uri,
        "yaw_deg": yaw_deg,
        "base_yaw_deg": yaw_deg,
        "flip_180_applied": False,
        "physics_enabled": bool(config.get("physics_enabled", False)),
        "fill_light_intensity": float(config.get("fill_light_intensity", 800.0)),
        "fill_light_jitter_ratio": float(config.get("fill_light_jitter_ratio", 0.0)),
    }
    if "scene_offset" in config:
        bg["scene_offset"] = config["scene_offset"]
    return {"appearance": {"background": bg}}


def _resolve_sim_device(*, tactile_enabled: bool = False) -> str:
    if bool(getattr(args_cli, "cpu", False)):
        if tactile_enabled:
            raise ValueError(
                "TacMap tactile collection requires GPU simulation; remove --cpu or disable "
                "modalities.tactile in the collect config."
            )
        return "cpu"
    requested = str(getattr(args_cli, "device", "") or "").strip()
    if tactile_enabled and (not requested or requested.lower().startswith("cpu")):
        print("[INFO] TacMap tactile enabled; using sim_device=cuda:0")
        return "cuda:0"
    return requested or "cpu"


def _ensure_tactile_supported_device(*, sim_device: str, tactile_enabled: bool) -> None:
    if not tactile_enabled:
        return
    if str(sim_device).lower().startswith("cuda"):
        return
    raise ValueError(
        "TacMap tactile collection requires GPU simulation because ray casting is unavailable on "
        f"CpuSimulationView (current device: {sim_device}). Rerun with --device cuda:0 or disable "
        "modalities.tactile in the collect config."
    )


def _smoothstep01(value: float) -> float:
    value = max(0.0, min(1.0, float(value)))
    return value * value * (3.0 - 2.0 * value)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        print(f"[WARN] Ignoring invalid integer env {name}={raw!r}; using {default}.")
        return int(default)


def _collector_has_camera_modalities(collector: DataCollector | None) -> bool:
    if collector is None:
        return False
    cfg = getattr(collector, "_cfg", None)
    if cfg is None:
        return False
    return bool(cfg.enabled("rgb") or cfg.enabled("depth"))


def _sync_collector_mounted_camera_poses(collector: DataCollector | None) -> None:
    if collector is None:
        return
    sync = getattr(collector, "sync_mounted_camera_poses", None)
    if callable(sync):
        sync()


def _warmup_scene_before_camera_recording(
    *,
    collector: DataCollector | None,
    sim,
    physics_dt: float,
    controlled_articulations: list[object],
    articulation_hold_targets: dict[int, object],
    interactive_objects: dict[str, object],
    pre_step_hooks: list[object],
    reason: str,
) -> None:
    """Advance unrecorded renders/steps so reset/teleport camera poses settle."""
    if not _collector_has_camera_modalities(collector):
        return

    render_steps = max(0, _env_int("DEX2BENCH_CAMERA_RENDER_WARMUP_STEPS", 5))
    if render_steps > 0:
        print(f"[INFO] Camera render warmup: {render_steps} render(s) after {reason}.")
        for _ in range(render_steps):
            _sync_collector_mounted_camera_poses(collector)
            sim.render()

    steps = max(1, _env_int("DEX2BENCH_CAMERA_RESET_WARMUP_STEPS", 1))
    print(f"[INFO] Camera recording warmup: {steps} unrecorded step(s) after {reason}.")
    for _ in range(steps):
        for hook in pre_step_hooks:
            hook()
        write_articulation_targets(controlled_articulations, articulation_hold_targets)
        sim.step(render=False)
        update_sim_objects(interactive_objects.values(), physics_dt)
        _sync_collector_mounted_camera_poses(collector)
        sim.render()

    camera_rig = getattr(collector, "_camera_rig", None)
    if camera_rig is not None:
        attempts = max(1, _env_int("DEX2BENCH_CAMERA_PREFLIGHT_ATTEMPTS", 2))
        for attempt in range(attempts):
            _sync_collector_mounted_camera_poses(collector)
            sim.render()
            frames = camera_rig.capture(dt=physics_dt)
            missing = [
                camera_id for camera_id in camera_rig.camera_ids
                if frames.get(camera_id) is None or getattr(frames[camera_id], "rgb", None) is None
            ]
            if missing:
                print(f"[WARN] Camera preflight missing RGB before recording: {missing}")
            if attempt + 1 < attempts:
                _sync_collector_mounted_camera_poses(collector)
                sim.render()


def _apply_scripted_demo_targets(
    *,
    sim_step: int,
    physics_dt: float,
    controlled_articulations: list[object],
    articulation_hold_targets: dict[int, object],
) -> None:
    """Optional one-off scripted motion for local camera debugging."""
    mode = os.environ.get("DEX2BENCH_SCRIPTED_DEMO", "").strip().lower()
    if mode not in {
        "beaker_bimanual_grasp",
        "wrist_camera_check",
        "side_object_grasp",
        "side_object_hover",
        "1",
        "true",
    }:
        return

    t = float(sim_step) * float(physics_dt)
    arm_alpha = _smoothstep01((t - 0.8) / 2.8)
    close_alpha = _smoothstep01((t - 3.2) / 1.2)
    lift_alpha = _smoothstep01((t - 5.2) / 1.2)

    if mode == "side_object_hover":
        arm_alpha = _smoothstep01((t - 1.0) / 6.0)
        close_alpha = 0.0
        approach_goal = {
            "shoulder_pan_joint": -0.1148,
            "shoulder_lift_joint": -1.2808,
            "elbow_joint": 1.4096,
            "wrist_1_joint": -0.1135,
            "wrist_2_joint": 1.3489,
            "wrist_3_joint": 0.0,
            "L_arm_shoulder_pan_joint": -0.1315,
            "L_arm_shoulder_lift_joint": 1.1104,
            "L_arm_elbow_joint": -1.2422,
            "L_arm_wrist_1_joint": 0.2014,
            "L_arm_wrist_2_joint": -1.3352,
            "L_arm_wrist_3_joint": 0.0,
        }
        lift_delta = {}
    elif mode == "side_object_grasp":
        arm_alpha = _smoothstep01((t - 0.8) / 3.2)
        close_alpha = _smoothstep01((t - 4.0) / 1.0)
        approach_goal = {
            "shoulder_pan_joint": -0.2518,
            "shoulder_lift_joint": -0.9826,
            "elbow_joint": 1.6232,
            "wrist_1_joint": -0.0786,
            "wrist_2_joint": 1.3697,
            "wrist_3_joint": 0.0,
            "L_arm_shoulder_pan_joint": -0.0458,
            "L_arm_shoulder_lift_joint": 0.9194,
            "L_arm_elbow_joint": -1.5197,
            "L_arm_wrist_1_joint": 0.1193,
            "L_arm_wrist_2_joint": -1.3441,
            "L_arm_wrist_3_joint": 0.0,
        }
        lift_delta = {}
    elif mode == "wrist_camera_check":
        approach_goal = {
            "shoulder_pan_joint": -1.15,
            "shoulder_lift_joint": -1.18,
            "elbow_joint": 1.38,
            "wrist_1_joint": -0.48,
            "wrist_2_joint": 1.20,
            "wrist_3_joint": -0.40,
            "L_arm_shoulder_pan_joint": 1.15,
            "L_arm_shoulder_lift_joint": 1.18,
            "L_arm_elbow_joint": -1.38,
            "L_arm_wrist_1_joint": 0.48,
            "L_arm_wrist_2_joint": -1.20,
            "L_arm_wrist_3_joint": 0.40,
        }
        lift_delta = {}
        close_alpha = 0.0
    else:
        approach_goal = {
            "shoulder_pan_joint": 1.18,
            "shoulder_lift_joint": -1.34,
            "elbow_joint": 1.96,
            "wrist_1_joint": -0.62,
            "wrist_2_joint": 1.40,
            "wrist_3_joint": -0.20,
            "L_arm_shoulder_pan_joint": -1.18,
            "L_arm_shoulder_lift_joint": 1.34,
            "L_arm_elbow_joint": -1.96,
            "L_arm_wrist_1_joint": 0.62,
            "L_arm_wrist_2_joint": -1.40,
            "L_arm_wrist_3_joint": 0.20,
        }
        lift_delta = {
            "shoulder_lift_joint": 0.10,
            "elbow_joint": -0.16,
            "wrist_1_joint": 0.08,
            "L_arm_shoulder_lift_joint": -0.10,
            "L_arm_elbow_joint": 0.16,
            "L_arm_wrist_1_joint": -0.08,
        }

    close_joint_tokens = (
        "thumb_1_joint",
        "thumb_2_joint",
        "thumb_3_joint",
        "thumb_4_joint",
        "index_1_joint",
        "index_2_joint",
        "middle_1_joint",
        "middle_2_joint",
        "ring_1_joint",
        "ring_2_joint",
        "little_1_joint",
        "little_2_joint",
    )

    for articulation in controlled_articulations:
        home = articulation.data.default_joint_pos
        target = home.clone()
        joint_names = list(getattr(articulation.data, "joint_names", []))
        name_to_index = {name: idx for idx, name in enumerate(joint_names)}

        for joint_name, goal in approach_goal.items():
            idx = name_to_index.get(joint_name)
            if idx is None:
                continue
            home_value = float(home[0, idx].item())
            value = home_value + arm_alpha * (float(goal) - home_value)
            value += lift_alpha * float(lift_delta.get(joint_name, 0.0))
            target[0, idx] = value

        for idx, joint_name in enumerate(joint_names):
            if not any(token in joint_name for token in close_joint_tokens):
                continue
            # RH56DFX fingers use 0 as open; positive targets close the fingers.
            target[0, idx] = float(home[0, idx].item()) + close_alpha * 0.85

        articulation_hold_targets[id(articulation)] = target

    if sim_step == 0:
        print(f"[INFO] Scripted demo active: {mode}")

def main() -> None:
    collector: DataCollector | None = None
    teleop_bridge = None
    keyboard_controls = None
    task_path = os.path.abspath(args_cli.task) if args_cli.task else None
    scene_ready = False
    sim_step = 0
    primary_error: BaseException | None = None

    try:
        if task_path is None:
            raise ValueError("--task is required.")
        task_dir = os.path.dirname(task_path)

        task = load_yaml(task_path)
        robot_key = DEFAULT_ROBOT_KEY
        default_collect_cfg = os.path.join(SCRIPT_DIR, "configs", "collect", "default.yaml")
        collect_cfg_path = args_cli.collect_config or default_collect_cfg
        fixed_background_sample = _build_fixed_background_sample(
            load_collect_scene_background(collect_cfg_path)
        )
        fixed_background = (
            fixed_background_sample.get("appearance", {}).get("background", {})
            if fixed_background_sample
            else {}
        )
        if fixed_background.get("enabled", False):
            print(
                "[INFO] Scene fixed background enabled: "
                f"{fixed_background.get('asset_uri') or '<none>'}"
            )
        default_generalization_cfg = os.path.join(SCRIPT_DIR, "configs", "scene", "generalization.yaml")
        generalization_cfg_path = args_cli.generalization_config or default_generalization_cfg
        generalization_raw = load_yaml(generalization_cfg_path) or {}
        overrides = task.get("generalization_overrides") if isinstance(task, dict) else None
        merged_generalization_raw = merge_scene_generalization_overrides(generalization_raw, overrides) if overrides else dict(generalization_raw)
        generalization_cfg = parse_scene_generalization_config(merged_generalization_raw, config_path=generalization_cfg_path)
        generalization_enabled = bool(args_cli.enable_generalization)
        collect_cfg = None
        tactile_enabled = False
        collection_seed_allocator = None
        current_seed_context = None
        if args_cli.collect:
            collect_cfg = load_collect_config(
                collect_cfg_path,
                output_root_override=args_cli.output_root,
                robot_key=robot_key,
            )
            tactile_enabled = collect_cfg.enabled("tactile")
            collection_base_seed = int(args_cli.seed) if args_cli.seed is not None else COLLECT_BASE_SEED_DEFAULT
            task_seed_id = resolve_task_seed_id(task_path, args_cli.task_seed_id)
            output_dir = collection_output_dir(collect_cfg.dataset_root, collect_cfg.dataset_name, task_path)
            collection_seed_allocator = CollectionSeedAllocator(
                base_seed=collection_base_seed,
                task_seed_id=task_seed_id,
                output_dir=output_dir,
            )
            current_seed_context = collection_seed_allocator.next()
            _seed_everything(current_seed_context.episode_seed)
            print(
                "[INFO] Collection seed context: "
                f"namespace={current_seed_context.seed_namespace} "
                f"task_seed_id={current_seed_context.task_seed_id} "
                f"task_base_seed={current_seed_context.task_base_seed} "
                f"episode_index={current_seed_context.episode_index} "
                f"episode_seed={current_seed_context.episode_seed} "
                f"existing_files={collection_seed_allocator.existing_file_count} "
                f"legacy_files={collection_seed_allocator.legacy_file_count}"
            )
        else:
            _seed_everything(args_cli.seed)
        scene_generalization_sample = sample_scene_generalization(
            generalization_cfg,
            enabled=generalization_enabled,
            task_asset_codes=collect_task_asset_codes(task),
            asset_split=args_cli.generalization_split,
        )
        if generalization_enabled:
            print(f"[INFO] Scene generalization enabled: {generalization_cfg_path}")
            print(f"[INFO] Scene generalization robot_key: {robot_key}")
            for line in scene_generalization_sample_debug_lines(scene_generalization_sample):
                print(line)

        sim_device = _resolve_sim_device(tactile_enabled=tactile_enabled)
        _ensure_tactile_supported_device(sim_device=sim_device, tactile_enabled=tactile_enabled)
        print(f"[INFO] sim_device={sim_device}")
        sim_cfg = sim_utils.SimulationCfg(
            dt=0.0166666,
            device=sim_device,
            physx=sim_utils.PhysxCfg(
                enable_ccd=True,
                enable_stabilization=True,
                bounce_threshold_velocity=0.01,
                gpu_max_rigid_contact_count=2**23,
                gpu_max_rigid_patch_count=2**22,
            ),
        )
        sim = sim_utils.SimulationContext(sim_cfg)
        physics_dt = sim.get_physics_dt()

        # Viewport 使用 cam_overhead 视角: 正上方俯视
        # sim.set_camera_view([0.0, -1.5, 2.0], [0.0, 0.0, 0.82])
        sim.set_camera_view([0.0, 0.0, 2.20], [0.0, 0.0, 0.82])

        runtime = build_scene(
            task,
            task_dir,
            generalization_enabled=bool(args_cli.enable_generalization),
            generalization_cfg=generalization_cfg,
            generalization_sample=scene_generalization_sample,
            fixed_background_sample=fixed_background_sample,
            robot_key=robot_key,
        )
        # Add success conditions and task instruction to runtime for collector
        runtime["success_conditions"] = task.get("success_conditions", [])
        runtime["task_instruction"] = task.get("description", None)

        # ── Ghost replay setup ──────────────────────────────────────────
        ghost_state: dict = {"enabled": False}  # mutable dict for in-loop updates
        if args_cli.ghost_hdf5 and os.path.exists(args_cli.ghost_hdf5):
            print(f"[Ghost] Loading {args_cli.ghost_hdf5}")
            ghost_data = ghost_replay.GhostReplayData(args_cli.ghost_hdf5)
            scene_obj_ids = list(runtime.get("interactive_objects", {}).keys())
            ghost_obj_ids = ghost_replay.resolve_ghost_object_ids(ghost_data, scene_obj_ids)
            obj_asset_paths = runtime.get("object_asset_paths", {})
            # Collect per-object scales from task YAML assets
            obj_scales = {}
            for obj in task.get("objects", []):
                oid = obj.get("id", "")
                if not oid:
                    continue
                asset_key = obj.get("asset", "")
                asset_spec = task.get("assets", {}).get(asset_key, {})
                sc = asset_spec.get("scale", [1.0, 1.0, 1.0])
                obj_scales[oid] = tuple(float(v) for v in sc)
            ghost_map = ghost_replay.spawn_ghost_objects(ghost_obj_ids, obj_asset_paths, obj_scales)
            # z-offset = current table height - recorded table height, so ghosts
            # sit on the current table instead of the (possibly different) one
            # baked into the replay HDF5.
            cur_table_off = ghost_replay.current_table_offset_m(scene_generalization_sample)
            ghost_z_offset = cur_table_off - ghost_data.table_height_offset_m
            ghost_state.update(
                enabled=True,
                data=ghost_data,
                map=ghost_map,
                start_sim_step=None,
                z_offset=ghost_z_offset,
            )
            print(
                f"[Ghost] Ready — {len(ghost_map)} ghost objects "
                f"(table z-offset {ghost_z_offset:+.3f}m: "
                f"cur {cur_table_off:+.3f} - rec {ghost_data.table_height_offset_m:+.3f})"
            )
        runtime["metrics"] = task.get("metrics", {})
        interactive_objects = runtime.get("interactive_objects", {})
        robot_runtime = runtime.get("robot_runtime", {})
        pre_step_hooks = list(robot_runtime.get("pre_step_hooks", []))

        if args_cli.collect:
            assert collect_cfg is not None
            needs_cameras = collect_cfg.enabled("rgb") or collect_cfg.enabled("depth")
            if needs_cameras and not getattr(args_cli, "enable_cameras", False):
                raise RuntimeError("RGB/Depth collection requires --enable_cameras.")
            collector = DataCollector(
                sim=sim,
                collect_cfg=collect_cfg,
                task_path=task_path,
                runtime=runtime,
                interactive_objects=interactive_objects,
                seed=getattr(current_seed_context, "episode_seed", args_cli.seed),
                base_seed=getattr(current_seed_context, "base_seed", None),
                task_seed_id=getattr(current_seed_context, "task_seed_id", None),
                task_base_seed=getattr(current_seed_context, "task_base_seed", None),
                seed_policy=getattr(current_seed_context, "seed_policy", None),
                seed_namespace=getattr(current_seed_context, "seed_namespace", None),
                episode_step_budget=args_cli.episode_steps,
            )
            if current_seed_context is not None:
                collector.set_episode_seed_context(current_seed_context)

        groups = classify_interactive_objects(interactive_objects)
        controlled_articulations = groups.controlled_articulations
        task_articulations = groups.task_articulations
        non_articulation_objects = groups.other_objects

        logger.debug("global_robot present: %s", "global_robot" in interactive_objects)
        logger.debug("robot classified: %s", groups.robot_articulation is not None)
        logger.debug("controlled articulations: %d", len(controlled_articulations))

        groups, controlled_articulations, task_articulations, non_articulation_objects, articulation_hold_targets = (
            initialize_scene_runtime_state(
                sim=sim,
                physics_dt=physics_dt,
                interactive_objects=interactive_objects,
                object_prim_paths=runtime.get("object_prim_paths", {}),
                object_display_colors=runtime.get("object_display_colors", {}),
                collector=collector,
                app_running_state_fn=_safe_app_running_state,
            )
        )

        # Create ContactSensorReader after sim.reset() if contact_pairs configured
        if collector is not None:
            from collector.contact_sensor_reader import install_contact_reader_for_runtime
            install_contact_reader_for_runtime(collector, runtime)

        # ── 禁用 Isaac Sim 内置实时节流, 改用外部时间对齐 ──────────
        try:
            import omni.kit.loop._loop as _omni_loop
            _loop_runner = _omni_loop.acquire_loop_interface()
            _loop_runner.set_manual_mode(False)
            print("[INFO] Disabled Isaac Sim loop runner manual mode (real-time throttle)")
        except Exception as _e:
            print(f"[WARN] Could not disable loop runner manual mode: {_e}")
        sim.set_setting("/app/runLoops/main/rateLimitEnabled", False)
        print("[INFO] Disabled carb rateLimitEnabled")
        # ── End 禁用节流 ─────────────────────────────────────────────

        # ── Teleop setup ──────────────────────────────────────────────
        teleop_bridge = None
        if args_cli.teleop:
            from teleop import TeleopController, TeleopIsaacLabBridge

            hand_type = _resolve_hand_type(DEFAULT_ROBOT_KEY)
            if hand_type is None:
                print(f"[WARN] Could not auto-detect hand type from '{DEFAULT_ROBOT_KEY}'. "
                      f"Teleop disabled.")
            else:
                robot_art = interactive_objects.get("global_robot")
                if robot_art is None:
                    print("[WARN] No global_robot found. Teleop disabled.")
                else:
                    ctrl = TeleopController(
                        hand_type=hand_type,
                        enable_right=True,
                        enable_left=True,
                    )
                    teleop_bridge = TeleopIsaacLabBridge(
                        ctrl, robot_art,
                        enable_arm_teleop=True,
                    )
                    if teleop_bridge.start(wait_timeout=3.0):
                        print(f"[INFO] Teleop active: {teleop_bridge}")
                    else:
                        print("[WARN] Teleop failed to start (SHM not available). "
                              "Running without teleop.")
                        teleop_bridge.stop()
                        teleop_bridge = None
        # ── End teleop setup ──────────────────────────────────────────

        episode_session = None
        _suppress_teleop = False
        _home_check_threshold_rad = 0.08
        # 回 home 时每帧最大关节变化速率 (rad/s)，设为约 1.0 使其以缓速回 home
        _HOME_RAMP_RATE = 1.5  # rad/s (比 velocity_limit_sim 稍慢，避免突然全速冲)
        _homing_start_sim_step: int | None = None  # 录制中开始 homing 的仿真步记录
        _homing_start_wall_time: float | None = None  # homing 开始的系统时间（超时检测用）
        _HOMING_TIMEOUT_S = 10.0  # homing 超时秒数，超时后瞬移到 home
        keyboard_episode_mode = bool(args_cli.collect) and not _env_truthy("DEX2BENCH_AUTO_RECORD")
        if keyboard_episode_mode:
            keyboard_controls = setup_episode_keyboard_controls()
            if keyboard_controls is None:
                print("[WARN] Falling back to immediate recording because keyboard controls are unavailable.")
                keyboard_episode_mode = False
            else:
                episode_session = EpisodeSessionController()
                print("[INFO] Episode keyboard controls: LEFT=home, DOWN=start recording (home first), RIGHT=stop recording (home first).")
                move_controlled_articulations_home(
                    controlled_articulations,
                    articulation_hold_targets,
                    teleop_bridge=teleop_bridge,
                    teleport=True,
                )

        auto_record_episode_target = 1
        auto_record_episode_index = 1
        auto_record_episode_start_step = 0
        auto_record_episode_step_budget = args_cli.episode_steps
        if collector is not None and not keyboard_episode_mode and _env_truthy("DEX2BENCH_AUTO_RECORD"):
            auto_record_episode_target = max(1, _env_int("DEX2BENCH_AUTO_RECORD_EPISODES", 1))
            if auto_record_episode_target > 1:
                if auto_record_episode_step_budget is None:
                    print(
                        "[WARN] DEX2BENCH_AUTO_RECORD_EPISODES requires --episode-steps; "
                        "falling back to one auto-recorded episode."
                    )
                    auto_record_episode_target = 1
                else:
                    print(
                        "[INFO] Auto-record multi-episode mode: "
                        f"episodes={auto_record_episode_target}, "
                        f"steps_per_episode={auto_record_episode_step_budget}"
                    )

        if collector is not None and not keyboard_episode_mode:
            _warmup_scene_before_camera_recording(
                collector=collector,
                sim=sim,
                physics_dt=physics_dt,
                controlled_articulations=controlled_articulations,
                articulation_hold_targets=articulation_hold_targets,
                interactive_objects=interactive_objects,
                pre_step_hooks=pre_step_hooks,
                reason="sim.reset()",
            )
            collector.start_episode(sim_step=0)
            auto_record_episode_start_step = 0
            if ghost_state["enabled"] and ghost_state["start_sim_step"] is None:
                ghost_state["start_sim_step"] = 0

        scene_ready = True
        print(f"[INFO] Scene ready: {task_path}")
        app_running_before_loop = _safe_app_running_state()
        logger.debug("simulation_app.is_running() before loop: %s", app_running_before_loop)
        if app_running_before_loop is False:
            print("[WARN] simulation_app.is_running() is False before the step loop; application may shut down immediately.")

        # ── Timing instrumentation ──────────────────────────────────
        import time as _time
        _wall_start = _time.monotonic()
        _timing_interval = 1000  # 每 N 步打印一次
        _last_timing_step = 0
        _last_timing_wall = _wall_start
        _teleop_sum_ms = 0.0
        _phys_sum_ms = 0.0
        _write_sum_ms = 0.0
        _sleep_sum_ms = 0.0
        _target_step_s = physics_dt   # 外部时间对齐目标 (e.g. 0.01s for 100Hz)
        # ── End timing setup ────────────────────────────────────────

        while simulation_app.is_running():
            _step_wall_t0 = _time.monotonic()

            if episode_session is not None and keyboard_controls is not None:
                _input_iface, _keyboard, _subscription, pending_commands, _global_listener = keyboard_controls

                while pending_commands:
                    command = pending_commands.popleft()
                    action = episode_session.handle(command)
                    if action.message:
                        _msg_map = {
                            "already_homing": "[INFO] Ignoring key — robot is already going home.",
                            "already_recording": "[INFO] Ignoring DOWN because an episode is already recording.",
                            "discard_recording": "[INFO] Discarding current episode (LEFT during recording).",
                        }
                        print(_msg_map.get(action.message, f"[INFO] {action.message}"))

                    if action.discard_recording and collector is not None:
                        collector.discard_episode()
                        print("[INFO] Episode discarded.")

                    if action.go_home:
                        move_controlled_articulations_home(
                            controlled_articulations,
                            articulation_hold_targets,
                            teleop_bridge=teleop_bridge,
                        )
                        # 记录 homing 开始时间（超时用）
                        _homing_start_wall_time = _time.monotonic()
                        # 如果正在录制，记录本帧 sim_step 供后续 HDF5 写入
                        if collector is not None and collector._episode_active:
                            _homing_start_sim_step = sim_step
                            if hasattr(collector, 'set_homing_start_sim_step'):
                                collector.set_homing_start_sim_step(sim_step)
                        print(f"[INFO] Going home... (state={episode_session.state.value}, pending={episode_session.pending_after_home.value})")

                    if action.suppress_teleop:
                        _suppress_teleop = True

                # Check if robot has reached home while homing
                if _suppress_teleop and episode_session.state is EpisodeSessionState.HOMING:
                    # 每帧将 hold target 向 home 渐进 (限速约 1 rad/s ≈ velocity_limit 的一半)
                    ramp_homing_targets(controlled_articulations, articulation_hold_targets,
                                        ramp_rate=_HOME_RAMP_RATE, dt=physics_dt)
                    home_reached = check_articulations_at_home(
                        controlled_articulations,
                        threshold_rad=_home_check_threshold_rad,
                    )
                    # Homing timeout: 超过 _HOMING_TIMEOUT_S 秒未回到 home 则直接瞬移
                    if not home_reached and _homing_start_wall_time is not None:
                        _homing_elapsed = _time.monotonic() - _homing_start_wall_time
                        if _homing_elapsed > _HOMING_TIMEOUT_S:
                            print(f"[WARN] Homing timeout ({_homing_elapsed:.1f}s > {_HOMING_TIMEOUT_S}s), teleporting to home.")
                            move_controlled_articulations_home(
                                controlled_articulations,
                                articulation_hold_targets,
                                teleport=True,
                                teleop_bridge=teleop_bridge,
                            )
                            home_reached = True
                    if not home_reached and sim_step % 30 == 0:
                        for line in summarize_articulations_home_errors(
                            controlled_articulations,
                            threshold_rad=_home_check_threshold_rad,
                        ):
                            print(line)
                    if home_reached:
                        _homing_start_wall_time = None  # 重置超时计时器
                        _suppress_teleop = False
                        # Reset teleop anchor so next ARKit frame becomes new reference
                        if teleop_bridge is not None:
                            reset_fn = getattr(teleop_bridge, "reset_input_state", None)
                            if callable(reset_fn):
                                reset_fn()
                        arrived_action = episode_session.handle(EpisodeCommand.HOME_REACHED)
                        print(f"[INFO] Home reached (state={episode_session.state.value})")

                        if arrived_action.stop_recording:
                            # Save episode if one was being recorded
                            if collector is not None and collector._episode_active:
                                episode_path = collector.finish_episode()
                                if episode_path is not None:
                                    print(f"[INFO] Episode finished: {episode_path}")
                            # Always resample scene (with or without active recording)
                            rebuild_result = resample_scene_for_next_episode(
                                sim=sim,
                                physics_dt=physics_dt,
                                task=task,
                                task_dir=task_dir,
                                generalization_enabled=generalization_enabled,
                                generalization_cfg=generalization_cfg,
                                fixed_background_sample=fixed_background_sample,
                                generalization_asset_split=args_cli.generalization_split,
                                seed_context=collection_seed_allocator.next() if collection_seed_allocator is not None else None,
                                current_robot_key=robot_key,
                                current_robot_runtime=robot_runtime,
                                collector=collector,
                                collect_cfg=collect_cfg if args_cli.collect else None,
                                app_running_state_fn=_safe_app_running_state,
                            )
                            if rebuild_result is not None:
                                runtime = rebuild_result["runtime"]
                                interactive_objects = rebuild_result["interactive_objects"]
                                groups = rebuild_result["groups"]
                                controlled_articulations = rebuild_result["controlled_articulations"]
                                task_articulations = rebuild_result["task_articulations"]
                                non_articulation_objects = rebuild_result["non_articulation_objects"]
                                articulation_hold_targets = rebuild_result["articulation_hold_targets"]
                                scene_generalization_sample = rebuild_result["scene_generalization_sample"]
                                robot_runtime = rebuild_result["robot_runtime"]
                                pre_step_hooks = list(robot_runtime.get("pre_step_hooks", []))
                                move_controlled_articulations_home(
                                    controlled_articulations,
                                    articulation_hold_targets,
                                    teleport=True,
                                )
                                # 机器人重新 spawn 后更新 teleop_bridge 的 articulation 引用
                                if teleop_bridge is not None:
                                    new_robot_art = interactive_objects.get("global_robot")
                                    if new_robot_art is not None:
                                        reset_fn = getattr(teleop_bridge, "reset_articulation", None)
                                        if callable(reset_fn):
                                            reset_fn(new_robot_art)
                            print("[INFO] Scene resampled.")
                            # Re-spawn ghost objects for new scene
                            if ghost_state["enabled"]:
                                scene_obj_ids = list(runtime.get("interactive_objects", {}).keys())
                                ghost_obj_ids = ghost_replay.resolve_ghost_object_ids(
                                    ghost_state["data"], scene_obj_ids,
                                )
                                obj_asset_paths = runtime.get("object_asset_paths", {})
                                ghost_state["map"] = ghost_replay.spawn_ghost_objects(
                                    ghost_obj_ids, obj_asset_paths, obj_scales,
                                )
                                ghost_state["start_sim_step"] = None
                                # Recompute z-offset against the resampled table height.
                                cur_table_off = ghost_replay.current_table_offset_m(
                                    scene_generalization_sample,
                                )
                                ghost_state["z_offset"] = (
                                    cur_table_off - ghost_state["data"].table_height_offset_m
                                )

                        if arrived_action.start_recording:
                            if collector is None:
                                print("[WARN] Cannot start recording — collector is disabled.")
                            else:
                                # Snap to exact HOME before the first recorded frame.
                                # The PD-ramp homing only reaches the configured threshold
                                # (a few mrad away from HOME).  If recording starts there,
                                # qpos[0] != HOME and any downstream code that assumes
                                # "frame 0 is HOME" (e.g. an older kinematic replay HOME
                                # offset) is wrong by exactly that few-mrad residual,
                                # which converts to ~1cm world-space link error.
                                # Only applied for the "press 2 → start recording" path;
                                # keys 1/3 do not start recording so they skip this.
                                move_controlled_articulations_home(
                                    controlled_articulations,
                                    articulation_hold_targets,
                                    teleport=True,
                                    teleop_bridge=teleop_bridge,
                                )
                                _warmup_scene_before_camera_recording(
                                    collector=collector,
                                    sim=sim,
                                    physics_dt=physics_dt,
                                    controlled_articulations=controlled_articulations,
                                    articulation_hold_targets=articulation_hold_targets,
                                    interactive_objects=interactive_objects,
                                    pre_step_hooks=pre_step_hooks,
                                    reason="HOME teleport",
                                )
                                collector.start_episode(sim_step=sim_step)
                                if ghost_state["enabled"] and ghost_state["start_sim_step"] is None:
                                    ghost_state["start_sim_step"] = sim_step
                                print("[INFO] Snapped to exact HOME and started recording.")
            # ── End episode keyboard/pedal controls ───────────────────────

            # Update hand targets from teleop if active (skip while going home).
            # Throttle to 20 Hz (every 3 physics steps at 60 Hz) to match ACT
            # training distribution and save ~2/3 of retarget CPU. PD targets
            # below still refresh every physics step using the held value.
            _t_teleop = _time.monotonic()
            TELEOP_STRIDE = 3  # 60 Hz physics / 3 = 20 Hz teleop, aligns with ACT
            if (
                teleop_bridge is not None
                and not _suppress_teleop
                and (sim_step % TELEOP_STRIDE == 0)
            ):
                teleop_bridge.update(articulation_hold_targets)
            _t_teleop_ms = (_time.monotonic() - _t_teleop) * 1000
            _teleop_sum_ms += _t_teleop_ms

            _apply_scripted_demo_targets(
                sim_step=sim_step,
                physics_dt=physics_dt,
                controlled_articulations=controlled_articulations,
                articulation_hold_targets=articulation_hold_targets,
            )

            # Capture (obs_t, action_t) — Convention A: action_t will be executed next
            if collector is not None:
                _t_before = _time.monotonic()
                collector.before_step(
                    sim_step=sim_step,
                    hold_targets=articulation_hold_targets,
                    controlled_articulations=controlled_articulations,
                    action_source="teleop" if teleop_bridge is not None else "scripted",
                )
                _t_before_step_ms = (_time.monotonic() - _t_before) * 1000

            # Per-step robot hooks (e.g. gravity + Coriolis feed-forward effort
            # for arms whose ImplicitActuator PD would otherwise leave a
            # gravity-induced steady-state error). Set effort buffers BEFORE
            # write_articulation_targets so the next write_data_to_sim call
            # flushes both position targets and the feed-forward effort.
            for _hook in pre_step_hooks:
                _hook()

            # Write action to robot controller
            _t_write = _time.monotonic()
            write_articulation_targets(controlled_articulations, articulation_hold_targets)
            _t_write_ms = (_time.monotonic() - _t_write) * 1000
            _write_sum_ms += _t_write_ms

            # Physics step
            _t_phys = _time.monotonic()
            sim.step(render=False)
            _t_phys_ms = (_time.monotonic() - _t_phys) * 1000
            _phys_sum_ms += _t_phys_ms

            # Refresh object states
            update_sim_objects(interactive_objects.values(), physics_dt)

            # ── Ghost replay update ────────────────────────────────────
            if ghost_state["enabled"] and ghost_state["start_sim_step"] is not None:
                ghost_time = (sim_step - ghost_state["start_sim_step"]) * physics_dt
                ghost_frame = int(ghost_time / ghost_state["data"].frame_dt)
                ghost_replay.update_ghost_objects(
                    ghost_state["map"], ghost_state["data"], ghost_frame, sim_step,
                    z_offset=ghost_state.get("z_offset", 0.0),
                )

            _sync_collector_mounted_camera_poses(collector)
            sim.render()

            # Capture observation AFTER step (includes success tracking)
            if collector is not None:
                _t_after = _time.monotonic()
                collector.after_step(sim_step=sim_step, dt=physics_dt)
                _t_after_step_ms = (_time.monotonic() - _t_after) * 1000

            # ── 外部时间对齐: 如果本步耗时 < physics_dt, sleep 补齐 ──
            _elapsed_s = _time.monotonic() - _step_wall_t0
            _remaining_s = _target_step_s - _elapsed_s
            if _remaining_s > 0.0005:  # 只在剩余 > 0.5ms 时才 sleep (避免精度损失)
                _time.sleep(_remaining_s)
            _t_sleep_ms = max(0.0, _remaining_s * 1000)
            _sleep_sum_ms += _t_sleep_ms
            # ── End 外部时间对齐 ────────────────────────────────────

            _step_wall_ms = (_time.monotonic() - _step_wall_t0) * 1000

            sim_step += 1

            # ── Periodic timing report ──────────────────────────────
            if args_cli.profile and sim_step % _timing_interval == 0:
                _now = _time.monotonic()
                _wall_elapsed = _now - _wall_start
                _sim_elapsed = sim_step * physics_dt
                _interval_wall = _now - _last_timing_wall
                _interval_steps = sim_step - _last_timing_step
                _interval_sim = _interval_steps * physics_dt
                _rt_ratio = _interval_sim / max(_interval_wall, 1e-9)
                _avg_step_ms = (_interval_wall / max(_interval_steps, 1)) * 1000
                _avg_teleop_ms = _teleop_sum_ms / max(_interval_steps, 1)
                _avg_phys_ms = _phys_sum_ms / max(_interval_steps, 1)
                _avg_write_ms = _write_sum_ms / max(_interval_steps, 1)
                _avg_sleep_ms = _sleep_sum_ms / max(_interval_steps, 1)
                _breakdown = (
                    f"last_step: total={_step_wall_ms:.1f}ms "
                    f"(teleop={_t_teleop_ms:.1f} write={_t_write_ms:.1f} "
                    f"phys={_t_phys_ms:.1f} sleep={_t_sleep_ms:.1f}"
                )
                if collector is not None:
                    _breakdown += f" before={_t_before_step_ms:.1f} after={_t_after_step_ms:.1f}"
                _breakdown += ")"
                _breakdown += (
                    f" | avg: teleop={_avg_teleop_ms:.1f} write={_avg_write_ms:.1f}"
                    f" phys={_avg_phys_ms:.1f} sleep={_avg_sleep_ms:.1f}"
                )
                logger.info(
                    "profile step=%d sim_time=%.2fs wall_time=%.2fs "
                    "rt_ratio=%.3fx avg_step=%.1fms %s",
                    sim_step,
                    _sim_elapsed,
                    _wall_elapsed,
                    _rt_ratio,
                    _avg_step_ms,
                    _breakdown,
                )
                _last_timing_step = sim_step
                _last_timing_wall = _now
                _teleop_sum_ms = 0.0
                _phys_sum_ms = 0.0
                _write_sum_ms = 0.0
                _sleep_sum_ms = 0.0
            # ── End timing report ───────────────────────────────────

            if (
                collector is not None
                and not keyboard_episode_mode
                and auto_record_episode_target > 1
                and auto_record_episode_step_budget is not None
                and sim_step - auto_record_episode_start_step >= auto_record_episode_step_budget
            ):
                episode_path = collector.finish_episode()
                if episode_path is not None:
                    print(
                        "[INFO] Auto-record episode "
                        f"{auto_record_episode_index}/{auto_record_episode_target} "
                        f"finished: {episode_path}"
                    )

                if auto_record_episode_index >= auto_record_episode_target:
                    print(
                        "[INFO] Auto-record target reached: "
                        f"{auto_record_episode_target} episode(s)."
                    )
                    break

                current_seed_context = (
                    collection_seed_allocator.next()
                    if collection_seed_allocator is not None
                    else None
                )
                rebuild_result = resample_scene_for_next_episode(
                    sim=sim,
                    physics_dt=physics_dt,
                    task=task,
                    task_dir=task_dir,
                    generalization_enabled=generalization_enabled,
                    generalization_cfg=generalization_cfg,
                    fixed_background_sample=fixed_background_sample,
                    generalization_asset_split=args_cli.generalization_split,
                    seed_context=current_seed_context,
                    current_robot_key=robot_key,
                    current_robot_runtime=robot_runtime,
                    collector=collector,
                    collect_cfg=collect_cfg if args_cli.collect else None,
                    app_running_state_fn=_safe_app_running_state,
                )
                if rebuild_result is None:
                    print("[WARN] Auto-record scene rebuild returned no result; stopping.")
                    break

                runtime = rebuild_result["runtime"]
                interactive_objects = rebuild_result["interactive_objects"]
                groups = rebuild_result["groups"]
                controlled_articulations = rebuild_result["controlled_articulations"]
                task_articulations = rebuild_result["task_articulations"]
                non_articulation_objects = rebuild_result["non_articulation_objects"]
                articulation_hold_targets = rebuild_result["articulation_hold_targets"]
                scene_generalization_sample = rebuild_result["scene_generalization_sample"]
                robot_runtime = rebuild_result["robot_runtime"]
                pre_step_hooks = list(robot_runtime.get("pre_step_hooks", []))
                move_controlled_articulations_home(
                    controlled_articulations,
                    articulation_hold_targets,
                    teleport=True,
                )
                auto_record_episode_index += 1
                _warmup_scene_before_camera_recording(
                    collector=collector,
                    sim=sim,
                    physics_dt=physics_dt,
                    controlled_articulations=controlled_articulations,
                    articulation_hold_targets=articulation_hold_targets,
                    interactive_objects=interactive_objects,
                    pre_step_hooks=pre_step_hooks,
                    reason="auto-record scene rebuild",
                )
                collector.start_episode(sim_step=sim_step)
                auto_record_episode_start_step = sim_step
                if ghost_state["enabled"] and ghost_state["start_sim_step"] is None:
                    ghost_state["start_sim_step"] = sim_step
                print(
                    "[INFO] Auto-record episode "
                    f"{auto_record_episode_index}/{auto_record_episode_target} started."
                )
                continue

            if (
                args_cli.episode_steps is not None
                and auto_record_episode_target <= 1
                and sim_step >= args_cli.episode_steps
            ):
                if args_cli.profile:
                    _wall_total = _time.monotonic() - _wall_start
                    _sim_total = sim_step * physics_dt
                    logger.info(
                        "profile final steps=%d sim_time=%.2fs wall_time=%.2fs "
                        "rt_ratio=%.3fx avg=%.1fms/step",
                        sim_step,
                        _sim_total,
                        _wall_total,
                        _sim_total / _wall_total,
                        _wall_total / sim_step * 1000,
                    )
                print(f"[INFO] Episode step limit reached: {args_cli.episode_steps}")
                break
    except Exception as exc:
        primary_error = exc
        log_runtime_exception(
            exc,
            task_path=task_path,
            scene_ready=scene_ready,
            sim_step=sim_step,
            app_running=_safe_app_running_state(),
        )
        raise
    finally:
        teardown_episode_keyboard_controls(keyboard_controls)
        if teleop_bridge is not None:
            teleop_bridge.stop()
        shutdown_with_diagnostics(collector=collector, simulation_app=simulation_app, primary_error=primary_error)


if __name__ == "__main__":
    main()
