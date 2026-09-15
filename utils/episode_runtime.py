from __future__ import annotations

from collections import deque
import logging
from typing import Any, Callable

import isaaclab.sim as sim_utils

from build import (
    apply_object_color_overrides,
    build_scene,
    collect_task_asset_codes,
    sample_scene_generalization,
    scene_generalization_sample_debug_lines,
)
from collector import DataCollector
from collector.contact_sensor_reader import install_contact_reader_for_runtime
from utils.episode_session import EpisodeCommand
from utils.runtime_helpers import classify_interactive_objects, initialize_articulation_state, initialize_sim_object_state
from utils.seed_policy import seed_everything
from utils.usd_prims import delete_prim_compat, delete_prim_if_valid, get_current_stage_compat


logger = logging.getLogger(__name__)


def ramp_homing_targets(
    controlled_articulations: list[object],
    articulation_hold_targets: dict[int, object],
    *,
    ramp_rate: float = 1.0,
    dt: float = 1.0 / 60,
) -> None:
    """每帧将 hold target 向 home 渐进，速率限制为 ramp_rate rad/s。

    在主循环的 homing 检查处每步调用，使手臂以受控速度回到 home 位，
    而不是以 velocity_limit_sim 全速冲向目标。

    Important: ramp from the *previous target*, not from the current joint
    position. Otherwise, if the joint can't keep up (or has drifted) the
    target gets stuck near ``cur ± step`` and never actually reaches
    ``home`` — the joint stays at its drifted position because the
    PD target keeps trailing it.
    """
    import torch
    step = ramp_rate * dt
    for art in controlled_articulations:
        home = art.data.default_joint_pos          # (1, n_joints) 目标
        prev_target = articulation_hold_targets.get(id(art))
        if prev_target is None:
            # First call: seed at current joint position so we don't snap.
            prev_target = art.data.joint_pos.clone()
        delta = (home - prev_target).clamp(-step, step)    # 每关节最大步幅
        articulation_hold_targets[id(art)] = (prev_target + delta).clone()


def move_controlled_articulations_home(
    controlled_articulations: list[object],
    articulation_hold_targets: dict[int, object],
    *,
    teleop_bridge=None,
    teleport: bool = False,
) -> None:
    """Set hold targets to home (default) joint positions.

    Args:
        teleport: If True, also write joint state to sim (instant reset).
                  If False (default), only set position targets so the PD
                  controller drives the joints there over multiple sim steps.
    """
    for articulation in controlled_articulations:
        home_pos = articulation.data.default_joint_pos.clone()
        if teleport:
            joint_vel = articulation.data.default_joint_vel.clone()
            root_state = articulation.data.default_root_state.clone()
            articulation.write_root_pose_to_sim(root_state[:, :7])
            articulation.write_root_velocity_to_sim(root_state[:, 7:])
            articulation.write_joint_state_to_sim(home_pos, joint_vel)
            articulation.reset()
        articulation_hold_targets[id(articulation)] = home_pos


def check_articulations_at_home(
    controlled_articulations: list[object],
    threshold_rad: float = 0.02,
) -> bool:
    """Return True if all controlled articulations are close to their default joint positions."""
    for articulation in controlled_articulations:
        home_pos = articulation.data.default_joint_pos
        current_pos = articulation.data.joint_pos
        max_err = (current_pos - home_pos).abs().max().item()
        if max_err > threshold_rad:
            return False
    return True


def summarize_articulations_home_errors(
    controlled_articulations: list[object],
    threshold_rad: float = 0.02,
    top_k: int = 5,
) -> list[str]:
    """Return human-readable summaries for joints still outside the home threshold."""
    summaries: list[str] = []
    for articulation_idx, articulation in enumerate(controlled_articulations):
        home_pos = articulation.data.default_joint_pos
        current_pos = articulation.data.joint_pos

        if getattr(home_pos, "ndim", 0) > 1:
            home_pos = home_pos[0]
        if getattr(current_pos, "ndim", 0) > 1:
            current_pos = current_pos[0]

        abs_err = (current_pos - home_pos).abs()
        max_err = float(abs_err.max().item())
        if max_err <= threshold_rad:
            continue

        joint_names = list(getattr(articulation.data, "joint_names", []))
        offender_count = min(int(abs_err.numel()), top_k)
        top_vals, top_indices = abs_err.topk(offender_count)
        offender_parts: list[str] = []
        for err_val, joint_idx in zip(top_vals.tolist(), top_indices.tolist()):
            joint_name = joint_names[joint_idx] if joint_idx < len(joint_names) else f"joint[{joint_idx}]"
            current_val = float(current_pos[joint_idx].item())
            home_val = float(home_pos[joint_idx].item())
            offender_parts.append(
                f"{joint_name}: err={err_val:.3f}, cur={current_val:.3f}, home={home_val:.3f}"
            )

        summaries.append(
            f"[HOMECHK] articulation[{articulation_idx}] max_err={max_err:.3f} rad; "
            f"top offenders: {'; '.join(offender_parts)}"
        )

    return summaries


def setup_episode_keyboard_controls():
    try:
        import carb.input
        import omni.appwindow
    except Exception as exc:
        logger.warning("Episode keyboard controls unavailable: %s", exc)
        return None

    app_window = omni.appwindow.get_default_app_window()
    if app_window is None:
        logger.warning("Episode keyboard controls unavailable: app window not found")
        return None

    input_iface = carb.input.acquire_input_interface()
    keyboard = app_window.get_keyboard()
    pending_commands: deque[EpisodeCommand] = deque()

    def _on_keyboard_event(event, *args):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            key = event.input

            if key == carb.input.KeyboardInput.NUMPAD_1:
                pending_commands.append(EpisodeCommand.HOME)
                logger.debug("HOME queued")
            elif key == carb.input.KeyboardInput.NUMPAD_2:
                pending_commands.append(EpisodeCommand.START)
                logger.debug("START queued")
            elif key == carb.input.KeyboardInput.NUMPAD_3:
                pending_commands.append(EpisodeCommand.STOP)
                logger.debug("STOP queued")

        return True

    subscription = input_iface.subscribe_to_keyboard_events(keyboard, _on_keyboard_event)

    # Optional global keyboard listener (works when IsaacSim is not focused).
    # X11 only; on Wayland pynput silently fails. Any failure is non-fatal.
    global_listener = _start_global_keyboard_listener(pending_commands)

    return (
        input_iface,
        keyboard,
        subscription,
        pending_commands,
        global_listener,
    )


def _start_global_keyboard_listener(pending_commands):
    try:
        from pynput import keyboard as _pkb
    except Exception as exc:
        logger.info("Global keyboard hotkeys disabled (pynput unavailable: %s)", exc)
        return None

    def _on_press(key):
        try:
            from pynput.keyboard import Key as _PKey
            if key == _PKey.page_down:
                pending_commands.append(EpisodeCommand.HOME)
                logger.debug("HOME queued (global)")
            elif key == _PKey.home:
                pending_commands.append(EpisodeCommand.START)
                logger.debug("START queued (global)")
            elif key == _PKey.end:
                pending_commands.append(EpisodeCommand.STOP)
                logger.debug("STOP queued (global)")
        except Exception as exc:
            logger.warning("Global hotkey handler error: %s", exc)

    try:
        listener = _pkb.Listener(on_press=_on_press, daemon=True)
        listener.start()
        logger.info("Global hotkeys active (LEFT=home, DOWN=start, RIGHT=stop); X11 only")
        return listener
    except Exception as exc:
        logger.info("Global keyboard hotkeys disabled (listener start failed: %s)", exc)
        return None


def teardown_episode_keyboard_controls(keyboard_controls) -> None:
    if keyboard_controls is None:
        return
    input_iface, keyboard, subscription = keyboard_controls[:3]
    try:
        input_iface.unsubscribe_to_keyboard_events(keyboard, subscription)
    except Exception as exc:
        print(f"[WARN] Failed to unsubscribe keyboard episode controls: {exc}")
    if len(keyboard_controls) >= 5:
        global_listener = keyboard_controls[4]
        if global_listener is not None:
            try:
                global_listener.stop()
            except Exception as exc:
                print(f"[WARN] Failed to stop global keyboard listener: {exc}")


def clear_scene_prims_preserve_robot(sim=None, *, preserve_robot: bool = True) -> None:
    full_rebuild = not preserve_robot
    restore_stop_guard = None
    if sim is not None and full_rebuild:
        restore_stop_guard = bool(getattr(sim, "_disable_app_control_on_stop_handle", False))
        sim._disable_app_control_on_stop_handle = True
        try:
            import omni.kit.app

            print("[scene rebuild] Stopping physics and releasing tensor views...")
            sim._timeline.stop()
            sim._timeline.commit()
            omni.kit.app.get_app().update()

            try:
                import omni.replicator.core as rep

                rep.vp_manager.destroy_hydra_textures("Replicator")
            except Exception as exc:
                print(f"[WARN] Failed to release Replicator render products: {exc}")
        except Exception:
            sim._disable_app_control_on_stop_handle = restore_stop_guard
            raise

    stage = get_current_stage_compat()
    objects_prim = stage.GetPrimAtPath("/World/Objects")
    if objects_prim.IsValid():
        for child in list(objects_prim.GetChildren()):
            child_path = str(child.GetPath())
            if preserve_robot and child_path == "/World/Objects/GlobalRobot":
                continue
            try:
                delete_prim_if_valid(stage, lambda path: delete_prim_compat(path, stage=stage), child_path)
            except Exception as exc:
                print(f"[WARN] Failed to delete prim '{child_path}': {exc}")

    for prim_path in (
        "/World/defaultGroundPlane",
        "/World/BackgroundDome",
        "/World/Environment",
        "/World/DistantLight",
        "/World/FillLight",
        "/World/Looks/TableSurfaceMaterial",
    ):
        try:
            delete_prim_if_valid(stage, lambda path: delete_prim_compat(path, stage=stage), prim_path)
        except Exception:
            pass

    # Flush stage so spawners see prims as deleted.
    if sim is not None:
        if full_rebuild:
            try:
                import omni.kit.app

                app = omni.kit.app.get_app()
                app.update()
                app.update()
                print("[scene rebuild] Old USD scene removed; ready to spawn the next episode.")
            finally:
                if restore_stop_guard is not None:
                    sim._disable_app_control_on_stop_handle = restore_stop_guard
        else:
            sim.render()
            sim.render()

    if full_rebuild and stage.GetPrimAtPath("/World/Objects/GlobalRobot").IsValid():
        from pxr import Sdf

        deleted = False
        for _ in range(3):
            try:
                stage.RemovePrim(Sdf.Path("/World/Objects/GlobalRobot"))
                deleted = True
                break
            except Exception:
                import omni.kit.app

                omni.kit.app.get_app().update()

        if not deleted or stage.GetPrimAtPath("/World/Objects/GlobalRobot").IsValid():
            raise RuntimeError(
                "Full scene rebuild left a stale /World/Objects/GlobalRobot prim. "
                "There may be a dangling PhysX/Fabric reference; restart the process "
                "between episodes or open a fresh USD stage."
            )


def initialize_scene_runtime_state(
    *,
    sim,
    physics_dt: float,
    interactive_objects: dict[str, object],
    object_prim_paths: dict[str, str] | None = None,
    object_display_colors: dict[str, tuple[float, float, float]] | None = None,
    collector: DataCollector | None,
    app_running_state_fn: Callable[[], bool | None],
):
    groups = classify_interactive_objects(interactive_objects)
    controlled_articulations = groups.controlled_articulations
    task_articulations = groups.task_articulations
    non_articulation_objects = groups.other_objects

    if collector is not None:
        collector.create_camera_rig_before_reset()

    sim.reset()
    logger.debug("simulation_app.is_running() after reset: %s", app_running_state_fn())

    # Ensure PhysX fabric re-initialisation has completed on the GPU before
    # any articulation tensor views are accessed (clone / write_joint_state_to_sim).
    # Without this barrier a dangling fabric address from a previous episode may
    # still be in-flight, causing "CUDA error: illegal memory access" after many
    # scene-rebuild cycles (observed around episode 39).
    import torch as _torch
    if _torch.cuda.is_available():
        _torch.cuda.synchronize()

    if object_prim_paths and object_display_colors:
        apply_object_color_overrides(
            object_prim_paths=object_prim_paths,
            object_colors=object_display_colors,
        )

    articulation_hold_targets: dict[int, object] = {}
    for obj in controlled_articulations:
        articulation_hold_targets[id(obj)] = initialize_articulation_state(obj)
    for obj in task_articulations:
        initialize_articulation_state(obj, set_hold_target=False)
    for obj in non_articulation_objects:
        initialize_sim_object_state(obj)

    if collector is not None:
        collector.initialize_after_reset()

    return groups, controlled_articulations, task_articulations, non_articulation_objects, articulation_hold_targets


def resample_scene_for_next_episode(
    *,
    sim,
    physics_dt: float,
    task: dict,
    task_dir: str,
    generalization_enabled: bool,
    generalization_cfg,
    fixed_background_sample,
    current_robot_key: str,
    current_robot_runtime: dict | None,
    collector: DataCollector | None,
    collect_cfg,
    app_running_state_fn: Callable[[], bool | None],
    seed_context=None,
    generalization_asset_split: str | None = None,
):
    if seed_context is not None:
        seed_everything(seed_context.episode_seed)
        print(
            "[INFO] Collection seed context: "
            f"namespace={seed_context.seed_namespace} "
            f"task_seed_id={seed_context.task_seed_id} "
            f"task_base_seed={seed_context.task_base_seed} "
            f"episode_index={seed_context.episode_index} "
            f"episode_seed={seed_context.episode_seed}"
        )
    scene_generalization_sample = sample_scene_generalization(
        generalization_cfg,
        enabled=generalization_enabled,
        task_asset_codes=collect_task_asset_codes(task),
        asset_split=generalization_asset_split,
    )
    if generalization_enabled:
        print(f"[INFO] Scene generalization resampled: robot_key={current_robot_key}")
    else:
        print(f"[INFO] Scene reset to base layout: robot_key={current_robot_key}")
    for line in scene_generalization_sample_debug_lines(scene_generalization_sample):
        print(line)

    if collector is not None:
        collector.prepare_scene_rebuild()

    clear_scene_prims_preserve_robot(sim=sim, preserve_robot=False)

    runtime = build_scene(
        task,
        task_dir,
        generalization_enabled=generalization_enabled,
        generalization_cfg=generalization_cfg,
        generalization_sample=scene_generalization_sample,
        fixed_background_sample=fixed_background_sample,
        robot_key=current_robot_key,
        existing_robot_runtime=None,  # 每次重新 spawn 机器人，确保 physics view 完整初始化
    )
    runtime["success_conditions"] = task.get("success_conditions", [])
    runtime["task_instruction"] = task.get("description", None)
    runtime["metrics"] = task.get("metrics", {})
    interactive_objects = runtime.get("interactive_objects", {})

    if collector is not None:
        collector.refresh_scene_runtime(runtime, interactive_objects)
        if seed_context is not None:
            collector.set_episode_seed_context(seed_context)

    groups, controlled_articulations, task_articulations, non_articulation_objects, articulation_hold_targets = (
        initialize_scene_runtime_state(
            sim=sim,
            physics_dt=physics_dt,
            interactive_objects=interactive_objects,
            object_prim_paths=runtime.get("object_prim_paths", {}),
            object_display_colors=runtime.get("object_display_colors", {}),
            collector=collector,
            app_running_state_fn=app_running_state_fn,
        )
    )
    install_contact_reader_for_runtime(collector, runtime)

    return {
        "runtime": runtime,
        "interactive_objects": interactive_objects,
        "groups": groups,
        "controlled_articulations": controlled_articulations,
        "task_articulations": task_articulations,
        "non_articulation_objects": non_articulation_objects,
        "articulation_hold_targets": articulation_hold_targets,
        "scene_generalization_sample": scene_generalization_sample,
        "robot_runtime": runtime.get("robot_runtime", current_robot_runtime or {}),
    }
