"""Helpers for articulation initialization and per-step state updates."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import sys
import traceback
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ArticulationLike(Protocol):
    """Structural interface for anything that looks like an IsaacLab Articulation."""

    data: Any

    def write_root_pose_to_sim(self, pose: Any) -> None: ...
    def write_root_velocity_to_sim(self, vel: Any) -> None: ...
    def write_joint_state_to_sim(self, pos: Any, vel: Any) -> None: ...
    def set_joint_position_target(self, target: Any) -> None: ...
    def write_data_to_sim(self) -> None: ...
    def reset(self) -> None: ...


@runtime_checkable
class SimObjectLike(Protocol):
    """Minimal interface for any interactive simulation object."""

    def reset(self) -> None: ...
    def update(self, dt: float) -> None: ...


@dataclass
class InteractiveObjectGroups:
    robot_articulation: ArticulationLike | None
    controlled_articulations: list[ArticulationLike]
    task_articulations: list[ArticulationLike]
    other_objects: list[object]


def _is_articulation_like(obj: object | None) -> bool:
    return (
        obj is not None
        and hasattr(obj, "write_joint_state_to_sim")
        and hasattr(obj, "set_joint_position_target")
    )


def classify_interactive_objects(interactive_objects: dict[str, object]) -> InteractiveObjectGroups:
    robot_articulation = interactive_objects.get("global_robot")
    if not _is_articulation_like(robot_articulation):
        robot_articulation = None

    controlled_articulations: list[object] = []
    if robot_articulation is not None:
        controlled_articulations.append(robot_articulation)

    task_articulations: list[object] = []
    other_objects: list[object] = []
    for obj_id, obj in interactive_objects.items():
        if obj_id == "global_robot":
            continue
        if _is_articulation_like(obj):
            task_articulations.append(obj)
        else:
            other_objects.append(obj)

    return InteractiveObjectGroups(
        robot_articulation=robot_articulation,
        controlled_articulations=controlled_articulations,
        task_articulations=task_articulations,
        other_objects=other_objects,
    )


def initialize_articulation_state(articulation: ArticulationLike, *, set_hold_target: bool = True):
    """Write the default root/joint state into sim and optionally seed a hold target."""
    root_state = articulation.data.default_root_state.clone()
    joint_pos = articulation.data.default_joint_pos.clone()
    joint_vel = articulation.data.default_joint_vel.clone()

    articulation.write_root_pose_to_sim(root_state[:, :7])
    articulation.write_root_velocity_to_sim(root_state[:, 7:])
    articulation.write_joint_state_to_sim(joint_pos, joint_vel)
    articulation.reset()
    if set_hold_target:
        articulation.set_joint_position_target(joint_pos)
        return joint_pos
    return None


def is_kinematic_rigid_object(obj: object) -> bool:
    """Return whether an IsaacLab rigid-object wrapper was configured kinematic.

    ``RigidObject.write_root_velocity_to_sim`` ultimately calls PhysX
    ``PxRigidDynamic::setLinearVelocity`` and ``setAngularVelocity``.  PhysX
    rejects both calls for kinematic bodies, so callers that restore root state
    must write the pose only for these fixed scene supports.

    Keep this helper structural so it is usable in lightweight unit tests and
    by utilities that do not import IsaacLab classes directly.
    """
    cfg = getattr(obj, "cfg", None)
    spawn = getattr(cfg, "spawn", None)
    rigid_props = getattr(spawn, "rigid_props", None)
    return bool(getattr(rigid_props, "kinematic_enabled", False))


def initialize_sim_object_state(obj: object) -> None:
    """Write a non-articulation object's default root state back into sim."""
    data = getattr(obj, "data", None)
    default_root_state = getattr(data, "default_root_state", None)
    write_root_pose = getattr(obj, "write_root_pose_to_sim", None)
    write_root_velocity = getattr(obj, "write_root_velocity_to_sim", None)
    if default_root_state is not None and callable(write_root_pose):
        root_state = default_root_state.clone()
        write_root_pose(root_state[:, :7])
        if callable(write_root_velocity) and not is_kinematic_rigid_object(obj):
            write_root_velocity(root_state[:, 7:])
    reset = getattr(obj, "reset", None)
    if callable(reset):
        reset()


def write_articulation_targets(articulation_objects: Iterable[ArticulationLike], hold_targets: dict[int, Any]) -> None:
    """Re-apply position targets every simulation step so the robot keeps its commanded pose."""
    for articulation in articulation_objects:
        target = hold_targets.get(id(articulation))
        if target is None:
            target = articulation.data.default_joint_pos.clone()
            hold_targets[id(articulation)] = target
        articulation.set_joint_position_target(target)
        articulation.write_data_to_sim()


def update_sim_objects(objects: Iterable[object], dt: float) -> None:
    """Refresh object state buffers after a simulation step when the wrapper exposes update()."""
    for obj in objects:
        update = getattr(obj, "update", None)
        if callable(update):
            update(dt)


def log_runtime_exception(
    exc: BaseException,
    *,
    task_path: str | None,
    scene_ready: bool,
    sim_step: int,
    app_running: bool | None,
) -> None:
    """Emit actionable runtime diagnostics to stderr before re-raising."""
    print("[ERROR] Unhandled exception during scene execution.", file=sys.stderr)
    context = [f"scene_ready={scene_ready}", f"sim_step={sim_step}"]
    if task_path is not None:
        context.insert(0, f"task_path={task_path}")
    if app_running is not None:
        context.append(f"app_running={app_running}")
    print(f"[ERROR] Context: {', '.join(context)}", file=sys.stderr)
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)


def log_shutdown_exception(exc: BaseException) -> None:
    """Emit shutdown diagnostics without suppressing the underlying traceback."""
    print("[ERROR][shutdown] Failed while closing runtime resources.", file=sys.stderr)
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)


def shutdown_runtime(collector=None, simulation_app=None):
    """Tear down collector-owned resources before closing the Isaac Sim application."""
    episode_path = None
    collector_error: Exception | None = None
    app_error: Exception | None = None

    if collector is not None:
        try:
            close = getattr(collector, "close", None)
            if callable(close):
                episode_path = close()
            else:
                flush_episode = getattr(collector, "flush_episode", None)
                if callable(flush_episode):
                    episode_path = flush_episode()
        except Exception as exc:
            collector_error = exc

    if simulation_app is not None:
        try:
            simulation_app.close()
        except Exception as exc:
            app_error = exc

    if collector_error is not None and app_error is not None:
        print("[ERROR][shutdown] Both collector and app raised exceptions.", file=sys.stderr)
        print("[ERROR][shutdown] App exception (secondary):", file=sys.stderr)
        traceback.print_exception(type(app_error), app_error, app_error.__traceback__, file=sys.stderr)
        raise collector_error
    if collector_error is not None:
        raise collector_error
    if app_error is not None:
        raise app_error

    return episode_path


def shutdown_with_diagnostics(collector=None, simulation_app=None, primary_error: BaseException | None = None):
    """Close runtime resources and avoid overwriting a previously logged runtime failure."""
    try:
        return shutdown_runtime(collector=collector, simulation_app=simulation_app)
    except Exception as exc:
        log_shutdown_exception(exc)
        if primary_error is None:
            raise
    return None
