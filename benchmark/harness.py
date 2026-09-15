"""Benchmark evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

from .metric_tracker import MetricTracker
from .metrics import EpisodeResult
from .results import write_results
from utils.seed_policy import (
    SEED_POLICY_ONE_INDEXED,
    episode_seed,
    resolve_benchmark_task_seed_id,
    seed_everything,
    task_base_seed,
)

HARNESS_SEED_POLICY = SEED_POLICY_ONE_INDEXED

try:
    from integration.starvla.task_registry import TaskSpec, get_task_spec
except ImportError:
    TaskSpec = Any
    get_task_spec = None


@dataclass
class TaskSpecFallback:
    """Minimal task spec when integration.starvla is unavailable."""
    scene_name: str
    success_conditions: list = field(default_factory=list)


@dataclass
class EvalConfig:
    scenes: list[str]
    num_episodes: int = 100
    step_limit: int = 1000
    output_dir: Path | str = "eval_result"
    policy_name: str = "unnamed"
    allow_legacy_success: bool = False
    base_seed: int = 0


@runtime_checkable
class BenchmarkTaskEnvProtocol(Protocol):
    """Protocol for benchmark task environments.

    Required methods must be implemented by all environments.
    Optional methods enable richer metrics when available:

    - get_robot_state() -> dict | None
        Return joint state dict with 'qpos' (np.ndarray) and 'qvel' keys.
        Used by MetricTracker for joint_limit violation detection.

    - get_joint_limits() -> tuple[np.ndarray, np.ndarray] | None
        Return (lower_limits, upper_limits) as float32 arrays of shape (num_joints,).
        Used by MetricTracker for joint_limit violation detection.

    - get_contact_data() -> dict[str, dict[str, list[float]]] | None
        Return per-object per-target contact forces: {sensor_id: {target_id: [fx,fy,fz]}}.
        Used by impact_count conditions for PhysX contact verification.

    - get_dt() -> float  or  get_physics_dt() -> float
        Return the physics timestep. Used for dwell-time calculations.
    """

    def reset(self, *, seed: int | None = None) -> None: ...
    def get_instruction(self) -> str: ...
    def get_observation(self) -> dict: ...
    def take_action(self, action: Any) -> None: ...
    def get_object_states(self) -> dict: ...


def _resolve_task_spec(scene_name: str) -> Any:
    if get_task_spec is not None:
        return get_task_spec(scene_name)
    return TaskSpecFallback(scene_name=scene_name)


def _get_task_metrics(task_spec: Any) -> dict:
    metrics = getattr(task_spec, "metrics", None)
    if isinstance(metrics, dict):
        return metrics
    if isinstance(task_spec, dict):
        metrics = task_spec.get("metrics")
        if isinstance(metrics, dict):
            return metrics
    return {}


def _get_env_dt(env: BenchmarkTaskEnvProtocol) -> float | None:
    for name in ("get_dt", "get_physics_dt"):
        fn = getattr(env, name, None)
        if callable(fn):
            return float(fn())
    return None


def _get_optional_payload(env: BenchmarkTaskEnvProtocol, name: str):
    fn = getattr(env, name, None)
    if callable(fn):
        return fn()
    return None


def rollout_scene(
    task_spec: Any,
    env: BenchmarkTaskEnvProtocol,
    policy_adapter: Any,
    episode_index: int,
    seed: int,
    step_limit: int,
    base_seed: int | None = None,
    allow_legacy_success: bool = False,
    perturbation_axis: str | None = None,
    robot_key: str | None = None,
) -> EpisodeResult:
    """Run a single episode and return the result."""
    conditions = getattr(task_spec, "success_conditions", [])
    metrics_spec = _get_task_metrics(task_spec)
    tracker = MetricTracker(
        metrics_spec,
        success_conditions=conditions,
        allow_legacy_success_conditions=allow_legacy_success,
        robot_key=robot_key,
    )
    seed_everything(seed)
    env.reset(seed=seed)
    if hasattr(policy_adapter, "reset"):
        policy_adapter.reset(env.get_instruction())

    steps = 0
    policy_exception: str | None = None

    try:
        for steps in range(1, step_limit + 1):
            obs = env.get_observation()
            example = {
                "lang": env.get_instruction(),
                "image": obs.get("image", []),
                "state": obs.get("state", np.zeros(0)),
            }
            action = policy_adapter.step(example, step=steps - 1)
            env.take_action(action)
            object_states = env.get_object_states()
            # Merge contact force data if available (env.get_contact_data() should
            # call ContactSensorReader.update(dt) internally before returning)
            _contact_data = _get_optional_payload(env, "get_contact_data")
            if _contact_data:
                for _obj_id, _forces in _contact_data.items():
                    if _obj_id in object_states:
                        object_states[_obj_id]["contact_forces"] = _forces
            if tracker.available:
                tracker.update(
                    object_states,
                    sim_step=steps,
                    dt=_get_env_dt(env),
                    robot_state=_get_optional_payload(env, "get_robot_state"),
                    joint_limits=_get_optional_payload(env, "get_joint_limits"),
                )
                if tracker.success:
                    break
    except Exception as exc:
        policy_exception = str(exc)

    # Determine terminated_reason
    if policy_exception is not None:
        terminated_reason = "error"
    elif tracker.success:
        terminated_reason = "stable_success"
    else:
        terminated_reason = "max_steps"

    scene = getattr(task_spec, "scene_name", str(task_spec))
    metric_result = tracker.finalize(
        steps=steps,
        terminated_reason=terminated_reason,
        policy_query_count=steps,
        policy_query_count_at_stable_success=steps if tracker.success else None,
    )
    return EpisodeResult(
        scene=scene,
        episode_index=episode_index,
        seed=seed,
        episode_seed=seed,
        base_seed=base_seed,
        seed_policy=HARNESS_SEED_POLICY if base_seed is not None else "episode_seed = seed argument",
        scene_generalization_sample=None,
        success=metric_result.stable_success,
        steps=steps,
        error=policy_exception,
        task_family=tracker.task_family,
        robot_key=robot_key,
        perturbation_axis=perturbation_axis,
        stable_success=metric_result.stable_success,
        ever_instant_success=metric_result.ever_instant_success,
        at_end_success=metric_result.at_end_success,
        at_end_success_observed=metric_result.at_end_success_observed,
        at_end_success_budget=metric_result.at_end_success_budget,
        first_success_step=metric_result.first_success_step,
        stable_success_step=metric_result.stable_success_step,
        first_stable_success_step=metric_result.first_stable_success_step,
        steps_to_stable_success=metric_result.steps_to_stable_success,
        policy_steps_to_stable_success=metric_result.policy_steps_to_stable_success,
        policy_queries_to_stable_success=metric_result.policy_queries_to_stable_success,
        time_to_stable_success_s=metric_result.time_to_stable_success_s,
        expert_time_s=metric_result.expert_time_s,
        expert_time_step=metric_result.expert_time_step,
        success_hold_s=metric_result.success_hold_s,
        terminal_success_rate=metric_result.terminal_success_rate,
        stage_completion_rate=metric_result.stage_completion_rate,
        normalized_progress_score=metric_result.normalized_progress_score,
        current_stage_completion_rate=metric_result.current_stage_completion_rate,
        current_normalized_progress_score=metric_result.current_normalized_progress_score,
        current_chain_depth=metric_result.current_chain_depth,
        latched_stage_completion_rate=metric_result.latched_stage_completion_rate,
        latched_normalized_progress_score=metric_result.latched_normalized_progress_score,
        latched_chain_depth=metric_result.latched_chain_depth,
        task_efficiency=metric_result.task_efficiency,
        kinematic_grasp_stability_index=metric_result.kinematic_grasp_stability_index,
        mean_kinematic_grasp_stability=metric_result.mean_kinematic_grasp_stability,
        per_object_best_kinematic_grasp_stability=metric_result.per_object_best_kinematic_grasp_stability,
        grasp_gsi_diagnostics=metric_result.grasp_gsi_diagnostics,
        robot_motion_metrics=metric_result.robot_motion_metrics,
        tool_selection_accuracy=metric_result.tool_selection_accuracy,
        tool_switch_success_rate=metric_result.tool_switch_success_rate,
        tool_switch_total=metric_result.tool_switch_total,
        tool_switch_successful=metric_result.tool_switch_successful,
        safety_violation_rate=metric_result.safety_violation_rate,
        safety_violation_rate_time=metric_result.safety_violation_rate_time,
        safety_violation_rate_event=metric_result.safety_violation_rate_event,
        safety_violation_step_rate=metric_result.safety_violation_step_rate,
        safety_violation_events_per_step=metric_result.safety_violation_events_per_step,
        episode_violation_rate=metric_result.episode_violation_rate,
        safety_hard_violation=metric_result.safety_hard_violation,
        drop_violation=metric_result.drop_violation,
        high_speed_violation=metric_result.high_speed_violation,
        excessive_impact_violation=metric_result.excessive_impact_violation,
        robot_constraint_violation=metric_result.robot_constraint_violation,
        finger_joint_limit_saturation=metric_result.finger_joint_limit_saturation,
        impact_metric_available=metric_result.impact_metric_available,
        robot_constraint_diagnostics=metric_result.robot_constraint_diagnostics,
        joint_limit_active_steps=metric_result.joint_limit_active_steps,
        joint_limit_active_step_rate=metric_result.joint_limit_active_step_rate,
        finger_joint_limit_saturation_steps=metric_result.finger_joint_limit_saturation_steps,
        finger_joint_limit_saturation_step_rate=metric_result.finger_joint_limit_saturation_step_rate,
        max_joint_limit_excess_rad=metric_result.max_joint_limit_excess_rad,
        max_joint_limit_excess_joint=metric_result.max_joint_limit_excess_joint,
        chain_depth=metric_result.chain_depth,
        stage_completion=metric_result.stage_completion,
        current_stage_completion=metric_result.current_stage_completion,
        stage_first_completion_step=metric_result.stage_first_completion_step,
        violation_counts=metric_result.violation_counts,
        policy_query_count=metric_result.policy_query_count,
        terminated_reason=metric_result.terminated_reason,
        timeseries=metric_result.timeseries,
        evaluation_protocol="reach_and_stop",
        early_stop=True,
        max_episode_steps=step_limit,
        dwell_time_s=tracker.dwell_time_s,
    )


def benchmark_scene_seed_context(namespace_base_seed: int, scene_name: str) -> tuple[int, int]:
    task_seed_id = resolve_benchmark_task_seed_id(scene_name)
    return task_seed_id, task_base_seed(namespace_base_seed, task_seed_id)


def _validate_unique_scene_seed_partitions(scenes: list[str], namespace_base_seed: int) -> dict[str, tuple[int, int]]:
    partitions: dict[str, tuple[int, int]] = {}
    owners: dict[int, str] = {}
    for scene_name in scenes:
        task_seed_id, scene_base_seed = benchmark_scene_seed_context(namespace_base_seed, scene_name)
        owner = owners.get(task_seed_id)
        if owner is not None and owner != scene_name:
            raise ValueError(
                "Benchmark scene seed partition collision: "
                f"{scene_name!r} and {owner!r} both map to task_seed_id={task_seed_id}. "
                "Use numeric task prefixes for stable explicit partitions."
            )
        owners[task_seed_id] = scene_name
        partitions[scene_name] = (task_seed_id, scene_base_seed)
    return partitions


def run_benchmark(
    config: EvalConfig,
    env_factory: Callable[[str], BenchmarkTaskEnvProtocol],
    policy_adapter: Any,
    execute_rollout: bool = True,
) -> list[EpisodeResult]:
    """Run benchmark across all scenes and episodes."""
    results: list[EpisodeResult] = []

    namespace_base_seed = config.base_seed
    scene_seed_contexts = _validate_unique_scene_seed_partitions(config.scenes, namespace_base_seed)
    for scene_name in config.scenes:
        task_spec = _resolve_task_spec(scene_name)
        env = env_factory(scene_name)
        _task_seed_id, scene_base_seed = scene_seed_contexts[scene_name]

        for episode_index in range(1, config.num_episodes + 1):
            current_episode_seed = episode_seed(scene_base_seed, episode_index)
            result = rollout_scene(
                task_spec=task_spec,
                env=env,
                policy_adapter=policy_adapter,
                episode_index=episode_index,
                seed=current_episode_seed,
                base_seed=scene_base_seed,
                step_limit=config.step_limit,
                allow_legacy_success=config.allow_legacy_success,
            )
            results.append(result)

    write_results(Path(config.output_dir), config.policy_name, results)
    return results
