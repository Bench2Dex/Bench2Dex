"""Benchmark evaluation metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, List

from .statistics import bootstrap_ci


METRIC_SCHEMA_VERSION = "core-v1.8-policy-step-query-split"
EVALUATION_PROTOCOL = "reach_and_stop"
SUCCESS_DEFINITION = "terminal.raw_condition holds for dwell_time_s"
MEASUREMENT_WINDOW = "episode_start_to_stable_success_or_failure_budget"
FIXED_HORIZON_MEASUREMENT_WINDOW = "episode_start_to_max_steps_or_error"
DEFAULT_MAX_EPISODE_STEPS = 1200
DEFAULT_DWELL_TIME_S = 0.5
FIELD_SEMANTICS = {
    "success_rate": (
        "primary protocol success rate: stable_success for reach_and_stop; "
        "terminal at_end_success_budget for fixed_horizon"
    ),
    "avg_time_to_success_s": "main efficiency metric: mean time_to_stable_success_s over reached-success episodes",
    "task_efficiency": (
        "benchmark-specific expert-normalized speed ratio: "
        "expert_time_s / time_to_stable_success_s over reached-success episodes"
    ),
    "avg_steps_to_success": (
        "diagnostic mean steps_to_stable_success over reached-success episodes, "
        "or None if no episode succeeds"
    ),
    "avg_policy_steps_to_success": (
        "diagnostic mean policy-rate control/action steps through stable success over "
        "reached-success episodes"
    ),
    "avg_policy_queries_to_success": (
        "diagnostic mean actual model inference queries through stable success over "
        "reached-success episodes; None when the first-success query count was not observed"
    ),
    "safe_success_rate": (
        "episode success rate with no task-level hard safety violation "
        "(drop or tracked-object high speed; joint limits are diagnostic by default)"
    ),
    "hard_violation_rate": (
        "episode rate with any task-level hard safety violation "
        "(drop or tracked-object high speed; joint limits are diagnostic by default)"
    ),
    "drop_rate": "episode rate with drop_off_table or drop_to_table_after_lift violation",
    "high_speed_violation_rate": "episode rate with any tracked object exceeding the configured speed threshold",
    "excessive_impact_rate": "deprecated contact-force impact metric; no longer populated by core safety",
    "robot_constraint_violation_rate": (
        "diagnostic episode rate with any asset hard-joint-limit observation; "
        "excluded from safe_success_rate unless include_in_core_safety is explicitly enabled"
    ),
    "finger_joint_limit_saturation_rate": (
        "episode rate with dexterous finger joints saturated beyond finger joint-limit thresholds; "
        "diagnostic by default and not included in safe_success_rate"
    ),
    "joint_limit_active_step_rate": (
        "diagnostic mean fraction of steps with any joint exceeding the soft diagnostic margin"
    ),
    "finger_joint_limit_saturation_step_rate": (
        "diagnostic mean fraction of steps with finger joint-limit saturation"
    ),
    "max_joint_limit_excess_rad": "diagnostic maximum operating/soft-limit excess observed in radians; hard-limit source and role validation are in robot_constraint_diagnostics",
    "safety_violation_rate_time": (
        "compatibility diagnostic; with fixed physics_dt this is equivalent to "
        "safety_violation_step_rate and is omitted from grouped summary output"
    ),
    "safety_violation_step_rate": (
        "core safety metric: fraction of executed steps with task-level safety violations; "
        "joint-limit observations are excluded by default"
    ),
    "safety_violation_rate_event": (
        "deprecated compatibility alias of safety_violation_events_per_step; "
        "omitted from grouped summary output"
    ),
    "safety_violation_events_per_step": "diagnostic event density; may exceed 1 when multiple violations happen in one step",
    "episode_violation_rate": (
        "compatibility alias of hard_violation_rate; omitted from grouped core output"
    ),
    "latched_stage_completion_rate": (
        "main progress metric; ever-reached stage completion rate at episode end"
    ),
    "current_stage_completion_rate": "diagnostic terminal-state stage completion rate at episode end",
    "current_normalized_progress_score": "chain-depth diagnostic; normalized progress score at episode end",
    "current_chain_depth_progress_score": (
        "alias of current_normalized_progress_score for ordered dependency-depth diagnostics"
    ),
    "latched_chain_depth_progress_score": (
        "alias of latched_normalized_progress_score for ever-reached dependency-depth diagnostics"
    ),
    "robustness": "success rates grouped by benchmark protocol category: cov, inv, and inv+cov",
}


@dataclass
class EpisodeResult:
    scene: str
    episode_index: int
    seed: int
    success: bool
    steps: int
    metric_schema_version: str = METRIC_SCHEMA_VERSION
    error: str | None = None
    episode_seed: int | None = None
    base_seed: int | None = None
    seed_policy: str | None = None
    scene_generalization_sample: dict | None = None
    task_family: str = "unknown"
    robot_key: str | None = None
    perturbation_axis: str | None = None
    stable_success: bool = False
    ever_instant_success: bool = False
    at_end_success: bool = False
    at_end_success_observed: bool = False
    at_end_success_budget: bool | None = None
    first_success_step: int | None = None
    stable_success_step: int | None = None
    first_stable_success_step: int | None = None
    steps_to_stable_success: int | None = None
    policy_steps_to_stable_success: int | None = None
    policy_queries_to_stable_success: int | None = None
    time_to_stable_success_s: float | None = None
    expert_time_s: float | None = None
    expert_time_step: int | None = None
    success_hold_s: float = 0.0
    terminal_success_rate: float = 0.0
    stage_completion_rate: float = 0.0
    normalized_progress_score: float = 0.0
    chain_depth_progress_score: float | None = None
    current_stage_completion_rate: float = 0.0
    current_normalized_progress_score: float = 0.0
    current_chain_depth_progress_score: float | None = None
    current_chain_depth: int = 0
    latched_stage_completion_rate: float = 0.0
    latched_normalized_progress_score: float = 0.0
    latched_chain_depth_progress_score: float | None = None
    latched_chain_depth: int = 0
    task_efficiency: float | None = None
    kinematic_grasp_stability_index: float | None = None
    mean_kinematic_grasp_stability: float | None = None
    per_object_best_kinematic_grasp_stability: dict[str, float] | None = None
    grasp_gsi_diagnostics: dict[str, float] | None = None
    robot_motion_metrics: dict[str, float] | None = None
    tool_selection_accuracy: float | None = None
    tool_switch_success_rate: float | None = None
    tool_switch_total: int = 0
    tool_switch_successful: int = 0
    safety_violation_rate: float = 0.0
    safety_violation_rate_time: float | None = None
    safety_violation_rate_event: float | None = None
    safety_violation_step_rate: float | None = None
    safety_violation_events_per_step: float | None = None
    episode_violation_rate: float | None = None
    safety_hard_violation: bool = False
    drop_violation: bool = False
    high_speed_violation: bool = False
    excessive_impact_violation: bool | None = None
    robot_constraint_violation: bool = False
    finger_joint_limit_saturation: bool = False
    impact_metric_available: bool = False
    chain_depth: int = 0
    robot_constraint_diagnostics: dict[str, Any] | None = None
    joint_limit_active_steps: int = 0
    joint_limit_active_step_rate: float | None = None
    finger_joint_limit_saturation_steps: int = 0
    finger_joint_limit_saturation_step_rate: float | None = None
    max_joint_limit_excess_rad: float | None = None
    max_joint_limit_excess_joint: str | None = None
    stage_completion: dict[str, bool] | None = None
    current_stage_completion: dict[str, bool] | None = None
    stage_first_completion_step: dict[str, int | None] | None = None
    violation_counts: dict[str, int] | None = None
    policy_query_count: int | None = None
    terminated_reason: str | None = None
    timeseries: dict | None = None
    evaluation_protocol: str = EVALUATION_PROTOCOL
    early_stop: bool = True
    max_episode_steps: int | None = None
    dwell_time_s: float | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        if payload["episode_seed"] is None:
            payload["episode_seed"] = self.seed
        if payload["chain_depth_progress_score"] is None:
            payload["chain_depth_progress_score"] = self.normalized_progress_score
        if payload["current_chain_depth_progress_score"] is None:
            payload["current_chain_depth_progress_score"] = self.current_normalized_progress_score
        if payload["latched_chain_depth_progress_score"] is None:
            payload["latched_chain_depth_progress_score"] = self.latched_normalized_progress_score
        return payload


def _mean_or_zero(values) -> float:
    values = list(values)
    return mean(values) if values else 0.0


def _mean_optional(results: List[EpisodeResult], attr: str) -> float | None:
    values = [getattr(r, attr) for r in results if getattr(r, attr) is not None]
    return mean(values) if values else None


def _max_optional(results: List[EpisodeResult], attr: str) -> float | None:
    values = [float(getattr(r, attr)) for r in results if getattr(r, attr) is not None]
    return max(values) if values else None


def _progress_alias(result: EpisodeResult, alias_attr: str, fallback_attr: str) -> float:
    value = getattr(result, alias_attr, None)
    if value is None:
        value = getattr(result, fallback_attr)
    return float(value)


def _sum_violation_counts(results: List[EpisodeResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        for name, value in (result.violation_counts or {}).items():
            counts[str(name)] = counts.get(str(name), 0) + int(value)
    return dict(sorted(counts.items()))


def _mean_float_mapping(results: List[EpisodeResult], attr: str) -> dict[str, float]:
    values_by_key: dict[str, list[float]] = {}
    for result in results:
        mapping = getattr(result, attr) or {}
        for key, value in mapping.items():
            if value is None:
                continue
            try:
                values_by_key.setdefault(str(key), []).append(float(value))
            except (TypeError, ValueError):
                continue
    return {
        key: mean(values)
        for key, values in sorted(values_by_key.items())
        if values
    }


def _sum_nested_int_counts(results: List[EpisodeResult], attr: str, key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        mapping = getattr(result, attr, None) or {}
        sub_mapping = mapping.get(key, {}) if isinstance(mapping, dict) else {}
        if not isinstance(sub_mapping, dict):
            continue
        for name, value in sub_mapping.items():
            counts[str(name)] = counts.get(str(name), 0) + int(value)
    return dict(sorted(counts.items()))


def _sum_nested_int(results: List[EpisodeResult], attr: str, key: str) -> int:
    total = 0
    for result in results:
        mapping = getattr(result, attr, None) or {}
        value = mapping.get(key, 0) if isinstance(mapping, dict) else 0
        try:
            total += int(value)
        except (TypeError, ValueError):
            continue
    return total


def _max_nested_float_mapping(results: List[EpisodeResult], attr: str, key: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for result in results:
        mapping = getattr(result, attr, None) or {}
        sub_mapping = mapping.get(key, {}) if isinstance(mapping, dict) else {}
        if not isinstance(sub_mapping, dict):
            continue
        for name, value in sub_mapping.items():
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                continue
            key_s = str(name)
            values[key_s] = max(values.get(key_s, value_f), value_f)
    return dict(sorted(values.items()))


def _diagnostic_boolean_coverage(results: List[EpisodeResult], key: str) -> dict[str, int]:
    """Count explicit true/false diagnostic values without erasing old rows."""
    coverage = {"available": 0, "unavailable": 0, "unknown": 0}
    for result in results:
        diagnostics = getattr(result, "robot_constraint_diagnostics", None) or {}
        if not isinstance(diagnostics, dict) or key not in diagnostics:
            coverage["unknown"] += 1
        elif bool(diagnostics[key]):
            coverage["available"] += 1
        else:
            coverage["unavailable"] += 1
    return coverage


def _diagnostic_string_counts(results: List[EpisodeResult], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        diagnostics = getattr(result, "robot_constraint_diagnostics", None) or {}
        value = diagnostics.get(key) if isinstance(diagnostics, dict) else None
        if value is not None and str(value):
            value_s = str(value)
            counts[value_s] = counts.get(value_s, 0) + 1
    return dict(sorted(counts.items()))


def _diagnostic_string_union(results: List[EpisodeResult], key: str) -> list[str]:
    values: set[str] = set()
    for result in results:
        diagnostics = getattr(result, "robot_constraint_diagnostics", None) or {}
        raw = diagnostics.get(key) if isinstance(diagnostics, dict) else None
        if isinstance(raw, (list, tuple, set)):
            values.update(str(value) for value in raw)
    return sorted(values)


def _baseline_joint_limit_summary(results: List[EpisodeResult]) -> dict[str, dict[str, int]]:
    """Summarize unscored reset/home hard-limit checks for auditability."""
    phases: dict[str, dict[str, int]] = {}
    for result in results:
        diagnostics = getattr(result, "robot_constraint_diagnostics", None) or {}
        checks = diagnostics.get("joint_limit_baseline_checks") if isinstance(diagnostics, dict) else None
        if not isinstance(checks, list):
            continue
        for check in checks:
            if not isinstance(check, dict):
                continue
            phase = str(check.get("phase") or "unknown")
            entry = phases.setdefault(phase, {"checked": 0, "hard_violation_episodes": 0})
            if check.get("checked"):
                entry["checked"] += 1
                if check.get("hard_violation"):
                    entry["hard_violation_episodes"] += 1
    return dict(sorted(phases.items()))


def _common_result_attr(results: List[EpisodeResult], attr: str, default: Any) -> Any:
    values = []
    for result in results:
        value = getattr(result, attr, None)
        if value is None:
            continue
        if value not in values:
            values.append(value)
    if not values:
        return default
    if len(values) == 1:
        return values[0]
    return values


def _measurement_window_for(protocol: Any, early_stop: Any) -> str:
    if protocol == "fixed_horizon" or early_stop is False:
        return FIXED_HORIZON_MEASUREMENT_WINDOW
    if isinstance(protocol, list) or isinstance(early_stop, list):
        return "mixed_protocols"
    return MEASUREMENT_WINDOW


def _pooled_timeseries_rate(results: List[EpisodeResult], key: str, fallback_attr: str) -> float:
    """Pool traces and legacy per-episode rates over their executed steps.

    Episode horizons differ under reach-and-stop, so averaging per-episode
    fractions does not equal the documented fraction of all executed steps.
    """
    total = 0
    positive = 0.0
    for result in results:
        values = (result.timeseries or {}).get(key)
        if isinstance(values, list) and values:
            total += len(values)
            positive += sum(bool(value) for value in values)
            continue
        # An empty trace is exact only for a zero-step episode.  If the result
        # records executed steps, use its persisted rate like any other legacy
        # row rather than dropping its denominator.
        rate = getattr(result, fallback_attr, None)
        steps = max(0, int(result.steps))
        if rate is not None and steps:
            total += steps
            positive += float(rate) * steps
    if total:
        return positive / float(total)
    return 0.0


def _pooled_episode_rate(
    results: List[EpisodeResult],
    attr: str,
    fallback_attr: str | None = None,
) -> float:
    """Pool persisted per-episode densities by their executed-step denominator."""
    total_steps = 0
    weighted_numerator = 0.0
    for result in results:
        rate = getattr(result, attr, None)
        if rate is None and fallback_attr is not None:
            rate = getattr(result, fallback_attr, None)
        steps = max(0, int(result.steps))
        if rate is None or steps <= 0:
            continue
        total_steps += steps
        weighted_numerator += float(rate) * steps
    if total_steps <= 0:
        return 0.0
    return weighted_numerator / float(total_steps)


def summarize(results: List[EpisodeResult]) -> dict:
    """Compute aggregate statistics from episode results."""
    total = len(results)
    if total == 0:
        success_rate_ci = {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
        return {
            "metric_schema_version": METRIC_SCHEMA_VERSION,
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "early_stop": True,
            "success_definition": SUCCESS_DEFINITION,
            "measurement_window": MEASUREMENT_WINDOW,
            "max_episode_steps": DEFAULT_MAX_EPISODE_STEPS,
            "dwell_time_s": DEFAULT_DWELL_TIME_S,
            "field_semantics": dict(FIELD_SEMANTICS),
            "base_seed": None,
            "seed_policy": None,
            "randomization_recorded": False,
            "total": 0,
            "n_episodes": 0,
            "successes": 0,
            "success_rate": 0.0,
            "success_rate_ci": success_rate_ci,
            "stable_success_rate_ci": success_rate_ci,
            "safe_success_rate": 0.0,
            "hard_violation_rate": 0.0,
            "drop_rate": 0.0,
            "high_speed_violation_rate": 0.0,
            "excessive_impact_rate": None,
            "robot_constraint_violation_rate": 0.0,
            "finger_joint_limit_saturation_rate": 0.0,
            "joint_limit_active_step_rate": 0.0,
            "finger_joint_limit_saturation_step_rate": 0.0,
            "max_joint_limit_excess_rad": None,
            "episode_violation_rate": 0.0,
            "safety_violation_step_rate": 0.0,
            "safety_violation_events_per_step": 0.0,
            "current_stage_completion_rate": 0.0,
            "current_chain_depth_progress_score": 0.0,
            "latched_stage_completion_rate": 0.0,
            "latched_chain_depth_progress_score": 0.0,
            "avg_steps": 0.0,
            "avg_steps_to_success": None,
            "avg_policy_steps_to_success": None,
            "avg_policy_queries_to_success": None,
            "avg_time_to_success_s": None,
            "core_metrics": {
                "completion": {
                    "success_rate": 0.0,
                    "success_rate_ci": success_rate_ci,
                },
                "progress": {
                    "latched_stage_completion_rate": 0.0,
                },
                "efficiency": {
                    "avg_time_to_success_s": None,
                },
                "safety": {
                    "safe_success_rate": 0.0,
                    "hard_violation_rate": 0.0,
                    "drop_rate": 0.0,
                    "high_speed_violation_rate": 0.0,
                    "safety_violation_step_rate": 0.0,
                },
                "robustness": {},
            },
            "diagnostic_metrics": {
                "efficiency": {
                    "task_efficiency": None,
                    "expert_time_s": None,
                    "expert_time_step": None,
                    "avg_steps_to_success": None,
                    "avg_policy_steps_to_success": None,
                    "avg_policy_queries_to_success": None,
                },
                "safety": {
                    "safety_violation_events_per_step": 0.0,
                    "violation_counts": {},
                    "finger_joint_limit_saturation_rate": 0.0,
                    "joint_limit_active_step_rate": 0.0,
                    "finger_joint_limit_saturation_step_rate": 0.0,
                    "max_joint_limit_excess_rad": None,
                    "robot_constraint_diagnostics": {},
                },
            },
        }
    successes = sum(1 for r in results if r.success)
    success_values = [1.0 if r.success else 0.0 for r in results]
    stable_values = [1.0 if r.stable_success else 0.0 for r in results]
    success_ci_mean, success_ci_lo, success_ci_hi = bootstrap_ci(success_values)
    stable_tsr_mean, stable_tsr_lo, stable_tsr_hi = bootstrap_ci(stable_values)
    success_rate_ci = {"mean": success_ci_mean, "ci_low": success_ci_lo, "ci_high": success_ci_hi}
    stable_success_rate_ci = {"mean": stable_tsr_mean, "ci_low": stable_tsr_lo, "ci_high": stable_tsr_hi}
    rs_at_p = _compute_rs_at_p(results)
    steps_to_success = [
        int(r.steps_to_stable_success)
        for r in results
        if r.success and r.steps_to_stable_success is not None
    ]
    policy_steps_to_success = [
        int(r.policy_steps_to_stable_success)
        for r in results
        if r.success and r.policy_steps_to_stable_success is not None
    ]
    policy_queries_to_success = [
        int(r.policy_queries_to_stable_success)
        for r in results
        if r.success and r.policy_queries_to_stable_success is not None
    ]
    time_to_success = [
        float(r.time_to_stable_success_s)
        for r in results
        if r.success and r.time_to_stable_success_s is not None
    ]
    evaluation_protocol = _common_result_attr(results, "evaluation_protocol", EVALUATION_PROTOCOL)
    early_stop = _common_result_attr(results, "early_stop", True)
    pooled_safety_step_rate = _pooled_timeseries_rate(
        results, "safety_violation", "safety_violation_step_rate"
    )
    pooled_joint_limit_active_rate = _pooled_timeseries_rate(
        results, "joint_limit_active", "joint_limit_active_step_rate"
    )
    pooled_finger_saturation_rate = _pooled_timeseries_rate(
        results, "finger_joint_limit_saturation_active", "finger_joint_limit_saturation_step_rate"
    )
    summary: dict[str, Any] = {
        "metric_schema_version": _common_result_attr(
            results,
            "metric_schema_version",
            METRIC_SCHEMA_VERSION,
        ),
        "evaluation_protocol": evaluation_protocol,
        "early_stop": early_stop,
        "success_definition": SUCCESS_DEFINITION,
        "measurement_window": _measurement_window_for(evaluation_protocol, early_stop),
        "max_episode_steps": _common_result_attr(results, "max_episode_steps", DEFAULT_MAX_EPISODE_STEPS),
        "dwell_time_s": _common_result_attr(results, "dwell_time_s", DEFAULT_DWELL_TIME_S),
        "field_semantics": dict(FIELD_SEMANTICS),
        "base_seed": _common_result_attr(results, "base_seed", None),
        "seed_policy": _common_result_attr(results, "seed_policy", None),
        "randomization_recorded": any(
            getattr(r, "scene_generalization_sample", None) is not None for r in results
        ),
        "total": total,
        "n_episodes": total,
        "successes": successes,
        "success_rate": successes / total,
        "success_rate_ci": success_rate_ci,
        "stable_success_rate_ci": stable_success_rate_ci,
        "at_end_success_rate": _mean_or_zero(1.0 if r.at_end_success else 0.0 for r in results),
        "ever_instant_success_rate": _mean_or_zero(1.0 if r.ever_instant_success else 0.0 for r in results),
        "safe_success_rate": _mean_or_zero(
            1.0 if (r.success and not r.safety_hard_violation) else 0.0
            for r in results
        ),
        "hard_violation_rate": _mean_or_zero(1.0 if r.safety_hard_violation else 0.0 for r in results),
        "drop_rate": _mean_or_zero(1.0 if r.drop_violation else 0.0 for r in results),
        "high_speed_violation_rate": _mean_or_zero(1.0 if r.high_speed_violation else 0.0 for r in results),
        "excessive_impact_rate": None,
        "robot_constraint_violation_rate": _mean_or_zero(
            1.0 if r.robot_constraint_violation else 0.0 for r in results
        ),
        "finger_joint_limit_saturation_rate": _mean_or_zero(
            1.0 if r.finger_joint_limit_saturation else 0.0 for r in results
        ),
        "joint_limit_active_step_rate": pooled_joint_limit_active_rate,
        "finger_joint_limit_saturation_step_rate": pooled_finger_saturation_rate,
        "max_joint_limit_excess_rad": _max_optional(results, "max_joint_limit_excess_rad"),
        "episode_violation_rate": _mean_or_zero(1.0 if r.safety_hard_violation else 0.0 for r in results),
        "current_stage_completion_rate": _mean_or_zero(r.current_stage_completion_rate for r in results),
        "current_chain_depth_progress_score": _mean_or_zero(
            float(r.current_normalized_progress_score) for r in results
        ),
        "latched_stage_completion_rate": _mean_or_zero(r.latched_stage_completion_rate for r in results),
        "latched_chain_depth_progress_score": _mean_or_zero(
            _progress_alias(
                r,
                "latched_chain_depth_progress_score",
                "latched_normalized_progress_score",
            )
            for r in results
        ),
        "task_efficiency": _mean_optional(results, "task_efficiency"),
        "expert_time_s": _common_result_attr(results, "expert_time_s", None),
        "expert_time_step": _common_result_attr(results, "expert_time_step", None),
        "kinematic_grasp_stability_index": _mean_optional(results, "kinematic_grasp_stability_index"),
        "mean_kinematic_grasp_stability": _mean_optional(results, "mean_kinematic_grasp_stability"),
        "per_object_best_kinematic_grasp_stability": _mean_float_mapping(
            results,
            "per_object_best_kinematic_grasp_stability",
        ),
        "grasp_gsi_diagnostics": _mean_float_mapping(results, "grasp_gsi_diagnostics"),
        "robot_motion_metrics": _mean_float_mapping(results, "robot_motion_metrics"),
        "tool_selection_accuracy": _mean_optional(results, "tool_selection_accuracy"),
        "tool_switch_success_rate": _mean_optional(results, "tool_switch_success_rate"),
        "tool_switch_total": sum(int(r.tool_switch_total) for r in results),
        "tool_switch_successful": sum(int(r.tool_switch_successful) for r in results),
        "safety_violation_rate_time": pooled_safety_step_rate,
        "safety_violation_step_rate": pooled_safety_step_rate,
        "safety_violation_rate_event": _pooled_episode_rate(
            results,
            "safety_violation_events_per_step",
            "safety_violation_rate_event",
        ),
        "safety_violation_events_per_step": _pooled_episode_rate(
            results,
            "safety_violation_events_per_step",
            "safety_violation_rate_event",
        ),
        "robot_constraint_diagnostics": {
            "joint_limit_soft_counts": _sum_nested_int_counts(
                results,
                "robot_constraint_diagnostics",
                "joint_limit_soft_counts",
            ),
            "joint_limit_hard_counts": _sum_nested_int_counts(
                results,
                "robot_constraint_diagnostics",
                "joint_limit_hard_counts",
            ),
            "joint_limit_hard_episode_events": _sum_nested_int(
                results,
                "robot_constraint_diagnostics",
                "joint_limit_hard_episode_events",
            ),
            "finger_joint_limit_saturation_counts": _sum_nested_int_counts(
                results,
                "robot_constraint_diagnostics",
                "finger_joint_limit_saturation_counts",
            ),
            "joint_limit_max_excess_by_joint": _max_nested_float_mapping(
                results,
                "robot_constraint_diagnostics",
                "joint_limit_max_excess_by_joint",
            ),
            "coverage": {
                "hard_limit_source": _diagnostic_boolean_coverage(
                    results, "joint_limit_hard_source_available"
                ),
                "joint_role_manifest": _diagnostic_boolean_coverage(
                    results, "joint_role_manifest_available"
                ),
            },
            "hard_limit_sources": _diagnostic_string_counts(
                results, "joint_limit_hard_source"
            ),
            "soft_limit_sources": _diagnostic_string_counts(
                results, "joint_limit_soft_source"
            ),
            "unknown_runtime_joint_names": _diagnostic_string_union(
                results, "unknown_runtime_joint_names"
            ),
            "baseline_checks": _baseline_joint_limit_summary(results),
        },
        "avg_current_chain_depth": _mean_or_zero(r.current_chain_depth for r in results),
        "avg_latched_chain_depth": _mean_or_zero(r.latched_chain_depth for r in results),
        "avg_steps": mean(r.steps for r in results),
        "avg_steps_to_success": mean(steps_to_success) if steps_to_success else None,
        "avg_policy_steps_to_success": mean(policy_steps_to_success) if policy_steps_to_success else None,
        "avg_policy_queries_to_success": (
            mean(policy_queries_to_success) if policy_queries_to_success else None
        ),
        "avg_time_to_success_s": mean(time_to_success) if time_to_success else None,
        "error_count": sum(1 for r in results if r.error is not None),
    }
    by_family: dict[str, list[EpisodeResult]] = {}
    for result in results:
        by_family.setdefault(result.task_family or "unknown", []).append(result)
    summary["task_family_terminal_success_rate"] = {
        family: _mean_or_zero(item.terminal_success_rate for item in items)
        for family, items in sorted(by_family.items())
    }
    if rs_at_p:
        summary["robust_success_rate"] = rs_at_p
    summary["violation_counts"] = _sum_violation_counts(results)
    summary["core_metrics"] = {
        "completion": {
            "success_rate": summary["success_rate"],
            "success_rate_ci": success_rate_ci,
        },
        "progress": {
            "latched_stage_completion_rate": summary["latched_stage_completion_rate"],
        },
        "efficiency": {
            "avg_time_to_success_s": summary["avg_time_to_success_s"],
        },
        "safety": {
            "safe_success_rate": summary["safe_success_rate"],
            "hard_violation_rate": summary["hard_violation_rate"],
            "drop_rate": summary["drop_rate"],
            "high_speed_violation_rate": summary["high_speed_violation_rate"],
            "safety_violation_step_rate": summary["safety_violation_step_rate"],
        },
        "robustness": rs_at_p,
    }
    summary["diagnostic_metrics"] = {
        "completion": {
            "ever_instant_success_rate": summary["ever_instant_success_rate"],
            "at_end_success_rate": summary["at_end_success_rate"],
            "task_family_terminal_success_rate": summary["task_family_terminal_success_rate"],
        },
        "progress": {
            "current_stage_completion_rate": summary["current_stage_completion_rate"],
            "current_chain_depth_progress_score": summary["current_chain_depth_progress_score"],
            "latched_chain_depth_progress_score": summary["latched_chain_depth_progress_score"],
            "avg_current_chain_depth": summary["avg_current_chain_depth"],
            "avg_latched_chain_depth": summary["avg_latched_chain_depth"],
        },
        "efficiency": {
            "task_efficiency": summary["task_efficiency"],
            "expert_time_s": summary["expert_time_s"],
            "expert_time_step": summary.get("expert_time_step"),
            "avg_steps_to_success": summary["avg_steps_to_success"],
            "avg_policy_steps_to_success": summary["avg_policy_steps_to_success"],
            "avg_policy_queries_to_success": summary["avg_policy_queries_to_success"],
        },
        "safety": {
            "safety_violation_events_per_step": summary["safety_violation_events_per_step"],
            "violation_counts": summary["violation_counts"],
            "robot_constraint_violation_rate": summary["robot_constraint_violation_rate"],
            "finger_joint_limit_saturation_rate": summary["finger_joint_limit_saturation_rate"],
            "joint_limit_active_step_rate": summary["joint_limit_active_step_rate"],
            "finger_joint_limit_saturation_step_rate": summary[
                "finger_joint_limit_saturation_step_rate"
            ],
            "max_joint_limit_excess_rad": summary["max_joint_limit_excess_rad"],
            "robot_constraint_diagnostics": summary["robot_constraint_diagnostics"],
        },
        "robot_motion": summary["robot_motion_metrics"],
        "grasp": {
            "kinematic_grasp_stability_index": summary["kinematic_grasp_stability_index"],
            "mean_kinematic_grasp_stability": summary["mean_kinematic_grasp_stability"],
            "per_object_best_kinematic_grasp_stability": summary[
                "per_object_best_kinematic_grasp_stability"
            ],
            "grasp_gsi_diagnostics": summary["grasp_gsi_diagnostics"],
        },
        "tool": {
            "tool_selection_accuracy": summary["tool_selection_accuracy"],
            "tool_switch_success_rate": summary["tool_switch_success_rate"],
            "tool_switch_total": summary["tool_switch_total"],
            "tool_switch_successful": summary["tool_switch_successful"],
        },
        "runtime": {
            "avg_steps": summary["avg_steps"],
            "error_count": summary["error_count"],
        },
    }
    return summary


_COVARIANT_AXES = {"object_pose", "table_height"}
_INVARIANT_AXES = {"background", "table_surface", "light", "clutter", "camera"}


def _robustness_category_for_result(result: EpisodeResult) -> str:
    """Map old axis strings and new protocol metadata into none/cov/inv/inv+cov."""
    sample = result.scene_generalization_sample or {}
    if isinstance(sample, dict):
        profile = str(sample.get("_generalization_profile") or "").strip().lower()
        if profile in {"cov_only", "cov-only", "cov"}:
            return "cov"
        if profile in {"inv_only", "inv-only", "inv"}:
            return "inv"
        if profile in {"inv_cov", "inv+cov", "full"}:
            return "inv+cov"

    raw = str(result.perturbation_axis or "none").strip().lower()
    if raw in {"", "none"}:
        return "none"
    if raw in {"cov", "cov_only", "cov-only"}:
        return "cov"
    if raw in {"inv", "inv_only", "inv-only"}:
        return "inv"
    if raw in {"inv_cov", "inv+cov", "full"}:
        return "inv+cov"

    axes = {axis.strip() for axis in raw.split(",") if axis.strip()}
    has_inv = bool(axes & _INVARIANT_AXES)
    has_cov = bool(axes & _COVARIANT_AXES)
    if has_inv and has_cov:
        return "inv+cov"
    if has_inv:
        return "inv"
    if has_cov:
        return "cov"
    return "inv+cov"


def _success_rate_bucket(group: list[EpisodeResult], baseline_sr: float | None = None) -> dict[str, Any]:
    values = [1.0 if result.success else 0.0 for result in group]
    ci_mean, ci_low, ci_high = bootstrap_ci(values)
    success_rate = _mean_or_zero(values)
    bucket: dict[str, Any] = {
        "n_episodes": len(group),
        "success_rate": success_rate,
        "success_rate_ci": {"mean": ci_mean, "ci_low": ci_low, "ci_high": ci_high},
    }
    if baseline_sr is not None and baseline_sr > 0:
        bucket["robust_ratio"] = success_rate / baseline_sr
    else:
        bucket["robust_ratio"] = None
    return bucket


def _compute_rs_at_p(results: List[EpisodeResult]) -> dict[str, Any]:
    if not results:
        return {}

    category_groups: dict[str, list[EpisodeResult]] = {}
    for result in results:
        category = _robustness_category_for_result(result)
        category_groups.setdefault(category, []).append(result)

    baseline = category_groups.get("none", [])
    baseline_values = [1.0 if result.success else 0.0 for result in baseline]
    if baseline_values:
        baseline_sr = _mean_or_zero(baseline_values)
        bl_ci_mean, bl_ci_low, bl_ci_high = bootstrap_ci(baseline_values)
        baseline_sr_ci: dict[str, float] | None = {"mean": bl_ci_mean, "ci_low": bl_ci_low, "ci_high": bl_ci_high}
    else:
        baseline_sr = None
        baseline_sr_ci = None

    per_category: dict[str, dict[str, Any]] = {}
    for category in ("cov", "inv", "inv+cov"):
        group = category_groups.get(category, [])
        if not group:
            continue
        per_category[category] = _success_rate_bucket(group, baseline_sr=baseline_sr)

    perturbed = [result for result in results if _robustness_category_for_result(result) != "none"]
    perturbed_values = [1.0 if result.success else 0.0 for result in perturbed]
    if perturbed_values:
        overall_sr = _mean_or_zero(perturbed_values)
        ov_ci_mean, ov_ci_low, ov_ci_high = bootstrap_ci(perturbed_values)
        overall_sr_ci: dict[str, float] | None = {"mean": ov_ci_mean, "ci_low": ov_ci_low, "ci_high": ov_ci_high}
    else:
        overall_sr = None
        overall_sr_ci = None

    # A per-channel summary cannot define robustness: the ratio requires both
    # the baseline and at least one perturbation bucket.  Returning partial
    # baseline-only or perturbation-only structures produced authoritative-
    # looking null ratios in every channel summary.  Cross-channel aggregation
    # is handled after the channel set is complete.
    if baseline_sr is None or overall_sr is None:
        return {}

    robust_ratio = None
    if overall_sr is not None and baseline_sr is not None and baseline_sr > 0:
        robust_ratio = overall_sr / baseline_sr

    return {
        "baseline_success_rate": baseline_sr,
        "baseline_success_rate_ci": baseline_sr_ci,
        "baseline_n": len(baseline),
        "per_category": per_category,
        "overall_perturbed_success_rate": overall_sr,
        "overall_perturbed_success_rate_ci": overall_sr_ci,
        "perturbed_n": len(perturbed),
        "robust_ratio": robust_ratio,
    }
