"""Helpers for serializing benchmark metrics into episode metadata."""

from __future__ import annotations

from typing import Any

from benchmark.metrics import METRIC_SCHEMA_VERSION


def build_metrics_episode_payload(metric_result: Any) -> dict[str, Any]:
    """Return the per-episode metrics payload persisted under /metrics/episode."""
    return {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "stable_success": metric_result.stable_success,
        "ever_instant_success": metric_result.ever_instant_success,
        "at_end_success_observed": metric_result.at_end_success_observed,
        "at_end_success_budget": metric_result.at_end_success_budget,
        "first_success_step": metric_result.first_success_step,
        "first_stable_success_step": metric_result.first_stable_success_step,
        "steps_to_stable_success": metric_result.steps_to_stable_success,
        "policy_steps_to_stable_success": metric_result.policy_steps_to_stable_success,
        "policy_queries_to_stable_success": metric_result.policy_queries_to_stable_success,
        "time_to_stable_success_s": metric_result.time_to_stable_success_s,
        "expert_time_s": metric_result.expert_time_s,
        "expert_time_step": metric_result.expert_time_step,
        "stage_completion_rate": metric_result.stage_completion_rate,
        "normalized_progress_score": metric_result.normalized_progress_score,
        "current_stage_completion_rate": metric_result.current_stage_completion_rate,
        "current_normalized_progress_score": metric_result.current_normalized_progress_score,
        "current_chain_depth": metric_result.current_chain_depth,
        "latched_stage_completion_rate": metric_result.latched_stage_completion_rate,
        "latched_normalized_progress_score": metric_result.latched_normalized_progress_score,
        "latched_chain_depth": metric_result.latched_chain_depth,
        "task_efficiency": metric_result.task_efficiency,
        "safety_violation_rate": metric_result.safety_violation_rate,
        "safety_violation_rate_time": metric_result.safety_violation_rate_time,
        "safety_violation_rate_event": metric_result.safety_violation_rate_event,
        "safety_violation_step_rate": metric_result.safety_violation_step_rate,
        "safety_violation_events_per_step": metric_result.safety_violation_events_per_step,
        "safety_hard_violation": metric_result.safety_hard_violation,
        "drop_violation": metric_result.drop_violation,
        "high_speed_violation": metric_result.high_speed_violation,
        "excessive_impact_violation": metric_result.excessive_impact_violation,
        "robot_constraint_violation": metric_result.robot_constraint_violation,
        "finger_joint_limit_saturation": metric_result.finger_joint_limit_saturation,
        "robot_constraint_diagnostics": metric_result.robot_constraint_diagnostics,
        "joint_limit_active_steps": metric_result.joint_limit_active_steps,
        "joint_limit_active_step_rate": metric_result.joint_limit_active_step_rate,
        "finger_joint_limit_saturation_steps": metric_result.finger_joint_limit_saturation_steps,
        "finger_joint_limit_saturation_step_rate": metric_result.finger_joint_limit_saturation_step_rate,
        "max_joint_limit_excess_rad": metric_result.max_joint_limit_excess_rad,
        "max_joint_limit_excess_joint": metric_result.max_joint_limit_excess_joint,
        "impact_metric_available": metric_result.impact_metric_available,
        "episode_violation_rate": metric_result.episode_violation_rate,
        "violation_counts": metric_result.violation_counts,
        "policy_query_count": metric_result.policy_query_count,
        "terminated_reason": metric_result.terminated_reason,
    }
