"""Write benchmark results to disk."""

from __future__ import annotations

import json
import os
from dataclasses import fields
from pathlib import Path
from typing import Any, List

from .metrics import EpisodeResult, summarize


def _write_json(path: Path, payload: object) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _ordered_summary_for_output(summary: dict[str, Any], policy_name: str | None = None) -> dict[str, Any]:
    """Arrange summary JSON for human-facing benchmark output.

    Grouped aggregate JSON (summary.json, per_task.json) omits compatibility
    aliases. per_episode.jsonl remains verbose for audit and backward
    compatibility. Flat fields in summarize() that duplicate grouped fields
    are not included in the output files.
    """
    robustness = summary.get("robust_success_rate") or summary.get("core_metrics", {}).get("robustness") or {}
    ordered_robustness = {}
    if robustness:
        ordered_robustness = {
            "baseline_success_rate": robustness.get("baseline_success_rate"),
            "baseline_success_rate_ci": robustness.get("baseline_success_rate_ci"),
            "baseline_n": robustness.get("baseline_n"),
            "overall_perturbed_success_rate": robustness.get("overall_perturbed_success_rate"),
            "overall_perturbed_success_rate_ci": robustness.get("overall_perturbed_success_rate_ci"),
            "perturbed_n": robustness.get("perturbed_n"),
            "robust_ratio": robustness.get("robust_ratio"),
            "per_category": robustness.get("per_category", robustness.get("per_axis", {})),
        }
    metadata: dict[str, Any] = {
        "policy_name": policy_name,
        "base_seed": summary.get("base_seed"),
        "seed_policy": summary.get("seed_policy"),
        "randomization_recorded": summary.get("randomization_recorded"),
        "n_episodes": summary.get("n_episodes", summary.get("total")),
        "successes": summary.get("successes"),
        "error_count": summary.get("error_count"),
        "evaluation_protocol": summary.get("evaluation_protocol"),
        "early_stop": summary.get("early_stop"),
        "success_definition": summary.get("success_definition"),
        "measurement_window": summary.get("measurement_window"),
        "max_episode_steps": summary.get("max_episode_steps"),
        "dwell_time_s": summary.get("dwell_time_s"),
        "field_semantics": summary.get("field_semantics", {}),
    }
    if policy_name is None:
        metadata.pop("policy_name")

    ordered = {
        "metric_schema_version": summary.get("metric_schema_version", "core-v1"),
        "metadata": metadata,
        "core_metrics": {
            "completion": {
                "success_rate": summary.get("success_rate"),
                "success_rate_ci": summary.get("success_rate_ci") or summary.get("stable_success_rate_ci"),
            },
            "progress": {
                "latched_stage_completion_rate": summary.get("latched_stage_completion_rate"),
            },
            "efficiency": {
                "avg_time_to_success_s": summary.get("avg_time_to_success_s"),
            },
            "safety": {
                "safe_success_rate": summary.get("safe_success_rate"),
                "hard_violation_rate": summary.get("hard_violation_rate"),
                "drop_rate": summary.get("drop_rate"),
                "high_speed_violation_rate": summary.get("high_speed_violation_rate"),
                "safety_violation_step_rate": summary.get("safety_violation_step_rate"),
            },
            "robustness": ordered_robustness,
        },
        "diagnostic_metrics": {
            "completion": {
                "ever_instant_success_rate": summary.get("ever_instant_success_rate"),
                "at_end_success_rate": summary.get("at_end_success_rate"),
                "task_family_success_rate": summary.get("task_family_terminal_success_rate", {}),
            },
            "progress": {
                "current_stage_completion_rate": summary.get("current_stage_completion_rate"),
                "current_chain_depth_progress_score": summary.get("current_chain_depth_progress_score"),
                "latched_chain_depth_progress_score": summary.get("latched_chain_depth_progress_score"),
                "avg_current_chain_depth": summary.get("avg_current_chain_depth"),
                "avg_latched_chain_depth": summary.get("avg_latched_chain_depth"),
            },
            "efficiency": {
                "task_efficiency": summary.get("task_efficiency"),
                "expert_time_s": summary.get("expert_time_s"),
                "expert_time_step": summary.get("expert_time_step"),
                "avg_steps_to_success": summary.get("avg_steps_to_success"),
                "avg_policy_steps_to_success": summary.get("avg_policy_steps_to_success"),
                "avg_policy_queries_to_success": summary.get("avg_policy_queries_to_success"),
            },
            "safety": {
                "safety_violation_events_per_step": summary.get("safety_violation_events_per_step"),
                "violation_counts": summary.get("violation_counts", {}),
                "robot_constraint_violation_rate": summary.get(
                    "robot_constraint_violation_rate"
                ),
                "finger_joint_limit_saturation_rate": summary.get("finger_joint_limit_saturation_rate"),
                "joint_limit_active_step_rate": summary.get("joint_limit_active_step_rate"),
                "finger_joint_limit_saturation_step_rate": summary.get(
                    "finger_joint_limit_saturation_step_rate"
                ),
                "max_joint_limit_excess_rad": summary.get("max_joint_limit_excess_rad"),
                "robot_constraint_diagnostics": summary.get("robot_constraint_diagnostics", {}),
            },
            "robot_motion": summary.get("robot_motion_metrics", {}),
            "grasp": {
                "kinematic_grasp_stability_index": summary.get("kinematic_grasp_stability_index"),
                "mean_kinematic_grasp_stability": summary.get("mean_kinematic_grasp_stability"),
                "per_object_best_kinematic_grasp_stability": summary.get(
                    "per_object_best_kinematic_grasp_stability",
                    {},
                ),
                "grasp_gsi_diagnostics": summary.get("grasp_gsi_diagnostics", {}),
            },
            "tool": {
                "tool_selection_accuracy": summary.get("tool_selection_accuracy"),
                "tool_switch_success_rate": summary.get("tool_switch_success_rate"),
                "tool_switch_total": summary.get("tool_switch_total"),
                "tool_switch_successful": summary.get("tool_switch_successful"),
            },
            "runtime": {
                "avg_steps": summary.get("avg_steps"),
                "error_count": summary.get("error_count"),
            },
        },
    }
    return ordered


def _write_summary_files(output_dir: Path, policy_name: str, results: List[EpisodeResult]) -> None:
    per_task: dict[str, list[EpisodeResult]] = {}
    for r in results:
        per_task.setdefault(r.scene, []).append(r)
    per_task_summary = {
        scene: _ordered_summary_for_output(summarize(items)) for scene, items in per_task.items()
    }
    _write_json(output_dir / "per_task.json", per_task_summary)

    overall = _ordered_summary_for_output(summarize(results), policy_name=policy_name)
    _write_json(output_dir / "summary.json", overall)


def _episode_result_from_dict(payload: dict[str, Any]) -> EpisodeResult:
    valid_fields = {field.name for field in fields(EpisodeResult)}
    kwargs = {key: value for key, value in payload.items() if key in valid_fields}
    if "metric_schema_version" not in payload:
        kwargs["metric_schema_version"] = "legacy-unversioned"
    # Existing per-episode rows may contain large traces. They are not needed
    # for aggregate summaries and keeping them would defeat chunked evaluation.
    kwargs["timeseries"] = None
    return EpisodeResult(**kwargs)


def episode_result_from_dict(payload: dict[str, Any]) -> EpisodeResult:
    """Build a compact EpisodeResult from a serialized per-episode row."""
    return _episode_result_from_dict(payload)


def load_episode_results(output_dir: Path | str) -> list[EpisodeResult]:
    """Load compact episode results from an existing per_episode.jsonl file."""
    output_dir = Path(output_dir)
    path = output_dir / "per_episode.jsonl"
    if not path.exists():
        return []

    results: list[EpisodeResult] = []
    with open(path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise TypeError(f"expected object, got {type(payload).__name__}")
                results.append(_episode_result_from_dict(payload))
            except Exception as exc:
                raise ValueError(f"Failed to load {path}:{line_no}: {exc}") from exc
    return results


def initialize_incremental_results(output_dir: Path | str, *, append: bool = False) -> None:
    """Create output directory and prepare the append-only per-episode file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(output_dir / "per_episode.jsonl", mode) as f:
        f.flush()
        os.fsync(f.fileno())


def append_episode_result(
    output_dir: Path | str,
    policy_name: str,
    result: EpisodeResult,
    results: List[EpisodeResult],
) -> None:
    """Append one episode result and refresh aggregate summary files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "per_episode.jsonl", "a") as f:
        f.write(json.dumps(result.to_dict()) + "\n")
        f.flush()
        os.fsync(f.fileno())

    _write_summary_files(output_dir, policy_name, results)


def write_summary_files(output_dir: Path | str, policy_name: str, results: List[EpisodeResult]) -> None:
    """Write per_task.json and summary.json without rewriting per_episode.jsonl."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_summary_files(output_dir, policy_name, results)


def write_results(output_dir: Path | str, policy_name: str, results: List[EpisodeResult]) -> None:
    """Write per-episode, per-task, and overall summary files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_episode = [r.to_dict() for r in results]
    with open(output_dir / "per_episode.jsonl", "w") as f:
        for row in per_episode:
            f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())

    _write_summary_files(output_dir, policy_name, results)
