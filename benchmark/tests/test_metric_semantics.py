from pathlib import Path

import pytest
import yaml

from benchmark.grasp_detector import GraspDetector
from benchmark.metric_tracker import MetricTracker
from benchmark.metrics import EpisodeResult, summarize
from benchmark.results import _ordered_summary_for_output
from benchmark.stage_tracker import StageTracker
from benchmark.tool_tracker import ToolTracker


def _object_state(z, lin=(0.0, 0.0, 0.0), ang=(0.0, 0.0, 0.0)):
    return {
        "pose_world": [0.0, 0.0, z],
        "lin_vel_world": list(lin),
        "ang_vel_world": list(ang),
    }


def test_scene_metric_stages_are_top_level_not_nested_under_terminal():
    scene_dir = Path(__file__).resolve().parents[2] / "scenes"
    offenders = []
    for scene_path in sorted(scene_dir.glob("*.yaml")):
        scene = yaml.safe_load(scene_path.read_text()) or {}
        metrics = scene.get("metrics") or {}
        terminal = metrics.get("terminal") or {}
        if isinstance(terminal, dict) and "stages" in terminal:
            offenders.append(scene_path.name)

    assert offenders == []


def test_drop_events_count_edges_not_active_duration():
    tracker = MetricTracker(
        {"safety": {"table_z": 0.75, "drop": {"enabled": True, "tracked_objects": ["obj"]}}},
        dt=0.1,
    )

    tracker.update({"obj": _object_state(0.5)}, sim_step=0)
    tracker.update({"obj": _object_state(0.5)}, sim_step=1)
    tracker.update({"obj": _object_state(0.8)}, sim_step=2)
    tracker.update({"obj": _object_state(0.5)}, sim_step=3)
    tracker.update({"obj": _object_state(0.5)}, sim_step=4)

    result = tracker.finalize()

    assert result.drop_violation is True
    assert result.violation_counts["drop_obj"] == 2
    assert result.safety_violation_step_rate == 4 / 5
    assert result.safety_violation_events_per_step == 2 / 5


def test_safety_events_per_step_is_not_a_bounded_rate():
    tracker = MetricTracker(
        {"safety": {"table_z": 0.75, "drop": {"enabled": True, "tracked_objects": ["obj_a", "obj_b"]}}},
        dt=0.1,
    )

    tracker.update(
        {
            "obj_a": _object_state(0.5),
            "obj_b": _object_state(0.5),
        },
        sim_step=0,
    )

    result = tracker.finalize()

    assert result.safety_violation_step_rate == 1.0
    assert result.safety_violation_events_per_step == 2.0
    assert result.safety_violation_rate_event == result.safety_violation_events_per_step


def test_off_table_drop_respects_drop_tracked_objects():
    tracker = MetricTracker(
        {"safety": {"table_z": 0.75, "drop": {"enabled": True, "tracked_objects": ["tracked"]}}},
        dt=0.1,
    )

    tracker.update(
        {
            "tracked": _object_state(0.5),
            "clutter": _object_state(0.5),
        },
        sim_step=0,
    )
    result = tracker.finalize()

    assert result.drop_violation is True
    assert result.violation_counts == {"drop_tracked": 1}
    assert result.safety_violation_events_per_step == 1.0


def test_off_table_drop_disabled_without_drop_enabled():
    tracker = MetricTracker({"safety": {"table_z": 0.75}}, dt=0.1)

    tracker.update({"obj": _object_state(0.5)}, sim_step=0)
    result = tracker.finalize()

    assert result.drop_violation is False
    assert result.safety_hard_violation is False
    assert result.violation_counts == {}


def test_drop_defaults_to_grasp_tracked_objects():
    tracker = MetricTracker(
        {
            "grasp": {"enabled": True, "tracked_objects": ["tracked"]},
            "safety": {"table_z": 0.75},
        },
        dt=0.1,
    )

    tracker.update(
        {
            "tracked": _object_state(0.5),
            "untracked": _object_state(0.5),
        },
        sim_step=0,
    )
    result = tracker.finalize()

    assert result.drop_violation is True
    assert result.violation_counts == {"drop_tracked": 1}


def test_allowed_drop_placement_exempts_fast_release_into_target():
    tracker = MetricTracker(
        {
            "safety": {
                "table_z": 0.75,
                "drop": {
                    "enabled": True,
                    "tracked_objects": ["obj"],
                    "allowed_placed_conditions": {
                        "obj": {
                            "type": "object_inside",
                            "object": "obj",
                            "container": "box",
                            "tolerance": 0.25,
                        }
                    },
                },
            }
        },
        dt=0.1,
    )

    tracker.update(
        {
            "obj": {"pose_world": [0.0, 0.0, 0.75], "lin_vel_world": [0.0, 0.0, 0.0], "ang_vel_world": [0, 0, 0]},
            "box": _object_state(0.75),
        },
        sim_step=0,
    )
    tracker.update(
        {
            "obj": {"pose_world": [0.0, 0.0, 0.90], "lin_vel_world": [0.0, 0.0, -0.5], "ang_vel_world": [0, 0, 0]},
            "box": _object_state(0.75),
        },
        sim_step=1,
    )
    tracker.update(
        {
            "obj": {"pose_world": [0.0, 0.0, 0.76], "lin_vel_world": [0.0, 0.0, -0.5], "ang_vel_world": [0, 0, 0]},
            "box": _object_state(0.75),
        },
        sim_step=2,
    )

    result = tracker.finalize()

    assert result.drop_violation is False
    assert result.violation_counts == {}


def test_high_speed_violation_replaces_contact_force_impact():
    tracker = MetricTracker(
        {"safety": {"high_speed": {"enabled": True, "threshold_mps": 1.0}}},
        dt=0.1,
    )

    tracker.update({"obj": _object_state(0.8, lin=(1.2, 0.0, 0.0))}, sim_step=0)
    tracker.update({"obj": _object_state(0.8, lin=(1.2, 0.0, 0.0))}, sim_step=1)

    result = tracker.finalize()

    assert result.high_speed_violation is True
    assert result.safety_hard_violation is True
    assert result.excessive_impact_violation is None
    assert result.impact_metric_available is False
    assert result.violation_counts["high_speed_obj"] == 1
    assert result.safety_violation_step_rate == 1.0

    summary = summarize(
        [
            EpisodeResult(
                scene="scene",
                episode_index=0,
                seed=0,
                success=False,
                steps=2,
                safety_hard_violation=result.safety_hard_violation,
                high_speed_violation=result.high_speed_violation,
                excessive_impact_violation=result.excessive_impact_violation,
                impact_metric_available=result.impact_metric_available,
            )
        ]
    )
    assert summary["high_speed_violation_rate"] == 1.0
    assert summary["core_metrics"]["safety"]["high_speed_violation_rate"] == 1.0
    assert "excessive_impact_rate" not in summary["core_metrics"]["safety"]
    assert "episode_violation_rate" not in summary["core_metrics"]["safety"]


def test_core_progress_uses_latched_stage_completion():
    summary = summarize(
        [
            EpisodeResult(
                scene="scene",
                episode_index=0,
                seed=0,
                success=False,
                steps=10,
                current_stage_completion_rate=0.0,
                latched_stage_completion_rate=0.75,
            )
        ]
    )

    assert summary["core_metrics"]["progress"]["latched_stage_completion_rate"] == 0.75
    assert "current_stage_completion_rate" not in summary["core_metrics"]["progress"]
    assert summary["diagnostic_metrics"]["progress"]["current_stage_completion_rate"] == 0.0


def test_current_progress_uses_current_predicate_with_latched_dependencies():
    tracker = StageTracker(
        [
            {
                "id": "open",
                "depends_on": [],
                "success_condition": {
                    "type": "joint_state",
                    "object": "door",
                    "joint": "joint_0",
                    "target": "open",
                    "tolerance": 0.5,
                },
            },
            {
                "id": "close",
                "depends_on": ["open"],
                "success_condition": {
                    "type": "joint_state",
                    "object": "door",
                    "joint": "joint_0",
                    "target": "closed",
                    "tolerance": 0.5,
                },
            },
        ],
        dt=0.1,
    )

    opened = tracker.update(
        {"door": {"qpos": {"joint_0": 1.0}}},
        sim_step=0,
    )
    closed = tracker.update(
        {"door": {"qpos": {"joint_0": 0.0}}},
        sim_step=1,
    )

    assert opened.current_completed == {"open": True, "close": False}
    assert closed.completed == {"open": True, "close": True}
    assert closed.current_completed == {"open": False, "close": True}
    assert closed.current_stage_completion_rate == 0.5
    assert closed.current_chain_depth == 2
    assert closed.current_normalized_progress_score == 1.0


def test_zero_current_progress_is_not_replaced_by_latched_progress():
    result = EpisodeResult(
        scene="scene",
        episode_index=0,
        seed=0,
        success=True,
        steps=1,
        normalized_progress_score=1.0,
        current_normalized_progress_score=0.0,
        latched_normalized_progress_score=1.0,
        # Reproduce the bad alias persisted by the pre-v1.7 writer.  Aggregate
        # output must use the canonical current field instead.
        current_chain_depth_progress_score=1.0,
    )

    payload = EpisodeResult(
        scene="scene",
        episode_index=1,
        seed=1,
        success=True,
        steps=1,
        normalized_progress_score=1.0,
        current_normalized_progress_score=0.0,
        latched_normalized_progress_score=1.0,
    ).to_dict()
    summary = summarize([result])

    assert payload["chain_depth_progress_score"] == 1.0
    assert payload["current_chain_depth_progress_score"] == 0.0
    assert summary["current_chain_depth_progress_score"] == 0.0


def test_grouped_safety_output_omits_redundant_rate_aliases():
    summary = summarize(
        [
            EpisodeResult(
                scene="scene",
                episode_index=0,
                seed=0,
                success=False,
                steps=10,
                safety_hard_violation=True,
                episode_violation_rate=1.0,
                safety_violation_rate_time=0.2,
                safety_violation_step_rate=0.2,
                safety_violation_rate_event=0.3,
                safety_violation_events_per_step=0.3,
            )
        ]
    )

    assert summary["episode_violation_rate"] == 1.0
    assert summary["safety_violation_rate_time"] == 0.2
    assert summary["safety_violation_rate_event"] == 0.3
    assert summary["core_metrics"]["safety"]["hard_violation_rate"] == 1.0
    assert "episode_violation_rate" not in summary["core_metrics"]["safety"]
    assert summary["core_metrics"]["safety"]["safety_violation_step_rate"] == 0.2
    assert summary["diagnostic_metrics"]["safety"]["safety_violation_events_per_step"] == 0.3
    assert "safety_violation_rate_time" not in summary["diagnostic_metrics"]["safety"]
    assert "safety_violation_rate_event" not in summary["diagnostic_metrics"]["safety"]

    ordered = _ordered_summary_for_output(summary)
    assert "episode_violation_rate" not in ordered["core_metrics"]["safety"]
    assert "safety_violation_rate_time" not in ordered["diagnostic_metrics"]["safety"]
    assert "safety_violation_rate_event" not in ordered["diagnostic_metrics"]["safety"]


def test_event_density_is_pooled_over_executed_steps():
    summary = summarize(
        [
            EpisodeResult(
                scene="scene",
                episode_index=0,
                seed=0,
                success=False,
                steps=1,
                safety_violation_events_per_step=1.0,
            ),
            EpisodeResult(
                scene="scene",
                episode_index=1,
                seed=1,
                success=False,
                steps=9,
                safety_violation_events_per_step=0.0,
            ),
        ]
    )

    assert summary["safety_violation_events_per_step"] == pytest.approx(0.1)
    assert summary["safety_violation_rate_event"] == pytest.approx(0.1)


def test_finger_joint_limit_saturation_is_diagnostic_in_summary():
    summary = summarize(
        [
            EpisodeResult(
                scene="scene",
                episode_index=0,
                seed=0,
                success=True,
                steps=10,
                safety_hard_violation=False,
                robot_constraint_violation=False,
                finger_joint_limit_saturation=True,
                joint_limit_active_step_rate=0.5,
                finger_joint_limit_saturation_step_rate=0.1,
                max_joint_limit_excess_rad=0.12,
                robot_constraint_diagnostics={
                    "finger_joint_limit_saturation_counts": {"right_thumb_2_joint": 1},
                    "joint_limit_max_excess_by_joint": {"right_thumb_2_joint": 0.12},
                },
            )
        ]
    )

    assert summary["safe_success_rate"] == 1.0
    assert summary["hard_violation_rate"] == 0.0
    assert summary["robot_constraint_violation_rate"] == 0.0
    assert summary["finger_joint_limit_saturation_rate"] == 1.0
    assert "robot_constraint_violation_rate" not in summary["core_metrics"]["safety"]
    assert "finger_joint_limit_saturation_rate" not in summary["core_metrics"]["safety"]
    diagnostic = summary["diagnostic_metrics"]["safety"]
    assert diagnostic["robot_constraint_violation_rate"] == 0.0
    assert diagnostic["finger_joint_limit_saturation_rate"] == 1.0
    assert diagnostic["joint_limit_active_step_rate"] == 0.5
    assert diagnostic["finger_joint_limit_saturation_step_rate"] == 0.1
    assert diagnostic["max_joint_limit_excess_rad"] == 0.12


def test_hard_joint_limit_keeps_safe_success_and_stays_out_of_core_safety():
    summary = summarize(
        [
            EpisodeResult(
                scene="scene",
                episode_index=0,
                seed=0,
                success=True,
                steps=10,
                safety_hard_violation=False,
                robot_constraint_violation=True,
                robot_constraint_diagnostics={
                    "joint_limit_hard_episode_events": 2,
                    "joint_limit_hard_counts": {"joint_a": 2},
                },
            )
        ]
    )

    assert summary["safe_success_rate"] == 1.0
    assert summary["hard_violation_rate"] == 0.0
    assert "robot_constraint_violation_rate" not in summary["core_metrics"]["safety"]
    diagnostic = summary["diagnostic_metrics"]["safety"]
    assert diagnostic["robot_constraint_violation_rate"] == 1.0
    assert diagnostic["robot_constraint_diagnostics"]["joint_limit_hard_episode_events"] == 2
    ordered = _ordered_summary_for_output(summary)
    assert "robot_constraint_violation_rate" not in ordered["core_metrics"]["safety"]
    assert ordered["diagnostic_metrics"]["safety"]["robot_constraint_violation_rate"] == 1.0


def test_fixed_horizon_summary_uses_primary_success_not_stable_success():
    summary = summarize(
        [
            EpisodeResult(
                scene="scene",
                episode_index=0,
                seed=0,
                success=True,
                steps=10,
                stable_success=False,
                at_end_success=True,
                at_end_success_observed=True,
                at_end_success_budget=True,
                evaluation_protocol="fixed_horizon",
                early_stop=False,
            )
        ]
    )

    assert summary["success_rate"] == 1.0
    assert summary["success_rate_ci"]["mean"] == 1.0
    assert summary["stable_success_rate_ci"]["mean"] == 0.0
    assert summary["measurement_window"] == "episode_start_to_max_steps_or_error"


def test_robustness_summary_separates_inv_only_from_inv_cov():
    summary = summarize(
        [
            EpisodeResult(scene="scene", episode_index=0, seed=0, success=True, steps=1),
            EpisodeResult(
                scene="scene",
                episode_index=1,
                seed=1,
                success=True,
                steps=1,
                scene_generalization_sample={"_generalization_profile": "cov_only"},
            ),
            EpisodeResult(
                scene="scene",
                episode_index=2,
                seed=2,
                success=False,
                steps=1,
                scene_generalization_sample={"_generalization_profile": "inv_only"},
            ),
            EpisodeResult(
                scene="scene",
                episode_index=3,
                seed=3,
                success=True,
                steps=1,
                perturbation_axis="background,object_pose",
            ),
        ]
    )

    per_category = summary["core_metrics"]["robustness"]["per_category"]
    assert set(per_category) == {"cov", "inv", "inv+cov"}
    assert per_category["cov"]["success_rate"] == 1.0
    assert per_category["inv"]["success_rate"] == 0.0
    assert per_category["inv+cov"]["success_rate"] == 1.0

    ordered = _ordered_summary_for_output(summary)
    assert "inv" in ordered["core_metrics"]["robustness"]["per_category"]


def test_terminal_success_rate_follows_evaluation_protocol():
    metrics_spec = {
        "task_family": "always_true",
        "terminal": {
            "dwell_time_s": 0.5,
            "raw_condition": {"type": "all", "conditions": []},
        },
    }
    fixed_tracker = MetricTracker(metrics_spec, dt=0.1)
    fixed_tracker.update({}, sim_step=0)
    fixed_result = fixed_tracker.finalize(
        terminated_reason="max_steps",
        evaluation_protocol="fixed_horizon",
    )

    reach_tracker = MetricTracker(metrics_spec, dt=0.1)
    reach_tracker.update({}, sim_step=0)
    reach_result = reach_tracker.finalize(
        terminated_reason="max_steps",
        evaluation_protocol="reach_and_stop",
    )

    assert fixed_result.stable_success is False
    assert fixed_result.at_end_success_budget is True
    assert fixed_result.terminal_success_rate == 1.0
    assert reach_result.terminal_success_rate == 0.0

    summary = summarize(
        [
            EpisodeResult(
                scene="scene",
                episode_index=0,
                seed=0,
                success=True,
                steps=1,
                stable_success=fixed_result.stable_success,
                at_end_success_budget=fixed_result.at_end_success_budget,
                terminal_success_rate=fixed_result.terminal_success_rate,
                task_family="always_true",
                evaluation_protocol="fixed_horizon",
                early_stop=False,
            )
        ]
    )
    assert summary["task_family_terminal_success_rate"]["always_true"] == 1.0


def test_avg_steps_to_success_is_none_without_successes():
    no_success_summary = summarize(
        [
            EpisodeResult(
                scene="scene",
                episode_index=0,
                seed=0,
                success=False,
                steps=10,
            )
        ]
    )
    empty_summary = summarize([])

    assert no_success_summary["avg_steps_to_success"] is None
    assert no_success_summary["core_metrics"]["efficiency"] == {"avg_time_to_success_s": None}
    assert "avg_steps_to_success" not in no_success_summary["core_metrics"]["efficiency"]
    assert no_success_summary["diagnostic_metrics"]["efficiency"]["avg_steps_to_success"] is None
    assert no_success_summary["diagnostic_metrics"]["efficiency"]["avg_policy_steps_to_success"] is None
    assert no_success_summary["diagnostic_metrics"]["efficiency"]["avg_policy_queries_to_success"] is None
    assert empty_summary["avg_steps_to_success"] is None
    assert empty_summary["core_metrics"]["efficiency"] == {"avg_time_to_success_s": None}
    assert "avg_steps_to_success" not in empty_summary["core_metrics"]["efficiency"]
    assert empty_summary["diagnostic_metrics"]["efficiency"]["avg_steps_to_success"] is None
    assert empty_summary["diagnostic_metrics"]["efficiency"]["avg_policy_steps_to_success"] is None
    assert empty_summary["diagnostic_metrics"]["efficiency"]["avg_policy_queries_to_success"] is None


def test_core_efficiency_uses_time_and_keeps_steps_diagnostic():
    summary = summarize(
        [
            EpisodeResult(
                scene="scene",
                episode_index=0,
                seed=0,
                success=True,
                steps=10,
                steps_to_stable_success=8,
                policy_steps_to_stable_success=2,
                policy_queries_to_stable_success=1,
                time_to_stable_success_s=0.8,
                task_efficiency=1.25,
                expert_time_s=1.0,
            )
        ]
    )

    assert summary["core_metrics"]["efficiency"] == {"avg_time_to_success_s": 0.8}
    assert summary["diagnostic_metrics"]["efficiency"]["task_efficiency"] == 1.25
    assert summary["diagnostic_metrics"]["efficiency"]["expert_time_s"] == 1.0
    assert summary["diagnostic_metrics"]["efficiency"]["avg_steps_to_success"] == 8
    assert summary["diagnostic_metrics"]["efficiency"]["avg_policy_steps_to_success"] == 2
    assert summary["diagnostic_metrics"]["efficiency"]["avg_policy_queries_to_success"] == 1


def test_policy_steps_and_model_queries_to_success_are_distinct():
    spec = {
        "terminal": {
            "dwell_time_s": 0.2,
            "raw_condition": {"type": "all", "conditions": []},
        }
    }
    tracker = MetricTracker(spec, dt=0.1)
    tracker.update({}, sim_step=0)
    tracker.update({}, sim_step=1)
    reach_result = tracker.finalize(
        terminated_reason="stable_success",
        policy_query_count=7,
        policy_stride=3,
        evaluation_protocol="reach_and_stop",
    )

    fixed_tracker = MetricTracker(spec, dt=0.1)
    fixed_tracker.update({}, sim_step=0)
    fixed_tracker.update({}, sim_step=1)
    fixed_result = fixed_tracker.finalize(
        terminated_reason="max_steps",
        policy_query_count=11,
        policy_query_count_at_stable_success=3,
        policy_stride=3,
        evaluation_protocol="fixed_horizon",
    )

    assert reach_result.steps_to_stable_success == 2
    assert reach_result.policy_steps_to_stable_success == 1
    assert reach_result.policy_queries_to_stable_success == 7
    assert fixed_result.policy_steps_to_stable_success == 1
    assert fixed_result.policy_queries_to_stable_success == 3


def test_grasp_proxy_is_unavailable_without_table_z():
    detector = GraspDetector(table_z=None, min_hold_frames=1, dt=0.1)

    detector.update({"obj": _object_state(1.2)}, sim_step=0)
    detector.update({"obj": _object_state(1.2)}, sim_step=1)

    state = detector.get_grasp_state("obj")
    assert detector.proxy_available is False
    assert state is not None
    assert state.is_lifted is False
    assert state.is_held is False

    tracker = MetricTracker({"grasp": {"enabled": True}}, dt=0.1)
    tracker.update({"obj": _object_state(1.2)}, sim_step=0)
    tracker.update({"obj": _object_state(1.2)}, sim_step=1)
    result = tracker.finalize()

    assert result.kinematic_grasp_stability_index is None
    assert result.mean_kinematic_grasp_stability is None
    assert result.per_object_best_kinematic_grasp_stability == {}


def test_grasp_proxy_respects_tracked_objects():
    detector = GraspDetector(table_z=0.75, min_hold_frames=1, dt=0.1, tracked_objects=["target"])

    detector.update(
        {
            "target": _object_state(0.9),
            "container": _object_state(1.2),
        },
        sim_step=0,
    )
    detector.update(
        {
            "target": _object_state(0.9),
            "container": _object_state(1.2),
        },
        sim_step=1,
    )

    assert detector.get_grasp_state("target") is not None
    assert detector.get_grasp_state("container") is None

    tracker = MetricTracker(
        {
            "grasp": {"enabled": True, "tracked_objects": ["target"]},
            "safety": {"table_z": 0.75, "drop": {"enabled": False}},
        },
        dt=0.1,
    )
    tracker.update({"target": _object_state(0.9), "container": _object_state(1.2)}, sim_step=0)
    tracker.update({"target": _object_state(0.9), "container": _object_state(1.2)}, sim_step=1)
    result = tracker.finalize()

    assert set(result.per_object_best_kinematic_grasp_stability) == {"target"}
    assert result.timeseries is None


def test_metrics_timeseries_is_opt_in_and_preserves_online_episode_metrics():
    spec = {
        "safety": {"table_z": 0.75, "drop": {"enabled": True, "tracked_objects": ["obj"]}},
    }
    default_tracker = MetricTracker(spec, dt=0.1)
    traced_tracker = MetricTracker({**spec, "metrics_trace": True}, dt=0.1)
    for step in range(2):
        state = {"obj": _object_state(0.5)}
        default_tracker.update(state, sim_step=step)
        traced_tracker.update(state, sim_step=step)

    default_result = default_tracker.finalize()
    traced_result = traced_tracker.finalize()

    assert default_result.timeseries is None
    assert traced_result.timeseries is not None
    assert traced_result.timeseries["safety_violation"] == [True, True]
    assert default_result.safety_violation_step_rate == traced_result.safety_violation_step_rate == 1.0
    assert default_result.violation_counts == traced_result.violation_counts == {"drop_obj": 1}


def test_joint_limit_small_transient_is_diagnostic_not_hard_violation():
    tracker = MetricTracker({"safety": {}}, dt=0.1)
    robot_state = {
        "joint_names": ["jnt_a"],
        "qpos": [1.02],
        "qvel": [0.0],
    }
    limits = ([0.0], [1.0])

    for step in range(4):
        tracker.update({}, sim_step=step, robot_state=robot_state, joint_limits=limits)
    result = tracker.finalize()

    assert result.robot_constraint_violation is False
    assert result.safety_hard_violation is False
    assert result.violation_counts == {}
    assert result.robot_constraint_diagnostics["joint_limit_soft_counts"] == {"jnt_a": 4}


def test_joint_limit_hard_violation_requires_duration_or_critical_excess():
    duration_tracker = MetricTracker({"safety": {}}, dt=0.1)
    robot_state = {
        "joint_names": ["jnt_a"],
        "qpos": [1.06],
        "qvel": [0.0],
    }
    limits = ([0.0], [1.0])

    for step in range(10):
        duration_tracker.update({}, sim_step=step, robot_state=robot_state, joint_limits=limits)
    duration_result = duration_tracker.finalize()

    assert duration_result.robot_constraint_violation is True
    assert duration_result.safety_hard_violation is False
    assert duration_result.violation_counts == {}
    assert duration_result.robot_constraint_diagnostics["joint_limit_hard_episode_events"] == 1
    assert duration_result.robot_constraint_diagnostics["joint_limit_hard_counts"] == {"jnt_a": 1}
    assert duration_result.max_joint_limit_excess_rad == pytest.approx(0.06)
    assert duration_result.max_joint_limit_excess_joint == "jnt_a"

    critical_tracker = MetricTracker({"safety": {}}, dt=0.1)
    critical_tracker.update(
        {},
        sim_step=0,
        robot_state={"joint_names": ["jnt_a"], "qpos": [1.20], "qvel": [0.0]},
        joint_limits=limits,
    )
    critical_result = critical_tracker.finalize()

    assert critical_result.robot_constraint_violation is True
    assert critical_result.safety_hard_violation is False
    assert critical_result.violation_counts == {}
    assert critical_result.robot_constraint_diagnostics["joint_limit_hard_episode_events"] == 1


def test_finger_joint_limit_saturation_is_diagnostic_by_default():
    tracker = MetricTracker({"safety": {}}, dt=0.1)
    robot_state = {
        "joint_names": ["right_thumb_2_joint"],
        "qpos": [1.10],
        "qvel": [0.0],
    }
    limits = ([0.0], [1.0])

    for step in range(15):
        tracker.update({}, sim_step=step, robot_state=robot_state, joint_limits=limits)
    result = tracker.finalize()

    assert result.finger_joint_limit_saturation is True
    assert result.robot_constraint_violation is False
    assert result.safety_hard_violation is False
    assert result.violation_counts == {}
    assert result.joint_limit_active_steps == 15
    assert result.finger_joint_limit_saturation_steps == 1
    assert result.robot_constraint_diagnostics["finger_joint_limit_saturation_counts"] == {
        "right_thumb_2_joint": 1
    }
    assert result.robot_constraint_diagnostics["joint_limit_soft_counts"] == {
        "right_thumb_2_joint": 15
    }


def test_joint_limit_can_be_explicitly_opted_into_core_safety():
    tracker = MetricTracker(
        {
            "safety": {
                "robot_constraint": {
                    "joint_limit": {
                        "include_in_core_safety": True,
                        "finger": {"as_hard_violation": True},
                    }
                }
            }
        },
        dt=0.1,
    )
    robot_state = {
        "joint_names": ["right_thumb_2_joint"],
        "qpos": [1.10],
        "qvel": [0.0],
    }
    limits = ([0.0], [1.0])

    for step in range(15):
        tracker.update({}, sim_step=step, robot_state=robot_state, joint_limits=limits)
    result = tracker.finalize()

    assert result.finger_joint_limit_saturation is True
    assert result.robot_constraint_violation is True
    assert result.safety_hard_violation is True
    assert result.violation_counts == {"joint_limit": 1}
    assert result.robot_constraint_diagnostics["joint_limit_hard_counts"] == {
        "right_thumb_2_joint": 1
    }


def test_tool_switch_timeout_without_target_counts_as_failure():
    detector = GraspDetector(table_z=0.75, min_hold_frames=1, dt=0.1)
    tracker = ToolTracker(
        stage_tools={"stage": ["tool_a", "tool_b"]},
        tool_equivalence_classes={},
        grasp_detector=detector,
        release_stable_s=0.0,
        switch_timeout_s=0.2,
        dt=0.1,
    )

    initial = {"tool_a": _object_state(0.75)}
    detector.update(initial, sim_step=0)
    tracker.update(initial, sim_step=0, stage_completion={})

    held = {"tool_a": _object_state(0.9)}
    detector.update(held, sim_step=1)
    tracker.update(held, sim_step=1, stage_completion={})

    released = {"tool_a": _object_state(0.75)}
    detector.update(released, sim_step=2)
    tracker.update(released, sim_step=2, stage_completion={})
    detector.update(released, sim_step=3)
    tracker.update(released, sim_step=3, stage_completion={})

    assert tracker.compute_tssr() == 0.0
    assert len(tracker.switch_events) == 1
    event = tracker.switch_events[0]
    assert event.from_tool == "tool_a"
    assert event.to_tool is None
    assert event.successful is False


# ════════════════════════════════════════════════════════════════════════════
# Joint name classification — finger vs arm handshake (v1.4 fix)
# ════════════════════════════════════════════════════════════════════════════

def test_schunk_hand_mimic_joints_are_finger_not_arm():
    """Schunk hand numeric mimic joints (j3, j14, index_spread, ring_spread)
    must be classified as finger, not arm — no hard violation."""
    tracker = MetricTracker({"safety": {}}, dt=0.1)
    robot_state = {
        "joint_names": [
            "shoulder_pan_joint",       # arm (shoulder_ pattern)
            "right_hand_j3",            # hand mimic (_hand_ pattern)
            "left_hand_j14",            # hand mimic (_hand_ pattern)
            "right_hand_index_spread",  # hand mimic (index_spread pattern)
            "right_hand_ring_spread",   # hand mimic (ring_spread pattern)
            "left_hand_j5",             # hand mimic (_hand_ pattern)
            "left_hand_Thumb_Flexion",  # hand active (thumb pattern, also _hand_)
        ],
        "qpos": [0.0, 1.10, 1.10, 1.10, 1.10, 1.10, 1.10],
        "qvel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    limits = ([0.0] * 7, [1.0] * 7)

    for step in range(15):
        tracker.update({}, sim_step=step, robot_state=robot_state, joint_limits=limits)
    result = tracker.finalize()

    # All hand joints saturated (15 steps at 0.10 excess > 0.08 finger margin)
    assert result.finger_joint_limit_saturation is True
    # Shoulder is at 0.0 — no arm violation
    assert result.robot_constraint_violation is False
    assert result.safety_hard_violation is False


def test_shadow_hand_coded_joints_are_finger():
    """Shadow hand alphanumeric codes (FFJ, MFJ, RFJ, LFJ, THJ, WRJ)
    must be classified as finger."""
    tracker = MetricTracker({"safety": {}}, dt=0.1)
    robot_state = {
        "joint_names": [
            "FFJ4", "MFJ3", "RFJ2", "LFJ1", "THJ5", "WRJ1",
            "shoulder_pan_joint", "elbow_joint",
        ],
        "qpos": [1.10] * 6 + [0.0] * 2,
        "qvel": [0.0] * 8,
    }
    limits = ([0.0] * 8, [1.0] * 8)

    for step in range(15):
        tracker.update({}, sim_step=step, robot_state=robot_state, joint_limits=limits)
    result = tracker.finalize()

    assert result.finger_joint_limit_saturation is True
    assert result.robot_constraint_violation is False


def test_panda_arm_joints_are_not_finger():
    """panda_joint1..7 must remain classified as ARM (not finger),
    even though they contain 'joint' which matches the Allegro joint_ pattern.
    The 'panda_joint' arm pattern takes priority."""
    tracker = MetricTracker({"safety": {}}, dt=0.1)
    robot_state = {
        "joint_names": [
            "panda_joint1",    # arm (panda_joint pattern → arm)
            "joint_0_0",       # Allegro hand (joint_ pattern → finger, no arm match)
        ],
        "qpos": [1.10, 1.10],
        "qvel": [0.0, 0.0],
    }
    limits = ([0.0] * 2, [1.0] * 2)

    # At 10 steps with 0.10 excess:
    #   panda_joint1  (arm):    10 >= 10, excess 0.10 > 0.05 → HARD VIOLATION
    #   joint_0_0     (finger): 10 <  15                    → not yet saturated
    for step in range(10):
        tracker.update({}, sim_step=step, robot_state=robot_state, joint_limits=limits)
    result = tracker.finalize()

    assert result.robot_constraint_violation is True   # from panda_joint1
    assert result.finger_joint_limit_saturation is False  # Allegro hasn't hit 15 steps


def test_existing_finger_patterns_still_work():
    """Existing descriptive finger name patterns (thumb/index/middle/ring/little/pinky/finger)
    continue to classify correctly after the arm-pattern addition."""
    tracker = MetricTracker({"safety": {}}, dt=0.1)
    robot_state = {
        "joint_names": [
            "right_thumb_2_joint",
            "left_index_1_joint",
            "middle_flexion_joint",
            "right_ring_mcp_joint",
            "left_little_dip_joint",
            "right_pinky_joint",
            "left_finger_proximal",
        ],
        "qpos": [1.10] * 7,
        "qvel": [0.0] * 7,
    }
    limits = ([0.0] * 7, [1.0] * 7)

    for step in range(15):
        tracker.update({}, sim_step=step, robot_state=robot_state, joint_limits=limits)
    result = tracker.finalize()

    assert result.finger_joint_limit_saturation is True
    assert result.robot_constraint_violation is False


def test_unmatched_numeric_arm_joints_default_to_arm():
    """Joints like xarm joint1-7 and kuka A1-A7 have no matching hand patterns
    and should default to ARM (not finger) behavior."""
    tracker = MetricTracker({"safety": {}}, dt=0.1)
    robot_state = {
        "joint_names": ["joint1", "A1", "joint7", "A7"],
        "qpos": [1.06, 1.06, 1.06, 1.06],
        "qvel": [0.0, 0.0, 0.0, 0.0],
    }
    limits = ([0.0] * 4, [1.0] * 4)

    for step in range(10):
        tracker.update({}, sim_step=step, robot_state=robot_state, joint_limits=limits)
    result = tracker.finalize()

    # All default to arm: 10 steps at 0.06 excess > 0.05 margin → hard violation
    assert result.robot_constraint_violation is True
    assert result.finger_joint_limit_saturation is False


def test_ur5_arm_joints_are_not_finger():
    """UR5 joints with shoulder_, elbow_, wrist_ prefix must remain as ARM
    even though 'wrist_' might superficially seem hand-related."""
    tracker = MetricTracker({"safety": {}}, dt=0.1)
    robot_state = {
        "joint_names": [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        ],
        "qpos": [1.06] * 6,
        "qvel": [0.0] * 6,
    }
    limits = ([0.0] * 6, [1.0] * 6)

    # At 1 step with 0.06 excess > 0.15 critical margin? No (0.06 < 0.15).
    # But at step 2: excess >= dynamic_margin (0.05) for 2 consecutive steps — not yet.
    # Let's run 10 steps to trigger the consecutive counter for arm.
    for step in range(10):
        tracker.update({}, sim_step=step, robot_state=robot_state, joint_limits=limits)
    result = tracker.finalize()

    # All are arm: 10 steps at 0.06 excess > 0.05 margin → hard violation
    assert result.robot_constraint_violation is True
    assert result.finger_joint_limit_saturation is False
