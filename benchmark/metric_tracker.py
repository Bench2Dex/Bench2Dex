"""Online benchmark metric tracking.

The tracker keeps episode-level benchmark metrics separate from raw data
collection.  It evaluates terminal success and stage progress every step so
rollout code can early-stop only after stable terminal success.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from success.engine import evaluate_condition_node

from .conditions import validate_terminal_raw_condition
from .grasp_detector import GraspDetector, GraspEvent
from .stage_tracker import StageTracker
from .tool_tracker import ToolTracker


DEFAULT_DWELL_TIME_S = 0.5
DEFAULT_HIGH_SPEED_MPS = 2.0
DEFAULT_DROP_MARGIN_M = 0.10
DEFAULT_DROP_LIFT_MARGIN_M = 0.05
DEFAULT_DROP_HEIGHT_M = 0.08
DEFAULT_DROP_SETTLE_MARGIN_M = 0.03
DEFAULT_DROP_FALL_VEL_MPS = 0.25
DEFAULT_JOINT_LIMIT_MARGIN_ABS = 0.05
DEFAULT_JOINT_LIMIT_MARGIN_RATIO = 0.03
DEFAULT_JOINT_LIMIT_MIN_CONSECUTIVE_STEPS = 10
DEFAULT_JOINT_LIMIT_CRITICAL_MARGIN_ABS = 0.15
DEFAULT_FINGER_JOINT_LIMIT_MARGIN_ABS = 0.08
DEFAULT_FINGER_JOINT_LIMIT_MARGIN_RATIO = 0.05
DEFAULT_FINGER_JOINT_LIMIT_MIN_CONSECUTIVE_STEPS = 15
DEFAULT_FINGER_JOINT_LIMIT_CRITICAL_MARGIN_ABS = 0.25
DEFAULT_FINGER_JOINT_LIMIT_AS_HARD_VIOLATION = False
DEFAULT_HARD_LIMIT_NUMERICAL_TOLERANCE_ABS = 0.005
DEFAULT_HARD_LIMIT_NUMERICAL_TOLERANCE_RATIO = 0.002
DEFAULT_FINGER_JOINT_NAME_PATTERNS = (
    # ── Descriptive finger names (legacy) ──
    "finger",
    "thumb",
    "index",
    "middle",
    "ring",
    "little",
    "pinky",
    # ── Numeric / coded hand-joint patterns ──
    "_hand_",            # Schunk: right_hand_j3, left_hand_Thumb_Flexion (all mimic+active)
    "_f_joint",          # Dexhand: r_f_joint, l_f_joint
    "ffj",               # Shadow hand: FFJ1-4 (index)
    "mfj",               # Shadow hand: MFJ1-4 (middle)
    "rfj",               # Shadow hand: RFJ1-4 (ring)
    "lfj",               # Shadow hand: LFJ1-5 (little)
    "thj",               # Shadow hand: THJ1-5 (thumb)
    "wrj",               # Shadow hand: WRJ1-2 (wrist)
    "leap_",             # Leap hand: leap_l_0..15, multi_leap_r_0..15
    "index_spread",      # Schunk spread mimic (index)
    "ring_spread",       # Schunk spread mimic (ring)
    "joint_",            # Allegro: joint_0_0..joint_15_0, multi_joint_0_0..multi_joint_15_0
)
DEFAULT_ARM_JOINT_NAME_PATTERNS = (
    # ── Definitive arm-joint patterns: checked BEFORE hand patterns ──
    # These take priority — a joint matching any of these is NEVER a finger joint.
    "shoulder_",         # UR5/xarm: shoulder_pan_joint, shoulder_lift_joint
    "elbow_",            # UR5: elbow_joint
    "wrist_",            # UR5: wrist_1_joint, wrist_2_joint, wrist_3_joint
    "panda_joint",       # Panda: panda_joint1..7 (excludes Allegro joint_0_0)
    "_arm_",             # Dual-arm prefix: L_arm_shoulder_pan_joint
    "l_joint_",          # Jaka left arm: l_joint_1..6 (prevents false positive from joint_)
)
LEGACY_JOINT_LIMIT_SOFT_MARGIN = 0.01


@dataclass
class MetricSnapshot:
    instant_success: bool = False
    stable_success: bool = False
    ever_instant_success: bool = False
    at_end_success_observed: bool = False
    first_success_step: int | None = None
    stable_success_step: int | None = None
    success_hold_s: float = 0.0

    @property
    def at_end_success(self) -> bool:
        return self.at_end_success_observed


@dataclass
class MetricResult(MetricSnapshot):
    terminal_success_rate: float = 0.0
    stage_completion_rate: float = 0.0
    normalized_progress_score: float = 0.0
    chain_depth: int = 0
    current_stage_completion_rate: float = 0.0
    current_normalized_progress_score: float = 0.0
    current_chain_depth: int = 0
    latched_stage_completion_rate: float = 0.0
    latched_normalized_progress_score: float = 0.0
    latched_chain_depth: int = 0
    stage_completion: dict[str, bool] = field(default_factory=dict)
    current_stage_completion: dict[str, bool] = field(default_factory=dict)
    stage_first_completion_step: dict[str, int | None] = field(default_factory=dict)
    kinematic_grasp_stability_index: float | None = None
    mean_kinematic_grasp_stability: float | None = None
    tool_selection_accuracy: float | None = None
    tool_switch_success_rate: float | None = None
    tool_switch_total: int = 0
    tool_switch_successful: int = 0
    per_object_best_kinematic_grasp_stability: dict[str, float] = field(default_factory=dict)
    grasp_gsi_diagnostics: dict[str, float] = field(default_factory=dict)
    robot_motion_metrics: dict[str, float] = field(default_factory=dict)
    safety_violation_rate: float = 0.0
    safety_hard_violation: bool = False
    drop_violation: bool = False
    high_speed_violation: bool = False
    excessive_impact_violation: bool | None = None
    robot_constraint_violation: bool = False
    finger_joint_limit_saturation: bool = False
    impact_metric_available: bool = False
    violation_counts: dict[str, int] = field(default_factory=dict)
    robot_constraint_diagnostics: dict[str, Any] = field(default_factory=dict)
    joint_limit_active_steps: int = 0
    joint_limit_active_step_rate: float | None = None
    finger_joint_limit_saturation_steps: int = 0
    finger_joint_limit_saturation_step_rate: float | None = None
    max_joint_limit_excess_rad: float | None = None
    max_joint_limit_excess_joint: str | None = None
    # --- New fields ---
    at_end_success_budget: bool | None = None
    first_stable_success_step: int | None = None
    steps_to_stable_success: int | None = None
    policy_steps_to_stable_success: int | None = None
    policy_queries_to_stable_success: int | None = None
    time_to_stable_success_s: float | None = None
    expert_time_s: float | None = None
    expert_time_step: int | None = None
    policy_query_count: int | None = None
    terminated_reason: str | None = None
    # Full per-step traces are debug data.  Normal evaluation keeps only
    # online aggregates and leaves this unset.
    timeseries: dict[str, list] | None = None
    safety_violation_rate_time: float | None = None
    safety_violation_rate_event: float | None = None
    safety_violation_step_rate: float | None = None
    safety_violation_events_per_step: float | None = None
    episode_violation_rate: float | None = None
    task_efficiency: float | None = None

    @property
    def at_end_success(self) -> bool:
        return self.at_end_success_observed


class MetricTracker:
    """Track terminal, progress, tool, and safety metrics online."""

    def __init__(
        self,
        metrics_spec: Mapping[str, Any] | None = None,
        *,
        success_conditions: list[dict[str, Any]] | None = None,
        allow_legacy_success_conditions: bool = False,
        dt: float = 1 / 60,
        table_height_offset: float = 0.0,
        robot_key: str | None = None,
    ) -> None:
        self._spec = dict(metrics_spec or {})
        self._table_height_offset = float(table_height_offset)
        self._dt = float(dt)
        self.task_family = str(self._spec.get("task_family", "unknown"))
        self.robot_key = str(robot_key) if robot_key else None
        # Exact roles come from the joint manifest generated from the combined
        # URDF and checked against the shipped USD.  Name heuristics below are
        # retained only for unit tests and legacy environments without a key.
        try:
            from robots.joint_safety import joint_roles
            self._joint_role_manifest = joint_roles(self.robot_key)
        except Exception:
            self._joint_role_manifest = None
        self._joint_role_manifest_available = self._joint_role_manifest is not None
        self._unknown_runtime_joint_names: set[str] = set()
        if self._spec.get("expert_steps") is not None:
            raise ValueError(
                "metrics.expert_steps is deprecated for official benchmark metrics; "
                "use metrics.expert_time_step (physics-simulation steps) instead."
            )
        self.expert_time_step = self._as_optional_int(self._spec.get("expert_time_step"))
        self.expert_time_s = self._as_optional_float(self._spec.get("expert_time_s"))
        self.policy_stride = max(1, int(self._spec.get("policy_stride", 1)))

        terminal = dict(self._spec.get("terminal") or {})
        # raw_condition is primary; condition is deprecated alias
        raw_cond = terminal.get("raw_condition")
        if raw_cond is not None:
            validate_terminal_raw_condition(raw_cond, allow_sequence=True)
            self._terminal_condition = dict(raw_cond)
            self._terminal_available = True
        elif terminal.get("condition") is not None:
            if not allow_legacy_success_conditions:
                raise ValueError(
                    "Official benchmark metrics require metrics.terminal.raw_condition; "
                    "set allow_legacy_success_conditions=True only for debug/legacy evaluation."
                )
            terminal_condition = terminal["condition"]
            validate_terminal_raw_condition(terminal_condition, allow_sequence=True)
            self._terminal_condition = dict(terminal_condition)
            self._terminal_available = True
        elif success_conditions:
            if not allow_legacy_success_conditions:
                raise ValueError(
                    "Official benchmark metrics require metrics.terminal.raw_condition; "
                    "set allow_legacy_success_conditions=True only for debug/legacy evaluation."
                )
            self._terminal_condition = {"type": "all", "conditions": list(success_conditions)}
            self._terminal_available = True
        else:
            self._terminal_condition = None
            self._terminal_available = False
        self.dwell_time_s = float(terminal.get("dwell_time_s", DEFAULT_DWELL_TIME_S))
        if self.dwell_time_s < 0:
            raise ValueError("metrics.terminal.dwell_time_s must be non-negative.")

        self._terminal_ctx: dict[str, Any] = {"dt": self._dt}
        self._terminal_hold_s = 0.0
        self._instant_success = False
        self._stable_success = False
        self._ever_instant_success = False
        self._at_end_success_observed = False
        self._first_success_step: int | None = None
        self._stable_success_step: int | None = None
        self._stable_success_time_s: float | None = None
        self._elapsed_time_s = 0.0

        stages = self._spec.get("stages", []) or []
        self._stages = [dict(s) for s in stages]
        tool_equiv = self._spec.get("tool_equivalence_classes") or {}
        self._stage_tracker = StageTracker(self._stages, dt=self._dt) if self._stages else None
        self._stage_dependencies = self._extract_stage_dependencies(self._stages)
        self._stage_tools = self._extract_stage_tools(self._stages, tool_equiv)
        self._terminal_tool = self._first_tool_id(self._terminal_condition)

        self._trace_enabled = bool(self._spec.get("metrics_trace", False))
        self._trace: list[dict[str, Any]] = []
        self._timeseries: list[dict[str, Any]] = []
        self._metric_update_count = 0
        self._stable_success_update_count: int | None = None
        # Keep the grasp aggregates that used to be reconstructed by scanning
        # the full trace at finalization.  This preserves episode metrics when
        # the optional debug trace is disabled.
        self._grasp_gsi_best_by_object: dict[str, float] = {}
        self._grasp_positive_gsi_sum = 0.0
        self._grasp_positive_gsi_count = 0
        self._grasp_component_sums: dict[str, float] = {}
        self._grasp_component_counts: dict[str, int] = {}
        self._violation_counts: dict[str, int] = {}
        self._last_step = 0
        self._known_violation_names: set[str] = set()
        self._current_step_violations: set[str] | None = None

        grasp_spec = dict(self._spec.get("grasp") or {})
        self._grasp_tracked_objects = self._parse_tracked_objects(grasp_spec.get("tracked_objects"))

        # Safety proxy configuration
        safety = self._spec.get("safety") or {}
        # safe_episode_score_lambda removed (safe_episode_score retired).
        # Accept the config key silently for backward compatibility.
        if "safe_episode_score_lambda" in safety or "safe_ep_lambda" in safety:
            import warnings
            warnings.warn(
                "safety.safe_episode_score_lambda is deprecated; safe_episode_score has been removed.",
                DeprecationWarning, stacklevel=2,
            )
        high_speed_spec = safety.get("high_speed") or {}
        if not isinstance(high_speed_spec, Mapping):
            high_speed_spec = {}
        self._high_speed_enabled = bool(high_speed_spec.get("enabled", safety.get("high_speed_enabled", True)))
        self._high_speed_threshold_mps = float(
            high_speed_spec.get(
                "threshold_mps",
                safety.get("high_speed_threshold_mps", safety.get("impact_speed_threshold", DEFAULT_HIGH_SPEED_MPS)),
            )
        )
        raw_high_speed_objects = high_speed_spec.get("tracked_objects") or safety.get("high_speed_tracked_objects") or []
        self._high_speed_tracked_objects = (
            {str(obj_id) for obj_id in raw_high_speed_objects} if raw_high_speed_objects else None
        )
        self._drop_margin = float(safety.get("drop_margin", DEFAULT_DROP_MARGIN_M))
        drop_spec = safety.get("drop") or {}
        if not isinstance(drop_spec, Mapping):
            drop_spec = {}
        self._drop_tracked_objects = self._parse_tracked_objects(drop_spec.get("tracked_objects"))
        if self._drop_tracked_objects is None:
            self._drop_tracked_objects = self._grasp_tracked_objects
        drop_enabled_default = self._table_z() is not None and self._drop_tracked_objects is not None
        self._drop_tracking_enabled = bool(drop_spec.get("enabled", drop_enabled_default))
        self._drop_lift_margin = float(drop_spec.get("lift_margin", DEFAULT_DROP_LIFT_MARGIN_M))
        self._drop_height = float(drop_spec.get("drop_height", DEFAULT_DROP_HEIGHT_M))
        self._drop_settle_margin = float(drop_spec.get("settle_margin", DEFAULT_DROP_SETTLE_MARGIN_M))
        self._drop_fall_vel_threshold = float(drop_spec.get("fall_vel_threshold", DEFAULT_DROP_FALL_VEL_MPS))
        raw_allowed = drop_spec.get("allowed_placed_conditions") or {}
        self._drop_allowed_conditions = dict(raw_allowed) if isinstance(raw_allowed, Mapping) else {}
        self._drop_allowed_ctx: dict[str, dict[str, Any]] = {}
        self._drop_initial_z: dict[str, float] = {}
        self._drop_max_z: dict[str, float] = {}
        self._drop_was_lifted: dict[str, bool] = {}
        self._drop_falling_fast: dict[str, bool] = {}
        self._drop_below_table_active: dict[str, bool] = {}
        self._drop_event_recorded: set[str] = set()
        self._drop_violation = False
        self._no_drop_tracked_objects_warned = False
        self._high_speed_violation = False
        self._high_speed_active: dict[str, bool] = {}
        self._impact_metric_available = False
        self._excessive_impact_violation: bool | None = None
        self._robot_constraint_violation = False
        robot_constraint_spec = safety.get("robot_constraint") or {}
        if not isinstance(robot_constraint_spec, Mapping):
            robot_constraint_spec = {}
        joint_limit_spec = robot_constraint_spec.get("joint_limit") or safety.get("joint_limit") or {}
        if not isinstance(joint_limit_spec, Mapping):
            joint_limit_spec = {}
        self._joint_limit_enabled = bool(joint_limit_spec.get("enabled", robot_constraint_spec.get("enabled", True)))
        # Joint-limit observations are asset/controller-health diagnostics by
        # default.  They are intentionally excluded from task-level safe
        # success until the asset's mimic/drive model has been validated.
        # A benchmark that has validated this execution layer may opt in.
        self._joint_limit_include_in_core_safety = bool(
            joint_limit_spec.get(
                "include_in_core_safety",
                robot_constraint_spec.get("include_in_core_safety", False),
            )
        )
        self._joint_limit_margin_abs = float(joint_limit_spec.get("hard_margin_abs", DEFAULT_JOINT_LIMIT_MARGIN_ABS))
        self._joint_limit_margin_ratio = float(joint_limit_spec.get("hard_margin_ratio", DEFAULT_JOINT_LIMIT_MARGIN_RATIO))
        self._joint_limit_min_consecutive_steps = max(
            1,
            int(joint_limit_spec.get("min_consecutive_steps", DEFAULT_JOINT_LIMIT_MIN_CONSECUTIVE_STEPS)),
        )
        self._joint_limit_critical_margin_abs = float(
            joint_limit_spec.get("critical_margin_abs", DEFAULT_JOINT_LIMIT_CRITICAL_MARGIN_ABS)
        )
        self._hard_limit_tolerance_abs = float(
            joint_limit_spec.get("numerical_tolerance_abs", DEFAULT_HARD_LIMIT_NUMERICAL_TOLERANCE_ABS)
        )
        self._hard_limit_tolerance_ratio = float(
            joint_limit_spec.get("numerical_tolerance_ratio", DEFAULT_HARD_LIMIT_NUMERICAL_TOLERANCE_RATIO)
        )
        finger_joint_limit_spec = joint_limit_spec.get("finger") or joint_limit_spec.get("finger_joint") or {}
        if not isinstance(finger_joint_limit_spec, Mapping):
            finger_joint_limit_spec = {}
        self._finger_joint_limit_margin_abs = float(
            finger_joint_limit_spec.get(
                "hard_margin_abs",
                joint_limit_spec.get("finger_hard_margin_abs", DEFAULT_FINGER_JOINT_LIMIT_MARGIN_ABS),
            )
        )
        self._finger_joint_limit_margin_ratio = float(
            finger_joint_limit_spec.get(
                "hard_margin_ratio",
                joint_limit_spec.get("finger_hard_margin_ratio", DEFAULT_FINGER_JOINT_LIMIT_MARGIN_RATIO),
            )
        )
        self._finger_joint_limit_min_consecutive_steps = max(
            1,
            int(
                finger_joint_limit_spec.get(
                    "min_consecutive_steps",
                    joint_limit_spec.get(
                        "finger_min_consecutive_steps",
                        DEFAULT_FINGER_JOINT_LIMIT_MIN_CONSECUTIVE_STEPS,
                    ),
                )
            ),
        )
        self._finger_joint_limit_critical_margin_abs = float(
            finger_joint_limit_spec.get(
                "critical_margin_abs",
                joint_limit_spec.get(
                    "finger_critical_margin_abs",
                    DEFAULT_FINGER_JOINT_LIMIT_CRITICAL_MARGIN_ABS,
                ),
            )
        )
        self._finger_joint_limit_as_hard_violation = bool(
            finger_joint_limit_spec.get(
                "as_hard_violation",
                joint_limit_spec.get(
                    "finger_as_hard_violation",
                    DEFAULT_FINGER_JOINT_LIMIT_AS_HARD_VIOLATION,
                ),
            )
        )
        raw_finger_patterns = finger_joint_limit_spec.get(
            "joint_name_patterns",
            joint_limit_spec.get("finger_joint_name_patterns", DEFAULT_FINGER_JOINT_NAME_PATTERNS),
        )
        if isinstance(raw_finger_patterns, str):
            raw_finger_patterns = [raw_finger_patterns]
        if raw_finger_patterns:
            self._finger_joint_name_patterns = tuple(str(pattern).lower() for pattern in raw_finger_patterns)
        else:
            self._finger_joint_name_patterns = DEFAULT_FINGER_JOINT_NAME_PATTERNS
        # Arm joint name patterns (negative filter — checked BEFORE hand patterns)
        raw_arm_patterns = joint_limit_spec.get(
            "arm_joint_name_patterns",
            DEFAULT_ARM_JOINT_NAME_PATTERNS,
        )
        if isinstance(raw_arm_patterns, str):
            raw_arm_patterns = [raw_arm_patterns]
        if raw_arm_patterns:
            self._arm_joint_name_patterns = tuple(str(pattern).lower() for pattern in raw_arm_patterns)
        else:
            self._arm_joint_name_patterns = DEFAULT_ARM_JOINT_NAME_PATTERNS
        self._joint_limit_consecutive: dict[str, int] = {}
        self._finger_joint_limit_consecutive: dict[str, int] = {}
        self._joint_limit_soft_counts: dict[str, int] = {}
        self._joint_limit_hard_counts: dict[str, int] = {}
        self._joint_limit_hard_active_joints: set[str] = set()
        self._finger_joint_limit_saturation_counts: dict[str, int] = {}
        self._joint_limit_max_excess_by_joint: dict[str, float] = {}
        self._joint_limit_active_steps: int = 0
        self._finger_joint_limit_saturation_steps: int = 0
        self._max_joint_limit_excess_rad: float = 0.0
        self._max_joint_limit_excess_joint: str | None = None
        self._last_joint_limit_active = False
        self._last_joint_limit_max_excess_rad = 0.0
        self._last_joint_limit_max_excess_joint: str | None = None
        self._last_finger_joint_limit_saturation_active = False
        self._last_arm_joint_limit_hard_active = False
        self._joint_limit_hard_active = False
        self._joint_limit_hard_episode_events = 0
        self._joint_limit_hard_source_available = False
        self._joint_limit_soft_source: str | None = None
        self._joint_limit_hard_source: str | None = None
        # Sparse diagnostics: one causal snapshot at the first observed
        # physical hard-limit violation, plus non-scored reset/home baselines.
        self._first_joint_limit_hard_event: dict[str, Any] | None = None
        self._joint_limit_baseline_checks: list[dict[str, Any]] = []
        self._finger_joint_limit_saturation = False
        self._safety_update_count: int = 0
        self._safety_total_time_s: float = 0.0
        self._violation_step_count: int = 0
        self._violation_active_time_s: float = 0.0
        self._no_table_z_warned: bool = False
        self._motion_update_count: int = 0
        self._motion_velocity_sq_sum: float = 0.0
        self._motion_acceleration_sq_sum: float = 0.0
        self._motion_jerk_sq_sum: float = 0.0
        self._motion_effort_sq_sum: float = 0.0
        self._motion_effort_count: int = 0
        self._prev_qvel: np.ndarray | None = None
        self._prev_joint_acceleration: np.ndarray | None = None

        self._grasp_enabled = bool(grasp_spec.get("enabled", False))
        self._grasp_detector: GraspDetector | None = None
        if self._grasp_enabled:
            self._grasp_detector = GraspDetector(
                table_z=self._grasp_table_z(grasp_spec),
                lift_margin=float(grasp_spec.get("lift_margin", 0.02)),
                ang_vel_threshold=float(grasp_spec.get("ang_vel_threshold", 1.0)),
                fall_vel_threshold=float(grasp_spec.get("fall_vel_threshold", 0.5)),
                min_hold_frames=int(grasp_spec.get("min_hold_frames", 3)),
                slip_lin_vel=float(grasp_spec.get("slip_lin_vel", 0.3)),
                slip_ang_vel=float(grasp_spec.get("slip_ang_vel", 2.0)),
                dt=self._dt,
                gsi_weights=grasp_spec.get("gsi_weights"),
                min_hold_s=float(grasp_spec.get("min_hold_s", 0.5)),
                max_slip_count=float(grasp_spec.get("max_slip_count", 3.0)),
                ang_vel_scale=float(grasp_spec.get("ang_vel_scale", 1.0)),
                lin_vel_scale=self._as_optional_float(grasp_spec.get("lin_vel_scale")),
                tracked_objects=self._grasp_tracked_objects,
            )

        self._tool_tracker: ToolTracker | None = None
        if self._grasp_detector is not None and (self._stage_tools or tool_equiv):
            self._tool_tracker = ToolTracker(
                stage_tools=self._stage_tools,
                tool_equivalence_classes=tool_equiv if isinstance(tool_equiv, Mapping) else {},
                grasp_detector=self._grasp_detector,
                gsi_threshold=float(grasp_spec.get("gsi_threshold", 0.5)),
                release_stable_s=float(grasp_spec.get("release_stable_s", 0.3)),
                switch_timeout_s=float(grasp_spec.get("switch_timeout_s", 5.0)),
                dt=self._dt,
                stage_dependencies=self._stage_dependencies,
                release_vel_threshold=float(grasp_spec.get("release_vel_threshold", 0.05)),
                grasp_stable_s=float(grasp_spec.get("switch_min_hold_s", grasp_spec.get("min_hold_s", 0.5))),
            )

    @property
    def available(self) -> bool:
        return self._terminal_available or bool(self._stages) or bool(getattr(self, "_grasp_enabled", False))

    @property
    def stable_success(self) -> bool:
        return self._stable_success

    @property
    def success(self) -> bool:
        return self._stable_success

    @property
    def trace(self) -> list[dict[str, Any]]:
        return list(self._trace)

    def update(
        self,
        object_states: dict[str, Any],
        *,
        sim_step: int,
        dt: float | None = None,
        robot_state: dict[str, Any] | None = None,
        joint_limits: tuple[Any, Any] | None = None,
        joint_command: Mapping[str, Any] | None = None,
    ) -> MetricSnapshot:
        if dt is not None:
            self._dt = float(dt)
        self._metric_update_count += 1
        self._elapsed_time_s += self._dt
        self._last_step = int(sim_step)
        self._update_terminal(object_states, sim_step=int(sim_step))
        stage_snapshot = self._update_stages(object_states, sim_step=int(sim_step))
        step_violations = self._update_safety_proxy(
            object_states,
            robot_state=robot_state,
            joint_limits=joint_limits,
            joint_command=joint_command,
        )
        self._update_robot_motion_metrics(robot_state)
        grasp_events: list[GraspEvent] = []
        if self._grasp_detector is not None:
            grasp_events = self._grasp_detector.update(object_states, sim_step=int(sim_step), dt=self._dt)
        if self._tool_tracker is not None:
            stage_completion = stage_snapshot.completed if stage_snapshot is not None else {}
            self._tool_tracker.update(
                object_states,
                sim_step=int(sim_step),
                stage_completion=stage_completion,
                dt=self._dt,
            )

        snapshot = self._snapshot()
        if snapshot.stable_success and self._stable_success_update_count is None:
            self._stable_success_update_count = self._metric_update_count

        grasp_components = self._update_grasp_statistics()

        # Full step-level traces are explicitly opt-in.  The online counters
        # above retain all official episode metrics without allocating a large
        # Python dict (and later a second dict-of-lists copy) every step.
        if self._trace_enabled:
            ts_entry: dict[str, Any] = {
            "sim_step": int(sim_step),
            "instant_success": snapshot.instant_success,
            "ever_instant_success": snapshot.ever_instant_success,
            "at_end_success_observed": snapshot.at_end_success_observed,
            "success_hold_s": snapshot.success_hold_s,
            "stable_success": snapshot.stable_success,
            "stage_completion_rate": stage_snapshot.stage_completion_rate if stage_snapshot is not None else 0.0,
            "normalized_progress_score": stage_snapshot.normalized_progress_score if stage_snapshot is not None else 0.0,
            "chain_depth": stage_snapshot.chain_depth if stage_snapshot is not None else 0,
            "current_stage_completion_rate": stage_snapshot.current_stage_completion_rate if stage_snapshot is not None else 0.0,
            "current_normalized_progress_score": stage_snapshot.current_normalized_progress_score if stage_snapshot is not None else 0.0,
            "current_chain_depth": stage_snapshot.current_chain_depth if stage_snapshot is not None else 0,
            "latched_stage_completion_rate": stage_snapshot.latched_stage_completion_rate if stage_snapshot is not None else 0.0,
            "latched_normalized_progress_score": stage_snapshot.latched_normalized_progress_score if stage_snapshot is not None else 0.0,
            "latched_chain_depth": stage_snapshot.latched_chain_depth if stage_snapshot is not None else 0,
            "selected_tool": self._selected_tool(stage_snapshot.completed if stage_snapshot else {}),
            "safety_violation": bool(step_violations),
            "violation_names": ",".join(sorted(step_violations)),
            "safety_hard_violation": bool(
                self._drop_violation
                or self._high_speed_violation
                or (
                    self._joint_limit_include_in_core_safety
                    and self._robot_constraint_violation
                )
            ),
            "drop_violation": bool(self._drop_violation),
            "high_speed_violation": bool(self._high_speed_violation),
            "excessive_impact_violation": self._excessive_impact_violation,
            "robot_constraint_violation": bool(self._robot_constraint_violation),
            "finger_joint_limit_saturation": bool(self._finger_joint_limit_saturation),
            "finger_joint_limit_saturation_active": bool(self._last_finger_joint_limit_saturation_active),
            "joint_limit_active": bool(self._last_joint_limit_active),
            "joint_limit_max_excess_rad": float(self._last_joint_limit_max_excess_rad),
            "joint_limit_max_excess_joint": self._last_joint_limit_max_excess_joint,
            "impact_metric_available": bool(self._impact_metric_available),
        }
            if grasp_events:
                ts_entry["grasp_events"] = ",".join(f"{event.obj_id}:{event.event_type}" for event in grasp_events)
            if stage_snapshot is not None:
                for stage_id, completed in stage_snapshot.completed.items():
                    ts_entry[f"stage_{stage_id}_completed"] = bool(completed)
                for stage_id, completed in stage_snapshot.current_completed.items():
                    ts_entry[f"stage_{stage_id}_current"] = bool(completed)
            for name in sorted(self._known_violation_names):
                ts_entry[f"violation_{name}"] = name in step_violations
            if self._grasp_detector is not None:
                for obj_id in sorted(self._grasp_detector.tracked_objects):
                    gs = self._grasp_detector.get_grasp_state(obj_id)
                    if gs is not None:
                        components = grasp_components.get(obj_id, {})
                        ts_entry[f"grasp_held_{obj_id}"] = bool(gs.is_held)
                        ts_entry[f"grasp_gsi_{obj_id}"] = float(components.get("gsi") or 0.0)
                        for name, value in components.items():
                            if name != "gsi":
                                ts_entry[f"grasp_{name}_{obj_id}"] = value
            self._timeseries.append(ts_entry)
            self._trace.append(
                {
                    "sim_step": int(sim_step),
                    "instant_success": snapshot.instant_success,
                    "stable_success": snapshot.stable_success,
                    "success_hold_s": snapshot.success_hold_s,
                    "stage_completion": dict(self._stage_tracker._completed) if self._stage_tracker else {},
                }
            )
        return snapshot

    def finalize(
        self,
        *,
        steps: int | None = None,
        terminated_reason: str | None = None,
        policy_query_count: int | None = None,
        policy_query_count_at_stable_success: int | None = None,
        max_steps: int | None = None,
        policy_stride: int | None = None,
        evaluation_protocol: str | None = None,
    ) -> MetricResult:
        steps = int(steps if steps is not None else self._metric_update_count)
        snapshot = self._snapshot()

        # Stage results from StageTracker.
        # Keep current and latched progress distinct.  A prior implementation
        # overwrote current with latched at finalization, making the diagnostic
        # unable to reveal a stage that was later undone.
        if self._stage_tracker is not None:
            stage_result = self._stage_tracker._result()
            stage_completion_rate = stage_result.latched_stage_completion_rate
            chain_depth = stage_result.latched_chain_depth
            normalized_progress_score = stage_result.latched_normalized_progress_score
            current_stage_completion_rate = stage_result.current_stage_completion_rate
            current_normalized_progress_score = stage_result.current_normalized_progress_score
            current_chain_depth = stage_result.current_chain_depth
            latched_stage_completion_rate = stage_result.latched_stage_completion_rate
            latched_normalized_progress_score = stage_result.latched_normalized_progress_score
            latched_chain_depth = stage_result.latched_chain_depth
            stage_completion = stage_result.completed
            current_stage_completion = stage_result.current_completed
            stage_first_completion_step = stage_result.first_completed_step
        else:
            stage_completion_rate = 0.0
            chain_depth = 0
            normalized_progress_score = 0.0
            current_stage_completion_rate = 0.0
            current_normalized_progress_score = 0.0
            current_chain_depth = 0
            latched_stage_completion_rate = 0.0
            latched_normalized_progress_score = 0.0
            latched_chain_depth = 0
            stage_completion = {}
            current_stage_completion = {}
            stage_first_completion_step = {}

        # at_end_success_budget: only set when terminated_reason=="max_steps"
        at_end_success_budget: bool | None = None
        if terminated_reason == "max_steps":
            at_end_success_budget = snapshot.at_end_success_observed

        protocol = str(evaluation_protocol or "reach_and_stop").strip().lower()
        if protocol == "fixed_horizon":
            primary_terminal_success = bool(at_end_success_budget)
        else:
            primary_terminal_success = bool(snapshot.stable_success)

        # Build timeseries dict-of-lists
        ts_dict: dict[str, list] | None = None
        if self._trace_enabled and self._timeseries:
            ts_dict = {}
            keys = sorted({key for entry in self._timeseries for key in entry.keys()})
            for k in keys:
                if k.startswith("violation_") and k != "violation_names":
                    ts_dict[k] = [bool(entry.get(k, False)) for entry in self._timeseries]
                else:
                    ts_dict[k] = [entry.get(k) for entry in self._timeseries]

        safety_violation_rate_time = self._safety_violation_rate_time()
        safety_violation_step_rate = self._safety_violation_step_rate()
        safety_violation_events_per_step = self._safety_violation_events_per_step()
        safety_violation_rate_event = safety_violation_events_per_step
        excessive_impact_violation = None
        safety_hard_violation = (
            bool(self._drop_violation)
            or bool(self._high_speed_violation)
            or (
                self._joint_limit_include_in_core_safety
                and bool(self._robot_constraint_violation)
            )
        )
        joint_limit_active_step_rate = (
            self._joint_limit_active_steps / float(self._safety_update_count)
            if self._safety_update_count > 0
            else 0.0
        )
        finger_joint_limit_saturation_step_rate = (
            self._finger_joint_limit_saturation_steps / float(self._safety_update_count)
            if self._safety_update_count > 0
            else 0.0
        )

        if self._tool_tracker is not None:
            self._tool_tracker.finalize(self._last_step)

        kinematic_grasp_stability_index: float | None = None
        mean_kinematic_grasp_stability: float | None = None
        per_object_best_kinematic_grasp_stability: dict[str, float] = {}
        grasp_gsi_diagnostics: dict[str, float] = {}
        if self._grasp_detector is not None and self._grasp_detector.proxy_available:
            for obj_id in sorted(self._grasp_detector.tracked_objects):
                per_object_best_kinematic_grasp_stability[obj_id] = self._grasp_gsi_best_by_object.get(obj_id, 0.0)
            if per_object_best_kinematic_grasp_stability:
                kinematic_grasp_stability_index = max(per_object_best_kinematic_grasp_stability.values())
            if self._grasp_positive_gsi_count:
                mean_kinematic_grasp_stability = self._grasp_positive_gsi_sum / self._grasp_positive_gsi_count
            grasp_gsi_diagnostics = self._grasp_component_diagnostics()

        tool_selection_accuracy: float | None = None
        tool_switch_success_rate: float | None = None
        tool_switch_total = 0
        tool_switch_successful = 0
        if self._tool_tracker is not None:
            tool_selection_accuracy = self._tool_tracker.compute_tsa()
            tool_switch_success_rate = self._tool_tracker.compute_tssr()
            tool_switch_total = len(self._tool_tracker.switch_events)
            tool_switch_successful = sum(1 for event in self._tool_tracker.switch_events if event.successful)

        # Task efficiency
        # steps_to_stable_success: number of tracker.update() calls from episode
        # start until stable success.  This is independent of sim_step numbering
        # (0-based in run_policy.py, 1-based in harness.py).
        steps_to_stable_success: int | None = None
        policy_steps_to_stable_success: int | None = None
        policy_queries_to_stable_success: int | None = None
        time_to_stable_success_s: float | None = None
        stride = max(1, int(policy_stride if policy_stride is not None else self.policy_stride))
        # Resolve expert_time_s: expert_time_step (steps) takes priority over
        # expert_time_s (seconds, backward-compat).  Steps are multiplied by
        # the actual simulator physics_dt, which may differ per embodiment.
        expert_time_s: float | None = None
        if self.expert_time_step is not None:
            expert_time_s = float(self.expert_time_step) * float(self._dt)
        elif self.expert_time_s is not None:
            expert_time_s = float(self.expert_time_s)
        task_efficiency: float | None = None
        if snapshot.stable_success and snapshot.stable_success_step is not None:
            steps_to_stable_success = self._stable_success_update_count or self._metric_update_count
            # Policy steps measure the 20 Hz control/action timeline, while
            # policy queries measure actual model invocations.  Chunked policies
            # can execute many policy steps from one query, so these diagnostics
            # must remain separate.
            policy_steps_to_stable_success = int(math.ceil(steps_to_stable_success / stride))
            if policy_query_count_at_stable_success is not None:
                policy_queries_to_stable_success = int(policy_query_count_at_stable_success)
            elif protocol != "fixed_horizon" and policy_query_count is not None:
                # A reach-and-stop rollout terminates at stable success, making
                # the final query count an exact fallback for older callers.
                policy_queries_to_stable_success = int(policy_query_count)
            time_to_stable_success_s = self._stable_success_time_s
            if time_to_stable_success_s is None:
                time_to_stable_success_s = float(steps_to_stable_success) * float(self._dt)
            if expert_time_s is not None and time_to_stable_success_s > 0:
                task_efficiency = float(expert_time_s) / float(time_to_stable_success_s)

        return MetricResult(
            instant_success=snapshot.instant_success,
            stable_success=snapshot.stable_success,
            ever_instant_success=snapshot.ever_instant_success,
            at_end_success_observed=snapshot.at_end_success_observed,
            first_success_step=snapshot.first_success_step,
            stable_success_step=snapshot.stable_success_step,
            first_stable_success_step=snapshot.stable_success_step,
            steps_to_stable_success=steps_to_stable_success,
            policy_steps_to_stable_success=policy_steps_to_stable_success,
            policy_queries_to_stable_success=policy_queries_to_stable_success,
            time_to_stable_success_s=time_to_stable_success_s,
            expert_time_s=expert_time_s,
            expert_time_step=self.expert_time_step,
            success_hold_s=snapshot.success_hold_s,
            terminal_success_rate=1.0 if primary_terminal_success else 0.0,
            stage_completion_rate=stage_completion_rate,
            normalized_progress_score=normalized_progress_score,
            chain_depth=chain_depth,
            current_stage_completion_rate=current_stage_completion_rate,
            current_normalized_progress_score=current_normalized_progress_score,
            current_chain_depth=current_chain_depth,
            latched_stage_completion_rate=latched_stage_completion_rate,
            latched_normalized_progress_score=latched_normalized_progress_score,
            latched_chain_depth=latched_chain_depth,
            stage_completion=stage_completion,
            current_stage_completion=current_stage_completion,
            stage_first_completion_step=stage_first_completion_step,
            kinematic_grasp_stability_index=kinematic_grasp_stability_index,
            mean_kinematic_grasp_stability=mean_kinematic_grasp_stability,
            tool_selection_accuracy=tool_selection_accuracy,
            tool_switch_success_rate=tool_switch_success_rate,
            tool_switch_total=tool_switch_total,
            tool_switch_successful=tool_switch_successful,
            per_object_best_kinematic_grasp_stability=per_object_best_kinematic_grasp_stability,
            grasp_gsi_diagnostics=grasp_gsi_diagnostics,
            robot_motion_metrics=self._robot_motion_metrics(),
            safety_violation_rate=safety_violation_rate_time,
            safety_violation_rate_time=safety_violation_rate_time,
            safety_violation_rate_event=safety_violation_rate_event,
            safety_violation_step_rate=safety_violation_step_rate,
            safety_violation_events_per_step=safety_violation_events_per_step,
            episode_violation_rate=1.0 if safety_hard_violation else 0.0,
            safety_hard_violation=safety_hard_violation,
            drop_violation=bool(self._drop_violation),
            high_speed_violation=bool(self._high_speed_violation),
            excessive_impact_violation=excessive_impact_violation,
            robot_constraint_violation=bool(self._robot_constraint_violation),
            finger_joint_limit_saturation=bool(self._finger_joint_limit_saturation),
            impact_metric_available=bool(self._impact_metric_available),
            violation_counts=dict(self._violation_counts),
            robot_constraint_diagnostics=self._robot_constraint_diagnostics(),
            joint_limit_active_steps=int(self._joint_limit_active_steps),
            joint_limit_active_step_rate=joint_limit_active_step_rate,
            finger_joint_limit_saturation_steps=int(self._finger_joint_limit_saturation_steps),
            finger_joint_limit_saturation_step_rate=finger_joint_limit_saturation_step_rate,
            max_joint_limit_excess_rad=float(self._max_joint_limit_excess_rad),
            max_joint_limit_excess_joint=self._max_joint_limit_excess_joint,
            at_end_success_budget=at_end_success_budget,
            policy_query_count=policy_query_count,
            terminated_reason=terminated_reason,
            timeseries=ts_dict,
            task_efficiency=task_efficiency,
        )

    def _update_grasp_statistics(self) -> dict[str, dict[str, float | None]]:
        """Update trace-independent grasp aggregates and return this step's values."""
        if self._grasp_detector is None:
            return {}
        output: dict[str, dict[str, float | None]] = {}
        for obj_id in sorted(self._grasp_detector.tracked_objects):
            components = self._grasp_detector.compute_gsi_components(obj_id)
            output[obj_id] = components
            gsi = float(components.get("gsi") or 0.0)
            self._grasp_gsi_best_by_object[obj_id] = max(self._grasp_gsi_best_by_object.get(obj_id, 0.0), gsi)
            if gsi > 0.0:
                self._grasp_positive_gsi_sum += gsi
                self._grasp_positive_gsi_count += 1
            for name, value in components.items():
                if name == "gsi" or value is None:
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                self._grasp_component_sums[name] = self._grasp_component_sums.get(name, 0.0) + numeric
                self._grasp_component_counts[name] = self._grasp_component_counts.get(name, 0) + 1
        return output

    def _grasp_component_diagnostics(self) -> dict[str, float]:
        component_names = (
            "legacy_gsi",
            "kinematic_stability_score",
            "lift_score",
            "hold_score",
            "motion_stability_score",
            "duration_score",
            "anti_slip_score",
        )
        diagnostics: dict[str, float] = {}
        for name in component_names:
            count = self._grasp_component_counts.get(name, 0)
            if count:
                diagnostics[f"mean_{name}"] = self._grasp_component_sums[name] / count
        return diagnostics

    def _update_robot_motion_metrics(self, robot_state: dict[str, Any] | None) -> None:
        if robot_state is None or robot_state.get("qvel") is None:
            self._prev_qvel = None
            self._prev_joint_acceleration = None
            return

        qvel = np.asarray(robot_state["qvel"], dtype=np.float32)
        if qvel.ndim != 1 or qvel.size == 0 or not np.all(np.isfinite(qvel)):
            self._prev_qvel = None
            self._prev_joint_acceleration = None
            return

        self._motion_update_count += 1
        self._motion_velocity_sq_sum += float(np.mean(np.square(qvel)))

        if self._prev_qvel is not None and self._prev_qvel.shape == qvel.shape:
            acceleration = (qvel - self._prev_qvel) / max(1e-9, self._dt)
            self._motion_acceleration_sq_sum += float(np.mean(np.square(acceleration)))
            if self._prev_joint_acceleration is not None and self._prev_joint_acceleration.shape == acceleration.shape:
                jerk = (acceleration - self._prev_joint_acceleration) / max(1e-9, self._dt)
                self._motion_jerk_sq_sum += float(np.mean(np.square(jerk)))
            self._prev_joint_acceleration = acceleration
        else:
            self._prev_joint_acceleration = None
        self._prev_qvel = qvel.copy()

        qeffort = robot_state.get("qeffort")
        if qeffort is not None:
            effort = np.asarray(qeffort, dtype=np.float32)
            if effort.ndim == 1 and effort.size > 0 and np.all(np.isfinite(effort)):
                self._motion_effort_count += 1
                self._motion_effort_sq_sum += float(np.mean(np.square(effort)))

    def _robot_motion_metrics(self) -> dict[str, float]:
        if self._motion_update_count <= 0:
            return {}
        metrics = {
            "joint_velocity_rms": math.sqrt(self._motion_velocity_sq_sum / self._motion_update_count),
        }
        if self._motion_update_count > 1:
            metrics["joint_acceleration_rms"] = math.sqrt(
                self._motion_acceleration_sq_sum / (self._motion_update_count - 1)
            )
        if self._motion_update_count > 2:
            metrics["joint_jerk_rms"] = math.sqrt(
                self._motion_jerk_sq_sum / (self._motion_update_count - 2)
            )
        if self._motion_effort_count > 0:
            metrics["joint_effort_rms"] = math.sqrt(self._motion_effort_sq_sum / self._motion_effort_count)
        return metrics

    def _update_terminal(self, object_states: dict[str, Any], *, sim_step: int) -> None:
        if not self._terminal_available or self._terminal_condition is None:
            self._instant_success = False
            self._at_end_success_observed = False
            return

        self._terminal_ctx["dt"] = self._dt
        instant = self._eval_condition(self._terminal_condition, object_states, self._terminal_ctx)
        self._instant_success = instant
        self._at_end_success_observed = instant
        if instant:
            self._ever_instant_success = True
            if self._first_success_step is None:
                self._first_success_step = int(sim_step)
            self._terminal_hold_s += self._dt
        else:
            self._terminal_hold_s = 0.0

        if (not self._stable_success) and instant and self._terminal_hold_s + 1e-12 >= self.dwell_time_s:
            self._stable_success = True
            self._stable_success_step = int(sim_step)
            self._stable_success_time_s = float(self._elapsed_time_s)

    def _update_stages(self, object_states: dict[str, Any], *, sim_step: int):
        if self._stage_tracker is not None:
            return self._stage_tracker.update(object_states, sim_step=sim_step)
        return None

    def _update_safety_proxy(
        self,
        object_states: dict[str, Any],
        *,
        robot_state: dict[str, Any] | None,
        joint_limits: Any | None,
        joint_command: Mapping[str, Any] | None = None,
    ) -> set[str]:
        self._safety_update_count += 1
        self._safety_total_time_s += self._dt
        self._current_step_violations = set()

        self._update_joint_limit_safety(
            robot_state=robot_state,
            joint_limits=joint_limits,
            joint_command=joint_command,
        )

        table_z = self._table_z()
        for obj_id, state in object_states.items():
            if self._should_track_high_speed(obj_id):
                lin_vel = np.asarray(state.get("lin_vel_world", [0.0, 0.0, 0.0]), dtype=np.float32)
                high_speed = bool(float(np.linalg.norm(lin_vel)) > self._high_speed_threshold_mps)
                high_speed_name = f"high_speed_{obj_id}"
                if high_speed:
                    self._high_speed_violation = True
                    self._mark_step_violation(high_speed_name)
                    if not self._high_speed_active.get(obj_id, False):
                        self._count_violation(high_speed_name)
                    self._high_speed_active[obj_id] = True
                else:
                    self._high_speed_active[obj_id] = False

            # Drop: object below table surface. Count the event on the transition
            # into a dropped state, but only for objects explicitly covered by
            # metrics.safety.drop.  This keeps containers/clutter out of the
            # official drop rate unless the scene opts them in.
            pose = state.get("pose_world")
            below_table = bool(
                self._should_track_lifted_drop(obj_id)
                and table_z is not None
                and pose is not None
                and float(pose[2]) < table_z - self._drop_margin
            )
            drop_name = f"drop_{obj_id}"
            if below_table:
                if self._is_allowed_drop_placement(obj_id, object_states):
                    self._drop_below_table_active[obj_id] = False
                else:
                    self._drop_violation = True
                    self._mark_step_violation(drop_name)
                    if not self._drop_below_table_active.get(obj_id, False):
                        self._count_violation(drop_name)
                    self._drop_below_table_active[obj_id] = True
            else:
                self._drop_below_table_active[obj_id] = False
            self._update_lifted_drop_state(obj_id, state, object_states)

        # Warn if table_z not configured
        if table_z is None and not self._no_table_z_warned and object_states:
            import logging
            logging.getLogger(__name__).info(
                "Safety: table_z not configured; drop detection disabled. "
                "Add metrics.safety.table_z to enable it."
            )
            self._no_table_z_warned = True
        if (
            self._drop_tracking_enabled
            and self._drop_tracked_objects is None
            and not self._no_drop_tracked_objects_warned
            and object_states
        ):
            import logging
            logging.getLogger(__name__).info(
                "Safety: drop detection enabled but no tracked_objects were configured; "
                "set metrics.safety.drop.tracked_objects or metrics.grasp.tracked_objects."
            )
            self._no_drop_tracked_objects_warned = True

        step_violations = set(self._current_step_violations or set())
        self._current_step_violations = None
        if step_violations:
            self._violation_step_count += 1
            self._violation_active_time_s += self._dt
        return step_violations

    def _update_joint_limit_safety(
        self,
        *,
        robot_state: dict[str, Any] | None,
        joint_limits: Any | None,
        joint_command: Mapping[str, Any] | None = None,
    ) -> None:
        self._last_joint_limit_active = False
        self._last_joint_limit_max_excess_rad = 0.0
        self._last_joint_limit_max_excess_joint = None
        self._last_finger_joint_limit_saturation_active = False
        self._last_arm_joint_limit_hard_active = False
        if not self._joint_limit_enabled:
            return
        if robot_state is None or joint_limits is None or robot_state.get("qpos") is None:
            return
        qpos = np.asarray(robot_state["qpos"], dtype=np.float32)
        soft_limits, hard_limits = self._split_joint_limits(joint_limits)
        if isinstance(joint_limits, Mapping):
            self._joint_limit_soft_source = str(joint_limits.get("soft_source") or self._joint_limit_soft_source or "runtime")
            if joint_limits.get("hard_source"):
                self._joint_limit_hard_source = str(joint_limits["hard_source"])
        if soft_limits is None:
            return
        lower = np.asarray(soft_limits[0], dtype=np.float32)
        upper = np.asarray(soft_limits[1], dtype=np.float32)
        if qpos.shape != lower.shape or qpos.shape != upper.shape or qpos.ndim != 1:
            return
        hard_lower = hard_upper = None
        if hard_limits is not None:
            hard_lower = np.asarray(hard_limits[0], dtype=np.float32)
            hard_upper = np.asarray(hard_limits[1], dtype=np.float32)
            if hard_lower.shape != qpos.shape or hard_upper.shape != qpos.shape:
                hard_lower = hard_upper = None
            else:
                self._joint_limit_hard_source_available = True

        joint_names = self._joint_names(robot_state, qpos.size)
        lower_excess = np.maximum(lower - qpos, 0.0)
        upper_excess = np.maximum(qpos - upper, 0.0)
        excess = np.maximum(lower_excess, upper_excess)
        ranges = np.maximum(upper - lower, 0.0)
        arm_dynamic_margin = np.maximum(
            self._joint_limit_margin_abs,
            self._joint_limit_margin_ratio * ranges,
        )
        finger_dynamic_margin = np.maximum(
            self._finger_joint_limit_margin_abs,
            self._finger_joint_limit_margin_ratio * ranges,
        )

        hard_joints: list[str] = []
        hard_excess_by_joint: dict[str, float] = {}
        hard_tolerance_by_joint: dict[str, float] = {}
        finger_saturated_joints: list[str] = []
        for idx, amount in enumerate(excess):
            name = joint_names[idx]
            amount_f = float(amount)
            role = self._joint_role(name)
            if amount_f > self._last_joint_limit_max_excess_rad:
                self._last_joint_limit_max_excess_rad = amount_f
                self._last_joint_limit_max_excess_joint = name
            if amount_f > self._max_joint_limit_excess_rad:
                self._max_joint_limit_excess_rad = amount_f
                self._max_joint_limit_excess_joint = name
            if amount_f > 0.0:
                prev = self._joint_limit_max_excess_by_joint.get(name, 0.0)
                if amount_f > prev:
                    self._joint_limit_max_excess_by_joint[name] = amount_f

            if amount_f > LEGACY_JOINT_LIMIT_SOFT_MARGIN:
                self._last_joint_limit_active = True
                self._joint_limit_soft_counts[name] = int(self._joint_limit_soft_counts.get(name, 0)) + 1

            # A hard-limit source is authoritative.  A true physical limit
            # violation is never downgraded merely because the joint is a
            # finger.  Tuple-only callers retain the legacy duration policy,
            # because historical environments supplied only soft limits.
            hard_excess = 0.0
            hard_tolerance = 0.0
            if hard_lower is not None and hard_upper is not None:
                hard_excess = float(max(hard_lower[idx] - qpos[idx], qpos[idx] - hard_upper[idx], 0.0))
                hard_range = max(float(hard_upper[idx] - hard_lower[idx]), 0.0)
                hard_tolerance = max(
                    self._hard_limit_tolerance_abs,
                    self._hard_limit_tolerance_ratio * hard_range,
                )
                if hard_excess > hard_tolerance:
                    hard_joints.append(name)
                    hard_excess_by_joint[name] = hard_excess
                    hard_tolerance_by_joint[name] = hard_tolerance

            if role in {"finger", "finger_mimic"}:
                if amount_f > float(finger_dynamic_margin[idx]):
                    self._finger_joint_limit_consecutive[name] = (
                        int(self._finger_joint_limit_consecutive.get(name, 0)) + 1
                    )
                else:
                    self._finger_joint_limit_consecutive[name] = 0

                if (
                    amount_f > self._finger_joint_limit_critical_margin_abs
                    or self._finger_joint_limit_consecutive[name]
                    >= self._finger_joint_limit_min_consecutive_steps
                ):
                    finger_saturated_joints.append(name)
                self._joint_limit_consecutive[name] = 0
                continue

            if hard_lower is not None:
                # Hard source was already evaluated above; proximity to a
                # soft operating limit remains diagnostic only.
                self._joint_limit_consecutive[name] = 0
                continue

            if amount_f > float(arm_dynamic_margin[idx]):
                self._joint_limit_consecutive[name] = int(self._joint_limit_consecutive.get(name, 0)) + 1
            else:
                self._joint_limit_consecutive[name] = 0

            if (
                amount_f > self._joint_limit_critical_margin_abs
                or self._joint_limit_consecutive[name] >= self._joint_limit_min_consecutive_steps
            ):
                hard_joints.append(name)

        if self._last_joint_limit_active:
            self._joint_limit_active_steps += 1

        if finger_saturated_joints:
            self._finger_joint_limit_saturation = True
            self._last_finger_joint_limit_saturation_active = True
            self._finger_joint_limit_saturation_steps += 1
            for name in finger_saturated_joints:
                self._finger_joint_limit_saturation_counts[name] = (
                    int(self._finger_joint_limit_saturation_counts.get(name, 0)) + 1
                )
            if self._finger_joint_limit_as_hard_violation:
                hard_joints.extend(finger_saturated_joints)

        if not hard_joints:
            self._joint_limit_hard_active = False
            self._joint_limit_hard_active_joints.clear()
            return

        self._last_arm_joint_limit_hard_active = True
        self._robot_constraint_violation = True
        if self._first_joint_limit_hard_event is None:
            self._first_joint_limit_hard_event = self._build_first_joint_limit_hard_event(
                joint_names=joint_names,
                qpos=qpos,
                soft_lower=lower,
                soft_upper=upper,
                hard_lower=hard_lower,
                hard_upper=hard_upper,
                hard_joints=hard_joints,
                hard_excess_by_joint=hard_excess_by_joint,
                hard_tolerance_by_joint=hard_tolerance_by_joint,
                joint_command=joint_command,
            )
        # Keep hard-limit entries as controller/asset diagnostics by default.
        # They become task-level safety only with an explicit validated-asset
        # opt-in; otherwise safe success reflects task/object safety rather
        # than a known simulator drive or mimic-model artifact.
        if not self._joint_limit_hard_active:
            self._joint_limit_hard_episode_events += 1
            if self._joint_limit_include_in_core_safety:
                self._count_violation("joint_limit")
        if self._joint_limit_include_in_core_safety:
            self._mark_step_violation("joint_limit")
        # Per-joint attribution follows each joint's rising edge, independently
        # of the aggregate event: a second joint may become unsafe while
        # another remains unsafe, and a joint may violate again after recovery.
        active_hard_joints = set(hard_joints)
        new_hard_joints = active_hard_joints - self._joint_limit_hard_active_joints
        for name in sorted(new_hard_joints):
            self._joint_limit_hard_counts[name] = int(self._joint_limit_hard_counts.get(name, 0)) + 1
        self._joint_limit_hard_active_joints = active_hard_joints
        self._joint_limit_hard_active = True

    def record_joint_limit_baseline(
        self,
        phase: str,
        *,
        robot_state: Mapping[str, Any] | None,
        joint_limits: Any | None,
    ) -> dict[str, Any]:
        """Record an unscored reset/home physical-limit baseline.

        This deliberately does not update violation counters or safe-success:
        it distinguishes an asset/reset/controller issue present before policy
        inference from a violation caused during rollout.
        """
        result: dict[str, Any] = {"phase": str(phase), "checked": False, "hard_violation": False}
        if robot_state is None or joint_limits is None or robot_state.get("qpos") is None:
            result["reason"] = "missing_robot_state_or_joint_limits"
            self._joint_limit_baseline_checks.append(result)
            return result
        _, hard_limits = self._split_joint_limits(joint_limits)
        if isinstance(joint_limits, Mapping):
            if joint_limits.get("soft_source"):
                self._joint_limit_soft_source = str(joint_limits["soft_source"])
            if joint_limits.get("hard_source"):
                self._joint_limit_hard_source = str(joint_limits["hard_source"])
        if hard_limits is None:
            result["reason"] = "hard_limit_source_unavailable"
            self._joint_limit_baseline_checks.append(result)
            return result
        qpos = np.asarray(robot_state["qpos"], dtype=np.float32)
        hard_lower = np.asarray(hard_limits[0], dtype=np.float32)
        hard_upper = np.asarray(hard_limits[1], dtype=np.float32)
        if qpos.ndim != 1 or qpos.shape != hard_lower.shape or qpos.shape != hard_upper.shape:
            result["reason"] = "joint_limit_shape_mismatch"
            self._joint_limit_baseline_checks.append(result)
            return result
        self._joint_limit_hard_source_available = True
        names = self._joint_names(robot_state, qpos.size)
        violations: list[dict[str, Any]] = []
        for idx, name in enumerate(names):
            excess = float(max(hard_lower[idx] - qpos[idx], qpos[idx] - hard_upper[idx], 0.0))
            tolerance = max(
                self._hard_limit_tolerance_abs,
                self._hard_limit_tolerance_ratio * max(float(hard_upper[idx] - hard_lower[idx]), 0.0),
            )
            if excess > tolerance:
                violations.append(
                    self._joint_limit_observation(
                        name=name,
                        index=idx,
                        qpos=qpos,
                        soft_lower=None,
                        soft_upper=None,
                        hard_lower=hard_lower,
                        hard_upper=hard_upper,
                        hard_excess=excess,
                        hard_tolerance=tolerance,
                        joint_command=None,
                    )
                )
        result.update(
            checked=True,
            hard_violation=bool(violations),
            hard_limit_source=self._joint_limit_hard_source,
            violating_joints=violations,
        )
        self._joint_limit_baseline_checks.append(result)
        return result

    def _build_first_joint_limit_hard_event(
        self,
        *,
        joint_names: list[str],
        qpos: np.ndarray,
        soft_lower: np.ndarray,
        soft_upper: np.ndarray,
        hard_lower: np.ndarray | None,
        hard_upper: np.ndarray | None,
        hard_joints: list[str],
        hard_excess_by_joint: Mapping[str, float],
        hard_tolerance_by_joint: Mapping[str, float],
        joint_command: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        by_name = {name: idx for idx, name in enumerate(joint_names)}
        violations = [
            self._joint_limit_observation(
                name=name,
                index=by_name[name],
                qpos=qpos,
                soft_lower=soft_lower,
                soft_upper=soft_upper,
                hard_lower=hard_lower,
                hard_upper=hard_upper,
                hard_excess=float(hard_excess_by_joint.get(name, 0.0)),
                hard_tolerance=float(hard_tolerance_by_joint.get(name, 0.0)),
                joint_command=joint_command,
            )
            for name in sorted(set(hard_joints))
            if name in by_name
        ]
        event: dict[str, Any] = {
            "sim_step": int(self._last_step),
            "hard_limit_source": self._joint_limit_hard_source,
            "soft_limit_source": self._joint_limit_soft_source,
            "shield_applied": bool((joint_command or {}).get("shield_applied", False)),
            "violating_joints": violations,
        }
        if joint_command is None:
            event["command_context_available"] = False
            return event
        event["command_context_available"] = True
        for key in (
            "active_joint_names",
            "raw_active_action",
            "full_joint_names",
            "expanded_full_target",
            "shielded_full_target",
        ):
            value = joint_command.get(key)
            if value is not None:
                event[key] = self._json_float_list(value) if key.endswith("target") or key == "raw_active_action" else list(value)
        return event

    def _joint_limit_observation(
        self,
        *,
        name: str,
        index: int,
        qpos: np.ndarray,
        soft_lower: np.ndarray | None,
        soft_upper: np.ndarray | None,
        hard_lower: np.ndarray | None,
        hard_upper: np.ndarray | None,
        hard_excess: float,
        hard_tolerance: float,
        joint_command: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        item: dict[str, Any] = {
            "joint": name,
            "role": self._joint_role(name),
            "qpos": float(qpos[index]),
            "hard_excess_rad": float(hard_excess),
            "hard_tolerance_rad": float(hard_tolerance),
        }
        if hard_lower is not None and hard_upper is not None:
            item["hard_lower"] = float(hard_lower[index])
            item["hard_upper"] = float(hard_upper[index])
        if soft_lower is not None and soft_upper is not None:
            item["operating_lower"] = float(soft_lower[index])
            item["operating_upper"] = float(soft_upper[index])
        if joint_command is not None:
            full_names = [str(value) for value in joint_command.get("full_joint_names") or []]
            if len(full_names) == len(qpos) and name in full_names:
                command_index = full_names.index(name)
                for key, output_key in (
                    ("expanded_full_target", "expanded_target"),
                    ("shielded_full_target", "shielded_target"),
                ):
                    values = joint_command.get(key)
                    if values is not None and len(values) == len(full_names):
                        item[output_key] = float(values[command_index])
            mimic_sources = joint_command.get("mimic_source_by_joint") or {}
            if isinstance(mimic_sources, Mapping) and name in mimic_sources:
                item["mimic_source_joint"] = str(mimic_sources[name])
        return item

    @staticmethod
    def _json_float_list(values: Any) -> list[float]:
        return [float(value) for value in np.asarray(values, dtype=np.float32).reshape(-1)]

    @staticmethod
    def _split_joint_limits(joint_limits: Any) -> tuple[tuple[Any, Any] | None, tuple[Any, Any] | None]:
        """Return ``(soft, hard)`` limits while accepting the legacy tuple API."""
        if isinstance(joint_limits, Mapping):
            soft = joint_limits.get("soft")
            hard = joint_limits.get("hard")
            if not (isinstance(soft, (tuple, list)) and len(soft) == 2):
                soft = hard
            if not (isinstance(hard, (tuple, list)) and len(hard) == 2):
                hard = None
            return soft, hard
        if isinstance(joint_limits, (tuple, list)) and len(joint_limits) == 2:
            return (joint_limits[0], joint_limits[1]), None
        return None, None

    def _joint_role(self, joint_name: str) -> str:
        if self._joint_role_manifest is not None:
            role = self._joint_role_manifest.get(joint_name)
            if role is not None:
                return role
            self._unknown_runtime_joint_names.add(joint_name)
            # Unknown asset joints are treated conservatively as arm joints;
            # diagnostics make the manifest mismatch explicit.
            return "unknown"
        return "finger" if self._is_finger_joint_legacy(joint_name) else "arm"

    def _is_finger_joint(self, joint_name: str) -> bool:
        """Classify a joint as finger/hand joint.

        Arm patterns are checked first (negative filter): joints matching arm
        patterns are NEVER finger joints, even if they also match hand patterns.
        Hand patterns are then checked for positive identification.
        Joints matching neither pattern default to arm behavior (legacy-compatible).
        """
        return self._joint_role(joint_name) in {"finger", "finger_mimic"}

    def _is_finger_joint_legacy(self, joint_name: str) -> bool:
        # Tier 1: definitive arm patterns (negative filter)
        if self._is_arm_joint(joint_name):
            return False
        # Tier 2: hand/finger patterns (positive filter)
        lower_name = joint_name.lower()
        return any(pattern and pattern in lower_name for pattern in self._finger_joint_name_patterns)

    def _is_arm_joint(self, joint_name: str) -> bool:
        """Check if joint_name matches known arm-joint patterns."""
        lower_name = joint_name.lower()
        return any(pattern and pattern in lower_name for pattern in self._arm_joint_name_patterns)

    @staticmethod
    def _joint_names(robot_state: Mapping[str, Any], joint_count: int) -> list[str]:
        raw_names = robot_state.get("joint_names")
        if raw_names is None:
            return [f"joint_{idx}" for idx in range(joint_count)]
        names = [str(name) for name in raw_names]
        if len(names) != joint_count:
            return [f"joint_{idx}" for idx in range(joint_count)]
        return names

    def _robot_constraint_diagnostics(self) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {}
        if self._joint_limit_soft_counts:
            diagnostics["joint_limit_soft_counts"] = dict(sorted(self._joint_limit_soft_counts.items()))
        if self._joint_limit_hard_counts:
            diagnostics["joint_limit_hard_counts"] = dict(sorted(self._joint_limit_hard_counts.items()))
        if self._finger_joint_limit_saturation_counts:
            diagnostics["finger_joint_limit_saturation_counts"] = dict(
                sorted(self._finger_joint_limit_saturation_counts.items())
            )
        if self._joint_limit_max_excess_by_joint:
            diagnostics["joint_limit_max_excess_by_joint"] = {
                name: float(value)
                for name, value in sorted(self._joint_limit_max_excess_by_joint.items())
            }
        if self._joint_limit_baseline_checks:
            diagnostics["joint_limit_baseline_checks"] = list(self._joint_limit_baseline_checks)
        if self._first_joint_limit_hard_event is not None:
            diagnostics["first_joint_limit_hard_event"] = dict(self._first_joint_limit_hard_event)
        diagnostics["joint_limit_active_steps"] = int(self._joint_limit_active_steps)
        diagnostics["joint_limit_hard_episode_events"] = int(
            self._joint_limit_hard_episode_events
        )
        diagnostics["joint_limit_in_core_safety"] = bool(
            self._joint_limit_include_in_core_safety
        )
        diagnostics["finger_joint_limit_saturation_steps"] = int(
            self._finger_joint_limit_saturation_steps
        )
        diagnostics["max_joint_limit_excess_rad"] = float(self._max_joint_limit_excess_rad)
        diagnostics["max_joint_limit_excess_joint"] = self._max_joint_limit_excess_joint
        diagnostics["joint_role_manifest_available"] = bool(self._joint_role_manifest_available)
        diagnostics["joint_role_manifest_robot_key"] = self.robot_key
        diagnostics["joint_limit_hard_source_available"] = bool(self._joint_limit_hard_source_available)
        diagnostics["joint_limit_soft_source"] = self._joint_limit_soft_source
        diagnostics["joint_limit_hard_source"] = self._joint_limit_hard_source
        if self._unknown_runtime_joint_names:
            diagnostics["unknown_runtime_joint_names"] = sorted(self._unknown_runtime_joint_names)
        return diagnostics

    def _snapshot(self) -> MetricSnapshot:
        return MetricSnapshot(
            instant_success=self._instant_success,
            stable_success=self._stable_success,
            ever_instant_success=self._ever_instant_success,
            at_end_success_observed=self._at_end_success_observed,
            first_success_step=self._first_success_step,
            stable_success_step=self._stable_success_step,
            success_hold_s=self._terminal_hold_s,
        )

    def _safety_violation_rate(self, steps: int) -> float:
        return self._safety_violation_rate_time()

    def _safety_violation_rate_time(self) -> float:
        if self._safety_total_time_s <= 0:
            return 0.0
        return self._violation_active_time_s / self._safety_total_time_s

    def _safety_violation_step_rate(self) -> float:
        if self._safety_update_count <= 0:
            return 0.0
        return self._violation_step_count / float(self._safety_update_count)

    def _safety_violation_events_per_step(self) -> float:
        if self._safety_update_count <= 0:
            return 0.0
        return sum(self._violation_counts.values()) / float(self._safety_update_count)

    def _safety_violation_rate_event(self) -> float:
        return self._safety_violation_events_per_step()

    def _count_violation(self, name: str) -> None:
        self._violation_counts[name] = int(self._violation_counts.get(name, 0)) + 1
        self._known_violation_names.add(name)
        if self._current_step_violations is not None:
            self._current_step_violations.add(name)

    def _mark_step_violation(self, name: str) -> None:
        self._known_violation_names.add(name)
        if self._current_step_violations is not None:
            self._current_step_violations.add(name)

    def _count_violation_once(self, name: str) -> None:
        if name in self._drop_event_recorded:
            return
        self._drop_event_recorded.add(name)
        self._count_violation(name)

    def _should_track_lifted_drop(self, obj_id: str) -> bool:
        return (
            self._drop_tracking_enabled
            and self._drop_tracked_objects is not None
            and obj_id in self._drop_tracked_objects
        )

    def _should_track_high_speed(self, obj_id: str) -> bool:
        return self._high_speed_enabled and (
            self._high_speed_tracked_objects is None or obj_id in self._high_speed_tracked_objects
        )

    def _update_lifted_drop_state(
        self,
        obj_id: str,
        state: Mapping[str, Any],
        object_states: Mapping[str, Any],
    ) -> None:
        if not self._should_track_lifted_drop(obj_id):
            return
        pose = state.get("pose_world")
        if pose is None:
            return
        z = float(pose[2])
        initial_z = self._drop_initial_z.setdefault(obj_id, z)
        previous_max_z = self._drop_max_z.get(obj_id, z)
        max_z = max(previous_max_z, z)
        self._drop_max_z[obj_id] = max_z

        if z > initial_z + self._drop_lift_margin:
            self._drop_was_lifted[obj_id] = True
        lin_vel = np.asarray(state.get("lin_vel_world", [0.0, 0.0, 0.0]), dtype=np.float32)
        if lin_vel.size >= 3 and float(lin_vel[2]) < -self._drop_fall_vel_threshold:
            self._drop_falling_fast[obj_id] = True

        if not self._drop_was_lifted.get(obj_id, False):
            return
        if not self._drop_falling_fast.get(obj_id, False):
            return
        if max_z - z <= self._drop_height:
            return
        if z > initial_z + self._drop_settle_margin:
            return
        if self._is_allowed_drop_placement(obj_id, object_states):
            return

        self._drop_violation = True
        self._count_violation_once(f"drop_to_table_{obj_id}")

    def _is_allowed_drop_placement(self, obj_id: str, object_states: Mapping[str, Any]) -> bool:
        condition = self._drop_allowed_conditions.get(obj_id)
        if not isinstance(condition, Mapping):
            return False
        ctx = self._drop_allowed_ctx.setdefault(obj_id, {"dt": self._dt})
        ctx["dt"] = self._dt
        return self._eval_condition(dict(condition), dict(object_states), ctx)

    def _table_z(self) -> float | None:
        safety = self._spec.get("safety") or {}
        if isinstance(safety, Mapping) and safety.get("table_z") is not None:
            return float(safety["table_z"]) + self._table_height_offset
        return None

    def _grasp_table_z(self, grasp_spec: Mapping[str, Any]) -> float | None:
        if grasp_spec.get("table_z") is not None:
            return float(grasp_spec["table_z"])
        return self._table_z()

    @staticmethod
    def _eval_condition(condition: dict[str, Any], states: dict[str, Any], ctx: dict[str, Any]) -> bool:
        return bool(evaluate_condition_node(condition, states, ctx))

    @staticmethod
    def _parse_tracked_objects(value: Any) -> set[str] | None:
        if value is None:
            return None
        if isinstance(value, (str, bytes)):
            return {str(value)}
        try:
            values = [str(item) for item in value]
        except TypeError:
            return {str(value)}
        return set(values) if values else None

    @staticmethod
    def _as_optional_float(value: Any) -> float | None:
        if value is None:
            return None
        return float(value)

    @staticmethod
    def _as_optional_int(value: Any) -> int | None:
        if value is None:
            return None
        return int(value)

    @staticmethod
    def _extract_stage_dependencies(stages: list[dict[str, Any]]) -> dict[str, list[str]]:
        deps: dict[str, list[str]] = {}
        for i, stage in enumerate(stages):
            stage_id = str(stage.get("id", f"stage_{i}"))
            deps[stage_id] = [str(dep) for dep in stage.get("depends_on", []) or []]
        return deps

    @classmethod
    def _extract_stage_tools(
        cls,
        stages: list[dict[str, Any]],
        tool_equivalence_classes: Mapping[str, Any] | None = None,
    ) -> dict[str, list[str]]:
        tools: dict[str, list[str]] = {}
        tool_equivalence_classes = tool_equivalence_classes or {}
        for i, stage in enumerate(stages):
            stage_id = str(stage.get("id", f"stage_{i}"))
            stage_tools = stage.get("tools")
            if stage_tools is not None:
                if isinstance(stage_tools, (str, bytes)):
                    tool_ids = [str(stage_tools)]
                else:
                    tool_ids = [str(tool_id) for tool_id in stage_tools]
            elif stage.get("tool_class") is not None:
                class_name = str(stage["tool_class"])
                raw_tool_ids = tool_equivalence_classes.get(class_name, []) if isinstance(tool_equivalence_classes, Mapping) else []
                tool_ids = [str(tool_id) for tool_id in raw_tool_ids]
            else:
                tool_id = cls._first_tool_id(stage.get("success_condition"))
                tool_ids = [tool_id] if tool_id is not None else []
            if tool_ids:
                tools[stage_id] = tool_ids
        return tools

    @classmethod
    def _first_tool_id(cls, node: Any) -> str | None:
        if isinstance(node, Mapping):
            tool_id = node.get("tool")
            if tool_id is not None:
                return str(tool_id)
            for value in node.values():
                found = cls._first_tool_id(value)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = cls._first_tool_id(item)
                if found is not None:
                    return found
        return None

    def _selected_tool(self, stage_completion: dict[str, bool]) -> str:
        for stage_id, tool_ids in self._stage_tools.items():
            if not stage_completion.get(stage_id, False):
                return tool_ids[0] if tool_ids else ""
        if self._stage_tools:
            last_tool_ids = next(reversed(self._stage_tools.values()))
            return last_tool_ids[0] if last_tool_ids else ""
        return self._terminal_tool or ""
