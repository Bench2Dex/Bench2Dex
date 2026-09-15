"""Proxy grasp-state detection and kinematic grasp-quality scoring.

The benchmark does not yet have reliable hand-object contact geometry or hand
FK for every robot embodiment.  GSI is therefore a kinematic diagnostic: lifted,
stably held, low object motion, sufficient duration, and no slip events.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np


@dataclass
class GraspState:
    is_lifted: bool = False
    is_stable: bool = False
    is_held: bool = False
    # Cumulative hold duration across all grasp intervals in this episode.
    # This value only increases while is_held is True and is never reset on
    # release.  It therefore represents the total time the object has been
    # stably held, not the duration of the current continuous grasp.
    hold_duration_s: float = 0.0
    slip_count: int = 0
    first_lift_step: int | None = None
    first_held_step: int | None = None
    lin_vel_norm: float = 0.0
    ang_vel_norm: float = 0.0
    initial_z: float | None = None
    _seen_once: bool = False
    _hold_candidate_frames: int = 0


@dataclass(frozen=True)
class GraspEvent:
    obj_id: str
    event_type: str
    step: int
    hold_duration_s: float = 0.0


class GraspDetector:
    def __init__(
        self,
        table_z: float | None,
        lift_margin: float = 0.02,
        ang_vel_threshold: float = 1.0,
        fall_vel_threshold: float = 0.5,
        min_hold_frames: int = 3,
        slip_lin_vel: float = 0.3,
        slip_ang_vel: float = 2.0,
        dt: float = 1 / 60,
        gsi_weights: tuple[float, float, float, float] | list[float] | None = None,
        min_hold_s: float = 0.5,
        max_slip_count: float = 3.0,
        ang_vel_scale: float = 1.0,
        lin_vel_scale: float | None = None,
        tracked_objects: Iterable[str] | None = None,
    ) -> None:
        self.table_z = None if table_z is None else float(table_z)
        self.lift_margin = float(lift_margin)
        self.ang_vel_threshold = float(ang_vel_threshold)
        self.fall_vel_threshold = float(fall_vel_threshold)
        self.min_hold_frames = max(1, int(min_hold_frames))
        self.slip_lin_vel = float(slip_lin_vel)
        self.slip_ang_vel = float(slip_ang_vel)
        self.dt = float(dt)
        weights = tuple(float(v) for v in (gsi_weights or (0.3, 0.3, 0.2, 0.2)))
        if len(weights) != 4:
            raise ValueError("gsi_weights must contain four values: pose, stability, duration, anti_slip.")
        if any(v < 0 for v in weights) or sum(weights) <= 0:
            raise ValueError("gsi_weights must be non-negative and sum to a positive value.")
        total = sum(weights)
        self.gsi_weights = tuple(v / total for v in weights)
        self.min_hold_s = max(1e-9, float(min_hold_s))
        self.max_slip_count = max(1e-9, float(max_slip_count))
        self.ang_vel_scale = max(1e-9, float(ang_vel_scale))
        self.lin_vel_scale = max(1e-9, float(self.slip_lin_vel if lin_vel_scale is None else lin_vel_scale))
        self._configured_tracked_objects = (
            {str(obj_id) for obj_id in tracked_objects}
            if tracked_objects is not None
            else None
        )
        self._states: dict[str, GraspState] = {}

    def update(
        self,
        object_states: dict[str, Any],
        sim_step: int,
        dt: float | None = None,
    ) -> list[GraspEvent]:
        if dt is not None:
            self.dt = float(dt)
        events: list[GraspEvent] = []
        present = set(str(obj_id) for obj_id in object_states)
        if self._configured_tracked_objects is not None:
            present &= self._configured_tracked_objects

        for obj_id in sorted(set(self._states) - present):
            state = self._states[obj_id]
            if state.is_held:
                events.append(GraspEvent(obj_id, "grasp_end", int(sim_step), state.hold_duration_s))
            state.is_lifted = False
            state.is_stable = False
            state.is_held = False
            state._hold_candidate_frames = 0
            state.lin_vel_norm = 0.0
            state.ang_vel_norm = 0.0

        for obj_id, raw_state in object_states.items():
            obj_key = str(obj_id)
            if self._configured_tracked_objects is not None and obj_key not in self._configured_tracked_objects:
                continue
            state = self._states.setdefault(obj_key, GraspState())
            pose = _get_value(raw_state, "pose_world", None)
            lin_vel = _as_vec3(_get_value(raw_state, "lin_vel_world", (0.0, 0.0, 0.0)))
            ang_vel = _as_vec3(_get_value(raw_state, "ang_vel_world", (0.0, 0.0, 0.0)))
            lin_norm = float(np.linalg.norm(lin_vel))
            ang_norm = float(np.linalg.norm(ang_vel))

            if state.initial_z is None and pose is not None:
                try:
                    state.initial_z = float(pose[2])
                except (TypeError, ValueError, IndexError):
                    state.initial_z = None
            lifted = self._is_lifted(pose, state)
            stable = ang_norm < self.ang_vel_threshold
            not_falling = float(lin_vel[2]) > -self.fall_vel_threshold
            if lifted and state.first_lift_step is None:
                state.first_lift_step = int(sim_step)

            prev_held = state.is_held
            state.is_lifted = lifted
            state.is_stable = stable
            state.lin_vel_norm = lin_norm
            state.ang_vel_norm = ang_norm

            if not state._seen_once:
                state._seen_once = True
                state._hold_candidate_frames = 0
                state.is_held = False
                continue

            if prev_held and (lin_norm > self.slip_lin_vel or ang_norm > self.slip_ang_vel):
                state.slip_count += 1
                events.append(GraspEvent(obj_key, "slip", int(sim_step), 0.0))

            grasp_like = lifted and stable and not_falling
            if grasp_like:
                state._hold_candidate_frames += 1
            else:
                state._hold_candidate_frames = 0

            if state._hold_candidate_frames >= self.min_hold_frames:
                state.is_held = True
                if not prev_held:
                    if state.first_held_step is None:
                        state.first_held_step = int(sim_step)
                    events.append(GraspEvent(obj_key, "grasp_start", int(sim_step), 0.0))
                state.hold_duration_s += self.dt
            else:
                state.is_held = False
                if prev_held:
                    events.append(GraspEvent(obj_key, "grasp_end", int(sim_step), state.hold_duration_s))

        return events

    def get_grasp_state(self, obj_id: str) -> GraspState | None:
        return self._states.get(str(obj_id))

    def compute_gsi(self, obj_id: str) -> float:
        return float(self.compute_gsi_components(obj_id)["gsi"])

    def compute_gsi_components(self, obj_id: str) -> dict[str, float | None]:
        state = self.get_grasp_state(obj_id)
        if state is None:
            return _empty_components()

        pose_score = 1.0 if state.is_held else (0.5 if state.is_lifted else 0.0)
        stability_score = math.exp(-state.ang_vel_norm / self.ang_vel_scale) if state.is_lifted else 0.0
        duration_score = min(1.0, state.hold_duration_s / self.min_hold_s)
        anti_slip_score = 1.0 - min(1.0, float(state.slip_count) / self.max_slip_count)

        weights = self.gsi_weights
        components = (pose_score, stability_score, duration_score, anti_slip_score)
        if self.table_z is None:
            non_pose = weights[1:]
            total = sum(non_pose)
            if total <= 0:
                legacy_score = 0.0
            else:
                legacy_score = sum((w / total) * c for w, c in zip(non_pose, components[1:]))
        else:
            legacy_score = sum(w * c for w, c in zip(weights, components))
        legacy_score = _clamp01(legacy_score)

        lift_score = 1.0 if state.is_lifted else 0.0
        if state.is_held:
            hold_score = 1.0
        elif state.is_lifted and state.is_stable:
            hold_score = 0.35
        elif state.is_lifted:
            hold_score = 0.15
        else:
            hold_score = 0.0
        lin_stability_score = math.exp(-state.lin_vel_norm / self.lin_vel_scale) if state.is_lifted else 0.0
        ang_stability_score = stability_score
        motion_stability_score = math.sqrt(lin_stability_score * ang_stability_score)
        duration_confidence = 0.75 + 0.25 * duration_score if state.is_held else 1.0
        kinematic_stability_score = _clamp01(
            lift_score * hold_score * motion_stability_score * duration_confidence * anti_slip_score
        )

        return {
            "gsi": kinematic_stability_score,
            "legacy_gsi": legacy_score,
            "kinematic_stability_score": kinematic_stability_score,
            "lift_score": lift_score,
            "hold_score": hold_score,
            "motion_stability_score": motion_stability_score,
            "duration_score": duration_score,
            "anti_slip_score": anti_slip_score,
        }

    @property
    def tracked_objects(self) -> set[str]:
        return set(self._states)

    @property
    def proxy_available(self) -> bool:
        """Whether lift-based kinematic grasp scoring is observable."""
        return self.table_z is not None

    def _is_lifted(self, pose: Any, state: GraspState) -> bool:
        if self.table_z is None:
            return False
        if pose is None:
            return False
        try:
            z = float(pose[2])
        except (TypeError, ValueError, IndexError):
            return False
        if state.initial_z is None:
            return False
        return z > state.initial_z + self.lift_margin


def _get_value(state: Any, key: str, default: Any) -> Any:
    if isinstance(state, Mapping):
        return state.get(key, default)
    return getattr(state, key, default)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _empty_components() -> dict[str, float | None]:
    return {
        "gsi": 0.0,
        "legacy_gsi": 0.0,
        "kinematic_stability_score": 0.0,
        "lift_score": 0.0,
        "hold_score": 0.0,
        "motion_stability_score": 0.0,
        "duration_score": 0.0,
        "anti_slip_score": 0.0,
    }


def _as_vec3(value: Any) -> np.ndarray:
    arr = np.asarray(value if value is not None else (0.0, 0.0, 0.0), dtype=np.float32)
    if arr.ndim == 0:
        return np.asarray((float(arr), 0.0, 0.0), dtype=np.float32)
    if arr.shape[0] < 3:
        padded = np.zeros(3, dtype=np.float32)
        padded[: arr.shape[0]] = arr
        return padded
    return arr[:3]
