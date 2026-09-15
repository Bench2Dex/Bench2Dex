"""Tool selection and switching metrics built on proxy grasp states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .grasp_detector import GraspDetector


@dataclass(frozen=True)
class ToolSwitchEvent:
    step: int
    from_tool: str | None
    to_tool: str | None
    from_stably_released: bool
    to_stably_grasped: bool
    switch_duration_s: float
    successful: bool


@dataclass
class _PendingSwitch:
    start_step: int
    from_tool: str
    to_tool: str | None = None
    elapsed_s: float = 0.0
    release_stable_s: float = 0.0
    grasp_stable_s: float = 0.0
    release_unsafe: bool = False


class ToolTracker:
    def __init__(
        self,
        stage_tools: Mapping[str, Any],
        tool_equivalence_classes: Mapping[str, list[str]],
        grasp_detector: GraspDetector,
        gsi_threshold: float = 0.5,
        release_stable_s: float = 0.3,
        switch_timeout_s: float = 5.0,
        dt: float = 1 / 60,
        stage_dependencies: Mapping[str, list[str]] | None = None,
        release_vel_threshold: float = 0.05,
        grasp_stable_s: float | None = None,
    ) -> None:
        self._grasp_detector = grasp_detector
        self._dt = float(dt)
        self._gsi_threshold = float(gsi_threshold)
        self._release_stable_required_s = max(0.0, float(release_stable_s))
        self._grasp_stable_required_s = max(
            0.0,
            float(release_stable_s if grasp_stable_s is None else grasp_stable_s),
        )
        self._switch_timeout_s = max(0.0, float(switch_timeout_s))
        self._release_vel_threshold = float(release_vel_threshold)
        self._tool_equiv = {
            str(class_name): [str(tool_id) for tool_id in tool_ids]
            for class_name, tool_ids in dict(tool_equivalence_classes or {}).items()
        }
        self._stage_dependencies = {
            str(stage_id): [str(dep) for dep in deps]
            for stage_id, deps in dict(stage_dependencies or {}).items()
        }
        self._stage_tool_options: dict[str, set[str]] = {
            str(stage_id): self._expand_tool_options(raw_tools)
            for stage_id, raw_tools in dict(stage_tools or {}).items()
        }
        self._stage_order = list(self._stage_tool_options)
        self._stage_selections: dict[str, str] = {}
        self._stage_selection_correct: dict[str, bool] = {}
        self._switch_events: list[ToolSwitchEvent] = []
        self._current_tool: str | None = None
        self._pending_switch: _PendingSwitch | None = None
        self._known_tools = self._collect_known_tools()

    def update(
        self,
        object_states: dict[str, Any],
        sim_step: int,
        stage_completion: dict[str, bool],
        dt: float | None = None,
    ) -> None:
        if dt is not None:
            self._dt = float(dt)
        current_tool = self._current_held_tool()
        self._update_stage_selections(current_tool, stage_completion)
        self._update_switches(current_tool, object_states, int(sim_step))

    def compute_tsa(self) -> float | None:
        if not self._stage_tool_options:
            return None
        correct = sum(1 for stage_id in self._stage_tool_options if self._stage_selection_correct.get(stage_id, False))
        return correct / float(len(self._stage_tool_options))

    def compute_tssr(self) -> float | None:
        if not self._switch_events:
            return None
        successful = sum(1 for event in self._switch_events if event.successful)
        return successful / float(len(self._switch_events))

    @property
    def switch_events(self) -> list[ToolSwitchEvent]:
        return list(self._switch_events)

    @property
    def stage_selections(self) -> dict[str, str]:
        return dict(self._stage_selections)

    def finalize(self, sim_step: int) -> None:
        """Close any in-flight switch attempt as a failed attempt."""
        pending = self._pending_switch
        if pending is None:
            return
        from_ok = (
            not pending.release_unsafe
            and pending.release_stable_s + 1e-12 >= self._release_stable_required_s
        )
        to_ok = False
        self._append_switch_event(pending, int(sim_step), from_ok, to_ok)
        self._pending_switch = None

    def _expand_tool_options(self, raw_tools: Any) -> set[str]:
        if raw_tools is None:
            values: list[Any] = []
        elif isinstance(raw_tools, (str, bytes)):
            values = [raw_tools]
        else:
            values = list(raw_tools)
        expanded: set[str] = set()
        for value in values:
            key = str(value)
            if key in self._tool_equiv:
                expanded.update(self._tool_equiv[key])
            else:
                expanded.add(key)
        return {tool_id for tool_id in expanded if tool_id}

    def _collect_known_tools(self) -> set[str]:
        known: set[str] = set()
        for tools in self._stage_tool_options.values():
            known.update(tools)
        for tools in self._tool_equiv.values():
            known.update(tools)
        return known

    def _current_held_tool(self) -> str | None:
        held: list[tuple[float, str]] = []
        for obj_id in self._grasp_detector.tracked_objects:
            state = self._grasp_detector.get_grasp_state(obj_id)
            if state is not None and state.is_held:
                held.append((float(state.hold_duration_s), obj_id))
        if not held:
            return None
        held.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return held[0][1]

    def _update_stage_selections(self, current_tool: str | None, stage_completion: dict[str, bool]) -> None:
        if current_tool is None:
            return
        active = [
            stage_id
            for stage_id in self._stage_order
            if stage_id not in self._stage_selections and self._stage_is_active(stage_id, stage_completion)
        ]
        if not active:
            return

        matching = [stage_id for stage_id in active if current_tool in self._stage_tool_options[stage_id]]
        if matching:
            for stage_id in matching:
                self._lock_stage_selection(stage_id, current_tool, True)
            return

        # If the selected object is not valid for any active stage, treat it as
        # the first pending stage's wrong selection instead of penalizing every
        # parallel active stage at once.
        self._lock_stage_selection(active[0], current_tool, False)

    def _stage_is_active(self, stage_id: str, stage_completion: dict[str, bool]) -> bool:
        deps = self._stage_dependencies.get(stage_id, [])
        return all(bool(stage_completion.get(dep, False)) for dep in deps)

    def _lock_stage_selection(self, stage_id: str, tool_id: str, correct: bool) -> None:
        self._stage_selections[stage_id] = tool_id
        self._stage_selection_correct[stage_id] = bool(correct)

    def _update_switches(self, current_tool: str | None, object_states: dict[str, Any], sim_step: int) -> None:
        previous = self._current_tool
        if current_tool == previous:
            self._advance_pending_switch(current_tool, object_states, sim_step)
            return

        if previous is not None and current_tool != previous:
            if self._counts_as_tool(previous) and (current_tool is None or self._counts_as_tool(current_tool)):
                self._pending_switch = _PendingSwitch(start_step=sim_step, from_tool=previous)
                if current_tool is not None:
                    self._pending_switch.to_tool = current_tool

        elif previous is None and current_tool is not None and self._pending_switch is not None:
            if self._counts_as_tool(current_tool):
                self._pending_switch.to_tool = current_tool

        self._current_tool = current_tool
        self._advance_pending_switch(current_tool, object_states, sim_step)

    def _counts_as_tool(self, tool_id: str) -> bool:
        return not self._known_tools or tool_id in self._known_tools

    def _advance_pending_switch(
        self,
        current_tool: str | None,
        object_states: dict[str, Any],
        sim_step: int,
    ) -> None:
        pending = self._pending_switch
        if pending is None:
            return
        pending.elapsed_s += self._dt

        from_state = self._grasp_detector.get_grasp_state(pending.from_tool)
        from_is_released = from_state is None or not from_state.is_held
        from_vel = _lin_vel_norm(object_states.get(pending.from_tool))
        if from_is_released and from_vel < self._release_vel_threshold:
            pending.release_stable_s += self._dt
        else:
            pending.release_stable_s = 0.0
            if from_is_released and from_vel >= self._release_vel_threshold:
                pending.release_unsafe = True

        if pending.to_tool is None and current_tool is not None and current_tool != pending.from_tool:
            pending.to_tool = current_tool

        if pending.to_tool is not None:
            to_state = self._grasp_detector.get_grasp_state(pending.to_tool)
            to_stable = (
                to_state is not None
                and to_state.is_held
                and self._grasp_detector.compute_gsi(pending.to_tool) >= self._gsi_threshold
            )
            if to_stable:
                pending.grasp_stable_s += self._dt
            else:
                pending.grasp_stable_s = 0.0

        from_ok = (
            not pending.release_unsafe
            and pending.release_stable_s + 1e-12 >= self._release_stable_required_s
        )
        to_ok = pending.grasp_stable_s + 1e-12 >= self._grasp_stable_required_s
        timed_out = pending.elapsed_s + 1e-12 >= self._switch_timeout_s
        if pending.to_tool is not None and (to_ok or timed_out):
            self._append_switch_event(pending, sim_step, from_ok, to_ok)
            self._pending_switch = None
        elif pending.to_tool is None and timed_out:
            self._append_switch_event(pending, sim_step, from_ok, False)
            self._pending_switch = None

    def _append_switch_event(
        self,
        pending: _PendingSwitch,
        sim_step: int,
        from_ok: bool,
        to_ok: bool,
    ) -> None:
        self._switch_events.append(
            ToolSwitchEvent(
                step=int(sim_step),
                from_tool=pending.from_tool,
                to_tool=pending.to_tool,
                from_stably_released=bool(from_ok),
                to_stably_grasped=bool(to_ok),
                switch_duration_s=float(pending.elapsed_s),
                successful=bool(from_ok and to_ok),
            )
        )


def _lin_vel_norm(state: Any) -> float:
    if state is None:
        return 0.0
    if isinstance(state, Mapping):
        lin = state.get("lin_vel_world", (0.0, 0.0, 0.0))
    else:
        lin = getattr(state, "lin_vel_world", (0.0, 0.0, 0.0))
    try:
        return float(sum(float(v) * float(v) for v in list(lin)[:3]) ** 0.5)
    except (TypeError, ValueError):
        return 0.0
