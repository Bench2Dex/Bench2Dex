"""Stage completion tracker extracted from MetricTracker for testability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from success.engine import evaluate_condition_node


@dataclass
class StageResult:
    completed: dict[str, bool] = field(default_factory=dict)
    current_completed: dict[str, bool] = field(default_factory=dict)
    stage_completion_rate: float = 0.0
    normalized_progress_score: float = 0.0
    chain_depth: int = 0
    current_stage_completion_rate: float = 0.0
    current_normalized_progress_score: float = 0.0
    current_chain_depth: int = 0
    latched_stage_completion_rate: float = 0.0
    latched_normalized_progress_score: float = 0.0
    latched_chain_depth: int = 0
    first_completed_step: dict[str, int | None] = field(default_factory=dict)


class StageTracker:
    """Track stage completion with depends_on gating."""

    def __init__(self, stages: list[dict[str, Any]], *, dt: float = 1 / 60) -> None:
        self._stages = [dict(s) for s in stages]
        self._stage_ids = [
            str(s.get("id", f"stage_{i}")) for i, s in enumerate(self._stages)
        ]
        self._completed: dict[str, bool] = {sid: False for sid in self._stage_ids}
        self._current_completed: dict[str, bool] = {sid: False for sid in self._stage_ids}
        self._first_step: dict[str, int | None] = {sid: None for sid in self._stage_ids}
        self._ctx: dict[str, dict[str, Any]] = {sid: {"dt": dt} for sid in self._stage_ids}

    def update(self, object_states: dict[str, Any], *, sim_step: int) -> StageResult:
        condition_true: dict[str, bool] = {}
        for stage, stage_id in zip(self._stages, self._stage_ids):
            condition = stage.get("success_condition")
            if not isinstance(condition, dict):
                condition_true[stage_id] = False
                continue
            ctx = self._ctx[stage_id]
            condition_true[stage_id] = evaluate_condition_node(condition, object_states, ctx)

        # ``current`` means that a stage's own predicate is true now, after its
        # dependencies have been reached at some point.  Requiring every
        # dependency predicate to remain true in the same frame makes current
        # progress meaningless for normal sequential tasks (for example an
        # open-door stage must become false again when the final close-door
        # stage succeeds).
        current_completed = self._resolve_current(
            condition_true,
            previously_completed=self._completed,
        )
        latched_completed = {
            stage_id: bool(self._completed[stage_id] or current_completed[stage_id])
            for stage_id in self._stage_ids
        }
        for stage_id, completed in latched_completed.items():
            if completed and not self._completed[stage_id]:
                self._completed[stage_id] = True
                self._first_step[stage_id] = int(sim_step)
        self._current_completed = current_completed
        return self._result()

    def _resolve_current(
        self,
        condition_true: dict[str, bool],
        *,
        previously_completed: dict[str, bool],
    ) -> dict[str, bool]:
        """Resolve predicates that are true now and dependency-eligible.

        A dependency is eligible when it was latched on an earlier frame or is
        newly satisfied on this frame.  A previously completed stage is *not*
        automatically current: its own predicate must still be true.
        """
        current = {sid: False for sid in self._stage_ids}
        changed = True
        while changed:
            changed = False
            for stage, stage_id in zip(self._stages, self._stage_ids):
                if current[stage_id] or not condition_true.get(stage_id, False):
                    continue
                deps = [str(d) for d in stage.get("depends_on", []) or []]
                if deps and not all(
                    previously_completed.get(dep, False) or current.get(dep, False)
                    for dep in deps
                ):
                    continue
                current[stage_id] = True
                changed = True
        return current

    def _result(self) -> StageResult:
        total = len(self._stage_ids)
        latched_done = sum(1 for v in self._completed.values() if v)
        current_done = sum(1 for v in self._current_completed.values() if v)
        max_chain_depth = self._max_chain_depth()
        current_chain_depth = self._current_milestone_depth()
        latched_chain_depth = self._depth_for(self._completed)
        current_stage_completion_rate = current_done / total if total else 0.0
        current_normalized_progress_score = (
            current_chain_depth / max_chain_depth if max_chain_depth else 0.0
        )
        latched_stage_completion_rate = latched_done / total if total else 0.0
        latched_normalized_progress_score = (
            latched_chain_depth / max_chain_depth if max_chain_depth else 0.0
        )
        return StageResult(
            completed=dict(self._completed),
            current_completed=dict(self._current_completed),
            stage_completion_rate=current_stage_completion_rate,
            normalized_progress_score=current_normalized_progress_score,
            chain_depth=current_chain_depth,
            current_stage_completion_rate=current_stage_completion_rate,
            current_normalized_progress_score=current_normalized_progress_score,
            current_chain_depth=current_chain_depth,
            latched_stage_completion_rate=latched_stage_completion_rate,
            latched_normalized_progress_score=latched_normalized_progress_score,
            latched_chain_depth=latched_chain_depth,
            first_completed_step=dict(self._first_step),
        )

    def _chain_depth(self) -> int:
        return self._depth_for(self._completed)

    def _current_chain_depth(self) -> int:
        return self._current_milestone_depth()

    def _current_milestone_depth(self) -> int:
        """Deepest dependency-eligible stage whose own predicate is true now.

        Dependency predicates need not remain simultaneously true.  The static
        DAG depth therefore measures the current milestone reached along a
        historically valid path, while ``current_stage_completion_rate`` still
        reports how many individual predicates remain true.
        """
        depth_by_stage = self._static_depth_by_stage()
        return max(
            (
                depth_by_stage.get(stage_id, 0)
                for stage_id, is_current in self._current_completed.items()
                if is_current
            ),
            default=0,
        )

    def _depth_for(self, completed: dict[str, bool]) -> int:
        """Longest completed path in the stage dependency DAG.

        Each completed root node has depth 1; each completed non-root has
        depth = max(depth of completed deps) + 1.  The result is the maximum
        depth across all completed stages, or 0 if none are completed.
        """
        dep_map: dict[str, list[str]] = {}
        for stage, stage_id in zip(self._stages, self._stage_ids):
            deps = [str(d) for d in stage.get("depends_on", []) or []]
            dep_map[stage_id] = deps

        memo: dict[str, int] = {}

        def _depth(sid: str) -> int:
            if sid in memo:
                return memo[sid]
            if not completed.get(sid, False):
                memo[sid] = 0
                return 0
            deps = dep_map.get(sid, [])
            if not deps:
                memo[sid] = 1
            else:
                memo[sid] = max(_depth(d) for d in deps) + 1
            return memo[sid]

        return max((_depth(sid) for sid in self._stage_ids), default=0)

    def _max_chain_depth(self) -> int:
        """Longest possible path in the stage dependency DAG."""
        return max(self._static_depth_by_stage().values(), default=0)

    def _static_depth_by_stage(self) -> dict[str, int]:
        """Return each stage's dependency depth independent of current state."""
        dep_map: dict[str, list[str]] = {}
        for stage, stage_id in zip(self._stages, self._stage_ids):
            deps = [str(d) for d in stage.get("depends_on", []) or []]
            dep_map[stage_id] = deps

        memo: dict[str, int] = {}

        def _depth(sid: str) -> int:
            if sid in memo:
                return memo[sid]
            deps = dep_map.get(sid, [])
            if not deps:
                memo[sid] = 1
            else:
                memo[sid] = max((_depth(d) for d in deps), default=0) + 1
            return memo[sid]

        return {sid: _depth(sid) for sid in self._stage_ids}
