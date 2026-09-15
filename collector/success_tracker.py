"""Persistent success tracking during episode collection."""

from __future__ import annotations

from typing import Any

from success import evaluate_success_conditions


class SuccessTracker:
    """Track success conditions incrementally during episode collection.

    This class evaluates success conditions every control cycle and maintains
    state across the episode, allowing for hold_duration and other temporal
    success conditions to work correctly.
    """

    def __init__(self, conditions: list[dict[str, Any]] | None, *, dt: float = 1 / 60) -> None:
        """Initialize the success tracker.

        Args:
            conditions: List of success condition dicts from scene YAML, or None if no conditions.
            dt: Default timestep in seconds (default 1/60 for 60Hz).
        """
        self._conditions = list(conditions or [])
        self._ctx: dict[str, Any] = {"dt": float(dt)}
        self.available = bool(self._conditions)
        self.success = False
        self._success_timestamps: list[int] = []  # sim_step when success first became True

    def update(self, object_states: dict[str, Any], sim_step: int, dt: float | None = None) -> bool:
        """Update success state every control cycle.

        Args:
            object_states: Current object states from scene
            sim_step: Current simulation step
            dt: Optional new dt to update context (for variable timestep)

        Returns:
            Current success state (True if all conditions satisfied).
        """
        # Always update dt if provided
        if dt is not None:
            self._ctx["dt"] = float(dt)

        if not self.available:
            self.success = False
            return False

        new_success = bool(
            evaluate_success_conditions(self._conditions, object_states, self._ctx)
        )

        # Record first timestamp when success becomes True
        if new_success and not self.success:
            self._success_timestamps.append(sim_step)

        self.success = new_success
        return self.success

    def get_success_timestamps(self) -> list[int]:
        """Return list of sim_step when success first occurred."""
        return self._success_timestamps.copy()

    def reset(self) -> None:
        """Reset success state for a new episode."""
        self.success = False
        self._success_timestamps.clear()
        self._ctx = {"dt": self._ctx.get("dt", 1 / 60)}