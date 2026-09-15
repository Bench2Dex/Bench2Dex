"""Validation for terminal raw conditions."""

from __future__ import annotations

from typing import Any


def validate_terminal_raw_condition(
    node: dict[str, Any],
    *,
    allow_sequence: bool = False,
) -> None:
    """Recursively reject hold_duration and (by default) sequence nodes.

    Raises ValueError if a disallowed node type is found.
    """
    ctype = node.get("type")
    if ctype == "hold_duration":
        raise ValueError(
            "hold_duration is not allowed in terminal raw_condition; "
            "use metrics.terminal.dwell_time_s instead."
        )
    if ctype == "sequence" and not allow_sequence:
        raise ValueError(
            "sequence is not allowed in terminal raw_condition by default; "
            "use stages for sequential tracking."
        )
    for child_key in ("conditions", "steps"):
        children = node.get(child_key)
        if isinstance(children, list):
            for child in children:
                validate_terminal_raw_condition(child, allow_sequence=allow_sequence)
    inner = node.get("condition")
    if isinstance(inner, dict):
        validate_terminal_raw_condition(inner, allow_sequence=allow_sequence)