"""
Success-condition evaluation entry point.

Scenes now use task-specific custom evaluators, so the engine only needs to
handle structural combinators and `custom` leaf nodes.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional

from success import condition_evaluator


def evaluate_success_conditions(
    conditions: List[Dict[str, Any]],
    object_states: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> bool:
    """Return *True* when **all** top-level conditions are satisfied."""
    if context is None:
        context = {}
    return all(_eval_node(condition, object_states, context) for condition in conditions)


def _eval_node(
    node: Dict[str, Any],
    states: Dict[str, Any],
    ctx: Dict[str, Any],
) -> bool:
    ctype = node["type"]

    if ctype == "all":
        return all(_eval_node(child, states, ctx) for child in node["conditions"])

    if ctype == "any":
        return any(_eval_node(child, states, ctx) for child in node["conditions"])

    if ctype == "sequence":
        key = f"_seq_{id(node)}"
        step_idx = int(ctx.get(key, 0))
        steps = node["steps"]
        if step_idx >= len(steps):
            return True
        if _eval_node(steps[step_idx], states, ctx):
            ctx[key] = step_idx + 1
        return int(ctx.get(key, 0)) >= len(steps)

    if ctype == "not":
        return not _eval_node(node["condition"], states, ctx)

    if ctype == "hold_duration":
        timer_key = f"_hold_{id(node)}"
        if _eval_node(node["condition"], states, ctx):
            ctx[timer_key] = float(ctx.get(timer_key, 0.0)) + float(ctx.get("dt", 1 / 60))
        else:
            ctx[timer_key] = 0.0
        return float(ctx[timer_key]) >= float(node["seconds"])

    if ctype == "custom":
        mod = importlib.import_module(node["evaluator"])
        return mod.check_success(states, ctx, node.get("params", {}))

    # Delegate semantic leaf types to the full evaluator
    return bool(condition_evaluator.evaluate_condition_tree(node, states, ctx))


_ENGINE_TYPES = frozenset({"all", "any", "not", "sequence", "hold_duration", "custom"})


def evaluate_condition_node(
    node: Dict[str, Any],
    object_states: Dict[str, Any],
    context: Dict[str, Any],
) -> bool:
    """Evaluate a single condition node.

    Handles structural combinators (all/any/not/sequence/hold_duration/custom)
    via the engine.  Delegates semantic leaf types (object_lifted, object_inside,
    etc.) to ``condition_evaluator.evaluate_condition_tree``.
    """
    ctype = node.get("type")
    if ctype in _ENGINE_TYPES:
        return _eval_node(node, object_states, context)
    return bool(condition_evaluator.evaluate_condition_tree(node, object_states, context))
