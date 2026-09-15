"""Small USD prim helpers that are easy to unit test."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


def get_current_stage_compat():
    """Return the current USD stage across IsaacLab versions."""
    try:
        from isaaclab.sim.utils import get_current_stage

        return get_current_stage()
    except (ImportError, AttributeError):
        import omni.usd

        return omni.usd.get_context().get_stage()


def create_prim_compat(prim_path: str, prim_type: str = "Xform", *, stage: Any | None = None):
    """Create a USD prim across IsaacLab versions."""
    try:
        import isaaclab.sim as sim_utils

        create_prim = getattr(sim_utils, "create_prim", None)
        if callable(create_prim):
            return create_prim(prim_path, prim_type, stage=stage)
    except Exception:
        pass

    stage = stage or get_current_stage_compat()
    return stage.DefinePrim(prim_path, prim_type)


def delete_prim_compat(prim_path: str, *, stage: Any | None = None):
    """Delete a USD prim across IsaacLab versions."""
    try:
        from isaaclab.sim import utils as sim_utils

        delete_prim = getattr(sim_utils, "delete_prim", None)
        if callable(delete_prim):
            try:
                return delete_prim(prim_path, stage=stage)
            except TypeError:
                return delete_prim(prim_path)
    except Exception:
        pass

    stage = stage or get_current_stage_compat()
    return stage.RemovePrim(prim_path)


def delete_prim_if_valid(stage: Any, delete_prim: Callable[[str], Any], prim_path: str) -> bool:
    """Delete a prim only when it exists on the stage."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return False
    delete_prim(prim_path)
    return True


def delete_children_if_valid(stage: Any, delete_prim: Callable[[str], Any], parent_path: str) -> int:
    """Delete direct children of a parent prim when the parent exists."""
    parent = stage.GetPrimAtPath(parent_path)
    if not parent.IsValid():
        return 0
    deleted = 0
    for child in list(parent.GetChildren()):
        child_path = str(child.GetPath())
        if delete_prim_if_valid(stage, delete_prim, child_path):
            deleted += 1
    return deleted


def delete_prims_if_valid(stage: Any, delete_prim: Callable[[str], Any], prim_paths: Sequence[str]) -> int:
    """Delete all existing prims from a path list."""
    deleted = 0
    for prim_path in prim_paths:
        if delete_prim_if_valid(stage, delete_prim, prim_path):
            deleted += 1
    return deleted


def existing_prim_paths(stage: Any, prim_paths: Sequence[str]) -> list[str]:
    """Return paths that currently resolve to valid prims on the stage."""
    return [prim_path for prim_path in prim_paths if stage.GetPrimAtPath(prim_path).IsValid()]


def wait_for_prims_absent(
    stage: Any,
    prim_paths: Sequence[str],
    flush_fn: Callable[[], Any] | None = None,
    *,
    attempts: int = 1,
) -> list[str]:
    """Flush and return prim paths that are still present after waiting."""
    remaining = existing_prim_paths(stage, prim_paths)
    for _ in range(max(0, attempts)):
        if not remaining:
            break
        if flush_fn is not None:
            flush_fn()
        remaining = existing_prim_paths(stage, prim_paths)
    return remaining


def delete_prims_and_wait(
    stage: Any,
    delete_prim: Callable[[str], Any],
    prim_paths: Sequence[str],
    flush_fn: Callable[[], Any] | None = None,
    *,
    attempts: int = 2,
) -> tuple[int, list[str]]:
    """Delete prims, flush stage updates, and report paths still present."""
    deleted = delete_prims_if_valid(stage, delete_prim, prim_paths)
    remaining = wait_for_prims_absent(stage, prim_paths, flush_fn, attempts=attempts)
    if remaining and hasattr(stage, "RemovePrim"):
        for prim_path in list(remaining):
            stage.RemovePrim(prim_path)
        remaining = wait_for_prims_absent(stage, prim_paths, flush_fn, attempts=attempts)
    return deleted, remaining
