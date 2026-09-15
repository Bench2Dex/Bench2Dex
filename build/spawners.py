"""Spawner helper functions for scene assets."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Dict, Tuple

from .object_initial_state import DEFAULT_ARTICULATION_DENSITY, DEFAULT_ARTICULATION_FIX_ROOT_LINK

if TYPE_CHECKING:
    import isaaclab.sim as sim_utils


def _resolve_path(path: str, task_dir: str) -> str:
    if not path or os.path.isabs(path) or "://" in path:
        resolved = path
    else:
        resolved = os.path.abspath(os.path.join(task_dir, path))
    if resolved and os.path.exists(resolved):
        return resolved
    usd2_fallback = resolved.replace("\\usd2\\", "\\usd\\").replace("/usd2/", "/usd/") if resolved else resolved
    if usd2_fallback and usd2_fallback != resolved and os.path.exists(usd2_fallback):
        return usd2_fallback
    return resolved



def _load_sim_utils():
    import isaaclab.sim as sim_utils

    return sim_utils


def _build_usd_cfg(
    path: str,
    scale: Tuple[float, float, float] | None,
    body: str,
    spawn: Mapping | None = None,
) -> Any:
    sim_utils = _load_sim_utils()
    usd_kw: Dict = {"usd_path": path}
    if scale is not None:
        usd_kw["scale"] = scale

    if body == "dynamic":
        rigid_props_cfg = spawn.get("rigid_props", {}) if isinstance(spawn, Mapping) else {}
        kinematic = bool(rigid_props_cfg.get("kinematic_enabled", False))
        disable_gravity = bool(rigid_props_cfg.get("disable_gravity", kinematic))
        usd_kw["rigid_props"] = sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=kinematic,           # True ⇒ 固定到世界, 不受物理推动, 但保留碰撞
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=disable_gravity,
        )
        usd_kw["collision_props"] = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
        # 支持在 YAML asset 定义中设置 mass 或 density (可选)
        mass_props = spawn.get("mass_props", {}) if isinstance(spawn, Mapping) else {}
        if mass_props:
            mass_val = mass_props.get("mass", None)
            density_val = mass_props.get("density", None)
            mp_kwargs = {}
            if mass_val is not None:
                mp_kwargs["mass"] = float(mass_val)
            if density_val is not None:
                mp_kwargs["density"] = float(density_val)
            if mp_kwargs:
                usd_kw["mass_props"] = sim_utils.MassPropertiesCfg(**mp_kwargs)
    elif body == "articulation":
        articulation_props = spawn.get("articulation_props", {}) if isinstance(spawn, Mapping) else {}
        mass_props = spawn.get("mass_props", {}) if isinstance(spawn, Mapping) else {}
        usd_kw["articulation_props"] = sim_utils.ArticulationRootPropertiesCfg(
            articulation_enabled=True,
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            fix_root_link=articulation_props.get("fix_root_link", DEFAULT_ARTICULATION_FIX_ROOT_LINK),
        )
        usd_kw["mass_props"] = sim_utils.MassPropertiesCfg(
            density=mass_props.get("density", DEFAULT_ARTICULATION_DENSITY)
        )
        usd_kw["collision_props"] = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
    elif body == "deformable":
        usd_kw["deformable_props"] = sim_utils.DeformableBodyPropertiesCfg(
            deformable_enabled=True,
            kinematic_enabled=False,
            solver_position_iteration_count=16,
            vertex_velocity_damping=0.02,
            self_collision=False,
            contact_offset=0.003,
            rest_offset=0.0,
            max_depenetration_velocity=5.0,
        )
    else:
        raise ValueError(f"Unsupported body_type '{body}'.")

    return sim_utils.UsdFileCfg(**usd_kw)
