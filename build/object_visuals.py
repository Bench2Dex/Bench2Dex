"""Helpers for object-level visual overrides."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Dict, Tuple

from utils.usd_prims import get_current_stage_compat


Color3 = Tuple[float, float, float]


def resolve_object_color(obj: Mapping) -> Color3 | None:
    """Parse an optional object-level color override from scene YAML."""
    raw_color = obj.get("color")
    if raw_color is None:
        return None
    if isinstance(raw_color, (str, bytes)) or not isinstance(raw_color, (list, tuple)) or len(raw_color) != 3:
        obj_id = str(obj.get("id", "<unknown>"))
        raise ValueError(f"Object '{obj_id}' color must be a numeric sequence of length 3.")
    return tuple(float(v) for v in raw_color)


def _get_current_stage():
    return get_current_stage_compat()


def _get_usd_visual_modules():
    from pxr import Gf, Sdf, UsdShade

    return Gf, Sdf, UsdShade


def _material_path_for_object(object_prim_path: str) -> str:
    object_name = object_prim_path.rsplit("/", 1)[-1]
    safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", object_name).strip("_") or "Object"
    return f"/World/Looks/ObjectColor_{safe_name}"


def apply_object_color_override(
    *,
    object_prim_path: str,
    color: Color3,
    stage=None,
) -> str:
    """Create and bind a PreviewSurface material that overrides object color."""
    stage = stage or _get_current_stage()
    object_prim = stage.GetPrimAtPath(object_prim_path)
    if not object_prim.IsValid():
        raise ValueError(f"Object prim '{object_prim_path}' was not found on the current stage.")

    Gf, Sdf, UsdShade = _get_usd_visual_modules()
    material_path = _material_path_for_object(object_prim_path)
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    surface_output = material.CreateSurfaceOutput()
    try:
        surface_output.ConnectToSource(shader.ConnectableAPI(), "surface")
    except Exception:
        shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        surface_output.ConnectToSource(shader_output)

    binding_api = UsdShade.MaterialBindingAPI.Apply(object_prim)
    binding_api.Bind(material, UsdShade.Tokens.strongerThanDescendants)
    return material_path


def apply_object_color_overrides(
    *,
    object_prim_paths: Mapping[str, str],
    object_colors: Mapping[str, Color3],
    stage=None,
) -> Dict[str, str]:
    """Apply all configured object color overrides and return material paths."""
    applied: Dict[str, str] = {}
    for obj_id, color in object_colors.items():
        prim_path = object_prim_paths.get(obj_id)
        if prim_path is None:
            raise ValueError(f"Object '{obj_id}' has a color override but no prim path was recorded.")
        applied[obj_id] = apply_object_color_override(
            object_prim_path=prim_path,
            color=color,
            stage=stage,
        )
    return applied
