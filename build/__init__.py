"""Public API for scene building."""

__all__ = [
    "GROUP_PATHS",
    "FillLightGeneralizationCfg",
    "SAFE_RESAMPLE_GROUPS",
    "SecondaryDistantLightGeneralizationCfg",
    "UNSAFE_RESAMPLE_GROUPS",
    "UsdSceneDistantLightCfg",
    "UsdSceneDomeLightCfg",
    "UsdSceneLightCfg",
    "active_perturbation_axes",
    "apply_object_color_overrides",
    "audit_clutter_assets",
    "build_scene",
    "collect_task_asset_codes",
    "dict_to_generalization_sample",
    "load_scene_generalization_config",
    "LightVisibilityGuardCfg",
    "merge_generalization_samples",
    "merge_scene_generalization_overrides",
    "parse_scene_generalization_config",
    "relativize_sample_paths",
    "resample_light",
    "resample_usd_scene_light",
    "resolve_sample_paths",
    "resolve_table_spec",
    "RobotAssetGeneralizationCfg",
    "rpy_deg_to_quat",
    "sample_scene_generalization",
    "scene_generalization_config_to_dict",
    "scene_generalization_sample_debug_lines",
    "scene_generalization_sample_to_dict",
    "validate_resolved_object_placement_keys",
]


def __getattr__(name):
    if name == "apply_object_color_overrides":
        from .object_visuals import apply_object_color_overrides

        return apply_object_color_overrides
    if name in {
        "GROUP_PATHS",
        "FillLightGeneralizationCfg",
        "LightVisibilityGuardCfg",
        "SAFE_RESAMPLE_GROUPS",
        "SecondaryDistantLightGeneralizationCfg",
        "UNSAFE_RESAMPLE_GROUPS",
        "UsdSceneDistantLightCfg",
        "UsdSceneDomeLightCfg",
        "UsdSceneLightCfg",
        "active_perturbation_axes",
        "audit_clutter_assets",
        "collect_task_asset_codes",
        "dict_to_generalization_sample",
        "load_scene_generalization_config",
        "merge_generalization_samples",
        "merge_scene_generalization_overrides",
        "parse_scene_generalization_config",
        "relativize_sample_paths",
        "resample_light",
        "resample_usd_scene_light",
        "resolve_sample_paths",
        "RobotAssetGeneralizationCfg",
        "sample_scene_generalization",
        "scene_generalization_config_to_dict",
        "scene_generalization_sample_debug_lines",
        "scene_generalization_sample_to_dict",
        "validate_resolved_object_placement_keys",
    }:
        from . import generalization as _generalization

        return getattr(_generalization, name)
    if name == "rpy_deg_to_quat":
        from .geometry import rpy_deg_to_quat

        return rpy_deg_to_quat
    if name in {"build_scene", "resolve_table_spec"}:
        from .scene_builder import build_scene, resolve_table_spec

        return {
            "build_scene": build_scene,
            "resolve_table_spec": resolve_table_spec,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
