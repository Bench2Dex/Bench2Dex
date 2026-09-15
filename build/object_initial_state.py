import re
from collections.abc import Mapping, Sequence
from typing import Any

DEFAULT_ARTICULATION_FIX_ROOT_LINK = False
DEFAULT_ARTICULATION_DENSITY = 100.0

DEFAULT_ARTICULATION_ACTUATOR_SPECS: dict[str, dict[str, Any]] = {
    "joints": {
        "joint_names_expr": [".*"],
        "effort_limit_sim": 400.0,
        "velocity_limit_sim": 100.0,
        "stiffness": 0.0,
        "damping": 1.0,
    }
}

_ACTUATOR_ALIAS_KEYS = {
    "effort_limit": "effort_limit_sim",
    "velocity_limit": "velocity_limit_sim",
}
_OPTIONAL_ACTUATOR_FLOAT_KEYS = (
    "effort_limit_sim",
    "velocity_limit_sim",
    "stiffness",
    "damping",
    "armature",
    "friction",
    "dynamic_friction",
    "viscous_friction",
)
_REQUIRED_ACTUATOR_FLOAT_KEYS = ("stiffness", "damping")
_ALLOWED_ACTUATOR_KEYS = frozenset({"joint_names_expr", *_OPTIONAL_ACTUATOR_FLOAT_KEYS, *_ACTUATOR_ALIAS_KEYS.keys()})

_ALLOWED_ARTICULATION_SPAWN_KEYS = frozenset({"articulation_props", "mass_props", "actuators", "joint_limits", "joint_pos"})
_ALLOWED_ARTICULATION_PROP_KEYS = frozenset({"fix_root_link"})
_MASS_PROP_ALIAS_KEYS = {"linear_density": "density"}
_ALLOWED_MASS_PROP_KEYS = frozenset({"density", *_MASS_PROP_ALIAS_KEYS.keys()})
_ALLOWED_JOINT_LIMIT_KEYS = frozenset({"lower", "upper", "unit"})
_ALLOWED_JOINT_LIMIT_UNITS = frozenset({"rad", "deg"})


def _as_float(value: Any, *, obj_id: str, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Object '{obj_id}' {field_name} must be numeric.") from exc


def _as_bool(value: Any, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError(f"{label} must be a boolean.")


def _normalize_joint_names_expr(value: Any, *, obj_id: str, group_name: str) -> list[str]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(
            f"Object '{obj_id}' actuator group '{group_name}' joint_names_expr must be a sequence of strings."
        )

    normalized: list[str] = []
    for expr in value:
        if not isinstance(expr, str) or not expr:
            raise TypeError(
                f"Object '{obj_id}' actuator group '{group_name}' joint_names_expr entries must be non-empty strings."
            )
        try:
            re.compile(expr)
        except re.error as exc:
            raise ValueError(
                f"Object '{obj_id}' actuator group '{group_name}' has invalid joint_names_expr '{expr}': {exc}."
            ) from exc
        normalized.append(expr)

    if not normalized:
        raise TypeError(f"Object '{obj_id}' actuator group '{group_name}' joint_names_expr must not be empty.")
    return normalized


def _resolve_actuator_limit_value(
    spec: Mapping[str, Any],
    *,
    obj_id: str,
    group_name: str,
    field_name: str,
) -> float | None:
    canonical_key = _ACTUATOR_ALIAS_KEYS.get(field_name, field_name)
    canonical_value = spec.get(canonical_key)
    alias_value = spec.get(field_name) if field_name in _ACTUATOR_ALIAS_KEYS else None

    if canonical_value is not None and alias_value is not None:
        canonical_float = _as_float(canonical_value, obj_id=obj_id, field_name=f"{canonical_key}")
        alias_float = _as_float(alias_value, obj_id=obj_id, field_name=f"{field_name}")
        if canonical_float != alias_float:
            raise ValueError(
                f"Object '{obj_id}' actuator group '{group_name}' sets both '{field_name}' and "
                f"'{canonical_key}' with different values."
            )
        return canonical_float

    value = canonical_value if canonical_value is not None else alias_value
    if value is None:
        return None
    return _as_float(value, obj_id=obj_id, field_name=f"actuator group '{group_name}' {canonical_key}")


def _normalize_actuator_group(
    group_name: str,
    spec: Mapping[str, Any],
    *,
    obj_id: str,
) -> dict[str, Any]:
    unknown_keys = sorted(set(spec.keys()) - _ALLOWED_ACTUATOR_KEYS)
    if unknown_keys:
        raise ValueError(
            f"Object '{obj_id}' actuator group '{group_name}' has unsupported keys: {', '.join(unknown_keys)}."
        )

    if "joint_names_expr" not in spec:
        raise TypeError(f"Object '{obj_id}' actuator group '{group_name}' must define joint_names_expr.")

    normalized: dict[str, Any] = {
        "joint_names_expr": _normalize_joint_names_expr(spec["joint_names_expr"], obj_id=obj_id, group_name=group_name)
    }

    for field_name in _REQUIRED_ACTUATOR_FLOAT_KEYS:
        if field_name not in spec:
            raise TypeError(f"Object '{obj_id}' actuator group '{group_name}' must define {field_name}.")
        normalized[field_name] = _as_float(
            spec[field_name], obj_id=obj_id, field_name=f"actuator group '{group_name}' {field_name}"
        )

    for field_name in ("effort_limit", "velocity_limit"):
        resolved = _resolve_actuator_limit_value(spec, obj_id=obj_id, group_name=group_name, field_name=field_name)
        if resolved is not None:
            normalized[_ACTUATOR_ALIAS_KEYS[field_name]] = resolved

    for field_name in ("armature", "friction", "dynamic_friction", "viscous_friction"):
        if field_name in spec and spec[field_name] is not None:
            normalized[field_name] = _as_float(
                spec[field_name], obj_id=obj_id, field_name=f"actuator group '{group_name}' {field_name}"
            )

    return normalized


def _copy_default_actuator_specs() -> dict[str, dict[str, Any]]:
    return {group_name: dict(spec) for group_name, spec in DEFAULT_ARTICULATION_ACTUATOR_SPECS.items()}


def _normalize_actuator_specs_mapping(actuators: Any, *, obj_id: str, source_label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(actuators, Mapping):
        raise TypeError(f"Object '{obj_id}' {source_label} must be a mapping of actuator group names to configs.")

    normalized: dict[str, dict[str, Any]] = {}
    for group_name, spec in actuators.items():
        if not isinstance(group_name, str) or not group_name:
            raise TypeError(f"Object '{obj_id}' {source_label} group names must be non-empty strings.")
        if not isinstance(spec, Mapping):
            raise TypeError(f"Object '{obj_id}' {source_label} group '{group_name}' must be a mapping.")
        normalized[group_name] = _normalize_actuator_group(group_name, spec, obj_id=obj_id)
    return normalized


def _normalize_joint_limits(joint_limits: Any, *, obj_id: str, source_label: str) -> dict[str, dict[str, float | str]]:
    if not isinstance(joint_limits, Mapping):
        raise TypeError(f"Object '{obj_id}' {source_label} must be a mapping of joint names to limit configs.")

    normalized: dict[str, dict[str, float | str]] = {}
    for joint_name, spec in joint_limits.items():
        if not isinstance(joint_name, str) or not joint_name:
            raise TypeError(f"Object '{obj_id}' {source_label} joint names must be non-empty strings.")

        if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
            values = list(spec)
            if len(values) != 2:
                raise TypeError(f"Object '{obj_id}' {source_label}.{joint_name} sequence must be [lower, upper].")
            lower = _as_float(values[0], obj_id=obj_id, field_name=f"{source_label}.{joint_name}.lower")
            upper = _as_float(values[1], obj_id=obj_id, field_name=f"{source_label}.{joint_name}.upper")
            unit = "rad"
        elif isinstance(spec, Mapping):
            unknown_keys = sorted(set(spec.keys()) - _ALLOWED_JOINT_LIMIT_KEYS)
            if unknown_keys:
                raise ValueError(
                    f"Object '{obj_id}' {source_label}.{joint_name} has unsupported keys: {', '.join(unknown_keys)}."
                )
            if "lower" not in spec or "upper" not in spec:
                raise TypeError(f"Object '{obj_id}' {source_label}.{joint_name} must define lower and upper.")
            lower = _as_float(spec["lower"], obj_id=obj_id, field_name=f"{source_label}.{joint_name}.lower")
            upper = _as_float(spec["upper"], obj_id=obj_id, field_name=f"{source_label}.{joint_name}.upper")
            unit = str(spec.get("unit", "rad")).lower()
            if unit not in _ALLOWED_JOINT_LIMIT_UNITS:
                raise ValueError(
                    f"Object '{obj_id}' {source_label}.{joint_name}.unit must be one of: "
                    f"{', '.join(sorted(_ALLOWED_JOINT_LIMIT_UNITS))}."
                )
        else:
            raise TypeError(f"Object '{obj_id}' {source_label}.{joint_name} must be a mapping or [lower, upper].")

        if lower > upper:
            raise ValueError(f"Object '{obj_id}' {source_label}.{joint_name} lower must be <= upper.")
        normalized[joint_name] = {"lower": lower, "upper": upper, "unit": unit}

    return normalized


def _normalize_joint_pos_mapping(joint_pos: Any, *, obj_id: str, source_label: str) -> dict[str, float]:
    if not isinstance(joint_pos, Mapping):
        raise TypeError(f"Object '{obj_id}' {source_label} must be a mapping of joint names to values.")

    normalized: dict[str, float] = {}
    for joint_name, value in joint_pos.items():
        if not isinstance(joint_name, str) or not joint_name:
            raise TypeError(f"Object '{obj_id}' {source_label} keys must be non-empty strings.")
        normalized[joint_name] = _as_float(value, obj_id=obj_id, field_name=f"{source_label}.{joint_name}")
    return normalized


def _normalize_spawn_config(spawn: Any, *, obj_id: str, source_label: str) -> dict[str, Any]:
    if spawn is None:
        return {}
    if not isinstance(spawn, Mapping):
        raise TypeError(f"Object '{obj_id}' {source_label} must be a mapping.")

    unknown_keys = sorted(set(spawn.keys()) - _ALLOWED_ARTICULATION_SPAWN_KEYS)
    if unknown_keys:
        raise ValueError(f"Object '{obj_id}' {source_label} has unsupported keys: {', '.join(unknown_keys)}.")

    normalized: dict[str, Any] = {}

    articulation_props = spawn.get("articulation_props")
    if articulation_props is not None:
        if not isinstance(articulation_props, Mapping):
            raise TypeError(f"Object '{obj_id}' {source_label}.articulation_props must be a mapping.")
        unknown_keys = sorted(set(articulation_props.keys()) - _ALLOWED_ARTICULATION_PROP_KEYS)
        if unknown_keys:
            raise ValueError(
                f"Object '{obj_id}' {source_label}.articulation_props has unsupported keys: {', '.join(unknown_keys)}."
            )
        normalized["articulation_props"] = {}
        if "fix_root_link" in articulation_props:
            normalized["articulation_props"]["fix_root_link"] = _as_bool(
                articulation_props["fix_root_link"],
                label=f"Object '{obj_id}' {source_label}.articulation_props.fix_root_link",
            )

    mass_props = spawn.get("mass_props")
    if mass_props is not None:
        if not isinstance(mass_props, Mapping):
            raise TypeError(f"Object '{obj_id}' {source_label}.mass_props must be a mapping.")
        unknown_keys = sorted(set(mass_props.keys()) - _ALLOWED_MASS_PROP_KEYS)
        if unknown_keys:
            raise ValueError(f"Object '{obj_id}' {source_label}.mass_props has unsupported keys: {', '.join(unknown_keys)}.")
        normalized["mass_props"] = {}
        density_value = mass_props.get("density")
        linear_density_value = mass_props.get("linear_density")
        if density_value is not None and linear_density_value is not None:
            density_float = _as_float(density_value, obj_id=obj_id, field_name=f"{source_label}.mass_props.density")
            linear_density_float = _as_float(
                linear_density_value,
                obj_id=obj_id,
                field_name=f"{source_label}.mass_props.linear_density",
            )
            if density_float != linear_density_float:
                raise ValueError(
                    f"Object '{obj_id}' {source_label}.mass_props sets both 'density' and 'linear_density' with different values."
                )
            normalized["mass_props"]["density"] = density_float
        elif density_value is not None:
            normalized["mass_props"]["density"] = _as_float(
                density_value,
                obj_id=obj_id,
                field_name=f"{source_label}.mass_props.density",
            )
        elif linear_density_value is not None:
            normalized["mass_props"]["density"] = _as_float(
                linear_density_value,
                obj_id=obj_id,
                field_name=f"{source_label}.mass_props.linear_density",
            )

    if "actuators" in spawn:
        normalized["actuators"] = _normalize_actuator_specs_mapping(
            spawn["actuators"], obj_id=obj_id, source_label=f"{source_label}.actuators"
        )

    if "joint_limits" in spawn:
        normalized["joint_limits"] = _normalize_joint_limits(
            spawn["joint_limits"], obj_id=obj_id, source_label=f"{source_label}.joint_limits"
        )

    if "joint_pos" in spawn:
        normalized["joint_pos"] = _normalize_joint_pos_mapping(
            spawn["joint_pos"], obj_id=obj_id, source_label=f"{source_label}.joint_pos"
        )

    return normalized


def _joint_is_covered_by_actuators(joint_name: str, actuator_specs: Mapping[str, Mapping[str, Any]]) -> bool:
    for spec in actuator_specs.values():
        for expr in spec.get("joint_names_expr", []):
            if re.fullmatch(expr, joint_name):
                return True
    return False


def _validate_joint_pos_coverage(obj_id: str, joint_pos: Mapping[str, float], actuator_specs: Mapping[str, Mapping[str, Any]]) -> None:
    if joint_pos:
        uncovered_joints = [joint_name for joint_name in joint_pos.keys() if not _joint_is_covered_by_actuators(joint_name, actuator_specs)]
        if uncovered_joints:
            raise ValueError(
                f"Object '{obj_id}' joint_pos joints are not covered by any actuator group: {', '.join(uncovered_joints)}."
            )


def resolve_articulation_spawn_config(asset_spec: Mapping[str, Any] | None, obj: Mapping[str, Any]) -> dict[str, Any]:
    obj_id = str(obj.get("id", "<unknown>"))
    asset_key = str(obj.get("asset", "<unknown>"))
    asset_spawn = _normalize_spawn_config(
        asset_spec.get("spawn") if isinstance(asset_spec, Mapping) else None,
        obj_id=obj_id,
        source_label=f"asset '{asset_key}' spawn",
    )
    object_spawn = _normalize_spawn_config(obj.get("spawn"), obj_id=obj_id, source_label="spawn")

    if "actuators" in object_spawn and obj.get("actuators") is not None:
        raise ValueError(f"Object '{obj_id}' defines both spawn.actuators and legacy top-level actuators.")

    merged: dict[str, Any] = {
        "articulation_props": {"fix_root_link": DEFAULT_ARTICULATION_FIX_ROOT_LINK},
        "mass_props": {"density": DEFAULT_ARTICULATION_DENSITY},
    }

    if "articulation_props" in asset_spawn:
        merged["articulation_props"].update(asset_spawn["articulation_props"])
    if "articulation_props" in object_spawn:
        merged["articulation_props"].update(object_spawn["articulation_props"])

    if "mass_props" in asset_spawn:
        merged["mass_props"].update(asset_spawn["mass_props"])
    if "mass_props" in object_spawn:
        merged["mass_props"].update(object_spawn["mass_props"])

    joint_limits: dict[str, dict[str, float | str]] = {}
    if "joint_limits" in asset_spawn:
        joint_limits.update(asset_spawn["joint_limits"])
    if "joint_limits" in object_spawn:
        joint_limits.update(object_spawn["joint_limits"])
    if joint_limits:
        merged["joint_limits"] = joint_limits

    joint_pos: dict[str, float] = {}
    if "joint_pos" in asset_spawn:
        joint_pos.update(asset_spawn["joint_pos"])
    if "joint_pos" in object_spawn:
        joint_pos.update(object_spawn["joint_pos"])
    if obj.get("joint_pos") is not None:
        joint_pos.update(_normalize_joint_pos_mapping(obj["joint_pos"], obj_id=obj_id, source_label="joint_pos"))
    if joint_pos:
        merged["joint_pos"] = joint_pos

    if "actuators" in object_spawn:
        actuator_specs = object_spawn["actuators"]
    elif "actuators" in asset_spawn:
        actuator_specs = asset_spawn["actuators"]
    elif obj.get("actuators") is not None:
        actuator_specs = _normalize_actuator_specs_mapping(obj["actuators"], obj_id=obj_id, source_label="actuators")
    else:
        actuator_specs = _copy_default_actuator_specs()

    _validate_joint_pos_coverage(obj_id, joint_pos, actuator_specs)
    merged["actuators"] = actuator_specs
    return merged


def normalize_articulation_actuator_specs(
    obj: Mapping[str, Any], asset_spec: Mapping[str, Any] | None = None
) -> dict[str, dict[str, Any]]:
    """Normalize final actuator groups for one articulation object."""
    return resolve_articulation_spawn_config(asset_spec, obj)["actuators"]


def build_articulation_actuator_cfgs_from_specs(actuator_specs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Build ImplicitActuatorCfg instances from normalized actuator specs."""
    from isaaclab.actuators import ImplicitActuatorCfg

    return {group_name: ImplicitActuatorCfg(**dict(spec_kwargs)) for group_name, spec_kwargs in actuator_specs.items()}


def build_articulation_actuator_cfgs(obj: Mapping[str, Any], asset_spec: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build ImplicitActuatorCfg instances for an articulation object from YAML data."""
    return build_articulation_actuator_cfgs_from_specs(normalize_articulation_actuator_specs(obj, asset_spec))


def build_articulation_init_state_kwargs(
    obj: Mapping[str, Any],
    pos: tuple[float, float, float],
    quat: tuple[float, float, float, float],
    joint_pos: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Build keyword arguments for articulation initial state from YAML object data."""
    init_state_kwargs: dict[str, Any] = {"pos": pos, "rot": quat}
    joint_pos = joint_pos if joint_pos is not None else obj.get("joint_pos")

    if joint_pos is None:
        return init_state_kwargs
    normalized_joint_pos = _normalize_joint_pos_mapping(
        joint_pos,
        obj_id=str(obj.get("id", "<unknown>")),
        source_label="joint_pos",
    )

    init_state_kwargs["joint_pos"] = normalized_joint_pos
    return init_state_kwargs
