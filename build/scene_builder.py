"""Scene builder for dex2scene."""

from __future__ import annotations

import os
from utils.seed_policy import get_py_rng as _rng
import re
import warnings
from typing import Dict, List, Mapping, Tuple

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, DeformableObject, DeformableObjectCfg, RigidObject, RigidObjectCfg
from robots import DEFAULT_ROBOT_KEY, spawn_robot_by_key
from utils.usd_prims import create_prim_compat, get_current_stage_compat

from .articulation_limits import apply_articulation_joint_limits
from .constants import Aabb, DEFAULT_BOUNDS_HALF, ENABLE_GLOBAL_ROBOT, SETTLE_STEPS, TABLETOP_ZONE_HEIGHT
from .generalization import (
    BaseLightCfg,
    SceneGeneralizationCfg,
    object_position_jitter_m,
    sample_distant_light_state,
    sample_explicit_yaw_offset_deg,
    scene_generalization_config_to_dict,
    scene_generalization_sample_to_dict,
    validate_resolved_object_placement_keys,
)
from .geometry import _compute_bounds, _rotate_bounds, _rpy_to_quat
from .layout import _inset_aabb_xy, _normalize_aabb, _normalize_aabbs, _resolve_explicit_position, _sample_explicit_position, _sample_position
from .object_initial_state import (
    build_articulation_actuator_cfgs_from_specs,
    build_articulation_init_state_kwargs,
    resolve_articulation_spawn_config,
)
from .object_visuals import resolve_object_color
from .spawners import _build_usd_cfg, _resolve_path
from .table_geometry import (
    RobotSupportTableGeometry,
    TableGeometry,
    TableHeights,
    ROBOT_SUPPORT_TABLE_COLOR,
    resolve_robot_support_table_geometry,
    resolve_table_geometry,
    resolve_table_heights,
)


def resolve_table_spec(task: Dict) -> Tuple[Tuple[float, float, float], float, Tuple[float, float, float]]:
    task_table = task.get("table", {}) or {}
    if not isinstance(task_table, dict):
        raise TypeError("Top-level 'table' must be a mapping if provided.")
    raw_size = task_table.get("size", [2.2, 1.1, 0.04])
    if isinstance(raw_size, (str, bytes)) or not isinstance(raw_size, (list, tuple)) or len(raw_size) != 3:
        raise ValueError("table.size must be a numeric sequence of length 3.")
    table_size = tuple(float(v) for v in raw_size)
    table_z = float(task_table.get("height", 0.75))
    raw_color = task_table.get("color", [0.45, 0.30, 0.20])
    if isinstance(raw_color, (str, bytes)) or not isinstance(raw_color, (list, tuple)) or len(raw_color) != 3:
        raise ValueError("table.color must be a numeric sequence of length 3.")
    color = tuple(float(v) for v in raw_color)
    return table_size, table_z, color


HDR_TEXTURE_EXTENSIONS = {".hdr", ".exr"}
HDR_BACKGROUND_DOME_INTENSITY = 1000.0


def _clutter_forbidden_xy_rect_aabb(clutter_cfg, table_z: float) -> Aabb | None:
    rect = getattr(clutter_cfg, "forbidden_xy_rect", None)
    if rect is None or not bool(getattr(rect, "enabled", False)):
        return None
    x0, x1 = (float(v) for v in getattr(rect, "x_range", (0.0, 0.0)))
    y0, y1 = (float(v) for v in getattr(rect, "y_range", (0.0, 0.0)))
    return (
        (min(x0, x1), min(y0, y1), float(table_z)),
        (max(x0, x1), max(y0, y1), float(table_z) + TABLETOP_ZONE_HEIGHT),
    )


def _build_background_dome_kwargs(sample_like) -> Dict | None:
    sample = dict(sample_like or {})
    appearance = sample.get("appearance", {}) if isinstance(sample, dict) else {}
    background = appearance.get("background", {}) if isinstance(appearance, dict) else {}
    if not background.get("enabled", False):
        return None

    kwargs: Dict = {"visible_in_primary_ray": True}
    explicit_intensity = background.get("intensity")
    if background.get("clean_background", True):
        kwargs["color"] = (0.8, 0.8, 0.8)
        if explicit_intensity is not None:
            kwargs["intensity"] = float(explicit_intensity)
        return kwargs

    texture_file = os.path.normpath(str(background.get("asset_uri", ""))).replace("\\", "/")
    if not texture_file:
        return None

    kwargs["texture_file"] = texture_file
    asset_kind = str(background.get("asset_kind", "")).strip().lower()
    ext = os.path.splitext(texture_file)[1].lower()
    if asset_kind == "image" or ext in HDR_TEXTURE_EXTENSIONS:
        kwargs["texture_format"] = "latlong"
        if explicit_intensity is not None:
            kwargs["intensity"] = float(explicit_intensity)
        else:
            kwargs["intensity"] = HDR_BACKGROUND_DOME_INTENSITY
    return kwargs


def _build_background_dome_orientation(sample_like):
    sample = dict(sample_like or {})
    appearance = sample.get("appearance", {}) if isinstance(sample, dict) else {}
    background = appearance.get("background", {}) if isinstance(appearance, dict) else {}
    if not background.get("enabled", False):
        return None
    yaw_deg = float(background.get("yaw_deg", 0.0) or 0.0)
    if yaw_deg == 0.0:
        return None
    return _rpy_to_quat((0.0, 0.0, yaw_deg))


def _set_light_shadow(path: str, cast_shadows: bool) -> None:
    try:
        from pxr import UsdLux
    except Exception:
        return
    try:
        prim = get_current_stage_compat().GetPrimAtPath(path)
        if prim.IsValid():
            shadow_api = UsdLux.ShadowAPI.Apply(prim)
            attr = shadow_api.CreateShadowEnableAttr(bool(cast_shadows))
            if hasattr(attr, "Set"):
                attr.Set(bool(cast_shadows))
    except Exception as exc:
        warnings.warn(f"Failed to set shadow state for {path}: {exc}", RuntimeWarning)


def _set_prim_attr(prim, attr_name: str, value):
    """Set *value* on *prim* attribute *attr_name*, creating it when absent.

    Handles the type mapping for ``float``, ``color3f``, ``quatf``, and
    ``token[]`` attributes so that newly-created attributes carry the correct
    Sdf type.
    """
    from pxr import Gf, Sdf

    attr = prim.GetAttribute(attr_name)
    if not attr:
        if isinstance(value, Gf.Quatf):
            type_name = Sdf.ValueTypeNames.Quatf
        elif isinstance(value, Gf.Vec3f):
            type_name = Sdf.ValueTypeNames.Color3f
        elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], str):
            type_name = Sdf.ValueTypeNames.TokenArray
        elif isinstance(value, bool):
            type_name = Sdf.ValueTypeNames.Bool
        elif isinstance(value, int):
            type_name = Sdf.ValueTypeNames.Int
        else:
            type_name = Sdf.ValueTypeNames.Float
        attr = prim.CreateAttribute(attr_name, type_name)
    attr.Set(value)


def _modify_usd_scene_lights(sample_dict: dict, *, env_prim_path: str = "/World/Environment") -> None:
    """Find ``scene_dir_light`` and ``scene_ibl_light`` prims under *env_prim_path*
    and apply sampled generalization parameters from *sample_dict*.

    Light-specific attributes (intensity, color, angle, exposure) are set through
    the USD schema API (``UsdLux.DistantLight`` / ``UsdLux.DomeLight``) so the
    renderer recognises them as authored schema inputs.  Xformable attributes
    (orientation, xformOpOrder) use the generic ``_set_prim_attr`` helper.
    """
    usd_light = (sample_dict.get("appearance", {}) or {}).get("usd_scene_light")
    if not usd_light or not usd_light.get("enabled", False):
        return

    from pxr import Gf, Usd, UsdLux

    stage = get_current_stage_compat()
    env_prim = stage.GetPrimAtPath(env_prim_path)
    if not env_prim.IsValid():
        return

    distant_sample = usd_light.get("distant")
    dome_sample = usd_light.get("dome")
    modified = 0

    for prim in Usd.PrimRange(env_prim):
        name = prim.GetName()
        if name == "scene_dir_light" and distant_sample is not None and prim.IsA(UsdLux.DistantLight):
            # --- schema attributes (use UsdLux API so the renderer respects them) ---
            dist_api = UsdLux.DistantLight(prim)
            dist_api.CreateIntensityAttr().Set(float(distant_sample["intensity"]))
            dist_api.CreateColorAttr().Set(Gf.Vec3f(*distant_sample["color"]))
            dist_api.CreateAngleAttr().Set(float(distant_sample["angle_deg"]))

            # --- direction: rotateZ (yaw) first, then rotateX (pitch) ---
            # The DistantLight points along -Z (straight down in Z-up coords).
            #  1. rotateZ(yaw)  spins around the world-vertical Z axis (azimuth).
            #  2. rotateX(pitch) tilts the beam from vertical (elevation).
            # This guarantees z_direction = -cos(pitch) < 0 for |pitch| < 90°,
            # so the light ALWAYS comes from above, and yaw covers full 360°.
            pitch = float(distant_sample["pitch_deg"])
            yaw = float(distant_sample["yaw_deg"])
            _set_prim_attr(prim, "xformOp:rotateZ", yaw)
            _set_prim_attr(prim, "xformOp:rotateX", pitch)
            from pxr import Vt
            _set_prim_attr(prim, "xformOpOrder",
                           Vt.TokenArray(["xformOp:rotateZ", "xformOp:rotateX"]))

            # Explicitly enable shadow casting.
            _set_light_shadow(str(prim.GetPath()), True)
            modified += 1

        elif name == "scene_ibl_light" and dome_sample is not None and prim.IsA(UsdLux.DomeLight):
            # --- schema attributes ---
            dome_api = UsdLux.DomeLight(prim)
            dome_api.CreateIntensityAttr().Set(float(dome_sample["intensity"]))
            dome_api.CreateColorAttr().Set(Gf.Vec3f(*dome_sample["color"]))
            dome_api.CreateExposureAttr().Set(float(dome_sample["exposure"]))
            modified += 1

    if modified:
        detail_parts = []
        if distant_sample is not None:
            detail_parts.append(f"distant intensity={distant_sample.get('intensity', 0.0):.1f}")
        if dome_sample is not None:
            detail_parts.append(f"dome intensity={dome_sample.get('intensity', 0.0):.1f}")
        detail = ", ".join(detail_parts)
        print(f"[INFO] Modified {modified} USD scene light prim(s) under {env_prim_path}: {detail}")


def _spawn_distant_light_from_sample(path: str, light_sample: Mapping, *, default_angle: float = 1.0) -> None:
    if not bool(light_sample.get("enabled", True)):
        return
    intensity = float(light_sample.get("intensity", 0.0) or 0.0)
    if intensity <= 0.0:
        return
    color = tuple(light_sample.get("color", BaseLightCfg.color))
    quat = tuple(light_sample.get("direction_quat", (1.0, 0.0, 0.0, 0.0)))
    angle = float(light_sample.get("angle_deg", default_angle))
    # Skip if prim already exists (e.g. during episode resampling).
    stage = get_current_stage_compat()
    if stage.GetPrimAtPath(path).IsValid():
        return
    light = sim_utils.DistantLightCfg(intensity=intensity, color=color, angle=angle)
    light.func(path, light, orientation=quat)
    _set_light_shadow(path, bool(light_sample.get("cast_shadows", True)))


def _legacy_fill_light_sample(bg_info: Mapping) -> Dict:
    fill_intensity = float(bg_info.get("fill_light_intensity", 0.0) or 0.0)
    if fill_intensity <= 0.0:
        return {"enabled": False}
    return {
        "enabled": True,
        "source": "legacy_background",
        "intensity": fill_intensity,
        "color": (1.0, 1.0, 1.0),
        "direction_quat": _rpy_to_quat((0.0, -70.0, 0.0)),
        "angle_deg": 5.0,
        "cast_shadows": False,
    }


def _normalize_usd_scene_light_metadata(sample_dict: Dict) -> None:
    appearance = sample_dict.setdefault("appearance", {})
    background = appearance.get("background", {})
    if str(background.get("asset_kind", "")).strip().lower() != "usd_scene":
        return

    usd_light = appearance.get("usd_scene_light")
    if not isinstance(usd_light, Mapping):
        appearance["usd_scene_light"] = {
            "enabled": False,
            "distant": None,
            "dome": None,
            "source": "usd_scene_defaults",
        }

    # Legacy light field no longer written; strip if present from anchor samples.
    if "light" in appearance:
        del appearance["light"]


def _merge_fixed_background_sample(sample_dict: Dict, fixed_background_sample) -> Dict:
    if not fixed_background_sample:
        return sample_dict

    fixed_dict = fixed_background_sample if isinstance(fixed_background_sample, dict) else scene_generalization_sample_to_dict(fixed_background_sample)
    fixed_background = fixed_dict.get("appearance", {}).get("background", {})
    if not fixed_background.get("enabled", False):
        return sample_dict

    merged = dict(sample_dict)
    appearance = dict(merged.get("appearance", {}))
    current_background = appearance.get("background", {})
    if isinstance(current_background, dict) and current_background.get("enabled", False):
        return merged

    appearance["background"] = dict(fixed_background)
    merged["appearance"] = appearance
    return merged


def _spawn_background_dome(sample_like) -> None:
    sample = sample_like if isinstance(sample_like, dict) else scene_generalization_sample_to_dict(sample_like)
    kwargs = _build_background_dome_kwargs(sample)
    if kwargs is None:
        return
    dome = sim_utils.DomeLightCfg(**kwargs)
    orientation = _build_background_dome_orientation(sample)
    if orientation is not None:
        dome.func("/World/BackgroundDome", dome, orientation=orientation)
    else:
        dome.func("/World/BackgroundDome", dome)


def _detect_scene_floor_z(scene_path: str) -> Tuple[float, float]:
    """Parse extentsHint from scene.usda to get floor Z and ceiling Z.

    Returns (floor_z, ceiling_z) in scene-local coordinates.  ceiling_z is
    inf when the value cannot be determined.
    """
    try:
        with open(scene_path, "r", encoding="utf-8") as f:
            content = f.read(32768)
        m = re.search(r"extentsHint\s*=\s*\[([^\]]+)\]", content)
        if m:
            nums = re.findall(r"[-\d.e+]+", m.group(1))
            if len(nums) >= 6:
                z_min = float(nums[2])
                z_max = float(nums[5])
                room_h = z_max - z_min
                print(f"[INFO] Scene extents Z: floor={z_min:.4f}, ceiling={z_max:.4f}, room_height={room_h:.2f}m")
                return z_min, z_max
            if len(nums) >= 3:
                z_min = float(nums[2])
                print(f"[INFO] Detected scene floor Z from extentsHint: {z_min:.4f} (ceiling unknown)")
                return z_min, float("inf")
        print(f"[WARN] No extentsHint found in '{scene_path}', using defaults")
    except Exception as exc:
        print(f"[WARN] Could not detect scene extents for '{scene_path}': {exc}")
    return 0.0, float("inf")


def _spawn_environment_scene(sample_like) -> str | None:
    """Spawn a USD room scene as background environment at /World/Environment."""
    sample = sample_like if isinstance(sample_like, dict) else scene_generalization_sample_to_dict(sample_like)
    background = sample.get("appearance", {}).get("background", {})
    if not background.get("enabled", False):
        return None
    if str(background.get("asset_kind", "")).strip().lower() != "usd_scene":
        return None

    scene_path = str(background.get("asset_uri", ""))
    if not scene_path or not os.path.isfile(scene_path):
        print(f"[WARN] USD scene file not found: {scene_path}")
        return None

    floor_z, ceiling_z = _detect_scene_floor_z(scene_path)
    raw_offset = background.get("scene_offset", (0.0, 0.0, 0.0))
    translation = tuple(float(v) for v in raw_offset)

    room_height = ceiling_z - floor_z if ceiling_z != float("inf") else float("inf")
    MIN_CLEARANCE_M = 2.0  # overhead camera 1.80m + robot arm ~1.95m
    if room_height < MIN_CLEARANCE_M:
        print(f"[WARN] Room height {room_height:.2f}m < {MIN_CLEARANCE_M}m — camera/robot may clip ceiling")

    yaw_deg = float(background.get("yaw_deg", 0.0) or 0.0)
    orientation = _rpy_to_quat((0.0, 0.0, yaw_deg)) if yaw_deg != 0.0 else None

    env_prim_path = "/World/Environment"
    cfg = sim_utils.UsdFileCfg(usd_path=scene_path)
    if orientation is not None:
        cfg.func(env_prim_path, cfg, translation=translation, orientation=orientation)
    else:
        cfg.func(env_prim_path, cfg, translation=translation)

    physics_enabled = background.get("physics_enabled", False)

    if physics_enabled:
        _fix_world_anchored_joints(env_prim_path, translation, orientation)

    _postprocess_environment_prims(env_prim_path, strip_physics=not physics_enabled)

    print(f"[INFO] USD environment scene spawned: {os.path.basename(os.path.dirname(scene_path))} "
          f"(floor_z={floor_z:.4f}, offset={translation}, yaw={yaw_deg:.1f}°, physics={physics_enabled})")
    return env_prim_path


def _fix_world_anchored_joints(env_prim_path: str, translation: tuple, orientation: tuple | None) -> None:
    """Adjust world-anchored FixedJoints in the environment scene for the scene offset.

    When body0 is unset (= world), localPos0/localRot0 are in world frame and don't
    move with the scene offset applied to /World/Environment.  We transform them so
    the joint anchor matches the relocated body position.
    """
    from pxr import Gf, Usd, UsdPhysics

    stage = get_current_stage_compat()
    env_prim = stage.GetPrimAtPath(env_prim_path)
    if not env_prim.IsValid():
        return

    tx, ty, tz = translation
    has_rotation = orientation is not None and tuple(orientation) != (1.0, 0.0, 0.0, 0.0)
    if has_rotation:
        w, x, y, z = orientation
        rot_quat = Gf.Quatd(w, Gf.Vec3d(x, y, z))
        rot_mat = Gf.Matrix3d(Gf.Rotation(rot_quat))

    fixed = 0
    for prim in Usd.PrimRange(env_prim):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        body0_rel = prim.GetRelationship("physics:body0")
        if body0_rel and body0_rel.GetTargets():
            continue

        pos_attr = prim.GetAttribute("physics:localPos0")
        if pos_attr and pos_attr.HasValue():
            old = pos_attr.Get()
            if has_rotation:
                rotated = rot_mat * Gf.Vec3d(float(old[0]), float(old[1]), float(old[2]))
                new_pos = Gf.Vec3f(float(rotated[0] + tx), float(rotated[1] + ty), float(rotated[2] + tz))
            else:
                new_pos = Gf.Vec3f(float(old[0]) + tx, float(old[1]) + ty, float(old[2]) + tz)
            pos_attr.Set(new_pos)

        if has_rotation:
            rot_attr = prim.GetAttribute("physics:localRot0")
            if rot_attr and rot_attr.HasValue():
                old_rot = rot_attr.Get()
                old_q = Gf.Quatd(float(old_rot.GetReal()), Gf.Vec3d(*[float(v) for v in old_rot.GetImaginary()]))
                new_q = rot_quat * old_q
                r = new_q.GetReal()
                im = new_q.GetImaginary()
                rot_attr.Set(Gf.Quatf(float(r), Gf.Vec3f(float(im[0]), float(im[1]), float(im[2]))))

        fixed += 1

    if fixed:
        print(f"[INFO] Fixed {fixed} world-anchored joint(s) in environment scene for offset ({translation})")


_LIGHT_FIXTURE_TOKENS = ("lightfixture", "lamp", "ceilinglight", "chandelier", "sconce", "lightbulb", "cylinder")
_STRUCTURAL_SHADOW_PREFIXES = ("ceiling_", "wall_")
_EXTRA_SHADOW_DISABLED_NAMES = {
    "tn__fp323base_a8968274f11925ff0a30aeb2dd96eb5c_1_1_0_vx1",
    "tn__fp326base1_bb20424a0012b130cc544f0e75950c1c_1_1_0_rz1",
    "sphere_ae09db496a0bb278a8a5784e8471eda5_1_1_0",
    "window_f519cfc5442b8d026e15e6b54af9a5c7_1_1_0",
    "cube_4a25a3ddc42cd4ade99fc164376f737b_1_1_0",
    "domelight_2e0383633aeb8ae68dc574b2bcfc158c_1_1_0",
    "tn__fp219cylinder_08f54d6eff6c736abfc4f58976b6a8f2_1_1_0_fa2",
}


def _should_disable_environment_shadow(prim_name: str, prim_path: str) -> bool:
    prim_name_lower = prim_name.lower()
    prim_path_lower = prim_path.lower()

    if any(tok in prim_path_lower for tok in _LIGHT_FIXTURE_TOKENS):
        return True
    if prim_name_lower in _EXTRA_SHADOW_DISABLED_NAMES:
        return True
    if any(f"/{name}" in prim_path_lower for name in _EXTRA_SHADOW_DISABLED_NAMES):
        return True
    return any(prim_name_lower.startswith(p) for p in _STRUCTURAL_SHADOW_PREFIXES) and "visual" in prim_name_lower


def _postprocess_environment_prims(env_prim_path: str, *, strip_physics: bool) -> None:
    """Single-pass post-processing of environment prim tree.

    Optionally strips physics APIs / deactivates joints, and disables shadow
    casting on light-fixture and structural (wall/ceiling) meshes — all in one
    traversal to avoid redundant ``Usd.PrimRange`` walks on large scenes.
    """
    from pxr import Sdf, Usd, UsdGeom

    if strip_physics:
        from pxr import UsdPhysics
        try:
            from pxr import PhysxSchema
        except ImportError:
            PhysxSchema = None

        PHYSICS_APIS = [
            UsdPhysics.RigidBodyAPI,
            UsdPhysics.CollisionAPI,
            UsdPhysics.MeshCollisionAPI,
            UsdPhysics.MassAPI,
            UsdPhysics.ArticulationRootAPI,
            UsdPhysics.FilteredPairsAPI,
        ]
        if PhysxSchema is not None:
            for name in ("PhysxRigidBodyAPI", "PhysxCollisionAPI", "PhysxArticulationAPI"):
                api = getattr(PhysxSchema, name, None)
                if api is not None:
                    PHYSICS_APIS.append(api)

    stage = get_current_stage_compat()
    env_prim = stage.GetPrimAtPath(env_prim_path)
    if not env_prim.IsValid():
        return

    joints_deactivated = 0
    apis_removed = 0
    shadows_disabled = 0

    for prim in Usd.PrimRange(env_prim):
        # --- physics stripping ---
        if strip_physics:
            if prim.IsA(UsdPhysics.Joint):
                prim.SetActive(False)
                joints_deactivated += 1
                continue
            for api_cls in PHYSICS_APIS:
                if prim.HasAPI(api_cls):
                    prim.RemoveAPI(api_cls)
                    apis_removed += 1

        # --- shadow disabling (light fixtures + structural meshes) ---
        if not prim.IsA(UsdGeom.Gprim):
            continue

        if not _should_disable_environment_shadow(prim.GetName(), str(prim.GetPath())):
            continue

        attr = prim.GetAttribute("primvars:doNotCastShadows")
        if not attr:
            attr = prim.CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool)
        attr.Set(True)
        shadows_disabled += 1

    physics_total = apis_removed + joints_deactivated
    if physics_total:
        print(f"[INFO] Disabled environment physics: removed {apis_removed} API(s), deactivated {joints_deactivated} joint(s)")
    if shadows_disabled:
        print(f"[INFO] Disabled shadow casting on {shadows_disabled} light fixture / structural mesh(es)")


def _set_table_uvs(table_prim_path: str) -> None:
    """Write face-varying UV primvar 'st' onto the table mesh.

    The top face (+Z) gets a full [0,1]x[0,1] mapping so the texture covers
    the entire surface as a single image.  All other faces map to (0.5, 0.5)
    so they pick up a uniform colour from the texture centre.
    """
    try:
        from pxr import Gf, Sdf, UsdGeom, Vt
    except Exception as exc:
        print(f"[WARN] Cannot set table UVs: {exc}")
        return

    stage = get_current_stage_compat()
    mesh_prim = stage.GetPrimAtPath(f"{table_prim_path}/geometry/mesh")
    if not mesh_prim.IsValid():
        print(f"[WARN] Table mesh prim not found at {table_prim_path}/geometry/mesh")
        return

    mesh = UsdGeom.Mesh(mesh_prim)
    points = mesh.GetPointsAttr().Get()
    face_counts = mesh.GetFaceVertexCountsAttr().Get()
    face_indices = mesh.GetFaceVertexIndicesAttr().Get()

    if points is None or face_counts is None or face_indices is None:
        print("[WARN] Table mesh has no geometry data for UV generation")
        return

    z_max = max(p[2] for p in points)
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    x_span = x_max - x_min if x_max > x_min else 1.0
    y_span = y_max - y_min if y_max > y_min else 1.0

    uvs = []
    idx = 0
    for count in face_counts:
        verts = [points[face_indices[idx + i]] for i in range(count)]
        is_top = all(abs(v[2] - z_max) < 1e-4 for v in verts)
        for i in range(count):
            if is_top:
                v = verts[i]
                u = (v[0] - x_min) / x_span
                t = (v[1] - y_min) / y_span
                uvs.append(Gf.Vec2f(u, t))
            else:
                uvs.append(Gf.Vec2f(0.5, 0.5))
        idx += count

    primvar_api = UsdGeom.PrimvarsAPI(mesh_prim)
    st_primvar = primvar_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray)
    st_primvar.SetInterpolation(UsdGeom.Tokens.faceVarying)
    st_primvar.Set(Vt.Vec2fArray(uvs))


def _bind_table_texture(table_prim_path: str, texture_uri: str) -> None:
    """Bind a texture image to the table surface using UsdPreviewSurface + UsdUVTexture."""
    if not texture_uri:
        return
    try:
        from pxr import Sdf, UsdShade
        from isaaclab.sim.utils import bind_visual_material
    except Exception as exc:
        print(f"[WARN] Table texture binding unavailable: {exc}")
        return

    _set_table_uvs(table_prim_path)

    try:
        stage = get_current_stage_compat()
        material_path = "/World/Looks/TableSurfaceMaterial"
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.95)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        st_reader = UsdShade.Shader.Define(stage, f"{material_path}/PrimvarReader")
        st_reader.CreateIdAttr("UsdPrimvarReader_float2")
        texture = UsdShade.Shader.Define(stage, f"{material_path}/DiffuseTexture")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture_uri).replace("\\", "/"))
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
        texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
        st_input = material.CreateInput("frame:stPrimvarName", Sdf.ValueTypeNames.Token)
        st_input.Set("st")
        st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).ConnectToSource(st_input)
        bind_visual_material(f"{table_prim_path}/geometry/mesh", material_path, stage=stage)
    except Exception as exc:
        print(f"[WARN] Failed to bind table texture '{texture_uri}': {exc}")


def _spawn_table(
    task: Dict,
    sample_dict: Dict,
    *,
    generalization_enabled: bool,
) -> Tuple[TableGeometry, TableHeights, Tuple[float, float, float]]:
    table_size, nominal_table_z, color = resolve_table_spec(task)
    table_heights = resolve_table_heights(
        nominal_table_z,
        sample_dict,
        generalization_enabled=generalization_enabled,
    )
    table_geometry = resolve_table_geometry(table_size, table_z=table_heights.table_z)

    table_prim_path = "/World/Objects/Table"
    tbl_cfg = sim_utils.MeshCuboidCfg(
        size=table_geometry.spawn_size,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    tbl_cfg.func(table_prim_path, tbl_cfg, translation=table_geometry.center)
    
    surface = (sample_dict or {}).get("appearance", {}).get("table_surface", {})
    if surface.get("enabled", False) and not surface.get("clean_surface", True):
        _bind_table_texture(table_prim_path, str(surface.get("asset_uri", "")))
    
    return table_geometry, table_heights, color


def _spawn_robot_support_table() -> RobotSupportTableGeometry | None:
    support_geometry = resolve_robot_support_table_geometry()
    if support_geometry is None:
        return None

    table_prim_path = "/World/Objects/RobotSupportTable"
    tbl_cfg = sim_utils.MeshCuboidCfg(
        size=support_geometry.size,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=ROBOT_SUPPORT_TABLE_COLOR),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    tbl_cfg.func(table_prim_path, tbl_cfg, translation=support_geometry.center)
    print(
        "[INFO] Robot support table spawned: "
        f"size={support_geometry.size}, center={support_geometry.center}"
    )
    return support_geometry


def _spawn_clutter(
    *,
    generalization_cfg: SceneGeneralizationCfg,
    sample_dict: Dict,
    task_dir: str,
    table_geometry: TableGeometry,
    table_z: float,
    robot_exclude_aabbs: List[Aabb],
    placed: List[Tuple[str, float, float, float]],
    interactive_objects: Dict[str, object],
    object_prim_paths: Dict[str, str],
    asset_local_bbox: Dict[str, Aabb],
    object_body_types: Dict[str, str],
    object_asset_paths: Dict[str, str],
    object_asset_keys: Dict[str, str],
) -> List[str]:
    clutter_sample = sample_dict.get("clutter", {}).get("tabletop", {})
    if not clutter_sample.get("enabled", False) or int(clutter_sample.get("count", 0) or 0) <= 0:
        return []

    clutter_cfg = generalization_cfg.clutter.tabletop
    zone = getattr(clutter_cfg, "zone", None)
    zone_mode = getattr(zone, "mode", "tabletop") if zone is not None else "tabletop"
    if zone_mode == "tabletop" or getattr(zone, "aabb", None) is None:
        clutter_aabb = table_geometry.tabletop_aabb(TABLETOP_ZONE_HEIGHT)
    else:
        clutter_aabb = zone.aabb
    forbidden = list(getattr(zone, "exclude_aabbs", ()) or [])
    forbidden_xy_rect = _clutter_forbidden_xy_rect_aabb(clutter_cfg, table_z)
    if forbidden_xy_rect is not None:
        forbidden.append(forbidden_xy_rect)
    forbidden.extend(robot_exclude_aabbs)

    distractor_ids: List[str] = []
    for index, asset in enumerate(clutter_sample.get("assets", [])):
        if asset.get("placement_failed"):
            continue
        body_type = str(asset.get("body_type", "dynamic")).lower()
        if body_type != "dynamic":
            raise ValueError(f"Clutter asset '{asset.get('name', index)}' must use body_type=dynamic.")
        asset_path = _resolve_path(asset["uri"], task_dir)
        asset_scale = tuple(float(v) for v in asset.get("scale", [1.0, 1.0, 1.0]))
        bounds = _compute_bounds(asset_path, asset_scale)
        if bounds is None:
            raise ValueError(f"Cannot compute clutter bounds for '{asset.get('name', index)}'.")
        lo, hi = bounds

        saved_pos = asset.get("spawn_position")
        saved_yaw = asset.get("spawn_yaw_deg")
        if saved_pos is not None and saved_yaw is not None:
            pos = tuple(float(v) for v in saved_pos)
            sampled_yaw = float(saved_yaw)
        else:
            yaw = _rng().uniform(0.0, 360.0)
            eff_min_z, _eff_max_z, footprint = _rotate_bounds(lo, hi, (0.0, 0.0, yaw))
            edge_margin_m = float(getattr(clutter_cfg, "edge_margin_m", 0.0) or 0.0)
            safe_clutter_aabb = _inset_aabb_xy(clutter_aabb, footprint + edge_margin_m)
            if safe_clutter_aabb is None:
                warnings.warn(f"Clutter placement zone too small for '{asset.get('name', index)}', skipping")
                asset["placement_failed"] = True
                continue
            try:
                pos, sampled_yaw = _sample_position(
                    safe_clutter_aabb,
                    table_z,
                    eff_min_z,
                    footprint,
                    placed,
                    (0.0, 360.0),
                    forbidden_aabbs=forbidden,
                )
            except Exception:
                warnings.warn(f"Clutter placement failed for '{asset.get('name', index)}', skipping")
                asset["placement_failed"] = True
                continue

        eff_min_z, _eff_max_z, footprint = _rotate_bounds(lo, hi, (0.0, 0.0, sampled_yaw))
        quat = _rpy_to_quat((0.0, 0.0, sampled_yaw))
        obj_id = asset.get("obj_id") or f"distractor_{index:02d}_{str(asset.get('name', 'asset')).replace(' ', '_')}"
        prim = f"/World/Objects/{obj_id}"
        wrapper_cfg = RigidObjectCfg(
            prim_path=prim,
            spawn=_build_usd_cfg(path=asset_path, scale=asset_scale, body=body_type),
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=quat),
        )
        interactive_objects[obj_id] = RigidObject(cfg=wrapper_cfg)
        object_prim_paths[obj_id] = prim
        asset_local_bbox[obj_id] = (lo, hi)
        object_body_types[obj_id] = body_type
        object_asset_paths[obj_id] = asset_path
        object_asset_keys[obj_id] = str(asset.get("asset_code") or asset.get("name", obj_id))
        placed.append((obj_id, pos[0], pos[1], footprint))
        distractor_ids.append(obj_id)

        asset["spawn_position"] = [float(pos[0]), float(pos[1]), float(pos[2])]
        asset["spawn_yaw_deg"] = float(sampled_yaw)
        asset["obj_id"] = obj_id
    return distractor_ids


def build_scene(
    task: Dict,
    task_dir: str,
    *,
    generalization_enabled: bool = False,
    generalization_cfg: SceneGeneralizationCfg | None = None,
    generalization_sample=None,
    fixed_background_sample=None,
    robot_key: str | None = None,
    existing_robot_runtime: Dict | None = None,
) -> Dict:
    if generalization_cfg is None or not hasattr(generalization_cfg, "appearance"):
        generalization_cfg = SceneGeneralizationCfg()
    sample_dict = scene_generalization_sample_to_dict(generalization_sample or {})
    sample_dict = _merge_fixed_background_sample(sample_dict, fixed_background_sample)
    _normalize_usd_scene_light_metadata(sample_dict)
    validate_resolved_object_placement_keys(
        sample_dict,
        (obj["id"] for obj in task.get("objects", [])),
    )
    config_dict = scene_generalization_config_to_dict(generalization_cfg)
    robot_key = robot_key or DEFAULT_ROBOT_KEY

    bg_info = sample_dict.get("appearance", {}).get("background", {})
    is_usd_scene = str(bg_info.get("asset_kind", "")).strip().lower() == "usd_scene"

    if is_usd_scene:
        ground = sim_utils.GroundPlaneCfg(visible=False)
        try:
            ground.func("/World/defaultGroundPlane", ground)
        except FileNotFoundError as exc:
            print(f"[WARN] Skipping ground plane spawn: {exc}")
        _spawn_environment_scene(sample_dict)
        _modify_usd_scene_lights(sample_dict)
    else:
        background_kwargs = _build_background_dome_kwargs(sample_dict)
        ground = sim_utils.GroundPlaneCfg(visible=not bool(background_kwargs and background_kwargs.get("texture_file")))
        try:
            ground.func("/World/defaultGroundPlane", ground)
        except FileNotFoundError as exc:
            print(f"[WARN] Skipping ground plane spawn: {exc}")
        _spawn_background_dome(sample_dict)

    if not is_usd_scene:
        light_sample = sample_dict.get("appearance", {}).get("light")
        if light_sample:
            light_intensity = float(light_sample.get("intensity", BaseLightCfg.intensity))
            light_color = tuple(light_sample.get("color", BaseLightCfg.color))
            light_quat = tuple(light_sample.get("direction_quat", (1.0, 0.0, 0.0, 0.0)))
            light_angle = float(light_sample.get("angle_deg", BaseLightCfg.angle_deg))
        else:
            light_intensity, light_color, light_quat, light_angle = sample_distant_light_state(BaseLightCfg(), generalization_cfg, generalization_enabled)
            light_sample = {}
        light = sim_utils.DistantLightCfg(intensity=light_intensity, color=light_color, angle=light_angle)
        light.func("/World/DistantLight", light, orientation=light_quat)
        _set_light_shadow("/World/DistantLight", bool(light_sample.get("cast_shadows", True)))

        fill_sample = light_sample.get("fill") if isinstance(light_sample, Mapping) else None
        if not isinstance(fill_sample, Mapping):
            fill_sample = _legacy_fill_light_sample(bg_info)
        _spawn_distant_light_from_sample("/World/FillLight", fill_sample, default_angle=5.0)
        if bool(fill_sample.get("enabled", False)):
            print(
                "[INFO] Created fill light at /World/FillLight "
                f"(intensity={float(fill_sample.get('intensity', 0.0))}, "
                f"shadows={'on' if bool(fill_sample.get('cast_shadows', False)) else 'off'})"
            )

        secondary_sample = light_sample.get("secondary_distant") if isinstance(light_sample, Mapping) else None
        if isinstance(secondary_sample, Mapping):
            _spawn_distant_light_from_sample("/World/SecondaryDistantLight", secondary_sample, default_angle=4.0)
    stage = get_current_stage_compat()
    if not stage.GetPrimAtPath("/World/Objects").IsValid():
        create_prim_compat("/World/Objects", "Xform", stage=stage)

    table_geometry, table_heights, _table_color = _spawn_table(
        task,
        sample_dict,
        generalization_enabled=generalization_enabled,
    )
    table_size = table_geometry.size
    table_z = table_heights.table_z
    robot_mount_height = table_heights.robot_mount_height

    interactive_objects: Dict[str, object] = {}
    object_prim_paths: Dict[str, str] = {}
    asset_local_bbox: Dict[str, Aabb] = {}
    object_body_types: Dict[str, str] = {}
    object_asset_paths: Dict[str, str] = {}
    object_asset_keys: Dict[str, str] = {}
    object_display_colors: Dict[str, Tuple[float, float, float]] = {}
    robot_exclude_aabbs: List[Aabb] = []

    robot_runtime: Dict = {}
    if ENABLE_GLOBAL_ROBOT:
        robot_runtime = existing_robot_runtime or spawn_robot_by_key(
            robot_key=robot_key,
            table_size=table_size,
            table_height=robot_mount_height,
            task_dir=task_dir,
        )
        interactive_objects.update(robot_runtime.get("interactive_objects", {}))
        robot_exclude_aabbs = _normalize_aabbs(robot_runtime.get("exclude_aabbs", []))
        if existing_robot_runtime is None:
            print(f"[INFO] Global robot loaded: key={robot_key}")
        else:
            print(f"[INFO] Reusing global robot: key={robot_key}")

    robot_support_table = (
        _spawn_robot_support_table()
        if ENABLE_GLOBAL_ROBOT
        else None
    )

    assets_cfg: Dict[str, Tuple] = {}
    assets_body_type: Dict[str, str] = {}
    for key, spec in task.get("assets", {}).items():
        path = _resolve_path(spec["path"], task_dir)
        scale = tuple(spec["scale"]) if "scale" in spec else None
        body = str(spec.get("body_type", "dynamic")).lower()
        bounds = _compute_bounds(path, scale)
        if bounds is None:
            r = DEFAULT_BOUNDS_HALF
            bounds = ((-r, -r, -r), (r, r, r))
            print(f"[WARN] Using default bounds for '{key}'.")
        assets_cfg[key] = (path, scale, spec, bounds[0], bounds[1])
        assets_body_type[key] = body

    zones = task.get("zones", {})
    layout = task.get("layout", {})
    layout_zone = layout.get("zone")
    layout_objs = set(layout.get("objects", []))
    yaw_range = tuple(layout.get("yaw_range", [0, 360]))
    zone_aabb: Aabb | None = None
    zone_exclude_aabbs: List[Aabb] = []
    if layout_zone and layout_zone in zones and isinstance(zones[layout_zone], dict):
        zone_spec = zones[layout_zone]
        zone_aabb = _normalize_aabb(zone_spec.get("aabb"))
        zone_exclude_aabbs.extend(_normalize_aabbs(zone_spec.get("exclude_aabbs", [])))
    if layout_zone == "tabletop":
        zone_aabb = table_geometry.tabletop_aabb(TABLETOP_ZONE_HEIGHT)
        zone_exclude_aabbs.extend(robot_exclude_aabbs)

    placed: List[Tuple[str, float, float, float]] = []
    resolved_object_placements: Dict[str, Dict] = {}
    saved_placements: Dict[str, Dict] = sample_dict.get("resolved_object_placements", {})
    position_jitter_m = object_position_jitter_m(generalization_sample or generalization_cfg, generalization_enabled)

    for obj in task.get("objects", []):
        obj_id = obj["id"]
        asset_key = obj["asset"]
        object_color = resolve_object_color(obj)
        asset_path, asset_scale, asset_spec, lo, hi = assets_cfg[asset_key]
        body_type = assets_body_type[asset_key]
        rpy = tuple(obj.get("rpy_deg", [0, 0, 0]))
        eff_min_z, _eff_max_z, footprint = _rotate_bounds(lo, hi, (rpy[0], rpy[1], 0.0))

        saved = saved_placements.get(obj_id)
        if saved is not None:
            pos = tuple(float(v) for v in saved["position"])
            saved_rpy = saved["rpy_deg"]
            yaw = float(saved_rpy[2])
        elif "position" in obj:
            pos = _resolve_explicit_position(obj["position"], obj_id=obj_id, table_z=table_z, min_z=eff_min_z)
            # Per-object generalization overrides:
            #   freeze_pose: true        → zero out both position and yaw jitter
            #   position_jitter_cm: X    → override global position jitter (0.0 = freeze)
            #   yaw_jitter_deg: X        → override global yaw jitter (0.0 = freeze)
            obj_gen = obj.get("generalization") if isinstance(obj, dict) else None
            obj_freeze = bool(obj_gen.get("freeze_pose", False)) if isinstance(obj_gen, dict) else False
            obj_pos_jitter_m: float | None = None  # None = use global
            obj_yaw_jitter_deg: float | None = None  # None = use global
            if isinstance(obj_gen, dict):
                if obj_freeze:
                    obj_pos_jitter_m = 0.0
                    obj_yaw_jitter_deg = 0.0
                if "position_jitter_cm" in obj_gen:
                    obj_pos_jitter_m = float(obj_gen["position_jitter_cm"]) * 0.01
                if "yaw_jitter_deg" in obj_gen:
                    obj_yaw_jitter_deg = float(obj_gen["yaw_jitter_deg"])
            # Resolve effective values
            _eff_pos_jitter_m = obj_pos_jitter_m if obj_pos_jitter_m is not None else position_jitter_m
            _eff_yaw_jitter_deg = obj_yaw_jitter_deg if obj_yaw_jitter_deg is not None else None  # None → use global later
            if _eff_pos_jitter_m > 0.0:
                pos = _sample_explicit_position(
                    pos,
                    obj_id=obj_id,
                    jitter_xy_m=_eff_pos_jitter_m,
                    aabb=zone_aabb,
                    footprint=footprint,
                    placed=placed,
                    forbidden_aabbs=zone_exclude_aabbs,
                    allow_anchor_fallback=generalization_enabled,
                )
            if _eff_yaw_jitter_deg is not None:
                yaw = rpy[2] + (_rng().uniform(-_eff_yaw_jitter_deg, _eff_yaw_jitter_deg) if _eff_yaw_jitter_deg > 0.0 else 0.0)
            else:
                yaw = rpy[2] + sample_explicit_yaw_offset_deg(generalization_sample or generalization_cfg, generalization_enabled)
        elif obj_id in layout_objs and zone_aabb is not None:
            pos, yaw = _sample_position(zone_aabb, table_z, eff_min_z, footprint, placed, yaw_range, forbidden_aabbs=zone_exclude_aabbs)
        else:
            raise ValueError(f"No position for '{obj_id}': set position or add to layout.")

        if saved is not None:
            full_rpy = tuple(float(v) for v in saved_rpy)
        elif "position" in obj:
            full_rpy = (rpy[0], rpy[1], yaw)
        else:
            full_rpy = (rpy[0], rpy[1], rpy[2] + yaw)
        quat = _rpy_to_quat(full_rpy)
        resolved_object_placements[obj_id] = {
            "position": [float(pos[0]), float(pos[1]), float(pos[2])],
            "rpy_deg": [float(full_rpy[0]), float(full_rpy[1]), float(full_rpy[2])],
        }
        prim = f"/World/Objects/{obj_id.replace(' ', '_')}"
        object_prim_paths[obj_id] = prim
        asset_local_bbox[obj_id] = (lo, hi)
        object_body_types[obj_id] = body_type
        object_asset_paths[obj_id] = asset_path
        object_asset_keys[obj_id] = asset_key

        if body_type != "articulation":
            if "actuators" in obj:
                raise ValueError(f"Object '{obj_id}' defines actuators but asset body_type is '{body_type}', not articulation.")
            # 对 dynamic 资产, 允许 spawn 中携带 rigid_props/mass_props (例如
            # kinematic_enabled, density, mass), 但禁止 articulation 专属字段.
            def _check_spawn_keys(src: Mapping, where: str) -> None:
                for k in src.keys():
                    if k not in {"rigid_props", "mass_props"}:
                        raise ValueError(
                            f"{where} declares spawn.{k} but body_type is '{body_type}', "
                            f"only 'rigid_props' and 'mass_props' are allowed for dynamic."
                        )
            if "spawn" in obj:
                _check_spawn_keys(obj["spawn"], f"Object '{obj_id}'")
            if isinstance(asset_spec, dict) and "spawn" in asset_spec:
                _check_spawn_keys(asset_spec["spawn"], f"Asset '{asset_key}'")

        if body_type == "articulation":
            resolved_spawn = resolve_articulation_spawn_config(asset_spec, obj)
        else:
            # 对 dynamic/deformable, 把 asset.spawn 和 object.spawn 合并后传下去
            asset_spawn = asset_spec.get("spawn") if isinstance(asset_spec, Mapping) else None
            object_spawn = obj.get("spawn")
            resolved_spawn = {}
            if isinstance(asset_spawn, Mapping):
                resolved_spawn.update(asset_spawn)
            if isinstance(object_spawn, Mapping):
                for k, v in object_spawn.items():
                    if isinstance(v, Mapping) and isinstance(resolved_spawn.get(k), Mapping):
                        merged = dict(resolved_spawn[k])
                        merged.update(v)
                        resolved_spawn[k] = merged
                    else:
                        resolved_spawn[k] = v
        usd_cfg = _build_usd_cfg(path=asset_path, scale=asset_scale, body=body_type, spawn=resolved_spawn)
        if body_type == "dynamic":
            wrapper_cfg = RigidObjectCfg(prim_path=prim, spawn=usd_cfg, init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=quat))
            interactive_objects[obj_id] = RigidObject(cfg=wrapper_cfg)
        elif body_type == "articulation":
            init_state_kwargs = build_articulation_init_state_kwargs(
                obj,
                pos=pos,
                quat=quat,
                joint_pos=resolved_spawn.get("joint_pos"),
            )
            actuator_cfgs = build_articulation_actuator_cfgs_from_specs(resolved_spawn["actuators"])
            wrapper_cfg = ArticulationCfg(
                prim_path=prim,
                spawn=usd_cfg,
                init_state=ArticulationCfg.InitialStateCfg(**init_state_kwargs),
                actuators=actuator_cfgs,
            )
            interactive_objects[obj_id] = Articulation(cfg=wrapper_cfg)
            apply_articulation_joint_limits(
                prim,
                resolved_spawn.get("joint_limits"),
                log_prefix=f"scene_builder:{obj_id}",
            )
        elif body_type == "deformable":
            wrapper_cfg = DeformableObjectCfg(prim_path=prim, spawn=usd_cfg, init_state=DeformableObjectCfg.InitialStateCfg(pos=pos, rot=quat))
            interactive_objects[obj_id] = DeformableObject(cfg=wrapper_cfg)
        else:
            usd_cfg.func(prim, usd_cfg, translation=pos, orientation=quat)
        if object_color is not None:
            object_display_colors[obj_id] = object_color
        placed.append((obj_id, pos[0], pos[1], footprint))

    sample_dict["resolved_object_placements"] = resolved_object_placements

    distractor_object_ids = _spawn_clutter(
        generalization_cfg=generalization_cfg,
        sample_dict=sample_dict,
        task_dir=task_dir,
        table_geometry=table_geometry,
        table_z=table_z,
        robot_exclude_aabbs=robot_exclude_aabbs,
        placed=placed,
        interactive_objects=interactive_objects,
        object_prim_paths=object_prim_paths,
        asset_local_bbox=asset_local_bbox,
        object_body_types=object_body_types,
        object_asset_paths=object_asset_paths,
        object_asset_keys=object_asset_keys,
    )

    return {
        "settle_steps": SETTLE_STEPS,
        "interactive_objects": interactive_objects,
        "object_prim_paths": object_prim_paths,
        "asset_local_bbox": asset_local_bbox,
        "object_body_types": object_body_types,
        "object_asset_paths": object_asset_paths,
        "object_asset_keys": object_asset_keys,
        "object_display_colors": object_display_colors,
        "distractor_object_ids": distractor_object_ids,
        "scene_generalization_config": config_dict,
        "scene_generalization_sample": sample_dict,
        "robot_key": robot_key,
        "robot_runtime": robot_runtime,
        "robot_pose": dict(robot_runtime.get("robot_pose", {})),
        "robot_support_table": robot_support_table.bounds_dict() if robot_support_table else {},
        "contact_pairs": _parse_contact_pairs(task, object_prim_paths),
        "table_z": table_z,
        "nominal_table_z": table_heights.nominal_table_z,
        "robot_mount_height": robot_mount_height,
        "table_bounds": table_geometry.bounds_dict(),
    }

def _parse_contact_pairs(
    task: Dict,
    object_prim_paths: Dict[str, str],
) -> list[dict]:
    """Extract contact_pairs from task metrics and activate PhysX contact reporting."""
    contact_pairs = task.get("metrics", {}).get("contact_pairs", [])
    if not contact_pairs:
        return []

    sensor_obj_ids = set()
    for pair in contact_pairs:
        sensor_obj_ids.add(str(pair["sensor"]))

    activated = 0
    for obj_id in sensor_obj_ids:
        prim_path = object_prim_paths.get(obj_id)
        if prim_path is None:
            warnings.warn(f"contact_pairs: sensor object '{obj_id}' not found in scene, skipping")
            continue
        try:
            sim_utils.activate_contact_sensors(prim_path, threshold=0.0)
            activated += 1
        except Exception as exc:
            warnings.warn(f"contact_pairs: failed to activate contact sensor on '{obj_id}': {exc}")

    if activated:
        print(f"[contact] Activated PhysX contact reporting on {activated} object(s)")
    return contact_pairs
