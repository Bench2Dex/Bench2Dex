"""TacMap tactile collection for robot hands with precomputed surface maps."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from collector.tacmap_configs import ROBOT_KEY_TO_TACMAP_CFG, TacMapHandCfg, resolve_tacmap_site_body_name
from collector.tactile import _body_name_in_articulation, _sanitize_name

if TYPE_CHECKING:
    from collector.tacmap_sensor import TacmapSensor, TacmapSensorCfg


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATASET_ROBOTS_ROOT = (_REPO_ROOT / "../dex2bench_dataset/Robots_p").resolve()
_ROBOT_PRIM_PATH = "/World/Objects/GlobalRobot"


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _require_file(path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"TacMap asset not found: {resolved.as_posix()}")
    return resolved.as_posix()


def _resolve_sensor_map_dir(cfg: TacMapHandCfg) -> Path:
    return (_DATASET_ROBOTS_ROOT / cfg.dataset_key / "tactile_sensor").resolve()


def _missing_asset_paths(cfg: TacMapHandCfg, sensor_map_dir: Path | None = None) -> list[Path]:
    sensor_dir = Path(sensor_map_dir).resolve() if sensor_map_dir is not None else _resolve_sensor_map_dir(cfg)
    missing: list[Path] = []
    for group in cfg.npy_groups:
        if not group.site_names:
            continue
        for filename in (group.points_npy, group.normals_npy):
            path = sensor_dir / filename
            if not path.is_file():
                missing.append(path.resolve())
    return missing


def tacmap_supported(robot_key: str) -> bool:
    """Return true when TacMap is configured and all required surface maps exist."""

    cfg = ROBOT_KEY_TO_TACMAP_CFG.get(str(robot_key))
    if cfg is None:
        return False
    return not _missing_asset_paths(cfg)


def _stage_has_prim(prim_path: str) -> bool | None:
    try:
        import omni.usd
    except ImportError:
        return None
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None
    return stage.GetPrimAtPath(prim_path).IsValid()


def _tacmap_raycast_geometry_types() -> set[str]:
    try:
        from isaaclab.utils.mesh import PRIMITIVE_MESH_TYPES
    except Exception:
        return {"Mesh"}
    return set(PRIMITIVE_MESH_TYPES) | {"Mesh"}


def _safe_tacmap_target_suffix(raw: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(raw)) or "link"


def _nearest_rigid_body_ancestor(prim, root_prim, usd_physics):
    current = prim
    while current.IsValid():
        if current.HasAPI(usd_physics.RigidBodyAPI):
            return current
        if current == root_prim:
            break
        parent = current.GetParent()
        if parent == current:
            break
        current = parent
    return None


def _articulation_link_targets_for_tacmap(stage, root_path: str, usd, usd_physics) -> list[str]:
    """Return mesh-bearing rigid links under an articulation root for TacMap ray casting."""

    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        print(f"[WARN][tacmap] articulation root prim not found: {root_path}")
        return []

    geometry_types = _tacmap_raycast_geometry_types()
    link_paths: dict[str, None] = {}
    for prim in usd.PrimRange(root_prim, usd.TraverseInstanceProxies()):
        if prim.GetTypeName() not in geometry_types:
            continue
        link_prim = _nearest_rigid_body_ancestor(prim, root_prim, usd_physics)
        if link_prim is None or not link_prim.IsValid():
            continue
        link_paths[link_prim.GetPath().pathString] = None
    return sorted(link_paths)


def build_tacmap_raycast_targets(
    object_prim_paths: dict[str, str],
    object_body_types: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Build raycast targets for TacMap, expanding articulations to per-link targets."""

    try:
        import omni.usd
        from pxr import Usd, UsdPhysics
    except ImportError:
        print("[WARN][tacmap] omni.usd/pxr unavailable; using object roots for raycast targets")
        target_paths = {
            obj_id: prim_path
            for obj_id, prim_path in object_prim_paths.items()
            if object_body_types.get(obj_id) in ("dynamic", "articulation")
        }
        return target_paths, {k: object_body_types[k] for k in target_paths}

    stage = omni.usd.get_context().get_stage()
    target_paths: dict[str, str] = {}
    target_types: dict[str, str] = {}
    expanded_articulations = 0
    expanded_links = 0

    for obj_id, prim_path in object_prim_paths.items():
        body_type = str(object_body_types.get(obj_id, "")).lower()
        if body_type == "dynamic":
            target_paths[obj_id] = prim_path
            target_types[obj_id] = body_type
            continue
        if body_type != "articulation":
            continue

        link_paths = _articulation_link_targets_for_tacmap(stage, prim_path, Usd, UsdPhysics)
        if not link_paths:
            print(
                f"[WARN][tacmap] articulation '{obj_id}' has no mesh-bearing rigid link targets; "
                f"falling back to root {prim_path}"
            )
            target_paths[obj_id] = prim_path
            target_types[obj_id] = body_type
            continue

        expanded_articulations += 1
        expanded_links += len(link_paths)
        print(f"[tacmap] articulation '{obj_id}' expanded to {len(link_paths)} link raycast targets")
        for link_idx, link_path in enumerate(link_paths):
            suffix = _safe_tacmap_target_suffix(link_path.rsplit("/", 1)[-1])
            target_id = f"{obj_id}__link_{link_idx:02d}_{suffix}"
            target_paths[target_id] = link_path
            target_types[target_id] = body_type

    print(
        f"[tacmap] Raycast targets: {len(target_paths)} total "
        f"({expanded_links} articulation links from {expanded_articulations} articulations)"
    )
    return target_paths, target_types


class TacMapRig:
    """Collect TacMap penetration-depth images for a configured tactile hand."""

    def __init__(
        self,
        *,
        robot_key: str,
        object_prim_paths: dict[str, str],
        object_body_types: dict[str, str] | None = None,
        robot_prim_path: str = _ROBOT_PRIM_PATH,
        resolution_step: int = 1,
        max_distance: float = 0.015,
        sensor_map_dir: str | Path | None = None,
    ) -> None:
        robot_key = str(robot_key)
        if robot_key not in ROBOT_KEY_TO_TACMAP_CFG:
            raise ValueError(f"TacMap not configured for {robot_key!r}.")

        self._hand_cfg = ROBOT_KEY_TO_TACMAP_CFG[robot_key]
        self._resolution_step = int(resolution_step)
        if self._resolution_step <= 0 or self._hand_cfg.native_resolution % self._resolution_step != 0:
            raise ValueError(
                "TacMap resolution_step must be positive and divide "
                f"{self._hand_cfg.native_resolution}, got {resolution_step!r}."
            )
        if not object_prim_paths:
            raise ValueError("TacMap collection requires at least one raycast object prim.")

        self._robot_key = robot_key
        self._robot_prim_path = str(robot_prim_path or _ROBOT_PRIM_PATH).rstrip("/")
        self._object_prim_paths = {str(k): str(v) for k, v in object_prim_paths.items()}
        self._object_body_types = {str(k): str(v) for k, v in (object_body_types or {}).items()}
        self._image_size = self._hand_cfg.native_resolution // self._resolution_step
        self._max_distance = float(max_distance)
        self._sensor_map_dir = (
            Path(sensor_map_dir).resolve() if sensor_map_dir is not None else _resolve_sensor_map_dir(self._hand_cfg)
        )
        self._initialized = False
        self._sensors: dict[str, Any] = {}
        self._tacmap_site_names = tuple(self._hand_cfg.site_names)
        self._site_to_npy = self._load_site_npy_map()
        self._site_to_body = self._build_site_body_map()
        self._tacmap_sensor_cls, self._tacmap_sensor_cfg_cls, self._grid_pattern_cfg_cls = _load_tacmap_types()

        self._build_sensors()

    @property
    def site_names(self) -> tuple[str, ...]:
        return self._tacmap_site_names

    @property
    def resolution_step(self) -> int:
        return self._resolution_step

    @property
    def image_size(self) -> int:
        return self._image_size

    @property
    def max_distance(self) -> float:
        return self._max_distance

    def _load_site_npy_map(self) -> dict[str, tuple[str, str]]:
        missing = _missing_asset_paths(self._hand_cfg, self._sensor_map_dir)
        if missing:
            missing_list = ", ".join(path.as_posix() for path in missing)
            raise FileNotFoundError(f"TacMap assets missing for {self._robot_key}: {missing_list}")

        site_to_npy: dict[str, tuple[str, str]] = {}
        for group in self._hand_cfg.npy_groups:
            if not group.site_names:
                continue
            points_npy = _require_file(self._sensor_map_dir / group.points_npy)
            normals_npy = _require_file(self._sensor_map_dir / group.normals_npy)
            for site_name in group.site_names:
                if site_name in site_to_npy:
                    raise ValueError(f"TacMap site {site_name!r} is assigned to more than one .npy group.")
                site_to_npy[site_name] = (points_npy, normals_npy)

        missing_sites = set(self._tacmap_site_names) - set(site_to_npy)
        if missing_sites:
            raise ValueError(f"TacMap sites without .npy group: {sorted(missing_sites)}")
        return site_to_npy

    def _build_site_body_map(self) -> dict[str, str]:
        return {
            site_name: resolve_tacmap_site_body_name(self._robot_key, site_name, self._robot_prim_path)
            for site_name in self._tacmap_site_names
        }

    def _sensor_prim_path(self, body_name: str) -> str:
        return f"{self._robot_prim_path}/{_sanitize_name(body_name)}"

    def _make_cfg(self, *, site_name: str, body_name: str):
        points_npy, normals_npy = self._site_to_npy[site_name]
        return self._tacmap_sensor_cfg_cls(
            prim_path=self._sensor_prim_path(body_name),
            mesh_prim_paths=[
                self._tacmap_sensor_cfg_cls.RaycastTargetCfg(prim_expr=prim_path, track_mesh_transforms=True)
                for prim_path in self._object_prim_paths.values()
            ],
            update_period=0.0,
            pattern_cfg=self._grid_pattern_cfg_cls(resolution=0.01, size=(0.5, 0.5)),
            offset=self._tacmap_sensor_cfg_cls.OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                convention="world",
            ),
            data_types=["distance_along_normal"],
            points_npy=points_npy,
            normals_npy=normals_npy,
            resolution_step=self._resolution_step,
            native_resolution=self._hand_cfg.native_resolution,
            max_distance=self._max_distance,
            debug_viz=False,
            correction_scale=self._hand_cfg.correction_scale,
            flip_normals=self._hand_cfg.flip_normals,
        )

    def _build_sensors(self) -> None:
        for site_name, body_name in self._site_to_body.items():
            self._sensors[site_name] = self._tacmap_sensor_cls(
                cfg=self._make_cfg(site_name=site_name, body_name=body_name)
            )

    def initialize_after_reset(self, robot_articulation: object | None) -> None:
        if robot_articulation is None:
            raise ValueError("TacMap tactile collection requires the global robot articulation after reset.")

        body_names = list(getattr(robot_articulation, "body_names", []) or [])
        missing_prims = []
        missing_articulation_bodies = []
        for site_name, body_name in self._site_to_body.items():
            prim_path = self._sensor_prim_path(body_name)
            exists = _stage_has_prim(prim_path)
            if exists is False:
                missing_prims.append(f"{site_name}->{prim_path}")
            if body_names and not _body_name_in_articulation(body_name, body_names):
                missing_articulation_bodies.append(f"{site_name}->{body_name}")
        if missing_prims:
            raise ValueError(f"TacMap site prims not found: {', '.join(missing_prims)}")
        if missing_articulation_bodies:
            raise ValueError(f"TacMap site bodies not found in articulation: {', '.join(missing_articulation_bodies)}")

        for sensor in self._sensors.values():
            sensor.reset()
        self._initialized = True

    def capture(self, dt: float) -> dict[str, object]:
        if not self._initialized:
            raise RuntimeError("TacMap rig must be initialized after sim.reset() before capture.")

        frames: dict[str, np.ndarray] = {}
        raw_depth_frames: dict[str, np.ndarray] = {}
        contact_masks: dict[str, np.ndarray] = {}
        for site_name, sensor in self._sensors.items():
            sensor.update(float(dt))
            output = sensor.data.output
            quantized = _to_numpy(output["distance_along_normal"])[0]
            frames[site_name] = quantized.reshape(self._image_size, self._image_size).astype(np.uint8, copy=False)
            if "distance_along_normal_m" in output:
                raw_depth = _to_numpy(output["distance_along_normal_m"])[0]
                raw_depth_frames[site_name] = raw_depth.reshape(self._image_size, self._image_size).astype(np.float32, copy=False)
            if "contact_mask" in output:
                mask = _to_numpy(output["contact_mask"])[0]
                contact_masks[site_name] = mask.reshape(self._image_size, self._image_size).astype(np.bool_, copy=False)

        return {
            "meta": {
                "sensor_type": "tacmap",
                "robot_key": self._robot_key,
                "dataset_key": self._hand_cfg.dataset_key,
                "site_names": list(self.site_names),
                "resolution_step": self._resolution_step,
                "native_resolution": self._hand_cfg.native_resolution,
                "image_size": self._image_size,
                "max_distance_m": self._max_distance,
                "sensor_map_dir": self._sensor_map_dir.as_posix(),
                "correction_scale": self._hand_cfg.correction_scale,
                "flip_normals": self._hand_cfg.flip_normals,
            },
            "tacmap": frames,
            "distance_along_normal_m": raw_depth_frames,
            "contact_mask": contact_masks,
        }

    def close(self) -> None:
        for sensor in self._sensors.values():
            try:
                destroy = getattr(sensor, "destroy", None)
                if callable(destroy):
                    destroy()
                    continue
                reset = getattr(sensor, "reset", None)
                if callable(reset):
                    reset()
            except Exception:
                pass
        self._sensors.clear()
        self._initialized = False


def _load_tacmap_types():
    from collector.tacmap_sensor import TacmapSensor, TacmapSensorCfg
    from isaaclab.sensors.ray_caster.patterns import GridPatternCfg

    return TacmapSensor, TacmapSensorCfg, GridPatternCfg
