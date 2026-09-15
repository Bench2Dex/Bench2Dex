"""Scene generalization config loading, sampling, and audit helpers."""

from __future__ import annotations

import copy
import hashlib
import os
from utils.seed_policy import get_py_rng as _rng
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from isaaclab.utils.io import load_yaml

from .geometry import _compute_bounds, _rpy_to_quat

DEFAULT_BACKGROUND_COLOR = (0.8, 0.8, 0.8)
_ASSET_CODE_PATTERN = re.compile(r"(?<!\d)(\d{3})(?:[-_]|$)")
_FLOORPLAN_PATTERN = re.compile(r"FloorPlan(\d+)_physics")
_TEXTURE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_VALID_ASSET_SPLITS = frozenset({"seen", "unseen", "all"})
_DEFAULT_SPLIT_RATIOS = {"seen": 0.8, "unseen": 0.2}


@dataclass(frozen=True)
class ObjectGeneralizationCfg:
    enabled: bool = True
    position_jitter_cm: float = 2.0
    yaw_jitter_deg: float = 10.0


@dataclass(frozen=True)
class TableHeightGeneralizationCfg:
    enabled: bool = False
    offset_m_range: Tuple[float, float] = (-0.05, 0.05)


@dataclass(frozen=True)
class RobotAssetGeneralizationCfg:
    enabled: bool = False
    selection_mode: str = "default"
    allowed_robot_keys: Tuple[str, ...] = ()
    default_robot_key: str = ""


@dataclass(frozen=True)
class BaseLightCfg:
    intensity: float = 2500.0
    color: Tuple[float, float, float] = (0.8, 0.8, 0.8)
    direction_yaw_deg: float = 0.0
    direction_pitch_deg: float = -60.0
    angle_deg: float = 1.0


@dataclass(frozen=True)
class ExtremeLightProfileCfg:
    intensity_range: Tuple[float, float] = (6000.0, 18000.0)
    color_min: Tuple[float, float, float] = (0.35, 0.35, 0.35)
    color_max: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    direction_yaw_range_deg: Tuple[float, float] = (-180.0, 180.0)
    direction_pitch_range_deg: Tuple[float, float] = (-85.0, -10.0)
    angle_range_deg: Tuple[float, float] = (0.2, 8.0)


@dataclass(frozen=True)
class FillLightGeneralizationCfg:
    enabled: bool = False
    source: str = "explicit"
    intensity: float = 800.0
    intensity_jitter_ratio: float = 0.0
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    direction_yaw_deg: float = 0.0
    direction_pitch_deg: float = -70.0
    angle_deg: float = 5.0
    cast_shadows: bool = False


@dataclass(frozen=True)
class SecondaryDistantLightGeneralizationCfg:
    enabled: bool = False
    intensity: float = 650.0
    intensity_jitter_ratio: float = 0.25
    color: Tuple[float, float, float] = (1.0, 0.96, 0.9)
    color_jitter_abs: float = 0.03
    direction_yaw_deg: float = -35.0
    direction_pitch_deg: float = -50.0
    direction_yaw_jitter_deg: float = 4.0
    direction_pitch_jitter_deg: float = 3.0
    angle_deg: float = 4.0
    angle_jitter_deg: float = 1.0
    cast_shadows: bool = True


@dataclass(frozen=True)
class LightVisibilityGuardCfg:
    enabled: bool = True
    min_total_lux: float = 1200.0
    min_color_channel: float = 0.25


@dataclass(frozen=True)
class UsdSceneDistantLightCfg:
    """Parameter ranges for randomising the iTHOR/USD ``scene_dir_light`` DistantLight prim.

    Defaults are anchored to the iTHOR payload baseline: intensity=1000, pitch=-10°,
    angle=0.53° (USD default), color=(1,1,1) (USD default).
    """
    intensity_range: Tuple[float, float] = (800.0, 1200.0)
    color_min: Tuple[float, float, float] = (0.7, 0.7, 0.7)
    color_max: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    pitch_range_deg: Tuple[float, float] = (-65.0, -15.0)
    yaw_range_deg: Tuple[float, float] = (-180.0, 180.0)
    angle_range_deg: Tuple[float, float] = (0.5, 5.0)


@dataclass(frozen=True)
class UsdSceneDomeLightCfg:
    """Parameter ranges for randomising the iTHOR/USD ``scene_ibl_light`` DomeLight prim.

    Defaults are anchored to the iTHOR payload baseline: intensity=500,
    color=(1.0, 0.853, 0.552), exposure=0.
    """
    intensity_range: Tuple[float, float] = (400.0, 700.0)
    color_min: Tuple[float, float, float] = (0.8, 0.75, 0.6)
    color_max: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    exposure_range: Tuple[float, float] = (-0.5, 0.5)


@dataclass(frozen=True)
class UsdSceneLightCfg:
    """Top-level configuration for randomising the lights embedded in a loaded USD scene."""
    enabled: bool = True
    distant: UsdSceneDistantLightCfg = field(default_factory=UsdSceneDistantLightCfg)
    dome: UsdSceneDomeLightCfg = field(default_factory=UsdSceneDomeLightCfg)


@dataclass(frozen=True)
class LightGeneralizationCfg:
    enabled: bool = True
    asset_split: str = "seen"
    profile_name: str = ""
    profile_split: str = ""
    profiles: Tuple[Any, ...] = ()
    priority: str = "scene_default"
    follow_background_yaw: bool = False
    intensity: float = 2500.0
    intensity_jitter_ratio: float = 0.15
    color: Tuple[float, float, float] = (0.8, 0.8, 0.8)
    color_jitter_abs: float = 0.06
    direction_yaw_deg: float = 0.0
    direction_pitch_deg: float = -60.0
    direction_yaw_jitter_deg: float = 12.0
    direction_pitch_jitter_deg: float = 8.0
    angle_deg: float = 1.0
    angle_jitter_deg: float = 0.5
    extreme_light_sampling_rate: float = 0.0
    extreme_profile: ExtremeLightProfileCfg = field(default_factory=ExtremeLightProfileCfg)
    fill: FillLightGeneralizationCfg = field(default_factory=FillLightGeneralizationCfg)
    secondary_distant: SecondaryDistantLightGeneralizationCfg = field(default_factory=SecondaryDistantLightGeneralizationCfg)
    visibility_guard: LightVisibilityGuardCfg = field(default_factory=LightVisibilityGuardCfg)
    usd_scene_light: UsdSceneLightCfg = field(default_factory=UsdSceneLightCfg)


class SceneGeneralizationCfg:
    def __init__(
        self,
        appearance=None,
        spatial=None,
        clutter=None,
        embodiment=None,
        object=None,
        light=None,
        asset_split: str = "seen",
    ):
        self.asset_split = _normalize_asset_split(asset_split, "asset_split")
        self.appearance = appearance if appearance is not None else _default_appearance()
        self.spatial = spatial if spatial is not None else _default_spatial()
        self.clutter = clutter if clutter is not None else _default_clutter()
        self.embodiment = embodiment if embodiment is not None else _default_embodiment()
        if object is not None:
            self.spatial.object_pose = object
        if light is not None:
            self.appearance.light = light

    @property
    def object(self):
        return self.spatial.object_pose

    @property
    def light(self):
        return self.appearance.light


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)


def _default_appearance():
    return _ns(
        background=_ns(
            enabled=False,
            asset_split="seen",
            clean_background_rate=0.02,
            intensity=1500.0,
            yaw_deg=0.0,
            random_flip_180_rate=0.0,
            physics_enabled=False,
            fill_light_intensity=0.0,
            fill_light_jitter_ratio=0.0,
            enabled_categories=(),
            assets=(),
        ),
        table_surface=_ns(enabled=False, asset_split="seen", clean_surface_rate=0.1, assets=()),
        light=LightGeneralizationCfg(),
    )


def _default_spatial():
    return _ns(
        object_pose=ObjectGeneralizationCfg(),
        table_height=TableHeightGeneralizationCfg(),
        camera=_ns(enabled=False, asset_split="seen", profile_name="", profile_split="", profiles=(), cameras={}),
    )


def _default_clutter():
    return _ns(
        tabletop=_ns(
            enabled=False,
            count=(0, 0),
            zone=_ns(mode="tabletop", aabb=None, exclude_aabbs=()),
            forbidden_xy_rect=_ns(enabled=False, x_range=(-0.25, 0.25), y_range=(-0.20, 0.35)),
            asset_pool=(),
            max_bbox_size_m=0.35,
            max_height_m=0.25,
            placement_retry_limit=50,
            scale_range=(1.0, 1.0),
            edge_margin_m=0.0,
            asset_split="seen",
            split_ratios=dict(_DEFAULT_SPLIT_RATIOS),
        )
    )


def _expect_mapping(data: Any, label: str) -> Dict[str, Any]:
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise TypeError(f"{label} must be a mapping.")
    return dict(data)


def _as_float(value: Any, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be numeric.") from exc


def _as_non_negative_float(value: Any, label: str) -> float:
    parsed = _as_float(value, label)
    if parsed < 0.0:
        raise ValueError(f"{label} must be non-negative.")
    return parsed


def _as_probability(value: Any, label: str) -> float:
    parsed = _as_float(value, label)
    if parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{label} must be within [0, 1].")
    return parsed


def _as_int(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be an integer.") from exc


def _as_vec3(value: Any, label: str, non_negative: bool = False) -> Tuple[float, float, float]:
    if isinstance(value, (int, float)):
        vec = (float(value), float(value), float(value))
    else:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
            raise ValueError(f"{label} must be a numeric sequence of length 3.")
        try:
            vec = tuple(float(v) for v in value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{label} must contain numeric values.") from exc
    if non_negative and any(v < 0.0 for v in vec):
        raise ValueError(f"{label} must be non-negative.")
    return vec


def _as_color3(value: Any, label: str) -> Tuple[float, float, float]:
    vec = _as_vec3(value, label)
    if any(v < 0.0 or v > 1.0 for v in vec):
        raise ValueError(f"{label} values must be within [0, 1].")
    return vec


def _as_range2(value: Any, label: str, non_negative: bool = False) -> Tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"{label} must be [min, max].")
    lo = _as_float(value[0], f"{label}[0]")
    hi = _as_float(value[1], f"{label}[1]")
    if hi < lo:
        raise ValueError(f"{label} must be ordered as [min, max].")
    if non_negative and (lo < 0.0 or hi < 0.0):
        raise ValueError(f"{label} must be non-negative.")
    return lo, hi


def _as_categories(value: Any, label: str) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a list of strings.")
    return tuple(str(item) for item in value if str(item))


def _parse_enabled_categories(value: Any, label: str) -> Dict[str, Tuple[str, ...]]:
    """Parse ``enabled_categories`` as either a flat list (backward-compat,
    maps to wildcard key ``"*"``) or a dict keyed by asset kind.

    Flat list form::

        enabled_categories: [iTHOR]

    Kind-aware dict form::

        enabled_categories:
          image: [Indoor, Nature]
          usd_scene: [iTHOR]
    """
    if value is None or (isinstance(value, Sequence) and not isinstance(value, (str, bytes, Mapping)) and len(value) == 0):
        return {}
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a list or dict, got {type(value).__name__}.")
    if isinstance(value, Sequence) and not isinstance(value, Mapping):
        cats = tuple(str(item) for item in value if str(item))
        return {"*": cats} if cats else {}
    if isinstance(value, Mapping):
        result: Dict[str, Tuple[str, ...]] = {}
        for kind, cats in value.items():
            kind_str = str(kind)
            if isinstance(cats, (str, bytes)):
                raise TypeError(f"{label}.{kind_str} must be a list of category strings.")
            if isinstance(cats, Sequence):
                cat_tuple = tuple(str(c) for c in cats if str(c))
                if cat_tuple:
                    result[kind_str] = cat_tuple
            else:
                raise TypeError(f"{label}.{kind_str} must be a list of category strings.")
        return result
    raise TypeError(f"{label} must be a list or dict, got {type(value).__name__}.")


# Legacy split values accepted only for backward-compat with existing HDF5
# datasets that recorded "train"/"test" before the rename to seen/unseen.
# New configs must use seen/unseen/all — these mappings exist purely for
# deserialization of old data, not as a public alias mechanism.
_LEGACY_SPLIT_MAP: Dict[str, str] = {"train": "seen", "val": "seen", "test": "unseen"}


def _normalize_asset_split(value: Any, label: str) -> str:
    split = str(value or "seen").strip().lower()
    if split in _LEGACY_SPLIT_MAP:
        return _LEGACY_SPLIT_MAP[split]
    if split not in _VALID_ASSET_SPLITS:
        raise ValueError(f"{label} must be one of {sorted(_VALID_ASSET_SPLITS)}.")
    return split


def _asset_split_matches(asset_split: str, requested_split: str) -> bool:
    requested = _normalize_asset_split(requested_split, "asset_split")
    if requested == "all":
        return True
    return _normalize_asset_split(asset_split, "asset.split") == requested


def _split_ratios(raw: Mapping[str, Any] | None, label: str) -> Dict[str, float]:
    data = dict(raw or {})
    ratios = {
        "seen": _as_non_negative_float(data.get("seen", _DEFAULT_SPLIT_RATIOS["seen"]), f"{label}.seen"),
        "unseen": _as_non_negative_float(data.get("unseen", _DEFAULT_SPLIT_RATIOS["unseen"]), f"{label}.unseen"),
    }
    total = sum(ratios.values())
    if total <= 0.0:
        raise ValueError(f"{label} must have a positive total.")
    return {key: value / total for key, value in ratios.items()}


def _hash_split(identifier: str, ratios: Mapping[str, float]) -> str:
    """Deterministic hash-based seen/unseen assignment.

    Uses MD5 hash of *identifier* so the same asset always maps to the
    same split, independent of sort order, filesystem, or other assets in
    the pool.  Adding or removing assets only affects the new/removed one.
    """
    h = int(hashlib.md5(identifier.encode()).hexdigest(), 16)
    threshold = int(ratios["seen"] * 10000)
    return "seen" if (h % 10000) < threshold else "unseen"


def _ensure_min_unseen(assets: list, group_key, n: int = 1) -> None:
    """After hash-based split assignment, guarantee each group has ≥*n* unseen.

    If a group has fewer than *n* unseen assets (and at least *n*+1 total),
    the highest-hash seen assets are flipped to unseen.  This prevents small
    categories from having zero unseen representatives while preserving the
    global hash distribution for all other assets.
    """
    groups: dict = {}
    for asset in assets:
        key = group_key(asset)
        groups.setdefault(key, []).append(asset)
    for group_assets in groups.values():
        if len(group_assets) < n + 1:
            continue
        unseen_count = sum(1 for a in group_assets if getattr(a, "split", "") == "unseen")
        if unseen_count >= n:
            continue
        seen_assets = sorted(
            [a for a in group_assets if getattr(a, "split", "") == "seen"],
            key=lambda a: int(hashlib.md5(a.asset_id.encode()).hexdigest(), 16),
            reverse=True,
        )
        for asset in seen_assets[: n - unseen_count]:
            asset.split = "unseen"


def _sanitize_asset_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    return text or "asset"


def _infer_asset_category_from_uri(uri: str) -> str:
    parent = os.path.basename(os.path.dirname(str(uri).replace("\\", "/")))
    return parent or ""


def _floorplan_number(uri: str) -> int | None:
    match = _FLOORPLAN_PATTERN.search(str(uri).replace("\\", "/"))
    return int(match.group(1)) if match else None


def _floorplan_asset_id(uri: str) -> str:
    number = _floorplan_number(uri)
    return f"FloorPlan{number}" if number is not None else os.path.basename(os.path.dirname(str(uri).replace("\\", "/")))


def _floorplan_family(number: int) -> int:
    return (int(number) // 100) * 100


def _expand_table_surface_assets(
    raw: Mapping[str, Any],
    *,
    config_dir: str | None,
    explicit_assets_raw: Sequence[Any],
) -> Tuple[SimpleNamespace, ...]:
    assets = []
    asset_root = str(raw.get("asset_root", "") or "")
    include_explicit = bool(raw.get("include_explicit_assets", False))
    split_ratios = _split_ratios(_expect_mapping(raw.get("split_ratios"), "appearance.table_surface.split_ratios"), "appearance.table_surface.split_ratios")

    if asset_root:
        root = _normalize_uri(asset_root, config_dir=config_dir)
        if not os.path.isdir(root):
            if not explicit_assets_raw:
                raise FileNotFoundError(f"appearance.table_surface.asset_root not found: {root}")
            asset_root = ""
        else:
            configured_categories = _as_categories(raw.get("categories", ()), "appearance.table_surface.categories")
            categories = configured_categories or tuple(
                entry for entry in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, entry))
            )
            for category in categories:
                category_dir = os.path.join(root, category)
                if not os.path.isdir(category_dir):
                    raise FileNotFoundError(f"appearance.table_surface category directory not found: {category_dir}")
                files = [
                    name for name in sorted(os.listdir(category_dir))
                    if os.path.isfile(os.path.join(category_dir, name)) and os.path.splitext(name)[1].lower() in _TEXTURE_EXTENSIONS
                ]
                for index, filename in enumerate(files):
                    uri = os.path.abspath(os.path.join(category_dir, filename)).replace("\\", "/")
                    stem, ext = os.path.splitext(filename)
                    asset_id = f"{category}/{filename}"
                    assets.append(_ns(
                        name=f"table_surface_{_sanitize_asset_name(category)}_{_sanitize_asset_name(stem)}_{_sanitize_asset_name(ext.lstrip('.'))}",
                        uri=uri,
                        category=str(category),
                        asset_id=asset_id,
                        split=_hash_split(asset_id, split_ratios),
                    ))

    if asset_root and assets:
        _ensure_min_unseen(assets, group_key=lambda a: a.category)

    if not asset_root or include_explicit:
        for index, item in enumerate(explicit_assets_raw):
            item_map = _expect_mapping(item, f"appearance.table_surface.assets[{index}]")
            uri = _normalize_uri(item_map.get("uri", ""), config_dir=config_dir)
            category = str(item_map.get("category", "") or _infer_asset_category_from_uri(uri))
            asset_id = str(item_map.get("asset_id", "") or (f"{category}/{os.path.basename(uri)}" if category else os.path.basename(uri)))
            assets.append(_ns(
                name=str(item_map.get("name", f"surface_{index}")),
                uri=uri,
                category=category,
                asset_id=asset_id,
                split=_normalize_asset_split(item_map.get("split", item_map.get("asset_split", "seen")), f"appearance.table_surface.assets[{index}].split"),
            ))
    return tuple(assets)


def _parse_background_assets(
    raw: Mapping[str, Any],
    *,
    config_dir: str | None,
    explicit_assets_raw: Sequence[Any],
) -> Tuple[SimpleNamespace, ...]:
    assets = []
    for index, item in enumerate(explicit_assets_raw):
        item_map = _expect_mapping(item, f"appearance.background.assets[{index}]")
        uri = _normalize_uri(item_map.get("uri", ""), config_dir=config_dir)
        kind = str(item_map.get("kind", "image"))
        category = str(item_map.get("category", ""))
        split_value = item_map.get("split", item_map.get("asset_split"))
        assets.append(_ns(
            kind=kind,
            category=category,
            uri=uri,
            asset_id=str(item_map.get("asset_id", "") or (_floorplan_asset_id(uri) if kind == "usd_scene" else os.path.basename(uri))),
            split=_normalize_asset_split(split_value, f"appearance.background.assets[{index}].split") if split_value is not None else "",
            yaw_deg=float(item_map.get("yaw_deg", 0.0)),
            scene_offset=tuple(float(v) for v in item_map.get("scene_offset", (0.0, 0.0, 0.0))),
        ))

    split_ratios = _split_ratios(_expect_mapping(raw.get("split_ratios"), "appearance.background.split_ratios"), "appearance.background.split_ratios")
    for asset in assets:
        if not asset.split:
            asset.split = _hash_split(asset.asset_id, split_ratios)

    ordered_ithor_seen_count = raw.get("ordered_ithor_seen_count")
    ithor_assets = [a for a in assets if a.kind == "usd_scene" and a.category == "iTHOR"]
    if ordered_ithor_seen_count is not None:
        seen_count = _as_int(ordered_ithor_seen_count, "appearance.background.ordered_ithor_seen_count")
        if seen_count < 0 or seen_count > len(ithor_assets):
            raise ValueError(
                "appearance.background.ordered_ithor_seen_count must satisfy "
                f"0 <= value <= {len(ithor_assets)}."
            )
        for index, asset in enumerate(ithor_assets):
            asset.split = "seen" if index < seen_count else "unseen"
    else:
        # Ensure each FloorPlan family has ≥1 unseen scene for fair evaluation.
        _ensure_min_unseen(
            ithor_assets,
            group_key=lambda a: _floorplan_family(_floorplan_number(a.uri) or 0),
        )
    return tuple(assets)


def _filter_assets_for_split(assets: Sequence[Any], requested_split: str) -> list[Any]:
    return [asset for asset in assets if _asset_split_matches(getattr(asset, "split", "seen"), requested_split)]


def _choose_category_balanced(assets: Sequence[Any]):
    by_category: Dict[str, list[Any]] = {}
    for asset in assets:
        by_category.setdefault(str(getattr(asset, "category", "") or ""), []).append(asset)
    categories = sorted(by_category)
    category = _rng().choice(categories)
    return _rng().choice(by_category[category])


def _normalize_uri(uri: str, config_dir: str | None = None) -> str:
    uri = str(uri)
    if not uri:
        return uri
    if "://" not in uri and not os.path.isabs(uri):
        uri = os.path.abspath(os.path.join(config_dir or os.getcwd(), uri))
    return uri.replace("\\", "/")


def _iter_split_profile_entries(raw_profiles: Any, label: str) -> Tuple[Dict[str, Any], ...]:
    """Return profile entries from ``profiles: {seen: [...], unseen: [...]}``.

    A split bucket may be a single mapping, a list of mappings, or a mapping
    from profile-name to profile mapping.  Each returned entry has explicit
    ``name`` and ``split`` keys.
    """
    if not raw_profiles:
        return ()
    if isinstance(raw_profiles, Sequence) and not isinstance(raw_profiles, (str, bytes, Mapping)):
        entries = []
        for index, item in enumerate(raw_profiles):
            entry = _expect_mapping(item, f"{label}[{index}]")
            split = _normalize_asset_split(
                entry.get("split", entry.get("profile_split", entry.get("asset_split", "seen"))),
                f"{label}[{index}].split",
            )
            entry["name"] = str(entry.get("name", entry.get("profile_name", f"{split}_{index}")))
            entry["split"] = split
            entries.append(entry)
        return tuple(entries)
    profiles_map = _expect_mapping(raw_profiles, label)
    entries: list[Dict[str, Any]] = []
    for split_key, split_value in profiles_map.items():
        split = _normalize_asset_split(split_key, f"{label}.{split_key}")
        if isinstance(split_value, Mapping):
            if "cameras" in split_value or any(key in split_value for key in (
                "intensity", "color", "direction_yaw_deg", "position_offset_m_range",
            )):
                bucket = [dict(split_value)]
            else:
                bucket = []
                for name, item in split_value.items():
                    item_map = _expect_mapping(item, f"{label}.{split_key}.{name}")
                    item_map.setdefault("name", str(name))
                    bucket.append(item_map)
        elif isinstance(split_value, Sequence) and not isinstance(split_value, (str, bytes)):
            bucket = [
                _expect_mapping(item, f"{label}.{split_key}[{index}]")
                for index, item in enumerate(split_value)
            ]
        else:
            raise TypeError(f"{label}.{split_key} must be a mapping or list of mappings.")
        for index, item in enumerate(bucket):
            entry = dict(item)
            entry.setdefault("name", f"{split}_{index}")
            entry["split"] = split
            entries.append(entry)
    return tuple(entries)


def _parse_usd_scene_light_cfg(raw: Mapping[str, Any], label: str) -> UsdSceneLightCfg:
    """Parse ``usd_scene_light`` block from YAML into a :class:`UsdSceneLightCfg`."""
    distant_raw = _expect_mapping(raw.get("distant"), f"{label}.distant")
    dome_raw = _expect_mapping(raw.get("dome"), f"{label}.dome")
    return UsdSceneLightCfg(
        enabled=bool(raw.get("enabled", UsdSceneLightCfg.enabled)),
        distant=UsdSceneDistantLightCfg(
            intensity_range=_as_range2(
                distant_raw.get("intensity_range", UsdSceneDistantLightCfg.intensity_range),
                f"{label}.distant.intensity_range", non_negative=True,
            ),
            color_min=_as_color3(
                distant_raw.get("color_min", UsdSceneDistantLightCfg.color_min),
                f"{label}.distant.color_min",
            ),
            color_max=_as_color3(
                distant_raw.get("color_max", UsdSceneDistantLightCfg.color_max),
                f"{label}.distant.color_max",
            ),
            pitch_range_deg=_as_range2(
                distant_raw.get("pitch_range_deg", UsdSceneDistantLightCfg.pitch_range_deg),
                f"{label}.distant.pitch_range_deg",
            ),
            yaw_range_deg=_as_range2(
                distant_raw.get("yaw_range_deg", UsdSceneDistantLightCfg.yaw_range_deg),
                f"{label}.distant.yaw_range_deg",
            ),
            angle_range_deg=_as_range2(
                distant_raw.get("angle_range_deg", UsdSceneDistantLightCfg.angle_range_deg),
                f"{label}.distant.angle_range_deg", non_negative=True,
            ),
        ),
        dome=UsdSceneDomeLightCfg(
            intensity_range=_as_range2(
                dome_raw.get("intensity_range", UsdSceneDomeLightCfg.intensity_range),
                f"{label}.dome.intensity_range", non_negative=True,
            ),
            color_min=_as_color3(
                dome_raw.get("color_min", UsdSceneDomeLightCfg.color_min),
                f"{label}.dome.color_min",
            ),
            color_max=_as_color3(
                dome_raw.get("color_max", UsdSceneDomeLightCfg.color_max),
                f"{label}.dome.color_max",
            ),
            exposure_range=_as_range2(
                dome_raw.get("exposure_range", UsdSceneDomeLightCfg.exposure_range),
                f"{label}.dome.exposure_range",
            ),
        ),
    )


def _light_cfg_from_raw(
    light_raw: Mapping[str, Any],
    *,
    background_raw: Mapping[str, Any],
    asset_split: str,
    label: str,
    profile_name: str = "",
    profile_split: str = "",
    profiles: Tuple[Any, ...] = (),
) -> LightGeneralizationCfg:
    extreme_raw = _expect_mapping(light_raw.get("extreme_profile"), f"{label}.extreme_profile")
    explicit_fill = "fill" in light_raw
    fill_raw = _expect_mapping(light_raw.get("fill"), f"{label}.fill")
    legacy_fill_present = "fill_light_intensity" in background_raw or "fill_light_jitter_ratio" in background_raw
    if explicit_fill:
        fill_cfg = FillLightGeneralizationCfg(
            enabled=bool(fill_raw.get("enabled", FillLightGeneralizationCfg.enabled)),
            source="explicit",
            intensity=_as_non_negative_float(
                fill_raw.get("intensity", FillLightGeneralizationCfg.intensity),
                f"{label}.fill.intensity",
            ),
            intensity_jitter_ratio=_as_non_negative_float(
                fill_raw.get("intensity_jitter_ratio", FillLightGeneralizationCfg.intensity_jitter_ratio),
                f"{label}.fill.intensity_jitter_ratio",
            ),
            color=_as_color3(fill_raw.get("color", FillLightGeneralizationCfg.color), f"{label}.fill.color"),
            direction_yaw_deg=_as_float(
                fill_raw.get("direction_yaw_deg", FillLightGeneralizationCfg.direction_yaw_deg),
                f"{label}.fill.direction_yaw_deg",
            ),
            direction_pitch_deg=_as_float(
                fill_raw.get("direction_pitch_deg", FillLightGeneralizationCfg.direction_pitch_deg),
                f"{label}.fill.direction_pitch_deg",
            ),
            angle_deg=_as_non_negative_float(
                fill_raw.get("angle_deg", FillLightGeneralizationCfg.angle_deg),
                f"{label}.fill.angle_deg",
            ),
            cast_shadows=bool(fill_raw.get("cast_shadows", FillLightGeneralizationCfg.cast_shadows)),
        )
    else:
        legacy_intensity = float(background_raw.get("fill_light_intensity", 0.0))
        fill_cfg = FillLightGeneralizationCfg(
            enabled=bool(legacy_fill_present and legacy_intensity > 0.0),
            source="legacy_background",
            intensity=legacy_intensity,
            intensity_jitter_ratio=float(background_raw.get("fill_light_jitter_ratio", 0.0)),
        )
    secondary_raw = _expect_mapping(light_raw.get("secondary_distant"), f"{label}.secondary_distant")
    secondary_cfg = SecondaryDistantLightGeneralizationCfg(
        enabled=bool(secondary_raw.get("enabled", SecondaryDistantLightGeneralizationCfg.enabled)),
        intensity=_as_non_negative_float(
            secondary_raw.get("intensity", SecondaryDistantLightGeneralizationCfg.intensity),
            f"{label}.secondary_distant.intensity",
        ),
        intensity_jitter_ratio=_as_non_negative_float(
            secondary_raw.get("intensity_jitter_ratio", SecondaryDistantLightGeneralizationCfg.intensity_jitter_ratio),
            f"{label}.secondary_distant.intensity_jitter_ratio",
        ),
        color=_as_color3(
            secondary_raw.get("color", SecondaryDistantLightGeneralizationCfg.color),
            f"{label}.secondary_distant.color",
        ),
        color_jitter_abs=_as_non_negative_float(
            secondary_raw.get("color_jitter_abs", SecondaryDistantLightGeneralizationCfg.color_jitter_abs),
            f"{label}.secondary_distant.color_jitter_abs",
        ),
        direction_yaw_deg=_as_float(
            secondary_raw.get("direction_yaw_deg", SecondaryDistantLightGeneralizationCfg.direction_yaw_deg),
            f"{label}.secondary_distant.direction_yaw_deg",
        ),
        direction_pitch_deg=_as_float(
            secondary_raw.get("direction_pitch_deg", SecondaryDistantLightGeneralizationCfg.direction_pitch_deg),
            f"{label}.secondary_distant.direction_pitch_deg",
        ),
        direction_yaw_jitter_deg=_as_non_negative_float(
            secondary_raw.get("direction_yaw_jitter_deg", SecondaryDistantLightGeneralizationCfg.direction_yaw_jitter_deg),
            f"{label}.secondary_distant.direction_yaw_jitter_deg",
        ),
        direction_pitch_jitter_deg=_as_non_negative_float(
            secondary_raw.get("direction_pitch_jitter_deg", SecondaryDistantLightGeneralizationCfg.direction_pitch_jitter_deg),
            f"{label}.secondary_distant.direction_pitch_jitter_deg",
        ),
        angle_deg=_as_non_negative_float(
            secondary_raw.get("angle_deg", SecondaryDistantLightGeneralizationCfg.angle_deg),
            f"{label}.secondary_distant.angle_deg",
        ),
        angle_jitter_deg=_as_non_negative_float(
            secondary_raw.get("angle_jitter_deg", SecondaryDistantLightGeneralizationCfg.angle_jitter_deg),
            f"{label}.secondary_distant.angle_jitter_deg",
        ),
        cast_shadows=bool(secondary_raw.get("cast_shadows", SecondaryDistantLightGeneralizationCfg.cast_shadows)),
    )
    visibility_raw = _expect_mapping(light_raw.get("visibility_guard"), f"{label}.visibility_guard")
    visibility_guard_cfg = LightVisibilityGuardCfg(
        enabled=bool(visibility_raw.get("enabled", LightVisibilityGuardCfg.enabled)),
        min_total_lux=_as_non_negative_float(
            visibility_raw.get("min_total_lux", LightVisibilityGuardCfg.min_total_lux),
            f"{label}.visibility_guard.min_total_lux",
        ),
        min_color_channel=_as_non_negative_float(
            visibility_raw.get("min_color_channel", LightVisibilityGuardCfg.min_color_channel),
            f"{label}.visibility_guard.min_color_channel",
        ),
    )
    if visibility_guard_cfg.min_color_channel > 1.0:
        raise ValueError(f"{label}.visibility_guard.min_color_channel must be within [0, 1].")
    usd_scene_light_raw = _expect_mapping(light_raw.get("usd_scene_light"), f"{label}.usd_scene_light")
    usd_scene_light_cfg = _parse_usd_scene_light_cfg(usd_scene_light_raw, f"{label}.usd_scene_light")
    return LightGeneralizationCfg(
        enabled=bool(light_raw.get("enabled", True)),
        asset_split=_normalize_asset_split(light_raw.get("asset_split", asset_split), f"{label}.asset_split"),
        profile_name=str(profile_name),
        profile_split=str(profile_split),
        profiles=profiles,
        priority=str(light_raw.get("priority", LightGeneralizationCfg.priority)),
        follow_background_yaw=bool(light_raw.get("follow_background_yaw", LightGeneralizationCfg.follow_background_yaw)),
        intensity=_as_non_negative_float(light_raw.get("intensity", LightGeneralizationCfg.intensity), f"{label}.intensity"),
        intensity_jitter_ratio=_as_non_negative_float(
            light_raw.get("intensity_jitter_ratio", LightGeneralizationCfg.intensity_jitter_ratio),
            f"{label}.intensity_jitter_ratio",
        ),
        color=_as_color3(light_raw.get("color", LightGeneralizationCfg.color), f"{label}.color"),
        color_jitter_abs=_as_non_negative_float(
            light_raw.get("color_jitter_abs", LightGeneralizationCfg.color_jitter_abs),
            f"{label}.color_jitter_abs",
        ),
        direction_yaw_deg=_as_float(light_raw.get("direction_yaw_deg", LightGeneralizationCfg.direction_yaw_deg), f"{label}.direction_yaw_deg"),
        direction_pitch_deg=_as_float(light_raw.get("direction_pitch_deg", LightGeneralizationCfg.direction_pitch_deg), f"{label}.direction_pitch_deg"),
        direction_yaw_jitter_deg=_as_non_negative_float(
            light_raw.get("direction_yaw_jitter_deg", LightGeneralizationCfg.direction_yaw_jitter_deg),
            f"{label}.direction_yaw_jitter_deg",
        ),
        direction_pitch_jitter_deg=_as_non_negative_float(
            light_raw.get("direction_pitch_jitter_deg", LightGeneralizationCfg.direction_pitch_jitter_deg),
            f"{label}.direction_pitch_jitter_deg",
        ),
        angle_deg=_as_non_negative_float(light_raw.get("angle_deg", LightGeneralizationCfg.angle_deg), f"{label}.angle_deg"),
        angle_jitter_deg=_as_non_negative_float(light_raw.get("angle_jitter_deg", LightGeneralizationCfg.angle_jitter_deg), f"{label}.angle_jitter_deg"),
        extreme_light_sampling_rate=_as_probability(
            light_raw.get("extreme_light_sampling_rate", LightGeneralizationCfg.extreme_light_sampling_rate),
            f"{label}.extreme_light_sampling_rate",
        ),
        extreme_profile=ExtremeLightProfileCfg(
            intensity_range=_as_range2(extreme_raw.get("intensity_range", ExtremeLightProfileCfg.intensity_range), f"{label}.extreme_profile.intensity_range", non_negative=True),
            color_min=_as_color3(extreme_raw.get("color_min", ExtremeLightProfileCfg.color_min), f"{label}.extreme_profile.color_min"),
            color_max=_as_color3(extreme_raw.get("color_max", ExtremeLightProfileCfg.color_max), f"{label}.extreme_profile.color_max"),
            direction_yaw_range_deg=_as_range2(extreme_raw.get("direction_yaw_range_deg", ExtremeLightProfileCfg.direction_yaw_range_deg), f"{label}.extreme_profile.direction_yaw_range_deg"),
            direction_pitch_range_deg=_as_range2(extreme_raw.get("direction_pitch_range_deg", ExtremeLightProfileCfg.direction_pitch_range_deg), f"{label}.extreme_profile.direction_pitch_range_deg"),
            angle_range_deg=_as_range2(extreme_raw.get("angle_range_deg", ExtremeLightProfileCfg.angle_range_deg), f"{label}.extreme_profile.angle_range_deg", non_negative=True),
        ),
        fill=fill_cfg,
        secondary_distant=secondary_cfg,
        visibility_guard=visibility_guard_cfg,
        usd_scene_light=usd_scene_light_cfg,
    )


def _parse_light_generalization_cfg(
    light_raw: Mapping[str, Any],
    *,
    background_raw: Mapping[str, Any],
    asset_split: str,
    label: str,
) -> LightGeneralizationCfg:
    base_raw = dict(light_raw)
    base_raw.pop("profiles", None)
    profiles = []
    for entry in _iter_split_profile_entries(light_raw.get("profiles"), f"{label}.profiles"):
        profile_raw = copy.deepcopy(base_raw)
        profile_name = str(entry.get("name", ""))
        profile_split = _normalize_asset_split(entry.get("split", asset_split), f"{label}.profiles.{profile_name}.split")
        profile_overrides = dict(entry)
        for key in ("name", "split", "profile_name", "profile_split", "profiles"):
            profile_overrides.pop(key, None)
        profile_raw.update(profile_overrides)
        profile_raw["asset_split"] = profile_split
        profiles.append(_light_cfg_from_raw(
            profile_raw,
            background_raw=background_raw,
            asset_split=profile_split,
            label=f"{label}.profiles.{profile_split}.{profile_name or len(profiles)}",
            profile_name=profile_name,
            profile_split=profile_split,
            profiles=(),
        ))
    return _light_cfg_from_raw(
        base_raw,
        background_raw=background_raw,
        asset_split=asset_split,
        label=label,
        profiles=tuple(profiles),
    )


def _parse_camera_entries(camera_entries_raw: Mapping[str, Any], label: str) -> Dict[str, SimpleNamespace]:
    if not isinstance(camera_entries_raw, Mapping):
        raise TypeError(f"{label} must be a mapping.")
    return {
        str(camera_id): _ns(
            enabled=bool(_expect_mapping(entry, f"{label}.{camera_id}").get("enabled", True)),
            stereo_group=str(_expect_mapping(entry, f"{label}.{camera_id}").get("stereo_group", "") or ""),
            position_offset_m_range=_as_vec3(_expect_mapping(entry, f"{label}.{camera_id}").get("position_offset_m_range", (0.0, 0.0, 0.0)), f"{label}.{camera_id}.position_offset_m_range", True),
            target_offset_m_range=_as_vec3(_expect_mapping(entry, f"{label}.{camera_id}").get("target_offset_m_range", (0.0, 0.0, 0.0)), f"{label}.{camera_id}.target_offset_m_range", True),
            distance_offset_m_range=_as_non_negative_float(_expect_mapping(entry, f"{label}.{camera_id}").get("distance_offset_m_range", 0.0), f"{label}.{camera_id}.distance_offset_m_range"),
            offset_xyz_offset_m_range=_as_vec3(_expect_mapping(entry, f"{label}.{camera_id}").get("offset_xyz_offset_m_range", (0.0, 0.0, 0.0)), f"{label}.{camera_id}.offset_xyz_offset_m_range", True),
            rpy_offset_deg_range=_as_vec3(_expect_mapping(entry, f"{label}.{camera_id}").get("rpy_offset_deg_range", (0.0, 0.0, 0.0)), f"{label}.{camera_id}.rpy_offset_deg_range", True),
        )
        for camera_id, entry in camera_entries_raw.items()
    }


def _deep_merge_mapping(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge_mapping(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _parse_camera_generalization_cfg(
    camera_raw: Mapping[str, Any],
    *,
    asset_split: str,
    label: str,
) -> SimpleNamespace:
    camera_entries_raw = camera_raw.get("cameras", {}) or {}
    base_raw = dict(camera_raw)
    base_raw.pop("profiles", None)
    base_cameras = _expect_mapping(camera_entries_raw, f"{label}.cameras")
    profiles = []
    for entry in _iter_split_profile_entries(camera_raw.get("profiles"), f"{label}.profiles"):
        profile_name = str(entry.get("name", ""))
        profile_split = _normalize_asset_split(entry.get("split", asset_split), f"{label}.profiles.{profile_name}.split")
        profile_overrides = dict(entry)
        for key in ("name", "split", "profile_name", "profile_split", "profiles"):
            profile_overrides.pop(key, None)
        profile_cameras = profile_overrides.pop("cameras", None)
        merged_profile_raw = _deep_merge_mapping(base_raw, profile_overrides)
        merged_cameras = copy.deepcopy(base_cameras)
        if profile_cameras is not None:
            merged_cameras = _deep_merge_mapping(
                merged_cameras,
                _expect_mapping(profile_cameras, f"{label}.profiles.{profile_split}.{profile_name}.cameras"),
            )
        profiles.append(_ns(
            name=profile_name,
            split=profile_split,
            enabled=bool(merged_profile_raw.get("enabled", camera_raw.get("enabled", False))),
            cameras=_parse_camera_entries(
                merged_cameras,
                f"{label}.profiles.{profile_split}.{profile_name or len(profiles)}.cameras",
            ),
        ))
    return _ns(
        enabled=bool(camera_raw.get("enabled", False)),
        asset_split=_normalize_asset_split(camera_raw.get("asset_split", asset_split), f"{label}.asset_split"),
        profile_name="",
        profile_split="",
        profiles=tuple(profiles),
        cameras=_parse_camera_entries(base_cameras, f"{label}.cameras"),
    )


def _as_aabb(value: Any, label: str):
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"{label} must be [[x,y,z],[x,y,z]].")
    p0 = _as_vec3(value[0], f"{label}[0]")
    p1 = _as_vec3(value[1], f"{label}[1]")
    return (
        (min(p0[0], p1[0]), min(p0[1], p1[1]), min(p0[2], p1[2])),
        (max(p0[0], p1[0]), max(p0[1], p1[1]), max(p0[2], p1[2])),
    )


def _as_aabb_list(value: Any, label: str):
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a list of AABBs.")
    return tuple(_as_aabb(item, f"{label}[{index}]") for index, item in enumerate(value))


def _as_count_range(value: Any, label: str) -> Tuple[int, int]:
    if isinstance(value, int):
        return (max(0, int(value)), max(0, int(value)))
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"{label} must be an integer or [min, max].")
    lo = _as_int(value[0], f"{label}[0]")
    hi = _as_int(value[1], f"{label}[1]")
    if lo < 0 or hi < 0 or hi < lo:
        raise ValueError(f"{label} must satisfy 0 <= min <= max.")
    return (lo, hi)


def _serialize(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _serialize(val) for key, val in value.items()}
    if hasattr(value, "__dict__"):
        return {str(key): _serialize(val) for key, val in vars(value).items()}
    return value


def extract_asset_code(text: Any) -> str:
    match = _ASSET_CODE_PATTERN.search(str(text)) if text is not None else None
    return match.group(1) if match else ""


def collect_task_asset_codes(task: Mapping[str, Any] | None) -> Tuple[str, ...]:
    codes = []
    assets = task.get("assets", {}) if isinstance(task, Mapping) else {}
    if isinstance(assets, Mapping):
        for asset_key, asset_spec in assets.items():
            code = extract_asset_code(asset_key)
            if not code and isinstance(asset_spec, Mapping):
                code = extract_asset_code(asset_spec.get("path"))
            if code and code not in codes:
                codes.append(code)
    return tuple(codes)


def scene_generalization_config_to_dict(cfg) -> Dict[str, Any]:
    return _serialize(cfg)


def scene_generalization_sample_to_dict(sample) -> Dict[str, Any]:
    return _serialize(sample)


DATASET_PATH_MARKERS = ("dex2bench_dataset/", "Dex2Assets/")


def _extract_relative_dataset_path(uri: str) -> str | None:
    for marker in DATASET_PATH_MARKERS:
        idx = uri.find(marker)
        if idx >= 0:
            return uri[idx:]
    return None


def _find_dataset_root(start_dir: str) -> str:
    """Walk up from *start_dir* to find the directory that contains
    ``dex2bench_dataset/`` or ``Dex2Assets/``."""
    d = os.path.abspath(start_dir)
    for _ in range(10):
        for marker in DATASET_PATH_MARKERS:
            if os.path.isdir(os.path.join(d, marker.rstrip("/"))):
                return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.abspath(start_dir)


def relativize_sample_paths(data):
    """Convert absolute dataset paths to relative (``dex2bench_dataset/...``)
    so the serialised sample is environment-independent."""
    if isinstance(data, dict):
        return {k: relativize_sample_paths(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(relativize_sample_paths(item) for item in data)
    if isinstance(data, str):
        rel = _extract_relative_dataset_path(data)
        if rel is not None:
            return rel
    return data


def resolve_sample_paths(data, base_dir: str):
    """Resolve dataset paths in a deserialised sample against *base_dir*.

    *base_dir* should be the directory that **contains** ``dex2bench_dataset/``
    (not a deeper config directory).  Use :func:`_find_dataset_root` to locate
    it automatically.

    Handles two cases:
    1. Relative paths (``dex2bench_dataset/...``) — produced by
       :func:`relativize_sample_paths`.
    2. Absolute paths from a different environment — extracts the relative
       portion (from the dataset marker onwards) and re-resolves.
    """
    if isinstance(data, dict):
        return {k: resolve_sample_paths(v, base_dir) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(resolve_sample_paths(item, base_dir) for item in data)
    if isinstance(data, str):
        for marker in DATASET_PATH_MARKERS:
            if data.startswith(marker):
                return os.path.normpath(os.path.join(base_dir, data)).replace("\\", "/")
        rel = _extract_relative_dataset_path(data)
        if rel is not None:
            return os.path.normpath(os.path.join(base_dir, rel)).replace("\\", "/")
    return data


def _deserialize(data: Any) -> Any:
    if isinstance(data, dict):
        return SimpleNamespace(**{k: _deserialize(v) for k, v in data.items()})
    if isinstance(data, list):
        return [_deserialize(item) for item in data]
    return data


def dict_to_generalization_sample(data: Dict[str, Any]) -> SimpleNamespace:
    """Recursively convert a dict (from JSON / ``_serialize``) back to nested
    ``SimpleNamespace``, the inverse of ``scene_generalization_sample_to_dict``.

    Note: ``_serialize`` converts tuples to lists, so the roundtrip yields
    lists where the original had tuples (e.g. color, position).  Downstream
    code uses iteration / indexing, so this is functionally benign.
    """
    if not isinstance(data, dict):
        raise TypeError("Expected a dict for generalization sample deserialization.")
    return _deserialize(data)


def validate_resolved_object_placement_keys(
    sample_dict: Mapping[str, Any],
    task_object_ids: Iterable[str],
    *,
    required: bool = False,
) -> None:
    """Fail when an anchor/replay placement map does not match the task IDs.

    A missing ID must never silently fall through to randomized placement: that
    would change task geometry in profiles such as ``none`` and ``inv_only``
    that are supposed to preserve the anchor geometry.  Freshly sampled
    profiles do not carry ``resolved_object_placements`` and are unaffected.
    """
    if "resolved_object_placements" not in sample_dict:
        if required:
            raise ValueError(
                "anchor/replay sample is missing resolved_object_placements. "
                "Refusing randomized-placement fallback."
            )
        return

    placements = sample_dict["resolved_object_placements"]
    if not isinstance(placements, Mapping):
        raise ValueError(
            "resolved_object_placements must be a mapping when supplied by "
            f"an anchor/replay sample; got {type(placements).__name__}. "
            "Refusing randomized-placement fallback."
        )

    expected = {str(object_id) for object_id in task_object_ids}
    actual = {str(object_id) for object_id in placements}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if not missing and not unexpected:
        return

    raise ValueError(
        "resolved_object_placements object-ID mismatch: "
        f"missing current task IDs={missing}; "
        f"unexpected/stale IDs={unexpected}. "
        "Refusing randomized-placement fallback. Update the anchor/replay "
        "metadata so its object IDs exactly match the current task YAML."
    )


SAFE_RESAMPLE_GROUPS = frozenset({"background", "table_surface", "light", "camera"})
UNSAFE_RESAMPLE_GROUPS = frozenset({"object_pose", "table_height", "clutter"})

GROUP_PATHS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "background": (("appearance", "background"),),
    "table_surface": (("appearance", "table_surface"),),
    "light": (("appearance", "usd_scene_light"),),
    "camera": (("spatial", "camera"),),
    "object_pose": (("spatial", "object_pose"),),
    "table_height": (("spatial", "table_height"),),
    "clutter": (("clutter", "tabletop"),),
}

assert SAFE_RESAMPLE_GROUPS | UNSAFE_RESAMPLE_GROUPS == frozenset(GROUP_PATHS), (
    "SAFE_RESAMPLE_GROUPS | UNSAFE_RESAMPLE_GROUPS must cover all GROUP_PATHS keys"
)


def merge_generalization_samples(
    base_ns: SimpleNamespace,
    new_ns: SimpleNamespace,
    resample_groups: set,
) -> SimpleNamespace:
    """Return a copy of *base_ns* with selected groups replaced by *new_ns* values.

    Only groups listed in *resample_groups* are taken from *new_ns*; everything
    else is kept from *base_ns*.  A logical group may cover multiple fields,
    e.g. ``light`` includes both the legacy global light sample and embedded USD
    scene light sample.  The top-level ``robot_key`` attribute is not covered by
    any ``GROUP_PATHS`` entry, so it is structurally preserved from *base_ns* —
    callers should still set it explicitly after merging.
    """
    merged = copy.deepcopy(base_ns)
    for group_name in resample_groups:
        if group_name not in GROUP_PATHS:
            raise ValueError(f"Unknown generalization group: {group_name!r}")
        for parent_attr, child_attr in GROUP_PATHS[group_name]:
            src_parent = getattr(new_ns, parent_attr, None)
            if src_parent is None:
                continue
            src_value = getattr(src_parent, child_attr, None)
            if src_value is None:
                continue
            dst_parent = getattr(merged, parent_attr, None)
            if dst_parent is None:
                continue
            setattr(dst_parent, child_attr, copy.deepcopy(src_value))
    if {"object_pose", "table_height"} & set(resample_groups):
        if hasattr(new_ns, "resolved_object_placements"):
            merged.resolved_object_placements = copy.deepcopy(new_ns.resolved_object_placements)
        elif hasattr(merged, "resolved_object_placements"):
            delattr(merged, "resolved_object_placements")
    return merged


def parse_scene_generalization_config(raw: Mapping[str, Any] | None, *, config_path: str | None = None) -> SceneGeneralizationCfg:
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise TypeError("Generalization config must be a mapping.")
    config_dir = os.path.dirname(os.path.abspath(config_path)) if config_path else None
    raw = dict(raw)
    asset_split = _normalize_asset_split(raw.get("asset_split", "seen"), "asset_split")

    appearance_raw = _expect_mapping(raw.get("appearance"), "appearance")
    spatial_raw = _expect_mapping(raw.get("spatial"), "spatial")
    clutter_raw = _expect_mapping(raw.get("clutter"), "clutter")
    embodiment_raw = _expect_mapping(raw.get("embodiment"), "embodiment")
    legacy_object_raw = _expect_mapping(raw.get("object"), "object")
    legacy_light_raw = _expect_mapping(raw.get("light"), "light")

    appearance = _default_appearance()
    background_raw = _expect_mapping(appearance_raw.get("background"), "appearance.background")
    background_assets_raw = background_raw.get("assets", []) or []
    if isinstance(background_assets_raw, (str, bytes)) or not isinstance(background_assets_raw, Sequence):
        raise TypeError("appearance.background.assets must be a list.")
    background_assets = _parse_background_assets(
        background_raw,
        config_dir=config_dir,
        explicit_assets_raw=background_assets_raw,
    )
    appearance.background = _ns(
        enabled=bool(background_raw.get("enabled", appearance.background.enabled)),
        asset_split=_normalize_asset_split(background_raw.get("asset_split", asset_split), "appearance.background.asset_split"),
        clean_background_rate=_as_probability(
            background_raw.get("clean_background_rate", appearance.background.clean_background_rate),
            "appearance.background.clean_background_rate",
        ),
        intensity=_as_non_negative_float(background_raw.get("intensity", appearance.background.intensity), "appearance.background.intensity"),
        yaw_deg=_as_float(background_raw.get("yaw_deg", appearance.background.yaw_deg), "appearance.background.yaw_deg"),
        random_flip_180_rate=_as_probability(
            background_raw.get("random_flip_180_rate", appearance.background.random_flip_180_rate),
            "appearance.background.random_flip_180_rate",
        ),
        physics_enabled=bool(background_raw.get("physics_enabled", getattr(appearance.background, "physics_enabled", False))),
        fill_light_intensity=float(background_raw.get("fill_light_intensity", getattr(appearance.background, "fill_light_intensity", 0.0))),
        fill_light_jitter_ratio=_as_non_negative_float(
            background_raw.get("fill_light_jitter_ratio", getattr(appearance.background, "fill_light_jitter_ratio", 0.0)),
            "appearance.background.fill_light_jitter_ratio",
        ),
        enabled_categories=_parse_enabled_categories(background_raw.get("enabled_categories", ()), "appearance.background.enabled_categories"),
        assets=background_assets,
    )

    table_surface_raw = _expect_mapping(appearance_raw.get("table_surface"), "appearance.table_surface")
    table_assets_raw = table_surface_raw.get("assets", []) or []
    if isinstance(table_assets_raw, (str, bytes)) or not isinstance(table_assets_raw, Sequence):
        raise TypeError("appearance.table_surface.assets must be a list.")
    table_assets = _expand_table_surface_assets(
        table_surface_raw,
        config_dir=config_dir,
        explicit_assets_raw=table_assets_raw,
    )
    appearance.table_surface = _ns(
        enabled=bool(table_surface_raw.get("enabled", appearance.table_surface.enabled)),
        asset_split=_normalize_asset_split(table_surface_raw.get("asset_split", asset_split), "appearance.table_surface.asset_split"),
        clean_surface_rate=_as_probability(
            table_surface_raw.get("clean_surface_rate", appearance.table_surface.clean_surface_rate),
            "appearance.table_surface.clean_surface_rate",
        ),
        assets=table_assets,
    )

    light_raw = _expect_mapping(appearance_raw.get("light"), "appearance.light") or dict(legacy_light_raw)
    if legacy_light_raw and "enabled" not in light_raw:
        light_raw["enabled"] = True
    appearance.light = _parse_light_generalization_cfg(
        light_raw,
        background_raw=background_raw,
        asset_split=asset_split,
        label="appearance.light",
    )

    spatial = _default_spatial()
    object_pose_raw = _expect_mapping(spatial_raw.get("object_pose"), "spatial.object_pose") or dict(legacy_object_raw)
    if legacy_object_raw and "enabled" not in object_pose_raw:
        object_pose_raw["enabled"] = True
    spatial.object_pose = ObjectGeneralizationCfg(
        enabled=bool(object_pose_raw.get("enabled", True)),
        position_jitter_cm=_as_non_negative_float(object_pose_raw.get("position_jitter_cm", ObjectGeneralizationCfg.position_jitter_cm), "spatial.object_pose.position_jitter_cm"),
        yaw_jitter_deg=_as_non_negative_float(object_pose_raw.get("yaw_jitter_deg", ObjectGeneralizationCfg.yaw_jitter_deg), "spatial.object_pose.yaw_jitter_deg"),
    )
    table_height_raw = _expect_mapping(spatial_raw.get("table_height"), "spatial.table_height")
    spatial.table_height = TableHeightGeneralizationCfg(
        enabled=bool(table_height_raw.get("enabled", TableHeightGeneralizationCfg.enabled)),
        offset_m_range=_as_range2(
            table_height_raw.get("offset_m_range", TableHeightGeneralizationCfg.offset_m_range),
            "spatial.table_height.offset_m_range",
        ),
    )
    camera_raw = _expect_mapping(spatial_raw.get("camera"), "spatial.camera")
    spatial.camera = _parse_camera_generalization_cfg(
        camera_raw,
        asset_split=asset_split,
        label="spatial.camera",
    )
    clutter = _default_clutter()
    tabletop_raw = _expect_mapping(clutter_raw.get("tabletop"), "clutter.tabletop")
    zone_raw = _expect_mapping(tabletop_raw.get("zone"), "clutter.tabletop.zone")
    forbidden_xy_rect_raw = _expect_mapping(tabletop_raw.get("forbidden_xy_rect"), "clutter.tabletop.forbidden_xy_rect")
    asset_pool_raw = tabletop_raw.get("asset_pool", []) or []
    if isinstance(asset_pool_raw, (str, bytes)) or not isinstance(asset_pool_raw, Sequence):
        raise TypeError("clutter.tabletop.asset_pool must be a list.")
    clutter_split_ratios = _split_ratios(
        _expect_mapping(tabletop_raw.get("split_ratios"), "clutter.tabletop.split_ratios"),
        "clutter.tabletop.split_ratios",
    )
    clutter_asset_pool = []
    for index, item in enumerate(asset_pool_raw):
        item_map = _expect_mapping(item, f"clutter.tabletop.asset_pool[{index}]")
        name = str(item_map.get("name", f"clutter_{index}"))
        uri = _normalize_uri(item_map.get("uri", ""), config_dir=config_dir)
        asset_id = str(
            item_map.get("asset_id", "")
            or name
            or extract_asset_code(item_map)
            or extract_asset_code(uri)
        )
        split_value = item_map.get("split", item_map.get("asset_split"))
        clutter_asset_pool.append(_ns(
            name=name,
            uri=uri,
            scale=_as_vec3(item_map.get("scale", (1.0, 1.0, 1.0)), f"clutter.tabletop.asset_pool[{index}].scale", True),
            body_type=str(item_map.get("body_type", "dynamic")).lower(),
            category=str(item_map.get("category", "")),
            asset_code=str(item_map.get("asset_code", extract_asset_code(item_map) or extract_asset_code(uri))),
            confusable_asset_codes=_as_categories(item_map.get("confusable_asset_codes", ()), f"clutter.tabletop.asset_pool[{index}].confusable_asset_codes"),
            asset_id=asset_id,
            split=_normalize_asset_split(split_value, f"clutter.tabletop.asset_pool[{index}].split") if split_value is not None else _hash_split(asset_id, clutter_split_ratios),
        ))
    _ensure_min_unseen(clutter_asset_pool, group_key=lambda a: a.category)
    clutter.tabletop = _ns(
        enabled=bool(tabletop_raw.get("enabled", False)),
        asset_split=_normalize_asset_split(tabletop_raw.get("asset_split", asset_split), "clutter.tabletop.asset_split"),
        split_ratios=clutter_split_ratios,
        count=_as_count_range(tabletop_raw.get("count", (0, 0)), "clutter.tabletop.count"),
        zone=_ns(
            mode=str(zone_raw.get("mode", "tabletop")).lower(),
            aabb=_as_aabb(zone_raw.get("aabb"), "clutter.tabletop.zone.aabb"),
            exclude_aabbs=_as_aabb_list(zone_raw.get("exclude_aabbs", ()), "clutter.tabletop.zone.exclude_aabbs"),
        ),
        forbidden_xy_rect=_ns(
            enabled=bool(forbidden_xy_rect_raw.get("enabled", clutter.tabletop.forbidden_xy_rect.enabled)),
            x_range=_as_range2(
                forbidden_xy_rect_raw.get("x_range", clutter.tabletop.forbidden_xy_rect.x_range),
                "clutter.tabletop.forbidden_xy_rect.x_range",
            ),
            y_range=_as_range2(
                forbidden_xy_rect_raw.get("y_range", clutter.tabletop.forbidden_xy_rect.y_range),
                "clutter.tabletop.forbidden_xy_rect.y_range",
            ),
        ),
        asset_pool=tuple(clutter_asset_pool),
        max_bbox_size_m=_as_non_negative_float(tabletop_raw.get("max_bbox_size_m", 0.35), "clutter.tabletop.max_bbox_size_m"),
        max_height_m=_as_non_negative_float(tabletop_raw.get("max_height_m", 0.25), "clutter.tabletop.max_height_m"),
        placement_retry_limit=max(1, _as_int(tabletop_raw.get("placement_retry_limit", 50), "clutter.tabletop.placement_retry_limit")),
        scale_range=_as_range2(tabletop_raw.get("scale_range", (1.0, 1.0)), "clutter.tabletop.scale_range", non_negative=True),
        edge_margin_m=_as_non_negative_float(tabletop_raw.get("edge_margin_m", 0.0), "clutter.tabletop.edge_margin_m"),
    )

    embodiment = _default_embodiment()
    robot_asset_raw = _expect_mapping(embodiment_raw.get("robot_asset"), "embodiment.robot_asset")
    selection_mode = str(robot_asset_raw.get("selection_mode", RobotAssetGeneralizationCfg.selection_mode))
    if selection_mode not in {"default", "random"}:
        raise ValueError("embodiment.robot_asset.selection_mode must be 'default' or 'random'.")
    embodiment.robot_asset = RobotAssetGeneralizationCfg(
        enabled=bool(robot_asset_raw.get("enabled", RobotAssetGeneralizationCfg.enabled)),
        selection_mode=selection_mode,
        allowed_robot_keys=_as_categories(
            robot_asset_raw.get("allowed_robot_keys", ()),
            "embodiment.robot_asset.allowed_robot_keys",
        ),
        default_robot_key=str(robot_asset_raw.get("default_robot_key", "")),
    )

    return SceneGeneralizationCfg(
        appearance=appearance,
        spatial=spatial,
        clutter=clutter,
        embodiment=embodiment,
        asset_split=asset_split,
    )


def load_scene_generalization_config(config_path: str) -> SceneGeneralizationCfg:
    return parse_scene_generalization_config(load_yaml(config_path), config_path=config_path)


def _sample_sym(value: float) -> float:
    return _rng().uniform(-value, value) if value > 0.0 else 0.0


def _sample_vec3_sym(value: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return tuple(_sample_sym(abs(v)) for v in value)


def _clamp_color_floor(color: Sequence[float], floor: float) -> Tuple[float, float, float]:
    return tuple(min(1.0, max(float(floor), float(channel))) for channel in color)


def _disabled_light_component(source: str = "explicit") -> SimpleNamespace:
    return _ns(
        enabled=False,
        source=source,
        intensity=0.0,
        color=(1.0, 1.0, 1.0),
        direction_quat=_rpy_to_quat((0.0, -70.0, 0.0)),
        angle_deg=0.0,
        cast_shadows=False,
    )


def _default_embodiment():
    return _ns(robot_asset=RobotAssetGeneralizationCfg())


def _sample_fill_light(cfg: FillLightGeneralizationCfg) -> SimpleNamespace:
    if not cfg.enabled or cfg.intensity <= 0.0:
        return _disabled_light_component(cfg.source)
    intensity = (
        cfg.intensity * _rng().uniform(1.0 - cfg.intensity_jitter_ratio, 1.0 + cfg.intensity_jitter_ratio)
        if cfg.intensity_jitter_ratio > 0.0
        else cfg.intensity
    )
    return _ns(
        enabled=True,
        source=str(cfg.source),
        intensity=float(max(0.0, intensity)),
        color=tuple(cfg.color),
        direction_quat=_rpy_to_quat((0.0, cfg.direction_pitch_deg, cfg.direction_yaw_deg)),
        angle_deg=float(cfg.angle_deg),
        cast_shadows=bool(cfg.cast_shadows),
    )


def _sample_secondary_distant_light(cfg: SecondaryDistantLightGeneralizationCfg) -> SimpleNamespace:
    if not cfg.enabled or cfg.intensity <= 0.0:
        return _ns(
            enabled=False,
            intensity=0.0,
            color=tuple(cfg.color),
            direction_quat=_rpy_to_quat((0.0, cfg.direction_pitch_deg, cfg.direction_yaw_deg)),
            angle_deg=float(cfg.angle_deg),
            cast_shadows=bool(cfg.cast_shadows),
        )
    intensity = (
        cfg.intensity * _rng().uniform(1.0 - cfg.intensity_jitter_ratio, 1.0 + cfg.intensity_jitter_ratio)
        if cfg.intensity_jitter_ratio > 0.0
        else cfg.intensity
    )
    color = (
        tuple(min(1.0, max(0.0, channel + _sample_sym(cfg.color_jitter_abs))) for channel in cfg.color)
        if cfg.color_jitter_abs > 0.0
        else tuple(cfg.color)
    )
    yaw_deg = cfg.direction_yaw_deg + _sample_sym(cfg.direction_yaw_jitter_deg)
    pitch_deg = cfg.direction_pitch_deg + _sample_sym(cfg.direction_pitch_jitter_deg)
    angle_deg = max(0.0, cfg.angle_deg + _sample_sym(cfg.angle_jitter_deg))
    return _ns(
        enabled=True,
        intensity=float(max(0.0, intensity)),
        color=tuple(color),
        direction_quat=_rpy_to_quat((0.0, pitch_deg, yaw_deg)),
        angle_deg=float(angle_deg),
        cast_shadows=bool(cfg.cast_shadows),
    )


def _apply_visibility_guard(
    *,
    guard: LightVisibilityGuardCfg,
    intensity: float,
    color: Tuple[float, float, float],
    fill: SimpleNamespace,
    secondary: SimpleNamespace,
) -> Tuple[float, Tuple[float, float, float], SimpleNamespace, SimpleNamespace, bool]:
    if not guard.enabled:
        return intensity, color, fill, secondary, False
    applied = False
    clamped_color = _clamp_color_floor(color, guard.min_color_channel)
    if clamped_color != tuple(color):
        color = clamped_color
        applied = True
    if bool(getattr(fill, "enabled", False)):
        clamped_fill_color = _clamp_color_floor(fill.color, guard.min_color_channel)
        if clamped_fill_color != tuple(fill.color):
            fill.color = clamped_fill_color
            applied = True
    if bool(getattr(secondary, "enabled", False)):
        clamped_secondary_color = _clamp_color_floor(secondary.color, guard.min_color_channel)
        if clamped_secondary_color != tuple(secondary.color):
            secondary.color = clamped_secondary_color
            applied = True
    fill_intensity = float(getattr(fill, "intensity", 0.0) or 0.0) if bool(getattr(fill, "enabled", False)) else 0.0
    secondary_intensity = float(getattr(secondary, "intensity", 0.0) or 0.0) if bool(getattr(secondary, "enabled", False)) else 0.0
    total_lux = float(intensity) + fill_intensity + secondary_intensity
    if total_lux < guard.min_total_lux:
        delta = float(guard.min_total_lux) - total_lux
        if not bool(getattr(fill, "enabled", False)):
            fill.enabled = True
            fill.source = "visibility_guard"
            fill.color = _clamp_color_floor(getattr(fill, "color", (1.0, 1.0, 1.0)), guard.min_color_channel)
            fill.direction_quat = _rpy_to_quat((0.0, -70.0, 0.0))
            fill.angle_deg = 5.0
            fill.cast_shadows = False
            fill.intensity = 0.0
        fill.intensity = float(getattr(fill, "intensity", 0.0) or 0.0) + delta
        applied = True
    return float(intensity), tuple(color), fill, secondary, applied


def _select_light_config(
    light_cfg: LightGeneralizationCfg,
    requested_split: str | None,
) -> tuple[LightGeneralizationCfg, str, int]:
    split = _normalize_asset_split(
        requested_split or getattr(light_cfg, "asset_split", "seen"),
        "appearance.light.asset_split",
    )
    profiles = tuple(getattr(light_cfg, "profiles", ()) or ())
    if not profiles:
        return light_cfg, split, 0
    pool = [
        profile for profile in profiles
        if _asset_split_matches(getattr(profile, "profile_split", "seen"), split)
    ]
    if not pool:
        raise ValueError(f"appearance.light.asset_split={split!r} filtered all configured profiles.")
    return _rng().choice(pool), split, len(pool)


def _select_camera_config(camera_cfg: Any, requested_split: str | None) -> tuple[Any, str, int]:
    split = _normalize_asset_split(
        requested_split or getattr(camera_cfg, "asset_split", "seen"),
        "spatial.camera.asset_split",
    )
    profiles = tuple(getattr(camera_cfg, "profiles", ()) or ())
    if not profiles:
        return camera_cfg, split, 0
    pool = [
        profile for profile in profiles
        if _asset_split_matches(getattr(profile, "split", "seen"), split)
    ]
    if not pool:
        raise ValueError(f"spatial.camera.asset_split={split!r} filtered all configured profiles.")
    selected = _rng().choice(pool)
    return _ns(
        enabled=bool(getattr(selected, "enabled", getattr(camera_cfg, "enabled", False))),
        asset_split=split,
        profile_name=str(getattr(selected, "name", "")),
        profile_split=str(getattr(selected, "split", "")),
        profiles=(),
        cameras=getattr(selected, "cameras", {}),
    ), split, len(pool)


def _sample_usd_scene_light(cfg: UsdSceneLightCfg, enabled: bool):
    """Sample DistantLight and DomeLight parameters for a loaded USD environment scene.

    Returns a :class:`~types.SimpleNamespace` with ``.distant`` and ``.dome``
    sub-namespaces, each holding the sampled attribute values ready to be applied
    to the corresponding USD prim.
    """
    if not enabled or not cfg.enabled:
        return _ns(enabled=False, distant=None, dome=None)

    distant = _ns(
        intensity=_rng().uniform(*cfg.distant.intensity_range),
        color=tuple(
            _rng().uniform(cfg.distant.color_min[i], cfg.distant.color_max[i])
            for i in range(3)
        ),
        pitch_deg=_rng().uniform(*cfg.distant.pitch_range_deg),
        yaw_deg=_rng().uniform(*cfg.distant.yaw_range_deg),
        angle_deg=_rng().uniform(*cfg.distant.angle_range_deg),
    )
    dome = _ns(
        intensity=_rng().uniform(*cfg.dome.intensity_range),
        color=tuple(
            _rng().uniform(cfg.dome.color_min[i], cfg.dome.color_max[i])
            for i in range(3)
        ),
        exposure=_rng().uniform(*cfg.dome.exposure_range),
    )
    return _ns(enabled=True, distant=distant, dome=dome)


def _is_usd_scene_background(asset_kind: Any) -> bool:
    return str(asset_kind or "").strip().lower() == "usd_scene"


def resample_usd_scene_light(
    cfg: SceneGeneralizationCfg,
    *,
    enabled: bool,
    background_asset_kind: str | None,
    asset_split: str | None = None,
) -> SimpleNamespace:
    light_cfg, _requested_split, _profile_pool_size = _select_light_config(
        cfg.appearance.light, asset_split or getattr(cfg, "asset_split", None),
    )
    return _sample_usd_scene_light(
        light_cfg.usd_scene_light,
        enabled=bool(enabled and _is_usd_scene_background(background_asset_kind)),
    )


def _sample_light(
    cfg: LightGeneralizationCfg,
    enabled: bool,
    base_light_cfg: BaseLightCfg | None = None,
    *,
    background_yaw_deg: float = 0.0,
):
    base = base_light_cfg or BaseLightCfg()
    if not enabled or not cfg.enabled:
        return _ns(
            enabled=False,
            is_extreme=False,
            jittered=False,
            visibility_guard_applied=False,
            intensity=base.intensity,
            color=base.color,
            direction_quat=_rpy_to_quat((0.0, base.direction_pitch_deg, base.direction_yaw_deg)),
            angle_deg=base.angle_deg,
            fill=_disabled_light_component(getattr(cfg.fill, "source", "explicit")),
            secondary_distant=_sample_secondary_distant_light(
                SecondaryDistantLightGeneralizationCfg(enabled=False)
            ),
        )
    is_extreme = _rng().random() < cfg.extreme_light_sampling_rate
    if is_extreme:
        profile = cfg.extreme_profile
        intensity = _rng().uniform(*profile.intensity_range)
        color = tuple(_rng().uniform(profile.color_min[i], profile.color_max[i]) for i in range(3))
        yaw_deg = _rng().uniform(*profile.direction_yaw_range_deg)
        pitch_deg = _rng().uniform(*profile.direction_pitch_range_deg)
        angle_deg = _rng().uniform(*profile.angle_range_deg)
    else:
        intensity = cfg.intensity * _rng().uniform(1.0 - cfg.intensity_jitter_ratio, 1.0 + cfg.intensity_jitter_ratio) if cfg.intensity_jitter_ratio > 0.0 else cfg.intensity
        color = tuple(min(1.0, max(0.0, channel + _sample_sym(cfg.color_jitter_abs))) for channel in cfg.color) if cfg.color_jitter_abs > 0.0 else tuple(cfg.color)
        yaw_deg = cfg.direction_yaw_deg + _sample_sym(cfg.direction_yaw_jitter_deg)
        pitch_deg = cfg.direction_pitch_deg + _sample_sym(cfg.direction_pitch_jitter_deg)
        angle_deg = max(0.0, cfg.angle_deg + _sample_sym(cfg.angle_jitter_deg))
    if cfg.follow_background_yaw:
        yaw_deg += background_yaw_deg
    fill = _sample_fill_light(cfg.fill)
    secondary = _sample_secondary_distant_light(cfg.secondary_distant)
    intensity, color, fill, secondary, visibility_guard_applied = _apply_visibility_guard(
        guard=cfg.visibility_guard,
        intensity=float(intensity),
        color=tuple(color),
        fill=fill,
        secondary=secondary,
    )
    jittered = (
        bool(is_extreme)
        or abs(float(intensity) - float(base.intensity)) > 1e-9
        or any(abs(float(color[i]) - float(base.color[i])) > 1e-9 for i in range(3))
        or abs(float(yaw_deg) - float(base.direction_yaw_deg)) > 1e-9
        or abs(float(pitch_deg) - float(base.direction_pitch_deg)) > 1e-9
        or abs(float(angle_deg) - float(base.angle_deg)) > 1e-9
        or bool(getattr(fill, "enabled", False))
        or bool(getattr(secondary, "enabled", False))
        or bool(visibility_guard_applied)
    )
    return _ns(
        enabled=True,
        follow_background_yaw=bool(cfg.follow_background_yaw),
        applied_background_yaw_deg=float(background_yaw_deg if cfg.follow_background_yaw else 0.0),
        is_extreme=is_extreme,
        jittered=jittered,
        visibility_guard_applied=bool(visibility_guard_applied),
        intensity=float(intensity),
        color=tuple(color),
        direction_quat=_rpy_to_quat((0.0, pitch_deg, yaw_deg)),
        angle_deg=float(angle_deg),
        fill=fill,
        secondary_distant=secondary,
    )


def resample_light(
    cfg: SceneGeneralizationCfg,
    *,
    enabled: bool,
    background_yaw_deg: float = 0.0,
    asset_split: str | None = None,
) -> SimpleNamespace:
    """Sample a fresh light configuration using the given *background_yaw_deg*.

    Thin public wrapper around ``_sample_light`` for selective resample
    scenarios where light is re-randomized independently of background.
    """
    light_cfg, requested_split, profile_pool_size = _select_light_config(
        cfg.appearance.light, asset_split or getattr(cfg, "asset_split", None),
    )
    sample = _sample_light(light_cfg, enabled=enabled, background_yaw_deg=background_yaw_deg)
    sample.asset_split = requested_split
    sample.profile_name = str(getattr(light_cfg, "profile_name", "") or "")
    sample.profile_split = str(getattr(light_cfg, "profile_split", "") or "")
    sample.profile_pool_size = profile_pool_size
    return sample


def filter_clutter_asset_pool(asset_pool: Iterable[Any], *, task_asset_codes: Iterable[str] = ()):
    task_codes = {str(code) for code in task_asset_codes if code}
    filtered = []
    for asset in asset_pool:
        asset_code = str(getattr(asset, "asset_code", "") or extract_asset_code(getattr(asset, "name", "")) or extract_asset_code(getattr(asset, "uri", "")))
        if asset_code and asset_code in task_codes:
            continue
        if any(str(code) in task_codes for code in getattr(asset, "confusable_asset_codes", ())):
            continue
        filtered.append(asset)
    return tuple(filtered)


def _sample_robot_asset(
    robot_cfg: RobotAssetGeneralizationCfg,
    *,
    enabled: bool,
    available_robot_keys: Iterable[str] = (),
):
    default_key = str(robot_cfg.default_robot_key or "")
    if not enabled or not robot_cfg.enabled:
        return default_key, _ns(
            enabled=False,
            selection_mode=robot_cfg.selection_mode,
            default_robot_key=default_key,
            selected_robot_key=default_key,
            allowed_robot_keys=tuple(robot_cfg.allowed_robot_keys),
        )

    allowed = tuple(str(key) for key in robot_cfg.allowed_robot_keys if str(key))
    if not allowed:
        allowed = (default_key,) if default_key else ()
    available = tuple(str(key) for key in available_robot_keys if str(key))
    if available:
        available_set = set(available)
        missing = [key for key in allowed if key not in available_set]
        if missing:
            raise ValueError(f"Unknown robot key in embodiment.robot_asset.allowed_robot_keys: {missing[0]}")
    if not allowed:
        raise ValueError("embodiment.robot_asset requires allowed_robot_keys or default_robot_key when enabled.")

    selected = _rng().choice(list(allowed)) if robot_cfg.selection_mode == "random" else (default_key or allowed[0])
    if available and selected not in set(available):
        raise ValueError(f"Unknown robot key in embodiment.robot_asset.default_robot_key: {selected}")
    return selected, _ns(
        enabled=True,
        selection_mode=robot_cfg.selection_mode,
        default_robot_key=default_key,
        selected_robot_key=selected,
        allowed_robot_keys=allowed,
    )


def sample_scene_generalization(
    cfg: SceneGeneralizationCfg,
    *,
    enabled: bool,
    task_asset_codes: Iterable[str] = (),
    asset_split: str | None = None,
    available_robot_keys: Iterable[str] = (),
    background_asset_index: int | None = None,
):
    background_cfg = cfg.appearance.background
    table_surface_cfg = cfg.appearance.table_surface
    table_height_cfg = cfg.spatial.table_height
    camera_cfg_base = cfg.spatial.camera
    clutter_cfg = cfg.clutter.tabletop
    robot_key, robot_asset_sample = _sample_robot_asset(
        cfg.embodiment.robot_asset,
        enabled=enabled,
        available_robot_keys=available_robot_keys,
    )

    background_enabled = bool(enabled and getattr(background_cfg, "enabled", False))
    clean_background = True
    asset_kind = None
    asset_category = None
    asset_id = None
    background_asset_split = None
    asset_uri = None
    background_asset_pool_size = 0
    asset_scene_offset = (0.0, 0.0, 0.0)
    background_intensity = float(getattr(background_cfg, "intensity", 1500.0)) if background_enabled else 0.0
    background_base_yaw_deg = float(getattr(background_cfg, "yaw_deg", 0.0)) if background_enabled else 0.0
    background_flip_180_applied = False
    background_yaw_deg = background_base_yaw_deg
    if background_enabled:
        force_background_asset = background_asset_index is not None and bool(background_cfg.assets)
        clean_background = (
            False if force_background_asset
            else _rng().random() < background_cfg.clean_background_rate or not background_cfg.assets
        )
        if not clean_background:
            assets = list(background_cfg.assets)
            if background_cfg.enabled_categories:
                enabled = background_cfg.enabled_categories  # Dict[str, Tuple[str, ...]]
                def _category_enabled(asset) -> bool:
                    asset_kind = str(getattr(asset, "kind", "") or "")
                    asset_category = str(getattr(asset, "category", "") or "")
                    kind_cats = enabled.get(asset_kind) or enabled.get("*", ())
                    return asset_category in set(kind_cats)
                assets = [asset for asset in assets if _category_enabled(asset)]
                if not assets:
                    raise ValueError("appearance.background.enabled_categories filtered all configured assets.")
            requested_background_split = _normalize_asset_split(
                asset_split or getattr(background_cfg, "asset_split", getattr(cfg, "asset_split", "seen")),
                "appearance.background.asset_split",
            )
            assets = _filter_assets_for_split(assets, requested_background_split)
            if not assets:
                raise ValueError(f"appearance.background.asset_split={requested_background_split!r} filtered all configured assets.")
            background_asset_pool_size = len(assets)
            if background_asset_index is None:
                selected_background = _rng().choice(assets)
            else:
                selected_background = assets[int(background_asset_index) % len(assets)]
            asset_kind = selected_background.kind
            asset_category = selected_background.category
            asset_id = getattr(selected_background, "asset_id", None)
            background_asset_split = getattr(selected_background, "split", requested_background_split)
            asset_uri = selected_background.uri
            if asset_kind == "usd_scene":
                background_base_yaw_deg = float(getattr(selected_background, "yaw_deg", 0.0))
                asset_scene_offset = tuple(getattr(selected_background, "scene_offset", (0.0, 0.0, 0.0)))
        background_flip_180_applied = _rng().random() < float(getattr(background_cfg, "random_flip_180_rate", 0.0))
        background_yaw_deg = (background_base_yaw_deg + (180.0 if background_flip_180_applied else 0.0)) % 360.0

    light_cfg, requested_light_split, light_profile_pool_size = _select_light_config(
        cfg.appearance.light, asset_split,
    )
    camera_cfg, requested_camera_split, camera_profile_pool_size = _select_camera_config(
        camera_cfg_base, asset_split,
    )

    table_surface_enabled = bool(enabled and getattr(table_surface_cfg, "enabled", False))
    clean_surface = True
    table_surface_name = None
    table_surface_uri = None
    table_surface_category = None
    table_surface_asset_id = None
    table_surface_asset_split = None
    table_surface_asset_pool_size = 0
    if table_surface_enabled:
        clean_surface = _rng().random() < table_surface_cfg.clean_surface_rate or not table_surface_cfg.assets
        if not clean_surface:
            requested_table_split = _normalize_asset_split(
                asset_split or getattr(table_surface_cfg, "asset_split", getattr(cfg, "asset_split", "seen")),
                "appearance.table_surface.asset_split",
            )
            table_assets = _filter_assets_for_split(list(table_surface_cfg.assets), requested_table_split)
            if not table_assets:
                raise ValueError(f"appearance.table_surface.asset_split={requested_table_split!r} filtered all configured assets.")
            table_surface_asset_pool_size = len(table_assets)
            selected_surface = _choose_category_balanced(table_assets)
            table_surface_name = selected_surface.name
            table_surface_uri = selected_surface.uri
            table_surface_category = getattr(selected_surface, "category", None)
            table_surface_asset_id = getattr(selected_surface, "asset_id", None)
            table_surface_asset_split = getattr(selected_surface, "split", requested_table_split)

    table_height_enabled = bool(enabled and getattr(table_height_cfg, "enabled", False))
    table_height_offset_m = (
        _rng().uniform(*tuple(getattr(table_height_cfg, "offset_m_range", (0.0, 0.0))))
        if table_height_enabled
        else 0.0
    )

    camera_samples = {}
    _stereo_group_cache: dict[str, Any] = {}
    if enabled and getattr(camera_cfg, "enabled", False):
        for camera_id, entry in camera_cfg.cameras.items():
            entry_enabled = bool(getattr(entry, "enabled", True))
            stereo_group = str(getattr(entry, "stereo_group", "") or "")
            if stereo_group and stereo_group in _stereo_group_cache:
                cached = _stereo_group_cache[stereo_group]
                camera_samples[camera_id] = _ns(
                    enabled=entry_enabled,
                    position_offset_m=cached.position_offset_m if entry_enabled else (0.0, 0.0, 0.0),
                    target_offset_m=cached.target_offset_m if entry_enabled else (0.0, 0.0, 0.0),
                    distance_offset_m=cached.distance_offset_m if entry_enabled else 0.0,
                    offset_xyz_offset_m=cached.offset_xyz_offset_m if entry_enabled else (0.0, 0.0, 0.0),
                    rpy_offset_deg=cached.rpy_offset_deg if entry_enabled else (0.0, 0.0, 0.0),
                )
            else:
                sample = _ns(
                    enabled=entry_enabled,
                    position_offset_m=_sample_vec3_sym(entry.position_offset_m_range) if entry_enabled else (0.0, 0.0, 0.0),
                    target_offset_m=_sample_vec3_sym(entry.target_offset_m_range) if entry_enabled else (0.0, 0.0, 0.0),
                    distance_offset_m=_sample_sym(entry.distance_offset_m_range) if entry_enabled else 0.0,
                    offset_xyz_offset_m=_sample_vec3_sym(entry.offset_xyz_offset_m_range) if entry_enabled else (0.0, 0.0, 0.0),
                    rpy_offset_deg=_sample_vec3_sym(entry.rpy_offset_deg_range) if entry_enabled else (0.0, 0.0, 0.0),
                )
                camera_samples[camera_id] = sample
                if stereo_group:
                    _stereo_group_cache[stereo_group] = sample
            camera_samples[camera_id].has_offset = _camera_sample_has_offset(camera_samples[camera_id])

    clutter_enabled = bool(enabled and getattr(clutter_cfg, "enabled", False))
    clutter_assets = ()
    clutter_count = 0
    clutter_asset_split = None
    clutter_asset_pool_size = 0
    if clutter_enabled:
        clutter_count = _rng().randint(clutter_cfg.count[0], clutter_cfg.count[1])
        if clutter_count > 0:
            requested_clutter_split = _normalize_asset_split(
                asset_split or getattr(clutter_cfg, "asset_split", getattr(cfg, "asset_split", "seen")),
                "clutter.tabletop.asset_split",
            )
            split_pool = _filter_assets_for_split(list(clutter_cfg.asset_pool), requested_clutter_split)
            if not split_pool:
                raise ValueError(f"clutter.tabletop.asset_split={requested_clutter_split!r} filtered all configured assets.")
            pool = list(filter_clutter_asset_pool(split_pool, task_asset_codes=task_asset_codes))
            if not pool:
                raise ValueError("clutter.tabletop.asset_pool is empty after task/confusable filtering.")
            clutter_asset_split = requested_clutter_split
            clutter_asset_pool_size = len(pool)
            scale_lo, scale_hi = clutter_cfg.scale_range
            selected_assets = [_rng().choice(pool) for _ in range(clutter_count)]
            clutter_assets = []
            for asset in selected_assets:
                factor = _rng().uniform(scale_lo, scale_hi)
                clutter_assets.append(_ns(
                    name=asset.name,
                    uri=asset.uri,
                    scale=tuple(v * factor for v in asset.scale),
                    body_type=asset.body_type,
                    category=asset.category,
                    asset_code=asset.asset_code,
                    asset_id=getattr(asset, "asset_id", None),
                    asset_split=getattr(asset, "split", requested_clutter_split),
                ))
            clutter_assets = tuple(clutter_assets)

    is_usd_scene_background = _is_usd_scene_background(asset_kind)
    light_sample = _sample_light(
        light_cfg,
        enabled=bool(enabled and not is_usd_scene_background),
        background_yaw_deg=background_yaw_deg,
    )
    light_sample.asset_split = requested_light_split
    light_sample.profile_name = str(getattr(light_cfg, "profile_name", "") or "")
    light_sample.profile_split = str(getattr(light_cfg, "profile_split", "") or "")
    light_sample.profile_pool_size = light_profile_pool_size

    usd_scene_light_sample = _sample_usd_scene_light(
        light_cfg.usd_scene_light,
        enabled=bool(enabled and is_usd_scene_background),
    )
    usd_scene_light_sample.profile_name = str(getattr(light_cfg, "profile_name", "") or "")
    usd_scene_light_sample.profile_split = str(getattr(light_cfg, "profile_split", "") or "")
    usd_scene_light_sample.profile_pool_size = light_profile_pool_size

    return _ns(
        robot_key=robot_key,
        appearance=_ns(
            background=_ns(
                enabled=background_enabled,
                clean_background=clean_background,
                intensity=background_intensity,
                yaw_deg=background_yaw_deg,
                base_yaw_deg=background_base_yaw_deg,
                flip_180_applied=background_flip_180_applied,
                physics_enabled=bool(getattr(background_cfg, "physics_enabled", False)) if background_enabled else False,
                fill_light_intensity=float(getattr(background_cfg, "fill_light_intensity", 0.0)) if background_enabled else 0.0,
                fill_light_jitter_ratio=float(getattr(background_cfg, "fill_light_jitter_ratio", 0.0)) if background_enabled else 0.0,
                asset_kind=asset_kind,
                asset_category=asset_category,
                asset_id=asset_id,
                asset_split=background_asset_split,
                asset_pool_size=background_asset_pool_size,
                asset_uri=asset_uri,
                scene_offset=asset_scene_offset,
            ),
            table_surface=_ns(
                enabled=table_surface_enabled,
                clean_surface=clean_surface,
                asset_name=table_surface_name,
                asset_uri=table_surface_uri,
                asset_category=table_surface_category,
                asset_id=table_surface_asset_id,
                asset_split=table_surface_asset_split,
                asset_pool_size=table_surface_asset_pool_size,
            ),
            usd_scene_light=usd_scene_light_sample,
        ),
        spatial=_ns(
            object_pose=_ns(enabled=bool(enabled and cfg.spatial.object_pose.enabled), position_jitter_cm=cfg.spatial.object_pose.position_jitter_cm if enabled and cfg.spatial.object_pose.enabled else 0.0, position_jitter_m=(cfg.spatial.object_pose.position_jitter_cm * 0.01) if enabled and cfg.spatial.object_pose.enabled else 0.0, yaw_jitter_deg=cfg.spatial.object_pose.yaw_jitter_deg if enabled and cfg.spatial.object_pose.enabled else 0.0),
            table_height=_ns(enabled=table_height_enabled, height_offset_m=float(table_height_offset_m), offset_m_range=tuple(getattr(table_height_cfg, "offset_m_range", (0.0, 0.0)))),
            camera=_ns(
                enabled=bool(enabled and getattr(camera_cfg, "enabled", False)),
                asset_split=requested_camera_split,
                profile_name=str(getattr(camera_cfg, "profile_name", "") or ""),
                profile_split=str(getattr(camera_cfg, "profile_split", "") or ""),
                profile_pool_size=camera_profile_pool_size,
                cameras=camera_samples,
            ),
        ),
        clutter=_ns(tabletop=_ns(
            enabled=clutter_enabled,
            count=clutter_count,
            asset_split=clutter_asset_split,
            asset_pool_size=clutter_asset_pool_size,
            assets=clutter_assets,
        )),
        embodiment=_ns(robot_asset=robot_asset_sample),
    )


def _camera_sample_has_offset(camera_sample: Any) -> bool:
    vector_fields = ("position_offset_m", "target_offset_m", "offset_xyz_offset_m", "rpy_offset_deg")
    for field_name in vector_fields:
        values = getattr(camera_sample, field_name, ()) or ()
        if any(abs(float(value)) > 1e-12 for value in values):
            return True
    return abs(float(getattr(camera_sample, "distance_offset_m", 0.0) or 0.0)) > 1e-12


def active_perturbation_axes(
    sample: SimpleNamespace,
    *,
    generalization_enabled: bool,
) -> str:
    """Return comma-separated active perturbation axes, or ``none``."""
    if not generalization_enabled or sample is None:
        return "none"

    axes: list[str] = []
    appearance = getattr(sample, "appearance", sample)
    background = getattr(appearance, "background", getattr(sample, "background", None))
    if background is not None and not bool(getattr(background, "clean_background", True)):
        axes.append("background")

    table_surface = getattr(appearance, "table_surface", getattr(sample, "table_surface", None))
    if table_surface is not None and not bool(getattr(table_surface, "clean_surface", True)):
        axes.append("table_surface")

    usd_scene_light = getattr(appearance, "usd_scene_light", None)
    light_active = usd_scene_light is not None and bool(getattr(usd_scene_light, "enabled", False))
    if light_active:
        axes.append("light")

    spatial = getattr(sample, "spatial", sample)
    object_pose = getattr(spatial, "object_pose", getattr(sample, "object_pose", None))
    position_jitter_m = float(getattr(object_pose, "position_jitter_m", 0.0) or 0.0) if object_pose is not None else 0.0
    if position_jitter_m <= 0.0 and object_pose is not None:
        position_jitter_m = float(getattr(object_pose, "position_jitter_cm", 0.0) or 0.0) * 0.01
    if position_jitter_m > 0.0:
        axes.append("object_pose")

    table_height = getattr(spatial, "table_height", getattr(sample, "table_height", None))
    if table_height is not None and abs(float(getattr(table_height, "height_offset_m", 0.0) or 0.0)) > 1e-12:
        axes.append("table_height")

    camera = getattr(spatial, "camera", getattr(sample, "camera", None))
    camera_entries: list[Any] = []
    if camera is not None:
        cameras = getattr(camera, "cameras", camera)
        if isinstance(cameras, Mapping):
            camera_entries = list(cameras.values())
        elif isinstance(cameras, Sequence) and not isinstance(cameras, (str, bytes)):
            camera_entries = list(cameras)
        else:
            camera_entries = [cameras]
    if any(bool(getattr(entry, "has_offset", False)) or _camera_sample_has_offset(entry) for entry in camera_entries):
        axes.append("camera")

    embodiment = getattr(sample, "embodiment", None)
    robot_asset = getattr(embodiment, "robot_asset", None)
    if robot_asset is not None:
        selected_robot_key = str(getattr(robot_asset, "selected_robot_key", "") or getattr(sample, "robot_key", "") or "")
        default_robot_key = str(getattr(robot_asset, "default_robot_key", "") or "")
        if selected_robot_key and default_robot_key and selected_robot_key != default_robot_key:
            axes.append("embodiment")

    clutter = getattr(sample, "clutter", None)
    tabletop = getattr(clutter, "tabletop", clutter)
    if tabletop is not None and int(getattr(tabletop, "count", 0) or 0) > 0:
        axes.append("clutter")

    return ",".join(sorted(set(axes))) or "none"


def object_position_jitter_m(cfg, enabled: bool) -> float:
    if not enabled:
        return 0.0
    object_pose = cfg.spatial.object_pose if hasattr(cfg, "spatial") else getattr(cfg, "object", None)
    if object_pose is None or not getattr(object_pose, "enabled", True):
        return 0.0
    return float(getattr(object_pose, "position_jitter_cm", 0.0)) * 0.01


def sample_explicit_yaw_offset_deg(cfg, enabled: bool) -> float:
    if not enabled:
        return 0.0
    object_pose = cfg.spatial.object_pose if hasattr(cfg, "spatial") else getattr(cfg, "object", None)
    yaw_jitter_deg = float(getattr(object_pose, "yaw_jitter_deg", 0.0)) if object_pose is not None else 0.0
    if object_pose is None or not getattr(object_pose, "enabled", True) or yaw_jitter_deg <= 0.0:
        return 0.0
    return _rng().uniform(-yaw_jitter_deg, yaw_jitter_deg)


def sample_distant_light_state(base_light_cfg: BaseLightCfg, cfg: SceneGeneralizationCfg, enabled: bool):
    light_cfg, _requested_split, _profile_pool_size = _select_light_config(
        cfg.appearance.light, getattr(cfg, "asset_split", None),
    )
    sample = _sample_light(light_cfg, enabled=enabled, base_light_cfg=base_light_cfg)
    return float(sample.intensity), tuple(sample.color), tuple(sample.direction_quat), float(sample.angle_deg)


def scene_generalization_sample_debug_lines(sample_like: Any) -> list[str]:
    sample = sample_like if isinstance(sample_like, Mapping) else scene_generalization_sample_to_dict(sample_like)
    appearance = sample.get("appearance", {})
    spatial = sample.get("spatial", {})
    background = appearance.get("background", {})
    table_surface = appearance.get("table_surface", {})
    light = appearance.get("light", {})
    usd_scene_light = appearance.get("usd_scene_light", {})
    table_height = spatial.get("table_height", {})
    camera = spatial.get("camera", {})
    clutter = sample.get("clutter", {}).get("tabletop", {})
    usd_distant = usd_scene_light.get("distant") or {}
    usd_dome = usd_scene_light.get("dome") or {}

    background_line = (
        "[INFO] Scene generalization background: "
        f"enabled={bool(background.get('enabled', False))} "
        f"clean_background={bool(background.get('clean_background', True))} "
        f"intensity={float(background.get('intensity', 0.0)):.3f} "
        f"yaw_deg={float(background.get('yaw_deg', 0.0)):.3f} "
        f"base_yaw_deg={float(background.get('base_yaw_deg', 0.0)):.3f} "
        f"flip_180_applied={bool(background.get('flip_180_applied', False))} "
        f"asset_kind={background.get('asset_kind') or '<none>'} "
        f"asset_category={background.get('asset_category') or '<none>'} "
        f"asset_split={background.get('asset_split') or '<none>'} "
        f"asset_pool_size={int(background.get('asset_pool_size', 0) or 0)} "
        f"asset_uri={background.get('asset_uri') or '<none>'}"
    )
    table_surface_line = (
        "[INFO] Scene generalization table_surface: "
        f"enabled={bool(table_surface.get('enabled', False))} "
        f"clean_surface={bool(table_surface.get('clean_surface', True))} "
        f"asset_name={table_surface.get('asset_name') or '<none>'} "
        f"asset_category={table_surface.get('asset_category') or '<none>'} "
        f"asset_split={table_surface.get('asset_split') or '<none>'} "
        f"asset_pool_size={int(table_surface.get('asset_pool_size', 0) or 0)} "
        f"asset_uri={table_surface.get('asset_uri') or '<none>'}"
    )
    table_height_line = (
        "[INFO] Scene generalization table_height: "
        f"enabled={bool(table_height.get('enabled', False))} "
        f"height_offset_m={float(table_height.get('height_offset_m', 0.0) or 0.0):.3f}"
    )
    light_line = (
        "[INFO] Scene generalization light: "
        f"enabled={bool(light.get('enabled', False))} "
        f"asset_split={light.get('asset_split') or '<none>'} "
        f"profile_name={light.get('profile_name') or '<none>'} "
        f"profile_split={light.get('profile_split') or '<none>'} "
        f"profile_pool_size={int(light.get('profile_pool_size', 0) or 0)}"
    )
    usd_light_line = (
        "[INFO] Scene generalization usd_scene_light: "
        f"enabled={bool(usd_scene_light.get('enabled', False))} "
        f"distant_intensity={float(usd_distant.get('intensity', 0.0) or 0.0):.3f} "
        f"distant_pitch_deg={float(usd_distant.get('pitch_deg', 0.0) or 0.0):.3f} "
        f"distant_yaw_deg={float(usd_distant.get('yaw_deg', 0.0) or 0.0):.3f} "
        f"dome_intensity={float(usd_dome.get('intensity', 0.0) or 0.0):.3f} "
        f"dome_exposure={float(usd_dome.get('exposure', 0.0) or 0.0):.3f}"
    )
    camera_line = (
        "[INFO] Scene generalization camera: "
        f"enabled={bool(camera.get('enabled', False))} "
        f"asset_split={camera.get('asset_split') or '<none>'} "
        f"profile_name={camera.get('profile_name') or '<none>'} "
        f"profile_split={camera.get('profile_split') or '<none>'} "
        f"profile_pool_size={int(camera.get('profile_pool_size', 0) or 0)}"
    )
    clutter_line = (
        "[INFO] Scene generalization clutter: "
        f"enabled={bool(clutter.get('enabled', False))} "
        f"count={int(clutter.get('count', 0) or 0)} "
        f"asset_split={clutter.get('asset_split') or '<none>'} "
        f"asset_pool_size={int(clutter.get('asset_pool_size', 0) or 0)}"
    )
    return [background_line, table_surface_line, light_line, usd_light_line, camera_line, clutter_line, table_height_line]


def merge_scene_generalization_overrides(base_config: Mapping[str, Any] | None, overrides: Mapping[str, Any] | None) -> Dict[str, Any]:
    merged = copy.deepcopy(dict(base_config or {}))
    blocked = {"appearance.background.assets", "appearance.table_surface.assets", "clutter.tabletop.asset_pool"}

    def _merge(dst: Dict[str, Any], src: Mapping[str, Any], path: Tuple[str, ...]):
        for key, value in src.items():
            current_path = path + (str(key),)
            dotted = ".".join(current_path)
            if dotted in blocked:
                raise ValueError(f"generalization_overrides may not redefine {dotted}.")
            if key not in dst:
                raise ValueError(f"Unknown generalization override key: {dotted}")
            if isinstance(value, Mapping):
                if not isinstance(dst.get(key), Mapping):
                    raise ValueError(f"generalization_overrides key {dotted} must override an existing mapping.")
                _merge(dst[key], value, current_path)
            else:
                dst[key] = copy.deepcopy(value)

    _merge(merged, dict(overrides or {}), ())
    return merged


def audit_clutter_assets(cfg_or_path):
    cfg = load_scene_generalization_config(cfg_or_path) if isinstance(cfg_or_path, str) else cfg_or_path
    tabletop = cfg.clutter.tabletop
    names = set()
    categories = set()
    report = []
    for asset in tabletop.asset_pool:
        if asset.name in names:
            raise ValueError(f"Duplicate clutter asset name: {asset.name}")
        names.add(asset.name)
        if asset.body_type != "dynamic":
            raise ValueError(f"Clutter asset '{asset.name}' must use body_type=dynamic.")
        if not asset.category:
            raise ValueError(f"Clutter asset '{asset.name}' must define a category.")
        categories.add(asset.category)
        asset_code = str(asset.asset_code or extract_asset_code(asset.name) or extract_asset_code(asset.uri))
        if asset_code and 80 <= int(asset_code) <= 125:
            raise ValueError(f"Clutter asset '{asset.name}' uses forbidden asset code {asset_code}.")
        if "://" not in asset.uri and not os.path.exists(asset.uri):
            raise FileNotFoundError(f"Clutter asset file not found: {asset.uri}")
        bounds = _compute_bounds(asset.uri, asset.scale)
        if bounds is None:
            raise ValueError(f"Cannot compute clutter bounds for '{asset.name}'.")
        lo, hi = bounds
        size = tuple(float(hi[i] - lo[i]) for i in range(3))
        if max(size) > tabletop.max_bbox_size_m:
            raise ValueError(f"Clutter asset '{asset.name}' exceeds max_bbox_size_m={tabletop.max_bbox_size_m:.3f}: {size}")
        if size[2] > tabletop.max_height_m:
            raise ValueError(f"Clutter asset '{asset.name}' exceeds max_height_m={tabletop.max_height_m:.3f}: {size[2]:.4f}")
        report.append({"name": asset.name, "category": asset.category, "asset_code": asset_code, "uri": asset.uri, "bbox_size_m": list(size)})
    return {"asset_count": len(report), "categories": sorted(categories), "assets": report}


__all__ = [
    "BaseLightCfg",
    "ExtremeLightProfileCfg",
    "FillLightGeneralizationCfg",
    "GROUP_PATHS",
    "LightGeneralizationCfg",
    "LightVisibilityGuardCfg",
    "ObjectGeneralizationCfg",
    "RobotAssetGeneralizationCfg",
    "SAFE_RESAMPLE_GROUPS",
    "SecondaryDistantLightGeneralizationCfg",
    "SceneGeneralizationCfg",
    "TableHeightGeneralizationCfg",
    "UNSAFE_RESAMPLE_GROUPS",
    "_VALID_ASSET_SPLITS",
    "active_perturbation_axes",
    "audit_clutter_assets",
    "collect_task_asset_codes",
    "dict_to_generalization_sample",
    "extract_asset_code",
    "filter_clutter_asset_pool",
    "load_scene_generalization_config",
    "merge_generalization_samples",
    "merge_scene_generalization_overrides",
    "object_position_jitter_m",
    "parse_scene_generalization_config",
    "relativize_sample_paths",
    "resample_light",
    "resample_usd_scene_light",
    "resolve_sample_paths",
    "sample_distant_light_state",
    "sample_explicit_yaw_offset_deg",
    "sample_scene_generalization",
    "scene_generalization_config_to_dict",
    "scene_generalization_sample_to_dict",
]
