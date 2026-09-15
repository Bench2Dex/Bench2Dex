"""Raw TacMap depth I/O shared by GR00T Cross training and inference."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


RAW_DEPTH_KEY = "distance_along_normal_m"
DEPTH_UNIT = "m"
DEFAULT_D_MAX_M = 0.015


@dataclass(frozen=True)
class TactileSchema:
    """Checkpoint-owned ordered layout and raw-depth normalization contract."""

    site_names: tuple[str, ...]
    image_shape: tuple[int, int]
    depth_key: str = RAW_DEPTH_KEY
    depth_unit: str = DEPTH_UNIT
    d_max_m: float = DEFAULT_D_MAX_M
    resolution_step: int = 1


def _decode_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_string(value.item())
    return str(value)


def _read_d_max_m(file: h5py.File, *, source: str) -> float:
    dataset = file.get("robot/tactile/meta/max_distance_m")
    value = DEFAULT_D_MAX_M if dataset is None else float(np.asarray(dataset[()]).item())
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{source}: robot/tactile/meta/max_distance_m must be finite and positive")
    return value


def _read_resolution_step(file: h5py.File, *, source: str) -> int:
    dataset = file.get("robot/tactile/meta/resolution_step")
    value = 1 if dataset is None else int(np.asarray(dataset[()]).item())
    if value <= 0:
        raise ValueError(f"{source}: robot/tactile/meta/resolution_step must be positive")
    return value


def read_tactile_schema(file: h5py.File, *, source: str = "HDF5") -> TactileSchema:
    """Read ordered float raw-depth TacMap metadata from one episode."""

    names_ds = file.get("robot/tactile/meta/site_names")
    if not isinstance(names_ds, h5py.Dataset):
        raise ValueError(f"{source}: missing robot/tactile/meta/site_names")
    site_names = tuple(_decode_string(value) for value in names_ds[:])
    if not site_names or len(set(site_names)) != len(site_names):
        raise ValueError(f"{source}: tactile site_names must be non-empty and unique")

    image_shape: tuple[int, int] | None = None
    for site_name in site_names:
        dataset = file.get(f"robot/tactile/{RAW_DEPTH_KEY}/{site_name}")
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"{source}: missing raw tactile depth for site {site_name!r}")
        if dataset.ndim != 3:
            raise ValueError(f"{source}: raw tactile site {site_name!r} must have shape [T,H,W]")
        if not np.issubdtype(dataset.dtype, np.floating):
            raise ValueError(f"{source}: raw tactile site {site_name!r} must be floating point, got {dataset.dtype}")
        current_shape = (int(dataset.shape[1]), int(dataset.shape[2]))
        if image_shape is None:
            image_shape = current_shape
        elif current_shape != image_shape:
            raise ValueError(f"{source}: raw tactile image shapes do not match")
    assert image_shape is not None
    return TactileSchema(
        site_names=site_names,
        image_shape=image_shape,
        depth_key=RAW_DEPTH_KEY,
        depth_unit=DEPTH_UNIT,
        d_max_m=_read_d_max_m(file, source=source),
        resolution_step=_read_resolution_step(file, source=source),
    )


def validate_tactile_schema(expected: TactileSchema, actual: TactileSchema, *, source: str = "HDF5") -> None:
    if expected.site_names != actual.site_names:
        raise ValueError(f"{source}: tactile site order does not match checkpoint")
    if expected.image_shape != actual.image_shape:
        raise ValueError(f"{source}: tactile image shape does not match checkpoint")
    if expected.depth_key != actual.depth_key or expected.depth_unit != actual.depth_unit:
        raise ValueError(f"{source}: tactile raw-depth contract does not match checkpoint")
    if not np.isclose(expected.d_max_m, actual.d_max_m, rtol=0.0, atol=1e-12):
        raise ValueError(f"{source}: tactile d_max does not match checkpoint")
    if expected.resolution_step != actual.resolution_step:
        raise ValueError(f"{source}: tactile resolution_step does not match checkpoint")


def read_tactile_checkpoint_schema(model_path: str | Path) -> TactileSchema:
    """Read the tactile sensor contract from a saved GR00T ``config.json``."""

    checkpoint = Path(model_path).expanduser()
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"tactile checkpoint config not found: {config_path}")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid tactile checkpoint config: {config_path}") from exc
    action_cfg = payload.get("action_head_cfg")
    if not isinstance(action_cfg, dict) or not bool(action_cfg.get("use_tactile", False)):
        raise ValueError(f"checkpoint does not contain a tactile action-head contract: {config_path}")

    site_names = tuple(str(name) for name in action_cfg.get("tactile_site_names", ()))
    if not site_names or len(set(site_names)) != len(site_names):
        raise ValueError("checkpoint tactile_site_names must be non-empty and unique")
    image_shape = (
        int(action_cfg.get("tactile_native_height", 0)),
        int(action_cfg.get("tactile_native_width", 0)),
    )
    if min(image_shape) <= 0:
        raise ValueError("checkpoint tactile native image dimensions must be positive")
    d_max_m = float(action_cfg.get("tactile_d_max_m", 0.0))
    if not np.isfinite(d_max_m) or d_max_m <= 0.0:
        raise ValueError("checkpoint tactile_d_max_m must be finite and positive")
    resolution_step = int(action_cfg.get("tactile_resolution_step", 0))
    if resolution_step <= 0:
        raise ValueError("checkpoint tactile_resolution_step must be positive")
    return TactileSchema(
        site_names=site_names,
        image_shape=image_shape,
        depth_key=str(action_cfg.get("tactile_depth_key", RAW_DEPTH_KEY)),
        depth_unit=str(action_cfg.get("tactile_depth_unit", DEPTH_UNIT)),
        d_max_m=d_max_m,
        resolution_step=resolution_step,
    )


def _resize_nearest(image: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    out_h, out_w = (int(output_shape[0]), int(output_shape[1]))
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"tactile output_shape must be positive, got {output_shape}")
    if image.shape == (out_h, out_w):
        return image
    row_idx = np.rint(np.linspace(0, image.shape[0] - 1, out_h)).astype(np.intp)
    col_idx = np.rint(np.linspace(0, image.shape[1] - 1, out_w)).astype(np.intp)
    return image[np.ix_(row_idx, col_idx)]


def _normalize_depth(depth: np.ndarray, d_max_m: float) -> np.ndarray:
    if not np.isfinite(depth).all():
        raise ValueError("raw tactile depth must be finite")
    return np.clip(depth, 0.0, d_max_m).astype(np.float32, copy=False) / np.float32(d_max_m)


def read_tactile_frame(
    file: h5py.File,
    frame_index: int,
    schema: TactileSchema,
    *,
    output_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Read one normalized raw-depth frame as ``[N,1,H,W]`` float32."""

    maps: list[np.ndarray] = []
    for site_name in schema.site_names:
        dataset = file[f"robot/tactile/{schema.depth_key}/{site_name}"]
        if not 0 <= int(frame_index) < len(dataset):
            raise IndexError(f"tactile frame index {frame_index} is out of range for site {site_name!r}")
        image = np.asarray(dataset[int(frame_index)], dtype=np.float32)
        if output_shape is not None:
            image = _resize_nearest(image, output_shape)
        maps.append(_normalize_depth(image, schema.d_max_m)[None, ...])
    return np.stack(maps, axis=0)


def stack_tactile_mapping(
    mapping: dict[str, Any],
    schema: TactileSchema,
    *,
    output_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Stack online raw float depth maps in checkpoint site order."""

    missing = [name for name in schema.site_names if name not in mapping]
    extra = [name for name in mapping if name not in schema.site_names]
    if missing or extra:
        raise ValueError(f"runtime tactile sites do not match checkpoint: missing={missing}, extra={extra}")
    maps: list[np.ndarray] = []
    for site_name in schema.site_names:
        image = np.asarray(mapping[site_name])
        if image.ndim != 2 or not np.issubdtype(image.dtype, np.floating):
            raise ValueError(f"runtime tactile site {site_name!r} must be a 2-D floating depth map")
        if tuple(image.shape) != schema.image_shape:
            raise ValueError(f"runtime tactile site {site_name!r} shape does not match checkpoint")
        if output_shape is not None:
            image = _resize_nearest(image, output_shape)
        maps.append(_normalize_depth(image.astype(np.float32, copy=False), schema.d_max_m)[None, ...])
    return np.stack(maps, axis=0)
