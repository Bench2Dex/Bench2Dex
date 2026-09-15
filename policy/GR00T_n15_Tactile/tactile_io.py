"""Pure HDF5/NumPy helpers for GR00T tactile observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np


@dataclass(frozen=True)
class TactileSchema:
    """Fixed tactile layout used by one task/embodiment checkpoint."""

    site_names: tuple[str, ...]
    image_shape: tuple[int, int]


def _decode_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_string(value.item())
    return str(value)


def read_tactile_schema(file: h5py.File, *, source: str = "HDF5") -> TactileSchema:
    """Validate and return the ordered TacMap site schema."""

    names_ds = file.get("robot/tactile/meta/site_names")
    if not isinstance(names_ds, h5py.Dataset):
        raise ValueError(f"{source}: missing robot/tactile/meta/site_names")
    site_names = tuple(_decode_string(value) for value in names_ds[:])
    if not site_names:
        raise ValueError(f"{source}: tactile site_names is empty")
    if len(set(site_names)) != len(site_names):
        raise ValueError(f"{source}: tactile site_names contains duplicates")

    image_shape: tuple[int, int] | None = None
    for site_name in site_names:
        dataset = file.get(f"robot/tactile/tacmap/{site_name}")
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"{source}: missing tactile map for site {site_name!r}")
        if dataset.ndim != 3:
            raise ValueError(f"{source}: tactile site {site_name!r} must have shape [T,H,W], got {dataset.shape}")
        if dataset.dtype != np.dtype(np.uint8):
            raise ValueError(f"{source}: tactile site {site_name!r} must be uint8, got {dataset.dtype}")
        current_shape = (int(dataset.shape[1]), int(dataset.shape[2]))
        if image_shape is None:
            image_shape = current_shape
        elif current_shape != image_shape:
            raise ValueError(
                f"{source}: tactile site {site_name!r} has shape {current_shape}, expected {image_shape}"
            )
    assert image_shape is not None
    return TactileSchema(site_names=site_names, image_shape=image_shape)


def validate_tactile_schema(expected: TactileSchema, actual: TactileSchema, *, source: str = "HDF5") -> None:
    if actual.site_names != expected.site_names:
        raise ValueError(
            f"{source}: tactile site order {actual.site_names!r} does not match checkpoint/dataset order "
            f"{expected.site_names!r}"
        )
    if actual.image_shape != expected.image_shape:
        raise ValueError(
            f"{source}: tactile image shape {actual.image_shape} does not match expected {expected.image_shape}"
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


def read_tactile_frame(
    file: h5py.File,
    frame_index: int,
    schema: TactileSchema,
    *,
    output_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Read one normalized tactile frame as ``[N,1,H,W]`` float32."""

    maps: list[np.ndarray] = []
    for site_name in schema.site_names:
        dataset = file[f"robot/tactile/tacmap/{site_name}"]
        if not 0 <= int(frame_index) < len(dataset):
            raise IndexError(f"tactile frame index {frame_index} is out of range for site {site_name!r}")
        image = np.asarray(dataset[int(frame_index)], dtype=np.uint8)
        if output_shape is not None:
            image = _resize_nearest(image, output_shape)
        maps.append(image[None, ...].astype(np.float32) / 255.0)
    return np.stack(maps, axis=0)


def stack_tactile_mapping(
    mapping: dict[str, Any],
    schema: TactileSchema,
    *,
    output_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Stack an online ``site_name -> uint8 map`` payload in checkpoint order."""

    missing = [name for name in schema.site_names if name not in mapping]
    extra = [name for name in mapping if name not in schema.site_names]
    if missing or extra:
        raise ValueError(f"runtime tactile sites do not match checkpoint: missing={missing}, extra={extra}")
    maps: list[np.ndarray] = []
    for site_name in schema.site_names:
        image = np.asarray(mapping[site_name])
        if image.ndim != 2:
            raise ValueError(f"runtime tactile site {site_name!r} must be 2-D, got {image.shape}")
        if image.dtype != np.dtype(np.uint8):
            raise ValueError(f"runtime tactile site {site_name!r} must be uint8, got {image.dtype}")
        if tuple(image.shape) != schema.image_shape:
            raise ValueError(
                f"runtime tactile site {site_name!r} has shape {tuple(image.shape)}, "
                f"expected {schema.image_shape}"
            )
        if output_shape is not None:
            image = _resize_nearest(image, output_shape)
        maps.append(image[None, ...].astype(np.float32) / 255.0)
    return np.stack(maps, axis=0)
