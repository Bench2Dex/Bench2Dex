from __future__ import annotations

import json
import os
import stat

import h5py
import numpy as np


_CAMERA_RGB_JPEG_QUALITY = 50
_GZIP_OPTS = {"compression": "gzip", "compression_opts": 4}


def ensure_file_writable(path: str) -> None:
    """Clear a copied file's read-only bit before appending HDF5 data."""
    mode = os.stat(path).st_mode
    os.chmod(path, mode | stat.S_IWRITE | stat.S_IWUSR)


def _encode_camera_rgb_jpeg(image: np.ndarray) -> np.ndarray:
    """JPEG-encode one RGB camera frame as variable-length uint8 bytes."""
    import cv2

    rgb = np.asarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3+), got {rgb.shape}.")
    if rgb.shape[-1] > 3:
        rgb = rgb[..., :3]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, _CAMERA_RGB_JPEG_QUALITY])
    if not ok:
        raise RuntimeError("Failed to JPEG-encode RGB camera frame.")
    return np.frombuffer(buf, dtype=np.uint8).copy()


def should_enable_tactile(*, cli_enable_tactile: bool, collect_tactile_enabled: bool) -> bool:
    return bool(cli_enable_tactile or collect_tactile_enabled)


def resolve_replay_sim_device(*, requested_device: str | None, enable_tactile: bool) -> str:
    requested = (requested_device or "").strip()
    if enable_tactile:
        if not requested:
            return "cuda:0"
        if requested.lower().startswith("cpu"):
            raise ValueError(
                "TacMap tactile replay requires GPU simulation; rerun with --device cuda:0 or disable tactile."
            )
        return requested
    return requested or "cpu"


def _read_positive_float(group: h5py.Group, key: str) -> float | None:
    if key not in group:
        return None
    value = group[key][()]
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode()
    try:
        scalar = float(np.asarray(value).item())
    except (TypeError, ValueError):
        return None
    return scalar if scalar > 0.0 else None


def resolve_replay_physics_dt(meta_group: h5py.Group, default_dt: float = 0.01) -> float:
    """Resolve the physics dt that replay should use for an HDF5 episode."""
    recorded_dt = _read_positive_float(meta_group, "physics_dt")
    if recorded_dt is not None:
        return recorded_dt

    step_stride = _read_positive_float(meta_group, "step_stride")
    if step_stride is not None:
        effective_fps = _read_positive_float(meta_group, "effective_fps")
        if effective_fps is not None:
            return 1.0 / (effective_fps * step_stride)

        fps = _read_positive_float(meta_group, "fps")
        if fps is not None:
            return 1.0 / (fps * step_stride)

    return float(default_dt)


def _write_meta_value(group: h5py.Group, key: str, value) -> None:
    str_dtype = h5py.string_dtype(encoding="utf-8")
    if isinstance(value, dict):
        child = group.create_group(str(key))
        for child_key, child_value in value.items():
            _write_meta_value(child, str(child_key), child_value)
        return
    if isinstance(value, str):
        group.create_dataset(key, data=np.asarray(value, dtype=str_dtype))
        return
    if isinstance(value, bool):
        group.create_dataset(key, data=np.asarray(value, dtype=np.bool_))
        return
    if isinstance(value, int):
        group.create_dataset(key, data=np.asarray(value, dtype=np.int64))
        return
    if isinstance(value, float):
        group.create_dataset(key, data=np.asarray(value, dtype=np.float64))
        return
    if isinstance(value, (list, tuple)) and value and all(isinstance(item, str) for item in value):
        group.create_dataset(key, data=np.asarray(list(value), dtype=str_dtype))
        return
    try:
        group.create_dataset(key, data=np.asarray(value))
    except TypeError:
        group.create_dataset(key, data=np.asarray(json.dumps(value), dtype=str_dtype))


def _decode_json_scalar(value) -> str:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.bytes_):
        return value.tobytes().decode()
    return str(value)


def _upsert_scalar_dataset(group: h5py.Group, key: str, value) -> None:
    if key in group:
        del group[key]
    _write_meta_value(group, key, value)


def _clear_buffer_if_requested(buffer, clear_after_write: bool) -> None:
    if clear_after_write and hasattr(buffer, "clear"):
        buffer.clear()


def _replace_dataset(group: h5py.Group, key: str, data, **kwargs) -> None:
    if key in group:
        del group[key]
    group.create_dataset(key, data=data, **kwargs)


def _write_sequence_dataset(
    group: h5py.Group,
    key: str,
    frames,
    *,
    dtype,
    chunks: bool = True,
) -> None:
    if key in group:
        del group[key]
    first = np.asarray(frames[0], dtype=dtype)
    dataset_kwargs = {}
    if chunks:
        dataset_kwargs["chunks"] = (1, *first.shape)
    ds = group.create_dataset(
        key,
        shape=(len(frames), *first.shape),
        dtype=dtype,
        **dataset_kwargs,
    )
    ds[0] = first
    for i, frame in enumerate(frames[1:], start=1):
        arr = np.asarray(frame, dtype=dtype)
        if arr.shape != first.shape:
            raise ValueError(
                f"Inconsistent shape for cameras dataset '{group.name}/{key}' "
                f"at frame {i}: expected {first.shape}, got {arr.shape}."
            )
        ds[i] = arr


def _mark_camera_modalities(f: h5py.File, *, has_rgb: bool, has_depth: bool) -> None:
    meta_grp = f.require_group("meta")
    if has_rgb:
        _upsert_scalar_dataset(meta_grp, "has_rgb", True)
        _upsert_scalar_dataset(meta_grp, "rgb_encoding", "jpeg")
        _upsert_scalar_dataset(meta_grp, "rgb_jpeg_quality", _CAMERA_RGB_JPEG_QUALITY)
    if has_depth:
        _upsert_scalar_dataset(meta_grp, "has_depth", True)

    if not (has_rgb or has_depth):
        return

    str_dtype = h5py.string_dtype(encoding="utf-8")
    modalities_path = "meta/modalities"
    if modalities_path in f:
        try:
            modalities = json.loads(_decode_json_scalar(f[modalities_path][()]))
        except Exception:
            return
    else:
        modalities = {}

    if isinstance(modalities, dict):
        if has_rgb:
            modalities["rgb"] = True
        if has_depth:
            modalities["depth"] = True
    elif isinstance(modalities, list):
        if has_rgb and "rgb" not in modalities:
            modalities.append("rgb")
        if has_depth and "depth" not in modalities:
            modalities.append("depth")
    else:
        return

    if modalities_path in f:
        del f[modalities_path]
    meta_grp.create_dataset("modalities", data=np.asarray(json.dumps(modalities), dtype=str_dtype))


def inject_camera_buffers_into_hdf5(
    output_path: str,
    cam_rgb_data: dict[str, object],
    cam_depth_data: dict[str, object],
    cam_intrinsics: dict[str, np.ndarray],
    cam_extrinsics: dict[str, object],
    cam_metadata: dict[str, dict[str, object]] | None = None,
    *,
    clear_after_write: bool = False,
) -> dict[str, dict[str, int]]:
    """Write buffered camera data to HDF5 without stacking full frame buffers.

    RGB frames are JPEG-encoded one by one into a variable-length uint8 dataset.
    Depth and extrinsics are also written frame by frame so callers do not need
    an additional contiguous copy of the full episode in memory.
    """
    counts: dict[str, dict[str, int]] = {}
    has_rgb = False
    has_depth = False

    with h5py.File(output_path, "a") as f:
        cameras_grp = f.require_group("cameras")
        all_cam_ids = (
            set(cam_rgb_data.keys())
            | set(cam_depth_data.keys())
            | set(cam_intrinsics.keys())
            | set(cam_extrinsics.keys())
            | set((cam_metadata or {}).keys())
        )
        for cam_id in sorted(all_cam_ids):
            cam_grp = cameras_grp.require_group(cam_id)
            counts[cam_id] = {"rgb": 0, "depth": 0, "extrinsic": 0}

            rgb_frames = cam_rgb_data.get(cam_id)
            if rgb_frames is not None and len(rgb_frames) > 0:
                if "rgb" in cam_grp:
                    del cam_grp["rgb"]
                vlen_dt = h5py.vlen_dtype(np.dtype("uint8"))
                rgb_ds = cam_grp.create_dataset("rgb", shape=(len(rgb_frames),), dtype=vlen_dt)
                for i, frame in enumerate(rgb_frames):
                    rgb_ds[i] = _encode_camera_rgb_jpeg(frame)
                counts[cam_id]["rgb"] = len(rgb_frames)
                has_rgb = True
                _clear_buffer_if_requested(rgb_frames, clear_after_write)

            depth_frames = cam_depth_data.get(cam_id)
            if depth_frames is not None and len(depth_frames) > 0:
                _write_sequence_dataset(cam_grp, "depth", depth_frames, dtype=np.float32)
                counts[cam_id]["depth"] = len(depth_frames)
                has_depth = True
                _clear_buffer_if_requested(depth_frames, clear_after_write)

            if cam_id in cam_intrinsics:
                if "intrinsic" in cam_grp:
                    del cam_grp["intrinsic"]
                cam_grp.create_dataset("intrinsic", data=np.asarray(cam_intrinsics[cam_id], dtype=np.float32))

            extrinsic_frames = cam_extrinsics.get(cam_id)
            if extrinsic_frames is not None and len(extrinsic_frames) > 0:
                _write_sequence_dataset(
                    cam_grp,
                    "extrinsic_world_from_cam",
                    extrinsic_frames,
                    dtype=np.float32,
                    chunks=False,
                )
                counts[cam_id]["extrinsic"] = len(extrinsic_frames)
                _clear_buffer_if_requested(extrinsic_frames, clear_after_write)

            metadata = (cam_metadata or {}).get(cam_id) or {}
            if metadata:
                str_dtype = h5py.string_dtype(encoding="utf-8")
                camera_model = str(metadata.get("camera_model") or "pinhole")
                _replace_dataset(cam_grp, "camera_model", np.asarray(camera_model, dtype=str_dtype))

                clipping_range = metadata.get("clipping_range")
                if clipping_range is not None:
                    _replace_dataset(
                        cam_grp,
                        "clipping_range",
                        np.asarray(clipping_range, dtype=np.float32),
                    )

                if camera_model == "fisheye":
                    fisheye_camera_matrix = metadata.get("fisheye_camera_matrix")
                    if fisheye_camera_matrix is not None:
                        _replace_dataset(
                            cam_grp,
                            "fisheye_camera_matrix",
                            np.asarray(fisheye_camera_matrix, dtype=np.float32),
                        )
                    distortion_coefficients = metadata.get("distortion_coefficients")
                    if distortion_coefficients is not None:
                        _replace_dataset(
                            cam_grp,
                            "distortion_coefficients",
                            np.asarray(distortion_coefficients, dtype=np.float32),
                        )
                    fisheye_valid_mask = metadata.get("fisheye_valid_mask")
                    if fisheye_valid_mask is not None:
                        _replace_dataset(
                            cam_grp,
                            "fisheye_valid_mask",
                            np.asarray(fisheye_valid_mask, dtype=np.bool_),
                            **_GZIP_OPTS,
                        )
                    _replace_dataset(
                        cam_grp,
                        "distortion_model",
                        np.asarray("opencv_fisheye_kannala_brandt", dtype=str_dtype),
                    )
                    _replace_dataset(
                        cam_grp,
                        "raw_image_semantics",
                        np.asarray("masked_fisheye_rectangular", dtype=str_dtype),
                    )

        _mark_camera_modalities(f, has_rgb=has_rgb, has_depth=has_depth)

    return counts


def _mark_tactile_modality(f: h5py.File) -> None:
    meta_grp = f.require_group("meta")
    _upsert_scalar_dataset(meta_grp, "has_tactile", True)

    str_dtype = h5py.string_dtype(encoding="utf-8")
    modalities_path = "meta/modalities"
    if modalities_path not in f:
        meta_grp.create_dataset("modalities", data=np.asarray(json.dumps({"tactile": True}), dtype=str_dtype))
        return

    try:
        mods = json.loads(_decode_json_scalar(f[modalities_path][()]))
    except Exception:
        return

    if isinstance(mods, dict):
        mods["tactile"] = True
    elif isinstance(mods, list):
        if "tactile" not in mods:
            mods.append("tactile")
    else:
        return

    del f[modalities_path]
    meta_grp.create_dataset("modalities", data=np.asarray(json.dumps(mods), dtype=str_dtype))


class IncrementalTacMapWriter:
    """Write TacMap tactile frames to HDF5 one at a time."""

    def __init__(self, output_path: str, frame_count: int) -> None:
        self._output_path = output_path
        self._frame_count = frame_count
        self._f: h5py.File | None = None
        self._tactile_grp: h5py.Group | None = None
        self._initialized = False
        self._frames_written = 0
        self._datasets: dict[str, h5py.Dataset] = {}
        self._raw_depth_datasets: dict[str, h5py.Dataset] = {}
        self._contact_mask_datasets: dict[str, h5py.Dataset] = {}

    def open(self) -> None:
        self._f = h5py.File(self._output_path, "a")
        robot_grp = self._f.require_group("robot")
        if "tactile" in robot_grp:
            del robot_grp["tactile"]
        self._tactile_grp = robot_grp.create_group("tactile")

    def write_frame(self, frame_idx: int, payload: dict[str, object]) -> None:
        if self._f is None or self._tactile_grp is None:
            raise RuntimeError("IncrementalTacMapWriter is not open.")
        if not self._initialized:
            self._init_datasets(payload)
            self._initialized = True
        self._write_data(frame_idx, payload)
        self._frames_written += 1

    def _init_datasets(self, first: dict[str, object]) -> None:
        assert self._tactile_grp is not None
        n = self._frame_count
        meta = dict(first.get("meta", {}))
        meta.setdefault("sensor_type", "tacmap")

        first_tacmap = dict(first.get("tacmap", {}))
        site_names = list(meta.get("site_names") or first_tacmap.keys())
        meta["site_names"] = [str(site_name) for site_name in site_names]

        meta_grp = self._tactile_grp.create_group("meta")
        for key, value in meta.items():
            _write_meta_value(meta_grp, str(key), value)

        tacmap_grp = self._tactile_grp.create_group("tacmap")
        raw_depth_payload = dict(first.get("distance_along_normal_m", {}))
        raw_depth_grp = self._tactile_grp.create_group("distance_along_normal_m") if raw_depth_payload else None
        contact_mask_payload = dict(first.get("contact_mask", {}))
        contact_mask_grp = self._tactile_grp.create_group("contact_mask") if contact_mask_payload else None
        for site_name in site_names:
            site_key = str(site_name)
            if site_key not in first_tacmap:
                raise ValueError(f"TacMap first frame is missing site: {site_key}")
            arr = np.asarray(first_tacmap[site_key], dtype=np.uint8)
            if arr.ndim != 2:
                raise ValueError(f"TacMap site '{site_key}' must be a 2-D uint8 image, got shape {arr.shape}.")
            self._datasets[site_key] = tacmap_grp.create_dataset(
                site_key,
                shape=(n, *arr.shape),
                dtype=np.uint8,
                chunks=(1, *arr.shape),
                fillvalue=0,
                **_GZIP_OPTS,
            )
            if raw_depth_grp is not None:
                if site_key not in raw_depth_payload:
                    raise ValueError(f"TacMap first frame is missing raw depth site: {site_key}")
                raw_arr = np.asarray(raw_depth_payload[site_key], dtype=np.float32)
                if raw_arr.shape != arr.shape:
                    raise ValueError(
                        f"TacMap raw depth site '{site_key}' shape {raw_arr.shape} does not match tacmap shape {arr.shape}."
                    )
                self._raw_depth_datasets[site_key] = raw_depth_grp.create_dataset(
                    site_key,
                    shape=(n, *raw_arr.shape),
                    dtype=np.float32,
                    chunks=(1, *raw_arr.shape),
                    fillvalue=0.0,
                    **_GZIP_OPTS,
                )
            if contact_mask_grp is not None:
                if site_key not in contact_mask_payload:
                    raise ValueError(f"TacMap first frame is missing contact mask site: {site_key}")
                mask_arr = np.asarray(contact_mask_payload[site_key], dtype=np.bool_)
                if mask_arr.shape != arr.shape:
                    raise ValueError(
                        f"TacMap contact mask site '{site_key}' shape {mask_arr.shape} does not match tacmap shape {arr.shape}."
                    )
                self._contact_mask_datasets[site_key] = contact_mask_grp.create_dataset(
                    site_key,
                    shape=(n, *mask_arr.shape),
                    dtype=np.bool_,
                    chunks=(1, *mask_arr.shape),
                    fillvalue=False,
                    **_GZIP_OPTS,
                )

    def _write_data(self, idx: int, payload: dict[str, object]) -> None:
        tacmap_payload = dict(payload.get("tacmap", {}))
        for site_name, ds in self._datasets.items():
            site_payload = tacmap_payload.get(site_name)
            if site_payload is None:
                continue
            arr = np.asarray(site_payload, dtype=np.uint8)
            if arr.shape != ds.shape[1:]:
                raise ValueError(
                    f"TacMap site '{site_name}' shape changed from {ds.shape[1:]} to {arr.shape}."
                )
            ds[idx] = arr
        raw_depth_payload = dict(payload.get("distance_along_normal_m", {}))
        for site_name, ds in self._raw_depth_datasets.items():
            site_payload = raw_depth_payload.get(site_name)
            if site_payload is None:
                continue
            arr = np.asarray(site_payload, dtype=np.float32)
            if arr.shape != ds.shape[1:]:
                raise ValueError(
                    f"TacMap raw depth site '{site_name}' shape changed from {ds.shape[1:]} to {arr.shape}."
                )
            ds[idx] = arr
        contact_mask_payload = dict(payload.get("contact_mask", {}))
        for site_name, ds in self._contact_mask_datasets.items():
            site_payload = contact_mask_payload.get(site_name)
            if site_payload is None:
                continue
            arr = np.asarray(site_payload, dtype=np.bool_)
            if arr.shape != ds.shape[1:]:
                raise ValueError(
                    f"TacMap contact mask site '{site_name}' shape changed from {ds.shape[1:]} to {arr.shape}."
                )
            ds[idx] = arr

    def close(self) -> None:
        if self._f is None:
            return
        if self._frames_written != self._frame_count:
            print(
                f"[WARN] TacMap frame count mismatch: expected {self._frame_count}, "
                f"got {self._frames_written}. Data may be misaligned with trajectory."
            )
        _mark_tactile_modality(self._f)
        self._f.close()
        self._f = None
        self._tactile_grp = None
