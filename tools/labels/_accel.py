"""Acceleration infrastructure: device detection, bulk HDF5 I/O, parallel execution.

Merged from _compute.py + _batch_io.py + _parallel.py to reduce file count.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import h5py
import numpy as np

from tools.labels._label_common import decode_hdf5_string, infer_camera_image_shape

if TYPE_CHECKING:
    import torch


# ===================================================================
# Section 1: Device detection and torch/numpy interop
# ===================================================================

try:
    import torch as _torch
except ImportError:
    _torch = None  # type: ignore[assignment]


class _CPUSentinel:
    """Lightweight stand-in for ``torch.device("cpu")`` when torch is missing."""

    type: str = "cpu"

    def __repr__(self) -> str:
        return "device(type='cpu')  # torch not installed"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _CPUSentinel):
            return True
        if _torch is not None and isinstance(other, _torch.device):
            return other.type == "cpu"
        return NotImplemented


def torch_available() -> bool:
    """Return True if torch is importable."""
    return _torch is not None


def get_device(preference: str = "auto") -> "torch.device":
    """Select compute device.

    Args:
        preference: ``"auto"`` (cuda > mps > cpu), ``"cuda"``, ``"mps"``, or ``"cpu"``.

    Returns:
        ``torch.device`` instance.  If torch is not installed, returns a
        lightweight sentinel whose ``.type`` attribute equals ``"cpu"`` so
        callers can branch on ``device.type == "cpu"`` without import guards.
    """
    if _torch is None:
        return _CPUSentinel()  # type: ignore[return-value]

    if preference == "auto":
        if _torch.cuda.is_available():
            return _torch.device("cuda")
        if hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
            return _torch.device("mps")
        return _torch.device("cpu")

    if preference == "cuda":
        if _torch.cuda.is_available():
            return _torch.device("cuda")
        warnings.warn("CUDA requested but not available, falling back to CPU")
        return _torch.device("cpu")

    if preference == "mps":
        if hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
            return _torch.device("mps")
        warnings.warn("MPS requested but not available, falling back to CPU")
        return _torch.device("cpu")

    return _torch.device("cpu")


def to_torch(
    arr: np.ndarray,
    device: "torch.device",
    dtype: "torch.dtype | None" = None,
) -> "torch.Tensor":
    """Convert a numpy array to a torch tensor on *device*."""
    if _torch is None:
        raise RuntimeError("torch is not installed")
    contiguous = np.ascontiguousarray(arr)
    return _torch.from_numpy(contiguous).to(device=device, dtype=dtype)


def to_numpy(t: "torch.Tensor") -> np.ndarray:
    """Convert a torch tensor to a contiguous numpy array on CPU."""
    return t.detach().cpu().numpy()


# ===================================================================
# Section 2: Bulk HDF5 episode reading
# ===================================================================

@dataclass
class CameraStaticData:
    """Per-camera data that is constant across all frames."""

    intrinsic: np.ndarray  # (3,3) float32
    camera_model: str  # "pinhole" | "fisheye"
    distortion: np.ndarray | None  # (4,) float32
    fisheye_matrix: np.ndarray | None  # (3,3) float32
    valid_mask: np.ndarray | None  # (H,W) bool
    image_shape: tuple[int, int]  # (H, W)
    clipping_range: tuple[float, float] | None
    depth_semantics: str


@dataclass
class EpisodeBatchData:
    """All data for one episode, read in bulk."""

    frame_valid: np.ndarray  # (F,) bool
    frame_errors: list[str]  # length F
    sim_steps: np.ndarray  # (F,) int64
    camera_ids: list[str]
    object_ids: list[str]

    # Camera data
    extrinsics: dict[str, np.ndarray] = field(default_factory=dict)  # cam -> (F,4,4)
    camera_static: dict[str, CameraStaticData] = field(default_factory=dict)
    depth_maps: dict[str, np.ndarray] | None = None  # cam -> (F,H,W)

    # Object data
    poses: dict[str, np.ndarray] = field(default_factory=dict)  # obj -> (F,7)
    qpos: dict[str, np.ndarray] = field(default_factory=dict)  # obj -> (F,J)
    joint_names: dict[str, list[str]] = field(default_factory=dict)

    # Box metadata
    local_bboxes: dict[str, tuple] | None = None

    # Expected forwards for extrinsic flip correction
    expected_forwards: dict[str, np.ndarray] = field(default_factory=dict)


def read_episode_batch(
    file: h5py.File,
    *,
    need_depth: bool = False,
    exclude_cameras: set[str] | None = None,
) -> EpisodeBatchData:
    """Read an entire episode from HDF5 in bulk.

    Replaces per-frame calls to ``read_camera_frames`` and
    ``read_object_states`` with single ``dataset[:]`` reads, and reads
    per-camera static metadata only once.
    """
    # ---- frame metadata ----
    frame_count = int(file["frame_valid"].shape[0]) if "frame_valid" in file else 0
    frame_valid = (
        np.asarray(file["frame_valid"][:], dtype=np.bool_)
        if "frame_valid" in file
        else np.ones(frame_count, dtype=np.bool_)
    )
    frame_errors = [
        str(item)
        for item in (
            file["frame_errors"][:] if "frame_errors" in file else np.asarray([""] * frame_count)
        )
    ]
    sim_steps = (
        np.asarray(file["time"]["sim_step"][:], dtype=np.int64)
        if "time" in file and "sim_step" in file["time"]
        else np.arange(frame_count, dtype=np.int64)
    )

    camera_ids = list(file["cameras"].keys()) if file.get("cameras") is not None else []
    object_ids = list(file["objects"].keys()) if file.get("objects") is not None else []

    batch = EpisodeBatchData(
        frame_valid=frame_valid,
        frame_errors=frame_errors,
        sim_steps=sim_steps,
        camera_ids=camera_ids,
        object_ids=object_ids,
    )

    # ---- expected forwards for extrinsic flip correction ----
    batch.expected_forwards = _load_expected_forwards(file)

    # ---- cameras ----
    exclude = exclude_cameras or set()
    depth_maps: dict[str, np.ndarray] = {}
    cameras_grp = file.get("cameras")
    if cameras_grp is not None:
        for cam_id in camera_ids:
            if cam_id in exclude:
                continue
            cam_grp = cameras_grp[cam_id]
            static = _read_camera_static(cam_grp)
            if static is None:
                continue
            batch.camera_static[cam_id] = static

            extr = np.asarray(cam_grp["extrinsic_world_from_cam"][:], dtype=np.float32)
            if cam_id in batch.expected_forwards:
                extr = _fix_extrinsic_flip_batch(extr, batch.expected_forwards[cam_id])
            batch.extrinsics[cam_id] = extr

            if need_depth:
                depth_key = _depth_key(cam_grp)
                if depth_key is not None:
                    depth_maps[cam_id] = np.asarray(cam_grp[depth_key][:], dtype=np.float32)

    batch.depth_maps = depth_maps if need_depth else None

    # ---- objects ----
    objects_grp = file.get("objects")
    if objects_grp is not None:
        for obj_id in object_ids:
            obj_grp = objects_grp[obj_id]
            batch.poses[obj_id] = np.asarray(obj_grp["pose_world"][:], dtype=np.float32)
            if "qpos" in obj_grp:
                batch.qpos[obj_id] = np.asarray(obj_grp["qpos"][:], dtype=np.float32)
            if "joint_names" in obj_grp:
                batch.joint_names[obj_id] = [
                    n.decode() if isinstance(n, bytes) else str(n)
                    for n in obj_grp["joint_names"][:]
                ]

    # ---- local bboxes ----
    meta = file.get("meta")
    if meta is not None and "local_bboxes" in meta:
        raw = json.loads(decode_hdf5_string(meta["local_bboxes"][()]))
        batch.local_bboxes = {obj_id: (tuple(b[0]), tuple(b[1])) for obj_id, b in raw.items()}

    return batch


def _read_camera_static(cam_grp: h5py.Group) -> CameraStaticData | None:
    """Read per-camera static metadata (called once per camera per episode)."""
    image_shape = infer_camera_image_shape(cam_grp)
    if image_shape is None:
        return None

    intrinsic = np.asarray(cam_grp["intrinsic"][()], dtype=np.float32)

    camera_model = "pinhole"
    if "camera_model" in cam_grp:
        camera_model = decode_hdf5_string(cam_grp["camera_model"][()])
    if camera_model in ("opencv_fisheye", "isaacsim_fisheye", "fisheye"):
        camera_model = "fisheye"

    distortion = None
    fisheye_matrix = None
    valid_mask = None
    if camera_model == "fisheye":
        if "distortion_coefficients" in cam_grp:
            distortion = np.asarray(cam_grp["distortion_coefficients"][()], dtype=np.float32)
        if "fisheye_camera_matrix" in cam_grp:
            fisheye_matrix = np.asarray(cam_grp["fisheye_camera_matrix"][()], dtype=np.float32)
        if "fisheye_valid_mask" in cam_grp:
            valid_mask = np.asarray(cam_grp["fisheye_valid_mask"][()], dtype=np.bool_)

    clipping_range = None
    if "clipping_range" in cam_grp:
        clipping_range = tuple(np.asarray(cam_grp["clipping_range"][()], dtype=np.float32).tolist())
    elif "render_clipping_range" in cam_grp:
        clipping_range = tuple(
            np.asarray(cam_grp["render_clipping_range"][()], dtype=np.float32).tolist()
        )

    depth_semantics = "distance_to_image_plane"
    if "depth_semantics" in cam_grp:
        ds = cam_grp["depth_semantics"]
        val = ds[()] if ds.shape == () else ds[0]
        decoded = decode_hdf5_string(val).strip()
        if decoded:
            depth_semantics = decoded

    return CameraStaticData(
        intrinsic=intrinsic,
        camera_model=camera_model,
        distortion=distortion,
        fisheye_matrix=fisheye_matrix,
        valid_mask=valid_mask,
        image_shape=image_shape,
        clipping_range=clipping_range,
        depth_semantics=depth_semantics,
    )


def _depth_key(cam_grp: h5py.Group) -> str | None:
    if "depth_m" in cam_grp:
        return "depth_m"
    if "depth" in cam_grp:
        return "depth"
    return None


def _load_expected_forwards(file: h5py.File) -> dict[str, np.ndarray]:
    meta = file.get("meta")
    if meta is None or "camera_definitions" not in meta:
        return {}
    try:
        raw = decode_hdf5_string(meta["camera_definitions"][()])
        defs = json.loads(raw)
    except Exception:
        return {}
    out: dict[str, np.ndarray] = {}
    for d in defs:
        if d.get("mount_type") != "world":
            continue
        pos = d.get("position")
        tgt = d.get("target")
        if pos is None or tgt is None:
            continue
        fwd = np.asarray(tgt, dtype=np.float64) - np.asarray(pos, dtype=np.float64)
        norm = np.linalg.norm(fwd)
        if norm < 1e-9:
            continue
        out[str(d["camera_id"])] = (fwd / norm).astype(np.float32)
    return out


def _fix_extrinsic_flip_batch(
    extrinsics: np.ndarray,
    expected_forward: np.ndarray,
) -> np.ndarray:
    """Vectorized 180-degree gimbal-lock flip correction for (F,4,4) arrays."""
    actual_forwards = extrinsics[:, :3, 0]
    dots = np.einsum("fi,i->f", actual_forwards, expected_forward)
    flip_mask = dots < -0.5
    if not np.any(flip_mask):
        return extrinsics
    result = extrinsics.copy()
    result[flip_mask, :3, 0] *= -1
    result[flip_mask, :3, 2] *= -1
    return result


# ===================================================================
# Section 3: Multi-episode parallel processing
# ===================================================================

def parallel_process(
    fn: Callable[[str], bool],
    paths: list[str],
    *,
    workers: int = 0,
    label: str = "Processing",
) -> tuple[int, int]:
    """Run *fn(path)* for each HDF5 path, optionally in parallel.

    Args:
        fn: Callable that takes an HDF5 file path and returns ``True`` if
            labels were written successfully.
        paths: List of HDF5 file paths to process.
        workers: Number of worker processes.
            ``0`` = auto, ``1`` = sequential (debug), ``>1`` = parallel.
        label: Human-readable label for progress messages.

    Returns:
        ``(success_count, error_count)``
    """
    total = len(paths)
    if total == 0:
        return 0, 0

    if workers == 0:
        workers = max(1, min((os.cpu_count() or 2) // 2, 8))

    if workers == 1:
        return _run_sequential(fn, paths, label)
    return _run_parallel(fn, paths, workers, label)


def _run_sequential(
    fn: Callable[[str], bool],
    paths: list[str],
    label: str,
) -> tuple[int, int]:
    n_ok = 0
    n_err = 0
    total = len(paths)
    for i, path in enumerate(paths, 1):
        try:
            ok = fn(path)
            if ok:
                n_ok += 1
            _progress(label, i, total, path)
        except Exception as exc:
            n_err += 1
            print(f"[ERROR] {path}: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
    return n_ok, n_err


def _run_parallel(
    fn: Callable[[str], bool],
    paths: list[str],
    workers: int,
    label: str,
) -> tuple[int, int]:
    n_ok = 0
    n_err = 0
    total = len(paths)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_to_path = {pool.submit(fn, p): p for p in paths}
        for i, future in enumerate(as_completed(future_to_path), 1):
            path = future_to_path[future]
            try:
                ok = future.result()
                if ok:
                    n_ok += 1
                _progress(label, i, total, path)
            except Exception as exc:
                n_err += 1
                print(f"[ERROR] {path}: {exc}", file=sys.stderr)

    return n_ok, n_err


def _progress(label: str, current: int, total: int, path: str) -> None:
    print(f"[{label}] {current}/{total} {os.path.basename(path)}")
