"""
Convert dex2bench raw HDF5 replay episodes to LeRobot Dataset v3.0 format.

Python dependencies are numpy, h5py, pandas, tqdm, and opencv-python. Video
encoding requires a system ffmpeg with libx264 support.
Produces a policy-neutral LeRobot v3 directory. Downstream training stacks may
still need their own dataset config or camera-key mapping.

First principles:
  1. Auto-detect schema from data — no hardcoded robot configs
  2. Vectorized validity filtering — every check is a numpy mask, not a Python loop
  3. Explicit camera selection/renaming — never silently drop or copy data
  4. Safe-by-default — refuse to overwrite unless --overwrite is passed
  5. Complete output — info.json + stats.json + optional adapter metadata
     + task metadata + episode metadata parquet + shared frame/video shards

Usage:
    # Minimal — auto‑detect everything, output to a local directory
    python tools/export/convert_bench2dex_to_lerobot_v3.py \\
        --input-dir outputs/dex2scene_dataset/scenes/06_fruit_upright \\
        --output-dir datasets/06_fruit_upright

    # With camera selection + renaming + task prompt
    python tools/export/convert_bench2dex_to_lerobot_v3.py \\
        --input-dir outputs/.../replay-generalization \\
        --output-dir datasets/my_task \\
        --camera-map overhead=cam_overhead wrist_right=cam_wrist_right \\
        --prompt "pick up the fruit and place it upright in the bowl" \\
        --robot-type "dual_ur5_rh56dfx" \\
        --overwrite

Output structure (LeRobot v3.0):
    {output_dir}/
    ├── meta/
    │   ├── info.json
    │   ├── stats.json
    │   ├── tasks.parquet
    │   ├── tasks.jsonl        # compatibility copy for tools that still read it
    │   ├── modality.json      # optional named-modality adapter metadata
    │   └── episodes/
    │       └── chunk-{ccc}/file-{fff}.parquet
    ├── data/
    │   └── chunk-{ccc}/
    │       └── file-{fff}.parquet
    └── videos/
        └── observation.images.{video_key}/
            └── chunk-{ccc}/
                └── file-{fff}.mp4
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNK_SIZE = 1000  # episodes per chunk directory (LeRobot convention)
DATA_FILE_SIZE_IN_MB = 100
VIDEO_FILE_SIZE_IN_MB = 500
DEFAULT_FPS = 20    # fallback when meta/effective_fps is absent
# Canonical task text lives in meta/tasks.parquet. This compatibility column
# stores task_index values for consumers that resolve language via task metadata.
TASK_DESCRIPTION_INDEX_KEY = "annotation.human.action.task_description"

_EMPTY_DIR_WARNING = """\
Output directory is not empty: {path}
Use --overwrite to replace ALL contents, or specify a different --output-dir.
"""

# ---------------------------------------------------------------------------
# HDF5 string decoding
# ---------------------------------------------------------------------------


def _decode(value: Any) -> str:
    """Decode an HDF5 scalar value (bytes, 0‑d ndarray, or plain str) to str."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode(value.item())
    return str(value)


def _read_hdf5_value(value: Any) -> Any:
    """Convert an HDF5 scalar/array value into a JSON/parquet-friendly object."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _read_hdf5_value(value.item())
        if value.dtype.kind in {"S", "O", "U"}:
            return [_read_hdf5_value(item) for item in value.tolist()]
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


_DEX2BENCH_META_KEYS = (
    "created_at",
    "dataset_version",
    "sensor_profile_id",
    "sensor_profile_version",
    "task_name",
    "scene_name",
    "scene_file",
    "instruction",
    "robot_key",
    "seed",
    "base_seed",
    "task_seed_id",
    "task_base_seed",
    "episode_index",
    "episode_seed",
    "seed_policy",
    "seed_namespace",
    "fps",
    "effective_fps",
    "step_stride",
    "control_mode",
    "action_type",
    "action_dim",
    "action_names",
    "demo_eligible",
    "success",
    "modalities",
    "camera_definitions",
    "pinhole_camera_ids",
    "fisheye_camera_ids",
    "collect_config",
    "scene_generalization_config",
    "scene_generalization_sample",
    "homing_start_sim_step",
)


def _read_dex2bench_episode_metadata(ep: h5py.File) -> dict[str, Any]:
    """Return Dex2Bench-only audit metadata.

    These columns are intentionally prefixed and written only to episode
    metadata. They are not declared in ``info.json/features`` and therefore are
    not consumed as policy training modalities.
    """
    if "meta" not in ep:
        return {}
    meta: dict[str, Any] = {}
    meta_group = ep["meta"]
    for key in _DEX2BENCH_META_KEYS:
        if key not in meta_group:
            continue
        value = _read_hdf5_value(meta_group[key][()])
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        meta[f"dex2bench/{key}"] = value
    return meta


def _contiguous_segments(indices: np.ndarray) -> list[np.ndarray]:
    """Split sorted frame indices into contiguous segments."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        return []
    breaks = np.where(np.diff(indices) != 1)[0] + 1
    return [segment for segment in np.split(indices, breaks) if len(segment) > 0]


# ---------------------------------------------------------------------------
# Episode metadata inspection
# ---------------------------------------------------------------------------


def _inspect_episode(hdf5_path: Path) -> dict[str, Any]:
    """Read schema‑relevant metadata from one HDF5 episode.

    Returns a dict with:
        camera_ids:       list[str]  — HDF5 camera group names
        camera_shape:     dict[str, tuple[int, int, int]]  — (H, W, C) per camera
        camera_models:    dict[str, str]  — "pinhole" or "fisheye" per camera
        has_fisheye:      bool  — at least one camera is fisheye
        state_dim:        int   — /robot/qpos.shape[1]
        action_dim:       int   — /action/commanded.shape[1]
        has_velocity:     bool  — /robot/qvel exists
        has_effort:       bool  — /robot/qeffort exists
        has_tactile:      bool  — /robot/tactile group exists
        episode_success:  bool | None  — from /episode/success or /meta/success
        fps:              int   — from meta/effective_fps or default
        task:             str   — from meta/instruction or scene_name
        joint_names:      list[str] — from /robot/joint_names or meta
        num_frames:       int
    """
    with h5py.File(hdf5_path, "r") as ep:
        # --- cameras ---
        camera_ids: list[str] = []
        camera_shape: dict[str, tuple[int, int, int]] = {}
        camera_models: dict[str, str] = {}
        if "cameras" in ep:
            for cam_id in sorted(ep["cameras"]):
                cam_grp = ep["cameras"][cam_id]
                if "rgb" not in cam_grp:
                    continue
                # Detect camera model
                cam_model = "pinhole"
                if "camera_model" in cam_grp:
                    cam_model = _decode(cam_grp["camera_model"][()])
                camera_models[cam_id] = cam_model
                rgb_ds = cam_grp["rgb"]
                if rgb_ds.ndim == 4:
                    # Uncompressed: (N, H, W, C)
                    _, h, w, c = rgb_ds.shape
                elif rgb_ds.ndim == 1:
                    # Compressed: (N,) bytes per frame — decode first frame to get shape
                    try:
                        img = _decode_rgb_frame(rgb_ds[0])
                    except Exception:
                        continue
                    h, w, c = img.shape
                else:
                    continue
                camera_shape[cam_id] = (int(h), int(w), int(c))
                camera_ids.append(cam_id)

        # --- state dim ---
        state_dim = 0
        if "robot" in ep and "qpos" in ep["robot"]:
            state_dim = int(ep["robot"]["qpos"].shape[1])

        # --- action dim ---
        action_dim = 0
        if "action" in ep and "commanded" in ep["action"]:
            action_dim = int(ep["action"]["commanded"].shape[1])

        # --- joint/action names ---
        joint_names: list[str] = []
        if "robot" in ep and "joint_names" in ep["robot"]:
            joint_names = [_decode(n) for n in ep["robot"]["joint_names"][:]]
        elif "meta" in ep and "action_names" in ep["meta"]:
            raw = _decode(ep["meta"]["action_names"][()])
            try:
                joint_names = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                joint_names = [s.strip() for s in raw.split(",") if s.strip()]

        # Pad generic names if needed
        if len(joint_names) < max(state_dim, action_dim):
            # Use whatever names we have, pad with generics for the larger dim
            max_dim = max(state_dim, action_dim)
            joint_names += [f"motor_{i}" for i in range(len(joint_names), max_dim)]

        action_names: list[str] = []
        if "action" in ep and "action_names" in ep["action"]:
            action_names = [_decode(n) for n in ep["action"]["action_names"][:]]
        elif "meta" in ep and "action_names" in ep["meta"]:
            raw = _decode(ep["meta"]["action_names"][()])
            try:
                action_names = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                action_names = [s.strip() for s in raw.split(",") if s.strip()]
        if len(action_names) < action_dim:
            action_names += [f"action_{i}" for i in range(len(action_names), action_dim)]

        # --- velocity / effort / tactile ---
        has_velocity = "robot" in ep and "qvel" in ep["robot"]
        has_effort = "robot" in ep and "qeffort" in ep["robot"]
        has_tactile = "robot" in ep and "tactile" in ep["robot"]

        # --- episode success ---
        episode_success: bool | None = None
        if "episode" in ep and "success" in ep["episode"]:
            try:
                episode_success = bool(ep["episode"]["success"][-1])
            except Exception:
                pass
        if episode_success is None and "meta" in ep and "success" in ep["meta"]:
            try:
                episode_success = bool(ep["meta"]["success"][()])
            except Exception:
                pass

        # --- fps ---
        fps = DEFAULT_FPS
        if "meta" in ep:
            for key in ("effective_fps", "fps"):
                if key in ep["meta"]:
                    fps = int(round(float(ep["meta"][key][()])))
                    break

        # --- task ---
        task = ""
        if "meta" in ep:
            for key in ("instruction", "task_instruction", "task_name", "scene_name"):
                if key in ep["meta"]:
                    raw = _decode(ep["meta"][key][()])
                    if raw and raw.lower() not in ("none", "null", ""):
                        task = raw
                        break
        if not task:
            task = hdf5_path.stem

        # --- frame count ---
        num_frames = 0
        if "frame_valid" in ep:
            num_frames = int(ep["frame_valid"].shape[0])
        elif "robot" in ep and "qpos" in ep["robot"]:
            num_frames = int(ep["robot"]["qpos"].shape[0])

    return {
        "camera_ids": camera_ids,
        "camera_shape": camera_shape,
        "camera_models": camera_models,
        "has_fisheye": any(m == "fisheye" for m in camera_models.values()),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "has_velocity": has_velocity,
        "has_effort": has_effort,
        "has_tactile": has_tactile,
        "episode_success": episode_success,
        "fps": fps,
        "task": task,
        "joint_names": joint_names,
        "action_names": action_names,
        "num_frames": num_frames,
    }


def _parse_split_spec(spec: str | None, n_episodes: int) -> dict[str, str]:
    """Parse a split specification like ``"train:val=0.9:0.1"`` into LeRobot info.json splits.

    Returns a dict mapping split name to episode range, e.g.
    ``{"train": "0:90", "val": "90:100"}``.
    When *spec* is None, returns ``{"train": f"0:{n_episodes}"}``.
    """
    if spec is None or n_episodes == 0:
        return {"train": f"0:{n_episodes}"}

    # Parse "name1:name2=prop1:prop2"
    if "=" not in spec:
        raise ValueError(f"Invalid split spec: {spec!r}. Expected 'name1:name2=prop1:prop2'.")
    names_part, props_part = spec.split("=", 1)
    names = [n.strip() for n in names_part.split(":")]
    props = [float(p.strip()) for p in props_part.split(":")]

    if len(names) != len(props):
        raise ValueError(
            f"Number of split names ({len(names)}) != number of proportions ({len(props)})"
        )
    if len(names) < 2:
        raise ValueError("At least two splits are required")

    total_prop = sum(props)
    if not np.isclose(total_prop, 1.0, atol=0.01):
        raise ValueError(f"Split proportions must sum to ~1.0, got {total_prop}")

    # Deterministic shuffle using episode indices
    rng = np.random.RandomState(42)
    indices = rng.permutation(n_episodes).tolist()

    splits: dict[str, str] = {}
    start = 0
    for name, prop in zip(names, props):
        count = int(round(prop * n_episodes))
        # Adjust last split to cover remaining episodes
        if name == names[-1]:
            count = n_episodes - start
        end = min(start + count, n_episodes)
        # Map shuffled indices to the original episode order expected by the dataset
        split_indices = sorted(indices[start:end])
        # Convert to LeRobot range format: "ep_i:ep_j:ep_k:..."
        # For contiguous ranges we use "start:end" format
        splits[name] = _indices_to_range_str(split_indices)
        start = end

    return splits


def _indices_to_range_str(indices: list[int]) -> str:
    """Convert sorted index list to LeRobot split range string.

    LeRobot format uses ``"start:end"`` range pairs (end is exclusive), joined by ``":"``.
    Example: ``[0, 1, 2, 5, 6, 9]`` → ``"0:3:5:7:9:10"``.
    """
    if not indices:
        return "0:0"
    # Compress consecutive indices into start:end (exclusive) range pairs
    ranges: list[str] = []
    range_start = indices[0]
    prev = indices[0]
    for i in indices[1:]:
        if i == prev + 1:
            prev = i
        else:
            ranges.append(f"{range_start}:{prev + 1}")
            range_start = i
            prev = i
    ranges.append(f"{range_start}:{prev + 1}")
    return ":".join(ranges)


def _validate_episode_consistency(ref: dict[str, Any], ep_path: Path) -> dict[str, Any]:
    """Check that *ep_path* matches the reference schema; warn on mismatches.

    Returns the current episode's schema dict (for downstream use, e.g. episode_success).
    Only warns on **missing** cameras and dimension mismatches — extra cameras
    beyond the reference are not flagged (they may be added by a later sensor profile).
    Warnings are accumulated across episodes; a summary is printed at the end by
    ``_flush_consistency_warnings``.
    """
    cur = _inspect_episode(ep_path)

    ref_cams = set(ref["camera_ids"])
    cur_cams = set(cur["camera_ids"])
    missing = ref_cams - cur_cams

    if cur["state_dim"] != ref["state_dim"] or cur["action_dim"] != ref["action_dim"] or missing:
        msg = f"  {ep_path.name}:"
        if cur["state_dim"] != ref["state_dim"]:
            msg += f" state_dim={cur['state_dim']}(expected {ref['state_dim']})"
        if cur["action_dim"] != ref["action_dim"]:
            msg += f" action_dim={cur['action_dim']}(expected {ref['action_dim']})"
        if missing:
            msg += f" cameras_missing={sorted(missing)}"
        _SCHEMA_MISMATCHES.append(msg)

    return cur


# Global accumulator for schema mismatch warnings (cleared per convert() call)
_SCHEMA_MISMATCHES: list[str] = []


def _flush_consistency_warnings() -> None:
    """Print accumulated schema mismatch warnings as a single summary block."""
    if not _SCHEMA_MISMATCHES:
        return
    print(
        f"\n[WARN] Schema mismatches vs episode 0 ({len(_SCHEMA_MISMATCHES)} episode(s)):",
        file=sys.stderr,
    )
    for msg in _SCHEMA_MISMATCHES:
        print(msg, file=sys.stderr)
    _SCHEMA_MISMATCHES.clear()


# ---------------------------------------------------------------------------
# Vectorized validity mask
# ---------------------------------------------------------------------------


def _build_valid_mask(
    ep: h5py.File,
    *,
    require_finite_action: bool = True,
) -> np.ndarray:
    """Return a boolean mask: True for frames usable for training.

    A frame is valid when ALL of:
      1. frame_valid == True  (sensor data captured correctly)
      2. action_valid == True  (real action, not placeholder)
      3. action values are all finite  (no NaN or Inf)
    """
    n = 0
    if "frame_valid" in ep:
        mask = np.asarray(ep["frame_valid"][:], dtype=bool)
        n = len(mask)
    elif "robot" in ep and "qpos" in ep["robot"]:
        n = int(ep["robot"]["qpos"].shape[0])
        mask = np.ones(n, dtype=bool)
    else:
        return np.zeros(0, dtype=bool)

    # qpos must be finite; HDF5 writer uses NaN for missing robot states.
    if "robot" in ep and "qpos" in ep["robot"]:
        qpos_arr = np.asarray(ep["robot"]["qpos"][:], dtype=np.float32)
        if qpos_arr.shape[0] == n:
            mask = mask & np.isfinite(qpos_arr).all(axis=1)
        else:
            print(
                f"[WARN] robot/qpos frame count mismatch: "
                f"{qpos_arr.shape[0]} != {n} in {ep.filename}",
                file=sys.stderr,
            )
            return np.zeros(n, dtype=bool)
    else:
        return np.zeros(n, dtype=bool)

    # action_valid
    if "action" in ep and "action_valid" in ep["action"]:
        action_valid = np.asarray(ep["action"]["action_valid"][:], dtype=bool)
        if action_valid.shape[0] == n:
            mask = mask & action_valid
        else:
            print(
                f"[WARN] action/action_valid frame count mismatch: "
                f"{action_valid.shape[0]} != {n} in {ep.filename}",
                file=sys.stderr,
            )
            return np.zeros(n, dtype=bool)

    # finite action values (vectorized — no per‑frame loop)
    if require_finite_action and "action" in ep and "commanded" in ep["action"]:
        action_arr = np.asarray(ep["action"]["commanded"][:], dtype=np.float32)
        if action_arr.shape[0] == n:
            finite = np.isfinite(action_arr).all(axis=1)
            mask = mask & finite
        else:
            print(
                f"[WARN] action/commanded frame count mismatch: "
                f"{action_arr.shape[0]} != {n} in {ep.filename}",
                file=sys.stderr,
            )
            return np.zeros(n, dtype=bool)

    return mask


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _compute_stats(values: np.ndarray) -> dict[str, list[float]]:
    """Compute per‑dimension normalization statistics."""
    if values.ndim == 1:
        values = values[:, None]
    return {
        "mean": values.mean(axis=0).astype(float).tolist(),
        "std": values.std(axis=0).astype(float).tolist(),
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).astype(float).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).astype(float).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).astype(float).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).astype(float).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).astype(float).tolist(),
        "count": int(values.shape[0]),
    }


def _flatten_episode_stats(prefix: str, values: np.ndarray) -> dict[str, list[float]]:
    """Return v3 episode metadata columns such as stats/action/min."""
    return {f"stats/{prefix}/{name}": stat for name, stat in _compute_stats(values).items()}


# ---------------------------------------------------------------------------
# Video writing
# ---------------------------------------------------------------------------


class _StreamingVideoWriter:
    """Write one LeRobot v3 video shard incrementally."""

    def __init__(self, out_path: Path, *, fps: float, shape: tuple[int, int, int]) -> None:
        import subprocess

        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required to write LeRobot MP4 videos")

        h, w, c = shape
        if c != 3:
            raise ValueError(f"Expected RGB frames with 3 channels, got {shape}")

        self.out_path = out_path
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.frame_count = 0
        self._proc = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-s",
                f"{w}x{h}",
                "-pix_fmt",
                "rgb24",
                "-r",
                str(fps),
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "medium",
                "-crf",
                "23",
                "-an",
                str(self.out_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if self._proc.stdin is None:
            raise RuntimeError("failed to open ffmpeg stdin")

    def write(self, frame: np.ndarray) -> None:
        self._proc.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
        self.frame_count += 1

    def close(self) -> None:
        if self._proc.stdin is not None and not self._proc.stdin.closed:
            self._proc.stdin.close()
        stderr = b""
        if self._proc.stderr is not None:
            stderr = self._proc.stderr.read()
        ret = self._proc.wait(timeout=300)
        if ret != 0:
            self.out_path.unlink(missing_ok=True)
            message = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg exited with code {ret}: {message}")


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Video feature metadata
# ---------------------------------------------------------------------------


def _video_feature(h: int, w: int, c: int, fps: float) -> dict[str, Any]:
    return {
        "dtype": "video",
        "shape": [h, w, c],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": float(fps),
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    }


# ---------------------------------------------------------------------------
# RGB frame decoding
# ---------------------------------------------------------------------------


def _decode_rgb_frame(frame: Any) -> np.ndarray:
    """Decode a single RGB frame from HDF5 (raw bytes or numpy array)."""
    if isinstance(frame, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(frame, dtype=np.uint8)
    else:
        arr = np.asarray(frame)
        if arr.shape == () and isinstance(arr.item(), (bytes, bytearray)):
            arr = np.frombuffer(arr.item(), dtype=np.uint8)

    if arr.ndim == 3:
        # Already decoded (H, W, C) — ensure uint8, drop alpha channel if present
        return arr[..., :3].astype(np.uint8, copy=False)

    # JPEG/PNG/etc. compressed — delegate to cv2
    import cv2

    decoded = cv2.imdecode(
        np.asarray(arr, dtype=np.uint8).reshape(-1), cv2.IMREAD_COLOR
    )
    if decoded is None:
        raise ValueError("Failed to decode compressed RGB frame")
    return decoded[:, :, ::-1].astype(np.uint8, copy=False)  # BGR → RGB


# ===================================================================
# Main converter
# ===================================================================


def convert(
    *,
    input_dir: Path,
    output_dir: Path,
    camera_map: dict[str, str] | None = None,
    prompt: str | None = None,
    robot_type: str = "dex2bench",
    overwrite: bool = False,
    chunk_size: int = CHUNK_SIZE,
    split: str | None = None,
    fps_override: int | None = None,
) -> Path:
    """Convert a directory of episode_*.hdf5 files to LeRobot v3 format.

    Args:
        input_dir: Directory containing ``episode_*.hdf5`` files.
        output_dir: Where to write the LeRobot v3 dataset.
        camera_map: ``{output_name: hdf5_camera_name}`` — selects and renames
                    cameras.  If None, ALL RGB cameras from the data are used
                    with their original HDF5 names.  Fisheye camera geometry is
                    kept in Dex2Bench metadata, but policy features receive only
                    ordinary RGB videos.
        prompt: Language instruction for the task.  If None, auto‑detected from
                the first episode's ``meta/instruction`` or ``meta/scene_name``.
        robot_type: Identifier written into ``info.json``.
        overwrite: If True, delete *output_dir* before converting.
        chunk_size: Episodes per ``chunk-NNN`` subdirectory.
        split: Optional train/val split specification, e.g. ``"train:val=0.9:0.1"``.
               When provided, episodes are deterministically shuffled and assigned
               to the named splits according to the given proportions.
        fps_override: Override the auto‑detected FPS from episode metadata.

    Returns:
        The resolved *output_dir* Path.
    """
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    # Clear accumulated schema warnings for this run
    _SCHEMA_MISMATCHES.clear()

    # --- discover episodes ---
    hdf5_paths = sorted(input_dir.glob("episode_*.hdf5"))
    if not hdf5_paths:
        raise FileNotFoundError(f"No episode_*.hdf5 files found in {input_dir}")
    print(f"Found {len(hdf5_paths)} episode(s) in {input_dir}")

    # --- output directory guard ---
    if output_dir.exists():
        if not overwrite:
            if any(output_dir.iterdir()):
                raise FileExistsError(_EMPTY_DIR_WARNING.format(path=output_dir))
        else:
            shutil.rmtree(output_dir)

    # --- inspect first episode for schema ---
    ref = _inspect_episode(hdf5_paths[0])
    effective_fps = fps_override if fps_override is not None else ref["fps"]
    print(f"  state dim   : {ref['state_dim']}")
    print(f"  action dim  : {ref['action_dim']}")
    print(f"  cameras     : {ref['camera_ids']}")
    print(f"  fps         : {effective_fps}{' (overridden)' if fps_override is not None else ''}")
    print(f"  task        : {ref['task']!r}")

    # --- fisheye camera note ---
    if ref["has_fisheye"]:
        fisheye_ids = [cid for cid, m in ref["camera_models"].items() if m == "fisheye"]
        print(
            f"[WARN] Fisheye camera(s) detected: {fisheye_ids}. "
            f"They will be exported as RGB videos; camera geometry remains Dex2Bench metadata only.",
            file=sys.stderr,
        )

    # --- tactile data warning ---
    if ref["has_tactile"]:
        print(
            "[WARN] Tactile (/robot/tactile) data detected in HDF5. "
            "Tactile data is NOT converted to LeRobot format. "
            "Only joint state (qpos) and camera (rgb) data are exported.",
            file=sys.stderr,
        )

    # --- resolve camera mapping ---
    if camera_map:
        # camera_map: {output_name: hdf5_camera_name}
        video_keys = list(camera_map.keys())
        # Validate all requested cameras exist
        for hdf5_name in camera_map.values():
            if hdf5_name not in ref["camera_ids"]:
                raise ValueError(
                    f"Camera '{hdf5_name}' not found in data. "
                    f"Available: {ref['camera_ids']}"
                )
    else:
        # Use all cameras with original names.
        video_keys = list(ref["camera_ids"])
        camera_map = {k: k for k in video_keys}
    if not video_keys:
        raise ValueError("No RGB cameras selected for export.")

    state_dim = ref["state_dim"]
    action_dim = ref["action_dim"]
    state_names = ref["joint_names"][:state_dim]
    action_names = ref["action_names"][:action_dim]

    # Pad names if joint_names was incomplete
    if len(state_names) < state_dim:
        state_names += [f"state_{i}" for i in range(len(state_names), state_dim)]
    if len(action_names) < action_dim:
        action_names += [f"action_{i}" for i in range(len(action_names), action_dim)]

    # --- resolve task prompt ---
    task_text = prompt or ref["task"]

    # --- prepare output directories ---
    meta_dir = output_dir / "meta"
    data_dir = output_dir / "data"
    video_dir = output_dir / "videos"
    for d in [meta_dir, data_dir, video_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # --- v3 shared-file conversion ---
    tasks: list[dict[str, Any]] = [{"task_index": 0, "task": task_text}]
    episode_rows: list[dict[str, Any]] = []
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_timestamps: list[np.ndarray] = []
    total_frames = 0

    # Pre-compute per-row byte estimate for sharding
    _per_row_est = (state_dim + action_dim) * 4 + 200  # float32 cols + list/dict overhead

    def _flush_shard(
        _chunk_index: int,
        _file_index: int,
        _data_rows: list[dict[str, Any]],
        _meta_rows: list[dict[str, Any]],
    ) -> None:
        """Write one data+meta+video shard to disk."""
        if _data_rows:
            out_data_dir = data_dir / f"chunk-{_chunk_index:03d}"
            out_data_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(_data_rows).to_parquet(
                out_data_dir / f"file-{_file_index:03d}.parquet",
                index=False,
            )
        if _meta_rows:
            out_meta_dir = meta_dir / "episodes" / f"chunk-{_chunk_index:03d}"
            out_meta_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(_meta_rows).to_parquet(
                out_meta_dir / f"file-{_file_index:03d}.parquet",
                index=False,
            )

    n_input_chunks = (len(hdf5_paths) + chunk_size - 1) // chunk_size
    for input_chunk_idx in range(n_input_chunks):
        chunk_paths = hdf5_paths[
            input_chunk_idx * chunk_size : (input_chunk_idx + 1) * chunk_size
        ]
        if not chunk_paths:
            continue

        chunk_index = input_chunk_idx
        file_index = 0
        data_rows: list[dict[str, Any]] = []
        chunk_episode_rows: list[dict[str, Any]] = []
        video_writers: dict[str, _StreamingVideoWriter] = {}
        video_frame_offsets: dict[str, int] = {k: 0 for k in video_keys}

        try:
            for ep_path in tqdm.tqdm(chunk_paths, desc=f"Converting chunk-{chunk_index:03d}", unit="ep"):
                cur_schema = _validate_episode_consistency(ref, ep_path)

                with h5py.File(ep_path, "r") as ep:
                    valid_mask = _build_valid_mask(ep, require_finite_action=True)
                    valid_idx = np.where(valid_mask)[0]
                    if len(valid_idx) == 0:
                        tqdm.tqdm.write(f"[SKIP] {ep_path.name}: 0 valid frames")
                        continue
                    segments = _contiguous_segments(valid_idx)
                    if len(segments) > 1:
                        tqdm.tqdm.write(
                            f"[WARN] {ep_path.name}: split into {len(segments)} contiguous valid segments"
                        )

                    # --- source episode success / Dex2Bench audit metadata ---
                    ep_success = cur_schema.get("episode_success", False) if cur_schema else False
                    if ep_success is None:
                        ep_success = False
                    dex2bench_meta = _read_dex2bench_episode_metadata(ep)

                    for source_segment_index, segment_idx in enumerate(segments):
                        state = np.asarray(ep["robot"]["qpos"][:], dtype=np.float32)[segment_idx]
                        action = np.asarray(ep["action"]["commanded"][:], dtype=np.float32)[segment_idx]
                        length = int(len(segment_idx))

                        output_ep_idx = len(episode_rows)
                        dataset_from_index = total_frames
                        dataset_to_index = total_frames + length
                        timestamps = (
                            (segment_idx.astype(np.float32) - float(segment_idx[0]))
                            / float(effective_fps)
                        )

                        rows = {
                            "observation.state": state.tolist(),
                            "action": action.tolist(),
                            "timestamp": timestamps.astype(float).tolist(),
                            TASK_DESCRIPTION_INDEX_KEY: [0] * length,
                            "task_index": [0] * length,
                            "episode_index": [output_ep_idx] * length,
                            "frame_index": list(range(length)),
                            "source_frame_index": segment_idx.astype(int).tolist(),
                            "index": list(range(dataset_from_index, dataset_to_index)),
                            "next.reward": [0.0] * length,
                            "next.done": [False] * length,
                            "next.success": [False] * length,
                        }
                        rows["next.done"][-1] = True
                        rows["next.success"][-1] = bool(ep_success and source_segment_index == len(segments) - 1)
                        data_rows.extend(pd.DataFrame(rows).to_dict("records"))

                        episode_meta: dict[str, Any] = {
                            "episode_index": output_ep_idx,
                            "source_episode": ep_path.name,
                            "source_segment_index": int(source_segment_index),
                            "source_frame_start": int(segment_idx[0]),
                            "source_frame_end": int(segment_idx[-1]) + 1,
                            "tasks": [0],
                            "length": length,
                            "data/chunk_index": chunk_index,
                            "data/file_index": file_index,
                            "dataset_from_index": dataset_from_index,
                            "dataset_to_index": dataset_to_index,
                            "episode_success": bool(ep_success and source_segment_index == len(segments) - 1),
                        }
                        episode_meta.update(dex2bench_meta)
                        episode_meta.update(_flatten_episode_stats("observation.state", state))
                        episode_meta.update(_flatten_episode_stats("action", action))

                        for out_key, hdf5_cam in camera_map.items():
                            cam_path = f"cameras/{hdf5_cam}/rgb"
                            if cam_path not in ep:
                                raise KeyError(f"Missing camera dataset: {cam_path} in {ep_path}")

                            video_key = f"observation.images.{out_key}"
                            if out_key not in video_writers:
                                h, w, c = ref["camera_shape"][hdf5_cam]
                                video_writers[out_key] = _StreamingVideoWriter(
                                    video_dir
                                    / video_key
                                    / f"chunk-{chunk_index:03d}"
                                    / f"file-{file_index:03d}.mp4",
                                    fps=effective_fps,
                                    shape=(h, w, c),
                                )

                            from_frame = video_frame_offsets[out_key]
                            ds = ep[cam_path]
                            for frame_i in segment_idx:
                                video_writers[out_key].write(_decode_rgb_frame(ds[int(frame_i)]))
                            to_frame = video_frame_offsets[out_key] + length
                            video_frame_offsets[out_key] = to_frame

                            episode_meta[f"videos/{video_key}/chunk_index"] = chunk_index
                            episode_meta[f"videos/{video_key}/file_index"] = file_index
                            episode_meta[f"videos/{video_key}/from_timestamp"] = (
                                float(from_frame) / float(effective_fps)
                            )
                            episode_meta[f"videos/{video_key}/to_timestamp"] = (
                                float(to_frame) / float(effective_fps)
                            )

                        chunk_episode_rows.append(episode_meta)
                        episode_rows.append(episode_meta)
                        all_states.append(state)
                        all_actions.append(action)
                        all_timestamps.append(timestamps)
                        total_frames = dataset_to_index

                        # --- size-based sharding: flush when approaching DATA_FILE_SIZE_IN_MB ---
                        _est_mb = (len(data_rows) * _per_row_est) / (1024 * 1024)
                        if _est_mb >= DATA_FILE_SIZE_IN_MB:
                            _flush_shard(chunk_index, file_index, data_rows, chunk_episode_rows)
                            for writer in video_writers.values():
                                writer.close()
                            video_writers.clear()
                            video_frame_offsets = {k: 0 for k in video_keys}
                            data_rows.clear()
                            chunk_episode_rows.clear()
                            file_index += 1

            # --- flush remaining data in this chunk ---
            for writer in video_writers.values():
                writer.close()
            _flush_shard(chunk_index, file_index, data_rows, chunk_episode_rows)

        except Exception:
            # Close video writers first
            for writer in video_writers.values():
                try:
                    writer.close()
                except Exception:
                    pass
            # Clean up partially-written output files for this chunk
            for pattern_dir, pattern_file in [
                (data_dir / f"chunk-{chunk_index:03d}", f"file-*.parquet"),
                (meta_dir / "episodes" / f"chunk-{chunk_index:03d}", f"file-*.parquet"),
            ]:
                if pattern_dir.exists():
                    for f in pattern_dir.glob(pattern_file):
                        try:
                            f.unlink()
                        except Exception:
                            pass
            for vk in video_keys:
                vdir = video_dir / f"observation.images.{vk}" / f"chunk-{chunk_index:03d}"
                if vdir.exists():
                    for f in vdir.glob("file-*.mp4"):
                        try:
                            f.unlink()
                        except Exception:
                            pass
            raise

    # --- build info.json ---
    features: dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [state_dim],
            "names": {"motors": state_names},
        },
        "action": {
            "dtype": "float32",
            "shape": [action_dim],
            "names": {"motors": action_names},
        },
        "timestamp": {"dtype": "float32", "shape": [1]},
        TASK_DESCRIPTION_INDEX_KEY: {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "source_frame_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "next.reward": {"dtype": "float32", "shape": [1]},
        "next.done": {"dtype": "bool", "shape": [1]},
        "next.success": {"dtype": "bool", "shape": [1]},
    }

    for out_key in video_keys:
        # Use shape from first camera mapping match
        hdf5_cam = camera_map.get(out_key, out_key)
        if hdf5_cam in ref["camera_shape"]:
            h, w, c = ref["camera_shape"][hdf5_cam]
            features[f"observation.images.{out_key}"] = _video_feature(h, w, c, effective_fps)

    n_episodes = len(episode_rows)
    n_chunks = len({int(row["data/chunk_index"]) for row in episode_rows}) if episode_rows else 0

    total_video_files = sum(
        1 for _path in video_dir.glob("observation.images.*/chunk-*/file-*.mp4")
    )

    info = {
        "codebase_version": "v3.0",
        "robot_type": robot_type,
        "total_episodes": n_episodes,
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": total_video_files,
        "total_episode_videos": n_episodes * len(video_keys),
        "total_chunks": n_chunks,
        "chunks_size": chunk_size,
        "data_files_size_in_mb": DATA_FILE_SIZE_IN_MB,
        "video_files_size_in_mb": VIDEO_FILE_SIZE_IN_MB,
        "fps": float(effective_fps),
        "splits": _parse_split_spec(split, n_episodes),
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
        "dex2bench_camera_models": {
            out_key: ref["camera_models"].get(camera_map.get(out_key, out_key), "pinhole")
            for out_key in video_keys
        },
    }

    # --- build stats.json ---
    stats: dict[str, Any] = {}
    if all_states:
        stats["observation.state"] = _compute_stats(np.concatenate(all_states, axis=0))
    if all_actions:
        stats["action"] = _compute_stats(np.concatenate(all_actions, axis=0))
    if all_timestamps:
        stats["timestamp"] = _compute_stats(np.concatenate(all_timestamps, axis=0))

    # --- build optional named-modality adapter metadata ---
    named_modality_adapter: dict[str, Any] = {
        "state": {
            "qpos": {
                "start": 0,
                "end": state_dim,
                "dtype": "float32",
                "absolute": True,
                "original_key": "observation.state",
            }
        },
        "action": {
            "qpos": {
                "start": 0,
                "end": action_dim,
                "dtype": "float32",
                "absolute": True,
                "original_key": "action",
            }
        },
        "video": {
            vk: {"original_key": f"observation.images.{vk}"}
            for vk in video_keys
        },
        "annotation": {
            "human.action.task_description": {},
        },
    }

    # --- write metadata ---
    if n_episodes == 0:
        raise RuntimeError("No valid episodes were converted")

    (meta_dir / "info.json").write_text(
        json.dumps(info, indent=4, ensure_ascii=False), encoding="utf-8"
    )
    (meta_dir / "stats.json").write_text(
        json.dumps(stats, indent=4, ensure_ascii=False), encoding="utf-8"
    )
    (meta_dir / "modality.json").write_text(
        json.dumps(named_modality_adapter, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_jsonl(meta_dir / "tasks.jsonl", tasks)
    tasks_df = pd.DataFrame(tasks, columns=["task_index", "task"])
    tasks_df.to_parquet(meta_dir / "tasks.parquet")

    # --- post‑conversion validation ---
    _validate_output(output_dir, n_episodes, video_keys, effective_fps)

    # --- summary ---
    print(f"\n{'='*60}")
    print(f"Conversion complete")
    print(f"  Episodes      : {n_episodes}")
    print(f"  Valid frames  : {total_frames}")
    print(f"  Cameras       : {video_keys}")
    print(f"  State dim     : {state_dim}")
    print(f"  Action dim    : {action_dim}")
    print(f"  Output        : {output_dir}")
    print(f"{'='*60}")

    # Flush accumulated schema warnings
    _flush_consistency_warnings()

    return output_dir


# ---------------------------------------------------------------------------
# Post‑conversion validation
# ---------------------------------------------------------------------------


def _validate_output(
    output_dir: Path,
    n_episodes: int,
    video_keys: list[str],
    fps: float,
) -> None:
    """Lightweight sanity checks on the LeRobot v3 output dataset."""
    errors: list[str] = []

    meta_dir = output_dir / "meta"
    for filename in ["info.json", "stats.json", "modality.json", "tasks.parquet"]:
        if not (meta_dir / filename).is_file():
            errors.append(f"Missing meta/{filename}")

    # Check info.json can be parsed
    info = json.loads((meta_dir / "info.json").read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        errors.append(f"Expected codebase_version v3.0, got {info.get('codebase_version')}")
    if info.get("fps") != float(fps):
        errors.append(f"FPS mismatch: info.json={info.get('fps')}, expected={fps}")
    if info.get("features") is None:
        errors.append("info.json missing 'features'")
    if info.get("data_path") != "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet":
        errors.append(f"Unexpected v3 data_path: {info.get('data_path')}")
    if info.get("video_path") != "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4":
        errors.append(f"Unexpected v3 video_path: {info.get('video_path')}")

    # Check episode metadata parquet
    episode_meta_paths = sorted((meta_dir / "episodes").glob("chunk-*/file-*.parquet"))
    if not episode_meta_paths:
        errors.append("No parquet files found in meta/episodes/")
        episode_meta = pd.DataFrame()
    else:
        episode_meta = pd.concat(pd.read_parquet(p) for p in episode_meta_paths)
        if len(episode_meta) != n_episodes:
            errors.append(
                f"episode metadata has {len(episode_meta)} entries, expected {n_episodes}"
            )
        for col in ["episode_index", "length", "data/chunk_index", "data/file_index",
                    "dataset_from_index", "dataset_to_index"]:
            if col not in episode_meta.columns:
                errors.append(f"episode metadata missing column {col!r}")

    # Check frame parquet shards
    parquets = sorted(output_dir.glob("data/chunk-*/file-*.parquet"))
    if not parquets:
        errors.append("No v3 parquet shards found in data/")
    else:
        total_rows = sum(len(pd.read_parquet(p)) for p in parquets)
        if total_rows != info.get("total_frames"):
            errors.append(f"total frame rows {total_rows} != info.total_frames {info.get('total_frames')}")

    # Check at least one non-empty v3 video shard per camera.
    for vk in video_keys:
        video_key = f"observation.images.{vk}"
        mp4s = sorted(output_dir.glob(f"videos/{video_key}/chunk-*/file-*.mp4"))
        if not mp4s:
            errors.append(f"No v3 videos found for camera {video_key!r}")
        for mp4 in mp4s:
            if mp4.stat().st_size == 0:
                errors.append(f"Empty video file: {mp4}")

    if errors:
        print("\n[VALIDATION ERRORS]", file=sys.stderr)
        for e in errors:
            print(f"  • {e}", file=sys.stderr)
        raise RuntimeError("LeRobot v3 output validation failed")
    else:
        print(f"\n[OK] Output validation passed")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_camera_map(items: list[str] | None) -> dict[str, str]:
    """Parse ``KEY=VALUE`` pairs into a camera map dict.

    Example:
        --camera-map overhead=cam_overhead wrist_right=cam_wrist_right
        -> {"overhead": "cam_overhead", "wrist_right": "cam_wrist_right"}
    """
    mapping: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(
                f"Camera map entry must be OUTPUT=HDF5_CAMERA, got {item!r}"
            )
        out, hdf5 = item.split("=", 1)
        out, hdf5 = out.strip(), hdf5.strip()
        if not out or not hdf5:
            raise ValueError(f"Invalid camera map entry: {item!r}")
        mapping[out] = hdf5
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", required=True, type=Path,
        help="Directory containing episode_*.hdf5 files.",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Directory to write the LeRobot v3 dataset.",
    )
    parser.add_argument(
        "--camera-map", nargs="*", default=None,
        help='Camera selection/renaming: "overhead=cam_overhead wrist_right=cam_wrist_right". '
             "Omit to use all cameras with original names.",
    )
    parser.add_argument(
        "--prompt", default=None,
        help="Language instruction for the task (overrides auto‑detection).",
    )
    parser.add_argument(
        "--robot-type", default="dex2bench",
        help="Robot identifier written to meta/info.json (default: dex2bench).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Delete output directory before converting.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=CHUNK_SIZE,
        help=f"Episodes per chunk-NNN directory (default: {CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--split", default=None,
        help='Train/val split specification, e.g. "train:val=0.9:0.1". '
             "When omitted, all episodes go to the train split.",
    )
    parser.add_argument(
        "--fps", type=int, default=None, dest="fps_override",
        help="Override the auto‑detected FPS from episode metadata.",
    )

    args = parser.parse_args()

    camera_map = _parse_camera_map(args.camera_map) if args.camera_map else None

    try:
        convert(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            camera_map=camera_map,
            prompt=args.prompt,
            robot_type=args.robot_type,
            overwrite=args.overwrite,
            chunk_size=args.chunk_size,
            split=args.split,
            fps_override=args.fps_override,
        )
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
