"""Convert dex2bench raw HDF5 replay episodes to GR00T LeRobot format."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import h5py
import numpy as np
import pandas as pd

_DEX2BENCH_ROOT = Path(__file__).resolve().parents[2]
if str(_DEX2BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEX2BENCH_ROOT))

from robots.active_dof_utils import (
    ActiveDofInfo,
    get_active_dof_info,
    get_active_dof_info_for_hdf5,
    robot_key_from_hdf5,
    select_active,
)


ANNOTATION_KEY = "annotation.human.action.task_description"
CHUNK_SIZE = 1000
DEFAULT_FPS = 20.0
DEFAULT_CAMERA_MAP = {
    "stereo_left": "cam_stereo_left",
    "stereo_right": "cam_stereo_right",
    "right_wrist": "cam_wrist_right",
    "left_wrist": "cam_wrist_left",
}


@dataclass(frozen=True)
class EpisodePayload:
    task: str
    state: np.ndarray
    action: np.ndarray
    images: dict[str, np.ndarray]
    fps: float


def _decode_hdf5_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_hdf5_string(value.item())
    return str(value)


def _task_from_episode(ep: h5py.File, fallback: str, prompt: str | None = None) -> str:
    if prompt:
        return prompt
    if "meta" in ep:
        for key in ("instruction", "task_name", "scene_name"):
            if key in ep["meta"]:
                value = _decode_hdf5_string(ep["meta"][key][()])
                if value and value.lower() not in {"none", "null"}:
                    return value
    return fallback


def _fps_from_episode(ep: h5py.File) -> float:
    if "meta" in ep:
        for key in ("effective_fps", "fps"):
            if key in ep["meta"]:
                return float(ep["meta"][key][()])
    return DEFAULT_FPS


def _valid_indices(ep: h5py.File) -> np.ndarray:
    if "robot" not in ep or "qpos" not in ep["robot"]:
        raise KeyError("Episode is missing /robot/qpos")
    frame_count = int(ep["robot"]["qpos"].shape[0])
    frame_valid = (
        np.asarray(ep["frame_valid"][:], dtype=np.bool_)
        if "frame_valid" in ep
        else np.ones(frame_count, dtype=np.bool_)
    )
    action_valid = (
        np.asarray(ep["action"]["action_valid"][:], dtype=np.bool_)
        if "action" in ep and "action_valid" in ep["action"]
        else np.ones(frame_count, dtype=np.bool_)
    )
    if "action" in ep and "commanded" in ep["action"]:
        action = np.asarray(ep["action"]["commanded"][:], dtype=np.float32)
        finite_action = np.isfinite(action).all(axis=1)
    else:
        raise ValueError("Episode is missing /action/commanded")
    return np.where(frame_valid & action_valid & finite_action)[0]


def _find_homing_cutoff(
    ep: h5py.File,
    ep_len_full: int,
    valid_indices: np.ndarray,
    *,
    truncate: bool = True,
    min_frames: int = 1,
) -> Tuple[Optional[int], str]:
    """Determine the frame index at which to truncate an episode.

    Reads ``meta/homing_start_sim_step`` and ``time/sim_step`` from the
    HDF5 file, following the same pattern as ACT's
    ``_resolve_effective_length`` (``policy/ACT/utils.py``).  The
    ``homing_start_sim_step`` is the sim-step at which the operator
    triggered "go home" — a reliable human label that the task is complete
    and subsequent frames are non-task noise.

    The returned cutoff is expressed as a count of *valid* frames to keep
    (i.e. an index into ``valid_indices``).

    Args:
        ep: Open h5py File handle for the episode.
        ep_len_full: Total number of frames in the episode (``len(time/sim_step)``).
        valid_indices: Sorted array of valid frame indices (from :func:`_valid_indices`).
        truncate: If False, immediately returns ``(None, info)`` — no
            truncation is applied.
        min_frames: Minimum number of valid frames to keep after truncation.

    Returns:
        ``(cutoff, info)`` where:
          * ``cutoff`` — int count of valid frames to retain (same semantics
            as ``len(valid_indices)``, i.e. ``valid_indices[:cutoff]`` are
            kept).  The homing frame itself is **excluded**.
            ``None`` if no truncation should be applied.
          * ``info`` — str describing what happened (for log output).
    """
    if not truncate:
        return None, "truncation disabled"

    # -- homing_start_sim_step ---------------------------------------------
    homing_start = ep.get("meta/homing_start_sim_step")
    if homing_start is None:
        return None, "homing_start_sim_step missing — keeping full episode"

    homing_val = int(homing_start[()])
    if homing_val < 0:
        return None, (
            f"homing_start_sim_step={homing_val} (no homing) "
            f"— keeping full episode"
        )

    # -- map sim step → frame index in the full episode -------------------
    time_sim_step = ep["time/sim_step"][:]  # (ep_len_full,) int64
    frame_idx = int(np.searchsorted(time_sim_step, homing_val))

    # Following ACT: require 0 < idx < total  (exclude the homing frame)
    if not (min_frames <= frame_idx < ep_len_full):
        return None, (
            f"homing at frame {frame_idx} out of valid range "
            f"[{min_frames}, {ep_len_full}) — keeping full episode"
        )

    cutoff_full = frame_idx  # exclusive: frames [:frame_idx] kept
    cutoff_full = max(cutoff_full, min_frames)
    cutoff_full = min(cutoff_full, ep_len_full)

    # -- map to valid_indices space ---------------------------------------
    valid_cutoff = int(np.searchsorted(valid_indices, cutoff_full, side="left"))
    valid_cutoff = max(valid_cutoff, min_frames)
    valid_cutoff = min(valid_cutoff, len(valid_indices))

    if valid_cutoff >= len(valid_indices):
        return None, (
            f"cutoff={valid_cutoff} >= valid_ep_len={len(valid_indices)} "
            f"— keeping full episode"
        )

    frames_removed = len(valid_indices) - valid_cutoff
    info = (
        f"homing at frame {frame_idx} (sim_step={homing_val}), "
        f"cutoff={valid_cutoff} (valid frames), removed={frames_removed}"
    )
    return valid_cutoff, info


def _decode_rgb_frame(frame: Any) -> np.ndarray:
    if isinstance(frame, (bytes, bytearray)):
        arr = np.frombuffer(frame, dtype=np.uint8)
    else:
        arr = np.asarray(frame)
        if arr.shape == () and isinstance(arr.item(), (bytes, bytearray)):
            arr = np.frombuffer(arr.item(), dtype=np.uint8)
    if arr.ndim == 3:
        return arr[..., :3].astype(np.uint8, copy=False)

    import cv2

    decoded = cv2.imdecode(np.asarray(arr, dtype=np.uint8).reshape(-1), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Failed to decode compressed RGB frame")
    return decoded[:, :, ::-1].astype(np.uint8, copy=False)


def _read_camera(ep: h5py.File, camera_id: str, indices: np.ndarray) -> np.ndarray:
    path = f"cameras/{camera_id}/rgb"
    if path not in ep:
        raise KeyError(f"Missing camera dataset: {path}")
    dataset = ep[path]
    frames = [_decode_rgb_frame(dataset[int(i)]) for i in indices]
    return np.stack(frames, axis=0)


def _read_episode(
    path: Path,
    camera_map: dict[str, str],
    *,
    prompt: str | None = None,
    state_dim: int | None = None,
    action_dim: int | None = None,
    active_dof_info: ActiveDofInfo | None = None,
    truncate_at_homing: bool = False,
) -> EpisodePayload:
    with h5py.File(path, "r") as ep:
        indices = _valid_indices(ep)
        if len(indices) == 0:
            raise ValueError(f"No valid frames in {path}")

        # -- truncate at homing frame --------------------------------------
        if truncate_at_homing:
            ep_len_full = int(ep["robot"]["qpos"].shape[0])
            cutoff, _trunc_info = _find_homing_cutoff(
                ep, ep_len_full, indices,
                truncate=True,
            )
            if cutoff is not None:
                indices = indices[:cutoff]
                if len(indices) == 0:
                    raise ValueError(f"No valid frames after truncation in {path}")

        state = np.asarray(ep["robot"]["qpos"][:], dtype=np.float32)[indices]
        action = np.asarray(ep["action"]["commanded"][:], dtype=np.float32)[indices]
        if state.ndim != 2:
            raise ValueError(f"{path} has invalid state shape {state.shape}; expected (T, D)")
        if action.ndim != 2:
            raise ValueError(f"{path} has invalid action shape {action.shape}; expected (T, D)")

        if active_dof_info is not None:
            state = select_active(state, active_dof_info)
            action = select_active(action, active_dof_info)

        if state_dim is not None and state.shape[1] != state_dim:
            raise ValueError(f"{path} has state dim {state.shape[1]}, expected {state_dim}")
        if action_dim is not None and action.shape[1] != action_dim:
            raise ValueError(f"{path} has action dim {action.shape[1]}, expected {action_dim}")
        images = {name: _read_camera(ep, cam_id, indices) for name, cam_id in camera_map.items()}
        return EpisodePayload(
            task=_task_from_episode(ep, path.stem, prompt=prompt),
            state=state,
            action=action,
            images=images,
            fps=_fps_from_episode(ep),
        )


def _stats(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "mean": values.mean(axis=0).astype(float).tolist(),
        "std": values.std(axis=0).astype(float).tolist(),
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).astype(float).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).astype(float).tolist(),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_video(path: Path, frames: np.ndarray, fps: float) -> None:
    import subprocess

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to write LeRobot MP4 videos")

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected RGB video frames with shape (T, H, W, 3), got {frames.shape}")

    path.parent.mkdir(parents=True, exist_ok=True)
    _, height, width, _ = frames.shape
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
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
        str(path),
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if proc.stdin is None:
        raise RuntimeError("failed to open ffmpeg stdin")
    for frame in frames:
        proc.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
    proc.stdin.close()
    ret = proc.wait(timeout=300)
    if ret != 0:
        raise RuntimeError(f"ffmpeg exited with code {ret} (missing libx264?)")


def _feature_for_video(frames: np.ndarray, fps: float) -> dict[str, Any]:
    height, width, channels = frames.shape[1:]
    return {
        "dtype": "video",
        "shape": [int(height), int(width), int(channels)],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": float(fps),
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    }


def convert_dataset(
    input_dir: Path,
    output_dir: Path,
    *,
    camera_map: dict[str, str] | None = None,
    task_name: str | None = None,
    prompt: str | None = None,
    state_dim: int | None = None,
    action_dim: int | None = None,
    overwrite: bool = False,
    use_active_dof: bool = False,
    robot_key: str | None = None,
    truncate_at_homing: bool = False,
) -> Path:
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    camera_map = dict(camera_map or DEFAULT_CAMERA_MAP)
    hdf5_files = sorted(input_dir.glob("episode_*.hdf5"))
    if not hdf5_files:
        raise ValueError(f"No episode_*.hdf5 files found in {input_dir}")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)

    meta_dir = output_dir / "meta"
    data_dir = output_dir / "data"
    video_dir = output_dir / "videos"
    meta_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)

    episodes: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    task_to_index: dict[str, int] = {}
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    total_frames = 0
    first_payload: EpisodePayload | None = None

    adi: ActiveDofInfo | None = None
    if use_active_dof:
        adi = get_active_dof_info_for_hdf5(str(hdf5_files[0]))
        if adi is None:
            rk = robot_key or robot_key_from_hdf5(str(hdf5_files[0]))
            if rk is not None:
                adi = get_active_dof_info(rk)

    fallback_prompt = prompt or (task_name.replace("_", " ") if task_name else None)

    for episode_index, ep_path in enumerate(hdf5_files):
        payload = _read_episode(
            ep_path,
            camera_map,
            prompt=fallback_prompt,
            state_dim=state_dim,
            action_dim=action_dim,
            active_dof_info=adi,
            truncate_at_homing=truncate_at_homing,
        )
        if first_payload is None:
            first_payload = payload
        if payload.task not in task_to_index:
            task_to_index[payload.task] = len(task_to_index)
            tasks.append({"task_index": task_to_index[payload.task], "task": payload.task})
        task_index = task_to_index[payload.task]
        length = int(payload.state.shape[0])
        chunk = episode_index // CHUNK_SIZE
        episode_data_dir = data_dir / f"chunk-{chunk:03d}"
        episode_data_dir.mkdir(parents=True, exist_ok=True)

        for video_key, frames in payload.images.items():
            _write_video(
                video_dir / f"chunk-{chunk:03d}" / f"observation.images.{video_key}" / f"episode_{episode_index:06d}.mp4",
                frames,
                payload.fps,
            )

        rows = []
        for frame_idx in range(length):
            global_index = total_frames + frame_idx
            rows.append(
                {
                    "observation.state": payload.state[frame_idx].astype(np.float32).tolist(),
                    "action": payload.action[frame_idx].astype(np.float32).tolist(),
                    "timestamp": float(frame_idx / payload.fps),
                    ANNOTATION_KEY: int(task_index),
                    "task_index": int(task_index),
                    "episode_index": int(episode_index),
                    "frame_index": int(frame_idx),
                    "index": int(global_index),
                    "next.reward": 0.0,
                    "next.done": bool(frame_idx == length - 1),
                }
            )
        pd.DataFrame(rows).to_parquet(
            episode_data_dir / f"episode_{episode_index:06d}.parquet",
            index=False,
        )
        episodes.append({"episode_index": episode_index, "tasks": [task_index], "length": length})
        all_states.append(payload.state)
        all_actions.append(payload.action)
        total_frames += length

    assert first_payload is not None
    state_dim = int(first_payload.state.shape[1])
    action_dim = int(first_payload.action.shape[1])
    motors = [f"joint_{i}" for i in range(state_dim)]
    action_names = [f"action_{i}" for i in range(action_dim)]

    features: dict[str, Any] = {
        "observation.state": {"dtype": "float32", "shape": [state_dim], "names": motors},
        "action": {"dtype": "float32", "shape": [action_dim], "names": action_names},
        "timestamp": {"dtype": "float64", "shape": [1]},
        ANNOTATION_KEY: {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "next.reward": {"dtype": "float64", "shape": [1]},
        "next.done": {"dtype": "bool", "shape": [1]},
    }
    for video_key, frames in first_payload.images.items():
        features[f"observation.images.{video_key}"] = _feature_for_video(frames, first_payload.fps)

    info = {
        "codebase_version": "v2.0",
        "robot_type": "dex2bench_qpos",
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": len(episodes) * len(camera_map),
        "total_chunks": (len(episodes) + CHUNK_SIZE - 1) // CHUNK_SIZE,
        "chunks_size": CHUNK_SIZE,
        "fps": float(first_payload.fps),
        "splits": {"train": "0:100"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    modality = {
        "state": {"qpos": {"start": 0, "end": state_dim, "dtype": "float32"}},
        "action": {"qpos": {"start": 0, "end": action_dim, "dtype": "float32"}},
        "video": {
            video_key: {"original_key": f"observation.images.{video_key}"}
            for video_key in camera_map
        },
        "annotation": {"human.action.task_description": {}},
    }
    stats = {
        "observation.state": _stats(np.concatenate(all_states, axis=0)),
        "action": _stats(np.concatenate(all_actions, axis=0)),
    }

    (meta_dir / "info.json").write_text(json.dumps(info, indent=4), encoding="utf-8")
    (meta_dir / "modality.json").write_text(json.dumps(modality, indent=4), encoding="utf-8")
    (meta_dir / "stats.json").write_text(json.dumps(stats, indent=4), encoding="utf-8")
    _write_jsonl(meta_dir / "episodes.jsonl", episodes)
    _write_jsonl(meta_dir / "tasks.jsonl", tasks)
    return output_dir


def _default_output_dir(repo_id: str) -> Path:
    return Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id


def _parse_camera_map(items: list[str] | None, use_default: bool = True) -> dict[str, str]:
    camera_map = dict(DEFAULT_CAMERA_MAP) if use_default else {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Camera map entries must be NAME=DEX2BENCH_CAMERA, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Invalid camera map entry: {item!r}")
        camera_map[key] = value
    if not camera_map:
        raise ValueError("Camera map is empty; provide at least one --camera-map NAME=DEX2BENCH_CAMERA")
    return camera_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--repo-id", default=None, help="Local LeRobot repo id, e.g. local/task_name")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--task-name", default=None, help="dex2bench task name used for defaults and logs.")
    parser.add_argument("--prompt", default=None, help="Language prompt stored in tasks.jsonl.")
    parser.add_argument("--state-dim", type=int, default=None, help="Expected qpos dimension. Omit to infer.")
    parser.add_argument("--action-dim", type=int, default=None, help="Expected action dimension. Omit to infer.")
    parser.add_argument("--use-active-dof", action="store_true", help="Convert qpos/action to active-only DOF using robot metadata.")
    parser.add_argument("--robot-key", default=None, help="Robot key fallback when HDF5 metadata is missing.")
    parser.add_argument(
        "--camera-map",
        action="append",
        default=None,
        help="Override camera mapping entry, e.g. head=cam_overhead. Can be repeated.",
    )
    parser.add_argument(
        "--no-default-camera-map",
        action="store_true",
        help="Start from an empty camera map instead of the default 4-camera map.",
    )
    parser.add_argument("--mode", choices=["video"], default="video")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--truncate-at-homing", action="store_true", default=False,
                        help="Truncate each episode at the homing-start frame (exclusive)")
    args = parser.parse_args()
    if args.output_dir is None and args.repo_id is None:
        parser.error("Either --output-dir or --repo-id is required.")
    output_dir = args.output_dir or _default_output_dir(args.repo_id)
    written = convert_dataset(
        args.input_dir,
        output_dir,
        camera_map=_parse_camera_map(args.camera_map, use_default=not args.no_default_camera_map),
        task_name=args.task_name,
        prompt=args.prompt,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        overwrite=args.overwrite,
        use_active_dof=args.use_active_dof,
        robot_key=args.robot_key,
        truncate_at_homing=args.truncate_at_homing,
    )
    print(written)


if __name__ == "__main__":
    main()
