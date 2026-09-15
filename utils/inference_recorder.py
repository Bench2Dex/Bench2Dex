"""Lightweight HDF5 recorder for policy inference episodes.

Saves episodes to success/ (and failure/ when record_all=True) subdirectories.
Low-bitrate JPG camera frames, object poses/velocities, joint qpos, and actions are stored.
Directory is cleared at construction time so each run starts fresh unless
``clear=False`` is passed for chunked evaluation append mode.

Tune with env vars (defaults are tiny for fast review, not quality):
  DEX2BENCH_RECORD_RES=240    # image max dimension in pixels
  DEX2BENCH_RECORD_JPEG=25    # JPEG quality 1-100
  DEX2BENCH_RECORD_STRIDE=3   # record every N policy steps (1=20fps, 3≈7fps)
  DEX2BENCH_RECORD_CAMS=cam_overhead,cam_wrist_right  # comma-separated camera ids (empty=all)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from typing import Any, Dict, List

import numpy as np


logger = logging.getLogger(__name__)


class InferenceRecorder:
    """Record per-episode data and write HDF5 to success/ (and failure/ when record_all)."""

    def __init__(self, record_dir: str, record_all: bool = False, *, clear: bool = True):
        self._record_all = record_all
        self._success_dir = os.path.join(record_dir, "success")
        self._failure_dir = os.path.join(record_dir, "failure")
        if clear and os.path.isdir(self._success_dir):
            shutil.rmtree(self._success_dir)
        os.makedirs(self._success_dir, exist_ok=True)
        if record_all:
            if clear and os.path.isdir(self._failure_dir):
                shutil.rmtree(self._failure_dir)
            os.makedirs(self._failure_dir, exist_ok=True)
        self._dir = record_dir
        self._ep: Dict[str, Any] | None = None

        self._img_res = int(os.environ.get("DEX2BENCH_RECORD_RES", 240))
        self._jpg_quality = int(os.environ.get("DEX2BENCH_RECORD_JPEG", 25))
        self._stride = max(1, int(os.environ.get("DEX2BENCH_RECORD_STRIDE", 3)))
        _cam_filter = os.environ.get("DEX2BENCH_RECORD_CAMS", "").strip()
        self._cam_ids = set(c.strip() for c in _cam_filter.split(",") if c.strip()) if _cam_filter else None

        self._step = 0
        logger.info(
            "Inference recorder: dir=%s record_all=%s clear=%s res=%spx jpeg=%s stride=%s cams=%s",
            record_dir,
            record_all,
            clear,
            self._img_res,
            self._jpg_quality,
            self._stride,
            sorted(self._cam_ids) if self._cam_ids else "all",
        )

    # ------------------------------------------------------------------
    def start_episode(self, object_ids: List[str], camera_ids: List[str],
                      episode_idx: int = -1, *, joint_names: List[str] | None = None,
                      action_names: List[str] | None = None,
                      metadata: Dict[str, Any] | None = None) -> None:
        self._step = 0
        self._episode_idx = episode_idx
        _cam_ids = [c for c in camera_ids if self._cam_ids is None or c in self._cam_ids]
        self._ep = {
            "obj_ids": list(object_ids),
            "qpos": [],
            "action": [],
            "joint_names": list(joint_names) if joint_names else None,
            "action_names": list(action_names) if action_names else None,
            "metadata": dict(metadata or {}),
            "objects": {
                oid: {
                    "pose_world": [],
                    "lin_vel_world": [],
                    "ang_vel_world": [],
                }
                for oid in object_ids
            },
            "cameras": {cid: [] for cid in _cam_ids},
        }

    # ------------------------------------------------------------------
    def record_step(
        self,
        qpos: np.ndarray,
        action: np.ndarray,
        object_states: Dict[str, Dict[str, np.ndarray]],
        camera_frames: Dict[str, np.ndarray | None],
    ) -> None:
        if self._ep is None:
            return
        self._ep["qpos"].append(np.asarray(qpos, dtype=np.float32).copy())
        self._ep["action"].append(np.asarray(action, dtype=np.float32).copy())
        for oid in self._ep["obj_ids"]:
            s = object_states.get(oid)
            obj_buf = self._ep["objects"][oid]
            obj_buf["pose_world"].append(
                np.asarray(s.get("pose_world", np.zeros(7, dtype=np.float32)), dtype=np.float32).copy()
                if s is not None
                else np.zeros(7, dtype=np.float32)
            )
            obj_buf["lin_vel_world"].append(
                np.asarray(s.get("lin_vel_world", np.zeros(3, dtype=np.float32)), dtype=np.float32).copy()
                if s is not None
                else np.zeros(3, dtype=np.float32)
            )
            obj_buf["ang_vel_world"].append(
                np.asarray(s.get("ang_vel_world", np.zeros(3, dtype=np.float32)), dtype=np.float32).copy()
                if s is not None
                else np.zeros(3, dtype=np.float32)
            )
            # Save articulation joint data when present (doors, faucets, etc.)
            if s is not None and "qpos" in s and "joint_names" in s:
                if "qpos" not in obj_buf:
                    obj_buf["qpos"] = []
                    obj_buf["joint_names"] = list(s["joint_names"])
                obj_buf["qpos"].append(np.asarray(s["qpos"], dtype=np.float32).copy())
                if "qvel" in s and s["qvel"] is not None:
                    if "qvel" not in obj_buf:
                        obj_buf["qvel"] = []
                    obj_buf["qvel"].append(np.asarray(s["qvel"], dtype=np.float32).copy())
        if self._step % self._stride == 0:
            self._append_camera_frames(camera_frames)
        self._step += 1

    def current_object_ids(self) -> List[str]:
        return self._ep["obj_ids"] if self._ep is not None else []

    # ------------------------------------------------------------------
    def finish_episode(self, success: bool) -> str | None:
        """Write HDF5 to success/ or (if record_all) failure/ subdirectory.  Returns path or None."""
        if self._ep is None or len(self._ep.get("qpos", [])) == 0:
            self._ep = None
            return None
        # Decide target subdirectory
        if success:
            target_dir = self._success_dir
        elif self._record_all:
            target_dir = self._failure_dir
        else:
            self._ep = None
            return None
        path = self._episode_path(target_dir)
        self._write(path)
        self._ep = None
        return path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _episode_path(self, target_dir: str) -> str:
        if self._episode_idx >= 0:
            stem = f"episode_{self._episode_idx:06d}"
            path = os.path.join(target_dir, f"{stem}.hdf5")
            if not os.path.exists(path):
                return path

            suffix = 1
            while True:
                path = os.path.join(target_dir, f"{stem}_dup{suffix:03d}.hdf5")
                if not os.path.exists(path):
                    return path
                suffix += 1

        existing = len([f for f in os.listdir(target_dir) if f.endswith(".hdf5")])
        return os.path.join(target_dir, f"episode_{existing:06d}.hdf5")

    def _append_camera_frames(self, camera_frames: Dict[str, np.ndarray | None]) -> None:
        import cv2

        res = self._img_res
        for cid in self._ep["cameras"]:
            rgb = camera_frames.get(cid)
            if rgb is not None:
                h, w = rgb.shape[:2]
                if max(h, w) > res:
                    scale = res / max(h, w)
                    rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)),
                                     interpolation=cv2.INTER_LINEAR)
                _, jpg = cv2.imencode(
                    '.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, self._jpg_quality],
                )
                self._ep["cameras"][cid].append(jpg.tobytes())
            else:
                self._ep["cameras"][cid].append(b'')

    def _write(self, path: str) -> None:
        import h5py

        n = len(self._ep["qpos"])
        with h5py.File(path, "w") as hf:
            hf.create_dataset("meta/frame_count", data=np.int64(n))
            meta = hf.require_group("meta")
            str_dtype = h5py.string_dtype(encoding="utf-8")
            for key, value in self._ep.get("metadata", {}).items():
                meta.create_dataset(
                    key,
                    data=np.asarray(json.dumps(value, sort_keys=True), dtype=str_dtype),
                )
            hf.create_dataset("robot/qpos", data=np.array(self._ep["qpos"], dtype=np.float32))
            hf.create_dataset("action/commanded", data=np.array(self._ep["action"], dtype=np.float32))
            if self._ep.get("joint_names"):
                hf.create_dataset("robot/joint_names", data=np.asarray(self._ep["joint_names"], dtype=h5py.string_dtype()))
            if self._ep.get("action_names"):
                hf.create_dataset("action/action_names", data=np.asarray(self._ep["action_names"], dtype=h5py.string_dtype()))
            obj_grp = hf.create_group("objects")
            for oid in self._ep["obj_ids"]:
                obj_data = self._ep["objects"][oid]
                obj_grp.create_dataset(
                    f"{oid}/pose_world",
                    data=np.array(obj_data["pose_world"], dtype=np.float32),
                )
                obj_grp.create_dataset(
                    f"{oid}/lin_vel_world",
                    data=np.array(obj_data["lin_vel_world"], dtype=np.float32),
                )
                obj_grp.create_dataset(
                    f"{oid}/ang_vel_world",
                    data=np.array(obj_data["ang_vel_world"], dtype=np.float32),
                )
                # Save articulation joint data when present
                if "qpos" in obj_data and "joint_names" in obj_data:
                    obj_grp.create_dataset(
                        f"{oid}/qpos",
                        data=np.array(obj_data["qpos"], dtype=np.float32),
                    )
                    obj_grp.create_dataset(
                        f"{oid}/joint_names",
                        data=np.asarray(obj_data["joint_names"], dtype=h5py.string_dtype()),
                    )
                    if "qvel" in obj_data:
                        obj_grp.create_dataset(
                            f"{oid}/qvel",
                            data=np.array(obj_data["qvel"], dtype=np.float32),
                        )
            cam_grp = hf.create_group("cameras")
            for cid, jpgs in self._ep["cameras"].items():
                cg = cam_grp.create_group(cid)
                ds = cg.create_dataset("rgb", (len(jpgs),), dtype=h5py.vlen_dtype(np.dtype(np.uint8)))
                for i, jpg in enumerate(jpgs):
                    ds[i] = np.frombuffer(jpg, dtype=np.uint8) if jpg else np.zeros(0, dtype=np.uint8)
