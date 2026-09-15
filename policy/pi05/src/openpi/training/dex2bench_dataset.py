"""dex2bench replay HDF5 dataset for OpenPI Pi0 training.

Reads dex2bench replay HDF5 files directly without any preprocessing step
(no `processed_data/`, no LeRobot conversion). Mirrors the approach used by
``policy/ACT/utils.py::Dex2BenchEpisodicDataset``.

HDF5 layout (produced by dex2bench teleop_record):
    robot/qpos              (T, 36) float32
    action/commanded        (T, 36) float32   # frame 0 may be NaN
    cameras/<cam>/rgb       (T,)    vlen uint8 JPEG bytes
    meta/success            scalar  bool      (optional)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

import cv2
import h5py
import numpy as np
import torch

_DEX2BENCH_ROOT = Path(__file__).resolve().parents[5]
if str(_DEX2BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEX2BENCH_ROOT))

from robots.active_dof_utils import (
    ActiveDofInfo,
    get_active_dof_info,
    get_active_dof_info_for_hdf5,
    robot_key_from_hdf5,
    select_active,
)


# Mapping: policy camera names -> dex2bench HDF5 group names.
# Four-camera layout: stereo pair + dual wrist cameras.
DEX2BENCH_CAM_MAP: dict[str, str] = {
    "cam_stereo_left":  "cam_stereo_left",
    "cam_stereo_right": "cam_stereo_right",
    "cam_wrist_left":   "cam_wrist_left",
    "cam_wrist_right":  "cam_wrist_right",
}


def _decode_jpeg(raw_bytes: bytes) -> np.ndarray:
    """Decode JPEG bytes to (H, W, 3) uint8 RGB array."""
    buf = np.frombuffer(raw_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _valid_indices(h5file: h5py.File) -> np.ndarray:
    """Return GR00T-style valid frame indices for qpos/action frames."""
    if "robot" not in h5file or "qpos" not in h5file["robot"]:
        raise KeyError("Episode is missing /robot/qpos")
    total = int(h5file["robot/qpos"].shape[0])
    frame_valid = (
        np.asarray(h5file["frame_valid"][:], dtype=bool)
        if "frame_valid" in h5file
        else np.ones(total, dtype=bool)
    )
    action_valid = (
        np.asarray(h5file["action/action_valid"][:], dtype=bool)
        if "action/action_valid" in h5file
        else np.ones(total, dtype=bool)
    )
    if "action/commanded" not in h5file:
        raise ValueError("Episode is missing /action/commanded")
    action = np.asarray(h5file["action/commanded"][:], dtype=np.float32)
    if action.shape[0] != total:
        raise ValueError(f"/action/commanded length {action.shape[0]} does not match /robot/qpos length {total}")
    finite_action = np.isfinite(action).all(axis=1)
    return np.where(frame_valid & action_valid & finite_action)[0].astype(np.int64)


def _truncate_valid_indices_at_homing(
    h5file: h5py.File,
    total: int,
    valid_indices: np.ndarray,
    *,
    min_frames: int = 1,
) -> np.ndarray:
    """Apply GR00T-style homing cutoff in valid-index space."""
    homing_start = h5file.get("meta/homing_start_sim_step")
    if homing_start is None or "time/sim_step" not in h5file:
        return valid_indices
    homing_val = int(homing_start[()])
    if homing_val < 0:
        return valid_indices

    sim_steps = h5file["time/sim_step"][:]
    frame_idx = int(np.searchsorted(sim_steps, homing_val))
    if not (min_frames <= frame_idx < total):
        return valid_indices

    cutoff = int(np.searchsorted(valid_indices, frame_idx, side="left"))
    cutoff = max(cutoff, min_frames)
    cutoff = min(cutoff, len(valid_indices))
    if cutoff >= len(valid_indices):
        return valid_indices
    return valid_indices[:cutoff]


def list_episode_files(dataset_dir: str, only_success: bool = False) -> list[str]:
    """List ``episode_*.hdf5`` files in ``dataset_dir`` (sorted)."""
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    files = sorted(
        os.path.join(dataset_dir, fn)
        for fn in os.listdir(dataset_dir)
        if fn.endswith(".hdf5")
    )
    if not files:
        raise FileNotFoundError(f"No HDF5 files in {dataset_dir}")
    if only_success:
        kept = []
        for p in files:
            with h5py.File(p, "r") as f:
                if "meta/success" in f and bool(f["meta/success"][()]):
                    kept.append(p)
        if not kept:
            raise RuntimeError(f"No successful episodes in {dataset_dir}")
        files = kept
    return files


class Dex2BenchPi0Dataset(torch.utils.data.Dataset):
    """Direct-read dataset for dex2bench → Pi0.

    Each ``__getitem__`` returns a dict with the keys expected by
    ``Dex2BenchInputs`` (see ``openpi.policies.dex2bench_policy``):

        {
            "state":   (state_dim,) float32,
            "images":  {cam_name: (H, W, 3) uint8 RGB},
            "actions": (action_horizon, state_dim) float32,
            "prompt":  str,
        }

    No image resize is performed here; ``ResizeImages`` (in
    ``ModelTransformFactory``) will handle resizing to 224x224.
    """

    def __init__(
        self,
        hdf5_paths: Sequence[str],
        action_horizon: int,
        prompt: str,
        camera_names: Sequence[str] = (
            "cam_stereo_left",
            "cam_stereo_right",
            "cam_wrist_left",
            "cam_wrist_right",
        ),
        use_active_dof: bool = True,
        robot_key: str | None = None,
    ) -> None:
        super().__init__()
        if not hdf5_paths:
            raise ValueError("hdf5_paths is empty")
        self.paths = list(hdf5_paths)
        self.action_horizon = int(action_horizon)
        self.prompt = prompt
        self.camera_names = list(camera_names)

        self._adi: ActiveDofInfo | None = None
        if use_active_dof:
            self._adi = get_active_dof_info_for_hdf5(self.paths[0])
            if self._adi is None:
                rk = robot_key or robot_key_from_hdf5(self.paths[0])
                if rk is not None:
                    self._adi = get_active_dof_info(rk)

        # GR00T-style indexing: filter invalid frames, truncate valid indices
        # before homing, and keep only starts with a full action horizon.
        kept_paths: list[str] = []
        self.valid_indices: list[np.ndarray] = []
        self.lengths: list[int] = []
        self.sample_lengths: list[int] = []
        max_delta = max(self.action_horizon - 1, 0)
        for p in self.paths:
            with h5py.File(p, "r") as f:
                total = int(f["robot/qpos"].shape[0])
                valid = _valid_indices(f)
                if valid.size == 0:
                    continue
                valid = _truncate_valid_indices_at_homing(f, total, valid)
                if valid.size == 0:
                    continue
                sample_length = max(int(valid.size) - max_delta, 0)
                if sample_length <= 0:
                    continue
                kept_paths.append(p)
                self.valid_indices.append(valid)
                self.lengths.append(int(valid.size))
                self.sample_lengths.append(sample_length)
        if not kept_paths:
            raise RuntimeError("No usable dex2bench samples after valid-frame filtering and homing truncation")
        self.paths = kept_paths
        self._cum = np.cumsum(self.sample_lengths)
        self._total = int(self._cum[-1])

    def __len__(self) -> int:
        return self._total

    def _locate(self, idx: int) -> tuple[int, int]:
        """Map a flat index to (episode_idx, time_idx)."""
        idx = idx % self._total
        ep = int(np.searchsorted(self._cum, idx, side="right"))
        prev = int(self._cum[ep - 1]) if ep > 0 else 0
        t = idx - prev
        return ep, t

    def __getitem__(self, idx: int) -> dict:
        ep, t = self._locate(idx)
        path = self.paths[ep]
        valid = self.valid_indices[ep]
        frame_idx = int(valid[t])
        with h5py.File(path, "r") as f:
            qpos_ds = f["robot/qpos"]
            qpos = np.asarray(qpos_ds[frame_idx], dtype=np.float32)

            valid_last = int(valid[-1])
            end = min(frame_idx + self.action_horizon, valid_last + 1)
            act = np.asarray(f["action/commanded"][frame_idx:end], dtype=np.float32)

            # This should only trigger for unusual valid-index gaps; it mirrors
            # GR00T's clipping to the last valid frame for future deltas.
            if act.shape[0] < self.action_horizon:
                pad_n = self.action_horizon - act.shape[0]
                if act.shape[0] == 0:
                    act = np.tile(qpos[None, :], (self.action_horizon, 1))
                else:
                    act = np.concatenate([act, np.tile(act[-1:], (pad_n, 1))], axis=0)

            images: dict[str, np.ndarray] = {}
            for cam_name in self.camera_names:
                d2b_key = DEX2BENCH_CAM_MAP.get(cam_name, cam_name)
                rgb_ds = f[f"cameras/{d2b_key}/rgb"]
                raw = bytes(rgb_ds[frame_idx])
                images[cam_name] = _decode_jpeg(raw)  # (H, W, 3) uint8 RGB

        if self._adi is not None:
            qpos = select_active(qpos, self._adi)
            act = select_active(act, self._adi)

        return {
            "state":   qpos.astype(np.float32),
            "images":  images,
            "actions": act.astype(np.float32),
            "prompt":  self.prompt,
        }
