from __future__ import annotations

import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np
from PIL import Image

_DEX2BENCH_ROOT = Path(__file__).resolve().parents[4]
if str(_DEX2BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEX2BENCH_ROOT))

from robots.active_dof_utils import (
    ActiveDofInfo,
    get_active_dof_info,
    get_active_dof_info_for_hdf5,
    robot_key_from_hdf5,
    select_active,
)

try:
    from diffusion_policy.dataset.obs_step_util import trim_observation_steps
except ModuleNotFoundError:  # allow direct file loading in lightweight tests
    from obs_step_util import trim_observation_steps

try:
    import torch
    from diffusion_policy.dataset.base_dataset import BaseImageDataset
except ModuleNotFoundError:  # allow lightweight unit tests without torch
    torch = None

    class BaseImageDataset:  # type: ignore[no-redef]
        pass


DEFAULT_CAMERA_OBS_MAP = {
    "cam_wrist_right": "right_cam",
    "cam_wrist_left": "left_cam",
    "cam_stereo_left": "stereo_left_cam",
    "cam_stereo_right": "stereo_right_cam",
}


@dataclass(frozen=True)
class _SampleIndex:
    episode_idx: int
    buffer_start: int
    buffer_end: int
    sample_start: int
    sample_end: int


def _as_bytes(raw) -> bytes:
    if isinstance(raw, bytes):
        return raw
    if isinstance(raw, np.ndarray):
        return raw.tobytes()
    return bytes(raw)


def _decode_rgb(raw) -> np.ndarray:
    data = _as_bytes(raw)
    try:
        import cv2

        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("cv2.imdecode returned None")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except ModuleNotFoundError:
        with Image.open(io.BytesIO(data)) as img:
            return np.asarray(img.convert("RGB"))


def _resize_rgb(img: np.ndarray, *, height: int, width: int) -> np.ndarray:
    if img.shape[0] == height and img.shape[1] == width:
        return img
    try:
        import cv2

        return cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    except ModuleNotFoundError:
        pil = Image.fromarray(img, mode="RGB")
        return np.asarray(pil.resize((width, height), resample=Image.BILINEAR))


def _fill_nan_action_rows(action: np.ndarray, qpos: np.ndarray) -> np.ndarray:
    action = action.astype(np.float32, copy=True)
    nan_mask = np.isnan(action).any(axis=1)
    if nan_mask.any():
        action[nan_mask] = qpos[nan_mask]
    return action


def _resolve_effective_length(h5file: h5py.File, total: int) -> int:
    """Return number of frames before homing starts.  Falls back to *total*.

    Follows the same pattern as ACT's ``_resolve_effective_length``
    (``policy/ACT/utils.py``): reads ``meta/homing_start_sim_step`` and
    maps it to a frame index via ``time/sim_step``.  The homing frame
    itself is **excluded**.

    Returns *total* unchanged when the homing metadata is missing,
    negative, or maps to a boundary position.
    """
    try:
        homing_start = h5file.get("meta/homing_start_sim_step")
        if homing_start is not None and "time/sim_step" in h5file:
            homing_val = int(homing_start[()])
            sim_steps = h5file["time/sim_step"][:]
            idx = int(np.searchsorted(sim_steps, homing_val))
            if 0 < idx < total:
                return idx
    except Exception:
        pass
    return total


def _make_val_mask(n_episodes: int, val_ratio: float) -> np.ndarray:
    mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0 or n_episodes <= 1:
        return mask
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    mask[-n_val:] = True
    return mask


def _build_indices(
    lengths: list[int],
    episode_mask: np.ndarray,
    *,
    sequence_length: int,
    pad_before: int,
    pad_after: int,
) -> list[_SampleIndex]:
    indices: list[_SampleIndex] = []
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)

    for episode_idx, length in enumerate(lengths):
        if not episode_mask[episode_idx]:
            continue
        min_start = -pad_before
        max_start = length - sequence_length + pad_after
        for start in range(min_start, max_start + 1):
            buffer_start = max(start, 0)
            buffer_end = min(start + sequence_length, length)
            sample_start = buffer_start - start
            sample_end = sequence_length - ((start + sequence_length) - buffer_end)
            indices.append(
                _SampleIndex(
                    episode_idx=episode_idx,
                    buffer_start=buffer_start,
                    buffer_end=buffer_end,
                    sample_start=sample_start,
                    sample_end=sample_end,
                )
            )
    return indices


class Dex2SceneHdf5Dataset(BaseImageDataset):
    """Read dex2scene replay HDF5 episodes directly for image diffusion policy."""

    def __init__(
        self,
        dataset_dir: str,
        horizon: int = 8,
        pad_before: int = 0,
        pad_after: int = 0,
        seed: int = 42,
        val_ratio: float = 0.0,
        batch_size: int = 8,
        max_train_episodes: int | None = None,
        camera_obs_map: Mapping[str, str] | None = None,
        image_shape: list[int] | tuple[int, int, int] | None = None,
        episode_mask: np.ndarray | None = None,
        _val_mask: np.ndarray | None = None,
        use_active_dof: bool = True,
        robot_key: str | None = None,
        n_obs_steps: int | None = None,
    ):
        super().__init__()
        self.dataset_dir = str(dataset_dir)
        self.horizon = int(horizon)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)
        self.seed = int(seed)
        self.val_ratio = float(val_ratio)
        self.batch_size = int(batch_size)
        self.camera_obs_map = dict(camera_obs_map or DEFAULT_CAMERA_OBS_MAP)
        self.camera_ids = list(self.camera_obs_map.keys())
        self.obs_camera_keys = list(self.camera_obs_map.values())
        self.n_obs_steps = None if n_obs_steps is None else int(n_obs_steps)
        self._use_active_dof = use_active_dof
        self._robot_key = robot_key
        self._adi: ActiveDofInfo | None = None

        if image_shape is None:
            self.image_shape = (3, 480, 640)
        else:
            if len(image_shape) != 3 or int(image_shape[0]) != 3:
                raise ValueError(f"image_shape must be [3, H, W], got {image_shape}")
            self.image_shape = tuple(int(x) for x in image_shape)

        self.hdf5_paths = sorted(str(p) for p in Path(self.dataset_dir).glob("episode_*.hdf5"))
        if not self.hdf5_paths:
            raise FileNotFoundError(f"No episode_*.hdf5 files found in {self.dataset_dir}")

        if self._use_active_dof:
            self._adi = get_active_dof_info_for_hdf5(self.hdf5_paths[0])
            if self._adi is None:
                rk = self._robot_key or robot_key_from_hdf5(self.hdf5_paths[0])
                if rk is not None:
                    self._adi = get_active_dof_info(rk)

        self.lengths = [self._episode_length(path) for path in self.hdf5_paths]
        if episode_mask is None:
            self.val_mask = _make_val_mask(len(self.hdf5_paths), self.val_ratio)
            episode_mask = ~self.val_mask
            if max_train_episodes is not None and int(max_train_episodes) < int(episode_mask.sum()):
                train_ids = np.nonzero(episode_mask)[0]
                rng = np.random.default_rng(self.seed)
                keep = rng.choice(train_ids, size=int(max_train_episodes), replace=False)
                episode_mask = np.zeros_like(episode_mask)
                episode_mask[keep] = True
        else:
            self.val_mask = np.array(_val_mask, dtype=bool) if _val_mask is not None else np.zeros(
                len(self.hdf5_paths), dtype=bool
            )
            episode_mask = np.array(episode_mask, dtype=bool)

        self.episode_mask = episode_mask
        self.indices = _build_indices(
            self.lengths,
            self.episode_mask,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
        )

    @staticmethod
    def _episode_length(path: str) -> int:
        with h5py.File(path, "r") as f:
            total = int(f["robot/qpos"].shape[0])
            return _resolve_effective_length(f, total)

    def get_validation_dataset(self) -> "Dex2SceneHdf5Dataset":
        return Dex2SceneHdf5Dataset(
            dataset_dir=self.dataset_dir,
            horizon=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            seed=self.seed,
            val_ratio=0.0,
            batch_size=self.batch_size,
            camera_obs_map=self.camera_obs_map,
            image_shape=self.image_shape,
            episode_mask=self.val_mask,
            _val_mask=self.val_mask,
            use_active_dof=self._use_active_dof,
            robot_key=self._robot_key,
            n_obs_steps=self.n_obs_steps,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def _read_camera_frame(self, rgb_ds, frame_idx: int) -> np.ndarray:
        _, height, width = self.image_shape
        img = _decode_rgb(rgb_ds[frame_idx])
        img = _resize_rgb(img, height=height, width=width)
        return np.moveaxis(img.astype(np.uint8), -1, 0)

    def _read_camera_sequence(self, f: h5py.File, camera_id: str, index: _SampleIndex) -> np.ndarray:
        out_len = self.n_obs_steps or self.horizon
        rgb_ds = f[f"cameras/{camera_id}/rgb"]
        p_start = max(index.sample_start, 0)
        p_end = min(index.sample_end, out_len)
        source_start = index.buffer_start + max(0, p_start - index.sample_start)
        source_end = source_start + max(0, p_end - p_start)

        first_frame = self._read_camera_frame(rgb_ds, index.buffer_start)
        out = np.empty((out_len, *first_frame.shape), dtype=first_frame.dtype)
        if index.sample_start > 0:
            out[:min(index.sample_start, out_len)] = first_frame

        if source_end > source_start:
            frames = [self._read_camera_frame(rgb_ds, i) for i in range(source_start, source_end)]
            out[p_start:p_end] = np.stack(frames, axis=0)

        if index.sample_end < out_len:
            out[index.sample_end:] = self._read_camera_frame(rgb_ds, index.buffer_end - 1)
        return out

    def _pad_sequence(self, arr: np.ndarray, index: _SampleIndex) -> np.ndarray:
        if index.sample_start == 0 and index.sample_end == self.horizon:
            return arr
        out = np.zeros((self.horizon, *arr.shape[1:]), dtype=arr.dtype)
        if index.sample_start > 0:
            out[:index.sample_start] = arr[0]
        if index.sample_end < self.horizon:
            out[index.sample_end:] = arr[-1]
        out[index.sample_start:index.sample_end] = arr
        return out

    def _sample_one(self, sample_idx: int) -> dict[str, np.ndarray]:
        index = self.indices[int(sample_idx)]
        path = self.hdf5_paths[index.episode_idx]
        result: dict[str, np.ndarray] = {}
        with h5py.File(path, "r") as f:
            qpos = f["robot/qpos"][index.buffer_start:index.buffer_end].astype(np.float32)
            action = f["action/commanded"][index.buffer_start:index.buffer_end].astype(np.float32)
            action = _fill_nan_action_rows(action, qpos)

            if self._adi is not None:
                qpos = select_active(qpos, self._adi)
                action = select_active(action, self._adi)

            result["state"] = self._pad_sequence(qpos, index)
            result["action"] = self._pad_sequence(action, index)
            for camera_id, obs_key in self.camera_obs_map.items():
                frames = self._read_camera_sequence(f, camera_id, index)
                result[obs_key] = frames
        return trim_observation_steps(
            result,
            n_obs_steps=self.n_obs_steps,
            obs_camera_keys=self.obs_camera_keys,
        )

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            raise NotImplementedError("slice indexing is not supported")
        if isinstance(idx, np.ndarray):
            samples = [self._sample_one(int(i)) for i in idx]
            return {key: np.stack([sample[key] for sample in samples], axis=0) for key in samples[0]}
        return self._sample_one(int(idx))

    def _load_all_state_action(self) -> tuple[np.ndarray, np.ndarray]:
        states = []
        actions = []
        for path in self.hdf5_paths:
            with h5py.File(path, "r") as f:
                total = int(f["robot/qpos"].shape[0])
                eff_len = _resolve_effective_length(f, total)
                qpos = f["robot/qpos"][:eff_len].astype(np.float32)
                action = f["action/commanded"][:eff_len].astype(np.float32)
                action = _fill_nan_action_rows(action, qpos)
                if self._adi is not None:
                    qpos = select_active(qpos, self._adi)
                    action = select_active(action, self._adi)
                states.append(qpos)
                actions.append(action)
        return np.concatenate(states, axis=0), np.concatenate(actions, axis=0)

    def get_normalizer(self, mode="limits", **kwargs):
        if torch is None:
            raise ModuleNotFoundError("torch is required to build the DP normalizer")
        from diffusion_policy.common.normalize_util import get_image_range_normalizer
        from diffusion_policy.model.common.normalizer import LinearNormalizer

        state, action = self._load_all_state_action()
        normalizer = LinearNormalizer()
        normalizer.fit(data={"action": action, "agent_pos": state}, last_n_dims=1, mode=mode, **kwargs)
        for obs_key in self.obs_camera_keys:
            normalizer[obs_key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self):
        if torch is None:
            raise ModuleNotFoundError("torch is required to return action tensors")
        _, action = self._load_all_state_action()
        return torch.from_numpy(action)

    def postprocess(self, samples, device):
        if torch is None:
            raise ModuleNotFoundError("torch is required to postprocess DP batches")

        obs = {}
        for obs_key in self.obs_camera_keys:
            obs[obs_key] = torch.from_numpy(samples[obs_key]).to(device, non_blocking=True).float() / 255.0
        obs["agent_pos"] = torch.from_numpy(samples["state"]).to(device, non_blocking=True).float()
        action = torch.from_numpy(samples["action"]).to(device, non_blocking=True).float()
        return {"obs": obs, "action": action}


__all__ = ["DEFAULT_CAMERA_OBS_MAP", "Dex2SceneHdf5Dataset"]
