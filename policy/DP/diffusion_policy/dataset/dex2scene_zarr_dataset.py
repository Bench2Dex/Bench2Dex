"""Zarr-backed dataset for dex2scene DP training.

Reads a preprocessed Zarr store (created by process_hdf5_to_zarr.py) with
zero JPEG-decode overhead.  Matches the HDF5 dataset's output dict format
exactly, so the workspace, normalizer, and deploy code need no changes.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.dataset.obs_step_util import trim_observation_steps
from diffusion_policy.model.common.normalizer import LinearNormalizer


# Camera keys stored in the Zarr (set at conversion time).
# Must match the obs keys expected by the policy shape_meta.
DEFAULT_OBS_CAMERA_KEYS = ["right_cam", "left_cam", "stereo_left_cam", "stereo_right_cam"]


class Dex2SceneZarrDataset(BaseImageDataset):
    """Dataset that reads pre-decoded camera frames from a Zarr store.

    Parameters
    ----------
    zarr_path : str
        Path to the ``.zarr`` directory produced by ``process_hdf5_to_zarr.py``.
    horizon : int
        Sequence length (= n_obs_steps for past + pred horizon).
    pad_before : int
        Pad frames before the episode start (0 = use boundary frame).
    pad_after : int
        Pad frames after the episode end.
    seed : int
        RNG seed for train/val split and episode down-sampling.
    val_ratio : float
        Fraction of episodes reserved for validation (0 = none).
    batch_size : int
        Expected batch size (used for the numba fast-path when idx is an
        ndarray — not strictly required but helps pre-allocation).
    max_train_episodes : int | None
        Down-sample training episodes to this count.
    obs_camera_keys : list[str] | None
        Zarr data keys that contain camera observations.  Defaults to
        ``["right_cam", "left_cam", "stereo_left_cam", "stereo_right_cam"]``.
    episode_mask : np.ndarray | None
        Explicit episode mask (used internally by get_validation_dataset).
    _val_mask : np.ndarray | None
        Explicit validation mask (used internally).
    """

    def __init__(
        self,
        zarr_path: str,
        horizon: int = 8,
        pad_before: int = 0,
        pad_after: int = 0,
        seed: int = 42,
        val_ratio: float = 0.0,
        batch_size: int = 8,
        max_train_episodes: int | None = None,
        obs_camera_keys: list[str] | None = None,
        n_obs_steps: int | None = None,
        episode_mask: np.ndarray | None = None,
        _val_mask: np.ndarray | None = None,
        **kwargs,  # absorb HDF5-only overrides (dataset_dir, image_shape, …)
    ) -> None:
        super().__init__()

        zarr_path = str(Path(zarr_path).expanduser())
        self.zarr_path = zarr_path
        self.batch_size = int(batch_size)
        self.obs_camera_keys = list(obs_camera_keys or DEFAULT_OBS_CAMERA_KEYS)
        self.n_obs_steps = None if n_obs_steps is None else int(n_obs_steps)

        # ------------------------------------------------------------------
        # Disk-backed Zarr (mode="r") — images stay on disk, only small
        # state/action arrays are loaded into memory by get_normalizer().
        # Each __getitem__ slices 8 frames × 4 cameras directly from disk
        # via zarr chunked reads — fast enough on SSD, zero RAM pressure.
        # ------------------------------------------------------------------
        self.replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode="r")

        # --- train / val split ---
        if episode_mask is None:
            val_mask = get_val_mask(
                n_episodes=self.replay_buffer.n_episodes,
                val_ratio=val_ratio,
                seed=seed,
            )
            train_mask = ~val_mask
            train_mask = downsample_mask(
                mask=train_mask,
                max_n=max_train_episodes,
                seed=seed,
            )
        else:
            train_mask = np.array(episode_mask, dtype=bool)
            val_mask = np.array(_val_mask, dtype=bool) if _val_mask is not None else np.zeros_like(train_mask)

        self.train_mask = train_mask
        self.val_mask = val_mask

        key_first_k = {}
        if self.n_obs_steps is not None:
            key_first_k = {key: self.n_obs_steps for key in [*self.obs_camera_keys, "state"]}

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
        )

        self.horizon = int(horizon)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)
        self.seed = int(seed)

    # ------------------------------------------------------------------
    # Validation split
    # ------------------------------------------------------------------
    def get_validation_dataset(self) -> "Dex2SceneZarrDataset":
        if not self.val_mask.any():
            return Dex2SceneZarrDataset(
                zarr_path=self.zarr_path,
                horizon=self.horizon,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                seed=self.seed,
                val_ratio=0.0,
                batch_size=self.batch_size,
                obs_camera_keys=self.obs_camera_keys,
                n_obs_steps=self.n_obs_steps,
                episode_mask=self.val_mask,
                _val_mask=self.val_mask,
            )
        key_first_k = {}
        if self.n_obs_steps is not None:
            key_first_k = {key: self.n_obs_steps for key in [*self.obs_camera_keys, "state"]}
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
            key_first_k=key_first_k,
        )
        val_set.train_mask = self.val_mask
        return val_set

    # ------------------------------------------------------------------
    # Normalizer (same logic as HDF5 dataset)
    # ------------------------------------------------------------------
    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        data = {
            "action": self.replay_buffer["action"][:],
            "agent_pos": self.replay_buffer["state"][:],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        for obs_key in self.obs_camera_keys:
            normalizer[obs_key] = get_image_range_normalizer()
        return normalizer

    # ------------------------------------------------------------------
    # Length
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.sampler)

    # ------------------------------------------------------------------
    # Get item
    # ------------------------------------------------------------------
    def __getitem__(self, idx) -> Dict[str, np.ndarray]:
        if isinstance(idx, slice):
            raise NotImplementedError("slice indexing is not supported")
        if isinstance(idx, np.ndarray):
            samples = [
                trim_observation_steps(
                    self.sampler.sample_sequence(int(i)),
                    n_obs_steps=self.n_obs_steps,
                    obs_camera_keys=self.obs_camera_keys,
                )
                for i in idx
            ]
            if not samples:
                return {}
            return {key: np.stack([s[key] for s in samples], axis=0) for key in samples[0]}
        return trim_observation_steps(
            self.sampler.sample_sequence(int(idx)),
            n_obs_steps=self.n_obs_steps,
            obs_camera_keys=self.obs_camera_keys,
        )

    # ------------------------------------------------------------------
    # Postprocess (same as HDF5: move to GPU, normalise images)
    # ------------------------------------------------------------------
    def postprocess(
        self,
        samples: Dict[str, np.ndarray],
        device: torch.device | str,
    ) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        obs: Dict[str, torch.Tensor] = {}
        for cam_key in self.obs_camera_keys:
            obs[cam_key] = (
                torch.from_numpy(samples[cam_key]).to(device, non_blocking=True).float() / 255.0
            )
        obs["agent_pos"] = (
            torch.from_numpy(samples["state"]).to(device, non_blocking=True).float()
        )
        action = torch.from_numpy(samples["action"]).to(device, non_blocking=True).float()
        return {"obs": obs, "action": action}


__all__ = ["Dex2SceneZarrDataset", "DEFAULT_OBS_CAMERA_KEYS"]
