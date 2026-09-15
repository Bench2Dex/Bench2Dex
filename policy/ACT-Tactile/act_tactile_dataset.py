"""ACT episode dataset with an aligned ordered TacMap observation."""

from __future__ import annotations

import cv2
import h5py
import numpy as np
import torch

from .act.utils import (
    ActiveDofInfo,
    Dex2BenchEpisodicDataset,
    _IMG_H,
    _IMG_W,
    _decode_jpeg,
    select_active,
)


def _read_site_names(h5file: h5py.File, path: str) -> tuple[str, ...]:
    metadata_path = "robot/tactile/meta/site_names"
    if metadata_path not in h5file:
        raise ValueError(f"{path}: missing {metadata_path}")

    raw_names = h5file[metadata_path][:]
    site_names = tuple(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in raw_names
    )
    if not site_names:
        raise ValueError(f"{path}: {metadata_path} must contain at least one site")
    if len(set(site_names)) != len(site_names):
        raise ValueError(f"{path}: {metadata_path} contains duplicate site names")
    if any(not name for name in site_names):
        raise ValueError(f"{path}: {metadata_path} contains an empty site name")
    return site_names


class ActTactileEpisodicDataset(Dex2BenchEpisodicDataset):
    """Keep ACT sampling exactly intact and append aligned raw TacMap frames.

    The output is a 5-tuple ``(image_data, qpos_data, action_data, is_pad, tactile)``
    where the first four values match :class:`Dex2BenchEpisodicDataset` under the
    same NumPy random state, and ``tactile`` is ``uint8 [num_sites, 1, H, W]``.
    """

    def __init__(
        self,
        hdf5_paths: list[str],
        camera_names: list[str],
        norm_stats: dict,
        max_action_len: int,
        active_dof_info: ActiveDofInfo | None = None,
    ):
        super().__init__(
            hdf5_paths,
            camera_names,
            norm_stats,
            max_action_len,
            active_dof_info=active_dof_info,
        )
        self.site_names: tuple[str, ...] | None = None
        self._validate_tactile_schema()

    def _validate_tactile_schema(self) -> None:
        for path in self.hdf5_paths:
            with h5py.File(path, "r") as h5file:
                site_names = _read_site_names(h5file, path)
                if self.site_names is None:
                    self.site_names = site_names
                elif site_names != self.site_names:
                    raise ValueError(
                        f"{path}: robot/tactile/meta/site_names order does not match "
                        "the first episode"
                    )

                frame_count = h5file["robot/qpos"].shape[0]
                for site_name in site_names:
                    map_path = f"robot/tactile/tacmap/{site_name}"
                    if map_path not in h5file:
                        raise ValueError(f"{path}: missing {map_path}")
                    tacmap = h5file[map_path]
                    if tacmap.ndim != 3:
                        raise ValueError(
                            f"{path}: {map_path} expected [T, H, W], got {tuple(tacmap.shape)}"
                        )
                    if tacmap.shape[0] != frame_count:
                        raise ValueError(
                            f"{path}: {map_path} has {tacmap.shape[0]} frames, "
                            f"but robot/qpos has {frame_count}"
                        )
                    if tacmap.shape[1] <= 0 or tacmap.shape[2] <= 0:
                        raise ValueError(f"{path}: {map_path} has an empty spatial dimension")

        assert self.site_names is not None

    def __getitem__(self, index: int):
        # ------------------------------------------------------------------
        # Replicate the parent class __getitem__ logic explicitly so that
        # the same start_ts is used for RGB, qpos, actions AND tactile, and
        # the HDF5 file is opened only once.  This avoids the fragile
        # numpy-random-state save/restore hack that depended on the parent
        # making exactly one np.random.randint call.
        # ------------------------------------------------------------------
        path = self.hdf5_paths[index]
        episode_len = self._episode_lengths[index]
        start_ts = np.random.randint(0, max(episode_len - 1, 1))

        with h5py.File(path, "r") as f:
            # --- qpos ---
            qpos = f["robot/qpos"][start_ts].astype(np.float32)

            # --- action chunk ---
            nan_mask = None
            if "action/commanded" in f:
                action_end = min(start_ts + self.max_action_len, episode_len)
                action = f["action/commanded"][start_ts:action_end].astype(np.float32)
                nan_mask = np.isnan(action).any(axis=1)
                if nan_mask.any():
                    action[nan_mask] = 0.0  # zero-fill; masked by is_pad below
            else:
                action_end = min(start_ts + self.max_action_len, episode_len)
                action = f["robot/qpos"][start_ts:action_end].astype(np.float32)

            if self._adi is not None:
                qpos = select_active(qpos, self._adi)
                action = select_active(action, self._adi)
            action_len = action.shape[0]

            # --- images ---
            image_dict = {}
            for act_name, d2b_key in zip(self.camera_names, self.d2b_cam_keys):
                raw = bytes(f[f"cameras/{d2b_key}/rgb"][start_ts])
                img = _decode_jpeg(raw)
                if img.shape[0] != _IMG_H or img.shape[1] != _IMG_W:
                    img = cv2.resize(img, (_IMG_W, _IMG_H), interpolation=cv2.INTER_LINEAR)
                image_dict[act_name] = img

            # --- tactile ---
            maps = [
                f[f"robot/tactile/tacmap/{site_name}"][start_ts]
                for site_name in self.site_names
            ]

        # --- zero-pad action chunk ---
        padded_action = np.zeros((self.max_action_len, action.shape[1]), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.ones(self.max_action_len, dtype=bool)
        is_pad[:action_len] = False
        if nan_mask is not None and nan_mask.any():
            is_pad[nan_mask] = True

        # --- assemble images ---
        all_cam_images = np.stack([image_dict[c] for c in self.camera_names], axis=0)
        image_data = torch.from_numpy(all_cam_images)
        image_data = torch.einsum("k h w c -> k c h w", image_data)

        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # --- normalize ---
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        # --- tactile: [num_sites, 1, H, W] uint8 ---
        tactile = torch.from_numpy(np.stack(maps, axis=0)).to(torch.uint8).unsqueeze(1)

        return image_data, qpos_data, action_data, is_pad, tactile
