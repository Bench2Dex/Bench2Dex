from __future__ import annotations

import os
import sys

_DEX2BENCH_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _DEX2BENCH_ROOT not in sys.path:
    sys.path.insert(0, _DEX2BENCH_ROOT)

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from robots.active_dof_utils import (
    ActiveDofInfo,
    get_active_dof_info,
    get_active_dof_info_for_hdf5,
    robot_key_from_hdf5,
    select_active,
)


# Camera key mapping: ACT name → dex2bench HDF5 group name
_D2B_CAM_MAP = {
    "cam_high":         "cam_overhead",
    "cam_right_wrist":  "cam_wrist_right",
    "cam_left_wrist":   "cam_wrist_left",
    "cam_chest":        "cam_chest",
    "cam_stereo_left":  "cam_stereo_left",
    "cam_stereo_right": "cam_stereo_right",
}
_IMG_W, _IMG_H = 640, 480


def _decode_jpeg(raw_bytes: bytes) -> np.ndarray:
    """Decode JPEG bytes to (H, W, 3) uint8 RGB array."""
    buf = np.frombuffer(raw_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _resolve_effective_length(h5file, total: int) -> int:
    """Return number of frames before homing starts. Falls back to *total*."""
    try:
        homing_start = h5file.get("meta/homing_start_sim_step")
        if homing_start is not None and "time/sim_step" in h5file:
            sim_steps = h5file["time/sim_step"][:]
            idx = int(np.searchsorted(sim_steps, homing_start[()]))
            if 0 < idx < total:
                return idx
    except Exception:
        pass
    return total


class Dex2BenchEpisodicDataset(torch.utils.data.Dataset):
    """Read dex2bench replay HDF5 files directly without pre-processing.

    Format mapping:
      robot/qpos[t]                            → qpos observation
      action/commanded[t:t+chunk]              → action targets (BC: commanded qpos)
      cameras/<d2b_cam>/rgb[t]                 → image observation (JPEG vlen bytes)
    """

    def __init__(
        self,
        hdf5_paths: list[str],
        camera_names: list[str],
        norm_stats: dict,
        max_action_len: int,
        active_dof_info: ActiveDofInfo | None = None,
    ):
        super().__init__()
        self.hdf5_paths = hdf5_paths
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.max_action_len = max_action_len
        self.d2b_cam_keys = [_D2B_CAM_MAP.get(c, c) for c in camera_names]
        self._adi = active_dof_info
        # Precompute effective episode lengths (truncated before homing)
        self._episode_lengths = []
        for p in hdf5_paths:
            with h5py.File(p, "r") as hf:
                self._episode_lengths.append(_resolve_effective_length(hf, hf["robot/qpos"].shape[0]))

    def __len__(self) -> int:
        return len(self.hdf5_paths)

    def __getitem__(self, index: int):
        path = self.hdf5_paths[index]
        with h5py.File(path, "r") as f:
            qpos_ds = f["robot/qpos"]
            episode_len = self._episode_lengths[index]
            start_ts = np.random.randint(0, max(episode_len - 1, 1))

            qpos = qpos_ds[start_ts].astype(np.float32)
            if "action/commanded" in f:
                action_end = min(start_ts + self.max_action_len, episode_len)
                action = f["action/commanded"][start_ts:action_end].astype(np.float32)
                nan_mask = np.isnan(action).any(axis=1)
                if nan_mask.any():
                    action[nan_mask] = f["robot/qpos"][start_ts:action_end][nan_mask]
            else:
                action_end = min(start_ts + self.max_action_len, episode_len)
                action = qpos_ds[start_ts:action_end].astype(np.float32)

            if self._adi is not None:
                qpos = select_active(qpos, self._adi)
                action = select_active(action, self._adi)
            action_len = action.shape[0]

            image_dict = {}
            for act_name, d2b_key in zip(self.camera_names, self.d2b_cam_keys):
                raw = bytes(f[f"cameras/{d2b_key}/rgb"][start_ts])
                img = _decode_jpeg(raw)
                if img.shape[0] != _IMG_H or img.shape[1] != _IMG_W:
                    img = cv2.resize(img, (_IMG_W, _IMG_H), interpolation=cv2.INTER_LINEAR)
                image_dict[act_name] = img

        padded_action = np.zeros((self.max_action_len, action.shape[1]), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.ones(self.max_action_len, dtype=bool)
        is_pad[:action_len] = False

        all_cam_images = np.stack([image_dict[c] for c in self.camera_names], axis=0)

        image_data = torch.from_numpy(all_cam_images)
        image_data = torch.einsum("k h w c -> k c h w", image_data)

        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats_dex2bench(
    hdf5_paths: list[str],
    active_dof_info: ActiveDofInfo | None = None,
) -> tuple[dict, int]:
    """Compute normalization statistics from dex2bench HDF5 files."""
    qpos_sum = None
    qpos_sq_sum = None
    action_sum = None
    action_sq_sum = None
    qpos_count = 0
    action_count = 0
    max_action_len = 0

    for path in hdf5_paths:
        with h5py.File(path, "r") as f:
            qpos = f["robot/qpos"][:].astype(np.float64)
            if "action/commanded" in f:
                commanded = f["action/commanded"][:].astype(np.float64)
                nan_mask = np.isnan(commanded).any(axis=1)
                if nan_mask.any():
                    commanded[nan_mask] = qpos[nan_mask]
            else:
                commanded = qpos

            eff_len = _resolve_effective_length(f, qpos.shape[0])
            if eff_len < qpos.shape[0]:
                qpos = qpos[:eff_len]
                commanded = commanded[:eff_len]

        if active_dof_info is not None:
            qpos = select_active(qpos, active_dof_info)
            commanded = select_active(commanded, active_dof_info)

        if qpos_sum is None:
            qpos_sum = np.zeros(qpos.shape[1], dtype=np.float64)
            qpos_sq_sum = np.zeros(qpos.shape[1], dtype=np.float64)
            action_sum = np.zeros(commanded.shape[1], dtype=np.float64)
            action_sq_sum = np.zeros(commanded.shape[1], dtype=np.float64)

        qpos_sum += qpos.sum(axis=0)
        qpos_sq_sum += np.square(qpos).sum(axis=0)
        action_sum += commanded.sum(axis=0)
        action_sq_sum += np.square(commanded).sum(axis=0)
        qpos_count += qpos.shape[0]
        action_count += commanded.shape[0]
        max_action_len = max(max_action_len, commanded.shape[0])

    qpos_mean = qpos_sum / qpos_count
    action_mean = action_sum / action_count
    # Higher floor for wuji hand (52 DOF, all independent, low variance in finger joints)
    var_floor = 0.0016 if (active_dof_info is not None and active_dof_info.full_dof >= 52) else 1e-4
    qpos_var = np.maximum(qpos_sq_sum / qpos_count - np.square(qpos_mean), var_floor)
    action_var = np.maximum(action_sq_sum / action_count - np.square(action_mean), var_floor)
    qpos_std = np.sqrt(qpos_var).astype(np.float32)
    action_std = np.sqrt(action_var).astype(np.float32)

    stats = {
        "action_mean": action_mean.astype(np.float32),
        "action_std": action_std,
        "qpos_mean": qpos_mean.astype(np.float32),
        "qpos_std": qpos_std,
    }
    return stats, max_action_len


def load_dex2bench_data(
    dataset_dir: str,
    camera_names: list[str],
    batch_size_train: int,
    batch_size_val: int,
    val_ratio: float = 0.0,
    only_success: bool = False,
    use_active_dof: bool = True,
    robot_key: str | None = None,
) -> tuple:
    """Build DataLoaders directly from dex2bench replay HDF5 files."""
    all_files = sorted(
        os.path.join(dataset_dir, fn)
        for fn in os.listdir(dataset_dir)
        if fn.endswith(".hdf5")
    )
    if not all_files:
        raise FileNotFoundError(f"No HDF5 files found in {dataset_dir}")

    if only_success:
        filtered = []
        for p in all_files:
            with h5py.File(p, "r") as f:
                if bool(f["meta/success"][()]):
                    filtered.append(p)
        skipped = len(all_files) - len(filtered)
        if skipped:
            print(f"[Dex2Bench] Skipped {skipped} failed episodes (only_success=True)")
        all_files = filtered

    if not all_files:
        raise RuntimeError("No episodes left after success filtering. Run label_success.py first.")

    print(f"[Dex2Bench] {len(all_files)} episodes from {dataset_dir}")

    adi: ActiveDofInfo | None = None
    if use_active_dof:
        adi = get_active_dof_info_for_hdf5(all_files[0])
        if adi is None and robot_key:
            adi = get_active_dof_info(robot_key)
        if adi is not None:
            print(f"[Dex2Bench] Active DOF mode: robot={adi.robot_key} "
                  f"full_dof={adi.full_dof} active_dof={adi.active_dof}"
                  f" (reindexed to HDF5 joint order)")

    train_paths = all_files
    val_paths = []
    if val_ratio > 0 and len(all_files) > 1:
        idx = np.random.permutation(len(all_files))
        n_val = min(max(1, int(len(all_files) * val_ratio)), len(all_files) - 1)
        val_paths = [all_files[i] for i in idx[:n_val]]
        train_paths = [all_files[i] for i in idx[n_val:]]
    print(f"[Dex2Bench] train episodes={len(train_paths)}, val episodes={len(val_paths)}")

    print("[Dex2Bench] Computing normalization stats...")
    norm_stats, max_action_len = get_norm_stats_dex2bench(all_files, active_dof_info=adi)
    print(f"[Dex2Bench] max_action_len={max_action_len}, state_dim={norm_stats['action_mean'].shape[0]}")

    train_dataset = Dex2BenchEpisodicDataset(train_paths, camera_names, norm_stats, max_action_len, active_dof_info=adi)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        prefetch_factor=1,
        persistent_workers=True,
    )
    val_dataloader = None
    if val_paths:
        val_dataset = Dex2BenchEpisodicDataset(val_paths, camera_names, norm_stats, max_action_len, active_dof_info=adi)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size_val,
            shuffle=False,
            pin_memory=True,
            num_workers=1,
            prefetch_factor=1,
            persistent_workers=True,
        )

    return train_dataloader, val_dataloader, norm_stats, True


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    return {k: v.detach().cpu() for k, v in d.items()}


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
