# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-embodiment HDF5 dataset for GR00T XE.

从 replay-generalization 读 RGB + 手部关节, 从 arm_ee_trajectories.hdf5 读机械臂末端位姿,
组合成统一的 64D state/action 空间。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from gr00t_hdf5_dataset import Dex2BenchHDF5Dataset, build_hdf5_dataset as _build_hdf5_dataset
from robots.active_dof_utils import select_active
from xe_hand_mapping import (
    build_name_to_slot_maps,
    robot_key_to_hand_name,
    validate_hand_coverage,
)
from xe_norm_stats import (
    ARTIFACT_NAME,
    apply_to_transform,
    check_task_against_artifact,
    load_artifact,
)


# Set on first use by the GR00T_XE_TILE_ACTION ablation switch, so the training
# log carries proof the switch actually reached the dataloader.
_TILE_MODE_ANNOUNCED = False


def _take_rows(dset: "h5py.Dataset", idxs: np.ndarray) -> np.ndarray:
    """Read ``dset`` rows by fancy index, tolerating repeats and disorder.

    h5py rejects duplicate and out-of-order indices alike ("Indexing elements
    must be in increasing order"), but the action chunk legitimately repeats the
    episode's last valid frame whenever the 16-frame horizon runs past its end
    (the ``np.clip`` in ``get_step_data``).  The single-dataset path never
    reaches those positions -- ``Dex2BenchHDF5Dataset._global_index`` reserves
    ``max_delta`` tail frames -- but ``MixtureDataset.sample_step`` picks
    ``rng.choice(trajectory_lengths)``, which does, so mixture training would
    otherwise die in the dataloader.  Read the distinct rows and scatter them
    back.
    """
    uniq = np.unique(idxs)
    return np.asarray(dset[uniq], dtype=np.float32)[np.searchsorted(uniq, idxs)]


class CrossEmbodimentHDF5Dataset(Dex2BenchHDF5Dataset):
    """从 replay HDF5 + arm_ee_trajectories.hdf5 构建统一 64D 数据."""

    def __init__(self, *args, **kwargs):
        # Initialize hand mapping BEFORE super().__init__() because
        # _compute_statistics (called during base init) needs it.
        self._xe_robot_key: str | None = None
        self._ee_file: h5py.File | None = None
        # Set only by build_xe_norm_stats.py, which has to measure every task's
        # statistics before the artifact it is about to write exists.  It is not
        # a fallback: a dataset built this way has no per-robot normalizers and
        # set_transforms_metadata refuses to let it train.
        self._xe_stats_only = bool(kwargs.pop("xe_stats_only", False))
        self._xe_task_stats: dict | None = None
        self._unified_hand_mask: np.ndarray | None = None
        self._hand_joint_to_slot: dict[str, int] = {}
        self._left_hand_joint_to_slot: dict[str, int] = {}
        self._first_joint_names: list[str] | None = None

        # Peek at first episode to get robot_key + real joint names, then build
        # AND validate the hand mapping before super().__init__() triggers
        # _compute_statistics.  validate_hand_coverage raises loudly if any
        # authored hand joint does not resolve (memory: no-silent-error-skipping).
        input_dir = kwargs.get("input_dir") or (args[0] if args else None)
        if input_dir is not None:
            hdf5_files = sorted(Path(input_dir).glob("episode_*.hdf5"))
            if hdf5_files:
                with h5py.File(hdf5_files[0], "r") as ep:
                    rk = ep.get("meta/robot_key")
                    if rk is not None:
                        val = rk[()]
                        self._xe_robot_key = val.decode("utf-8") if isinstance(val, bytes) else str(val)
                    self._first_joint_names = [
                        n.decode() if isinstance(n, bytes) else str(n)
                        for n in ep["robot/joint_names"][:]
                    ]
                if self._xe_robot_key:
                    self._build_hand_mapping(self._first_joint_names)

        self._state_dim = 64
        self._action_dim = 64
        # Don't pass state_dim/action_dim to base — let it compute raw dims,
        # then we override to 64 after init.
        if "state_dim" in kwargs:
            kwargs.pop("state_dim")
        if "action_dim" in kwargs:
            kwargs.pop("action_dim")

        # XE uses 64D unified space — do NOT apply active DOF selection, because
        # the base would then set _state_dim = _adi.active_dof (e.g. 52) instead
        # of 64.  We keep _adi=None and let the base use raw dims.
        if "active_dof_info" in kwargs:
            kwargs.pop("active_dof_info")

        # Open EE file BEFORE super().__init__() so _compute_statistics can use it
        self._ee_file: h5py.File | None = None
        if input_dir is not None:
            ee_path = Path(input_dir).parent / "arm_ee_trajectories.hdf5"
            if ee_path.is_file():
                self._ee_file = h5py.File(ee_path, "r")
                print(f"[GR00T_XE] Loaded EE trajectories from {ee_path}")
            else:
                raise FileNotFoundError(
                    f"[GR00T_XE] arm_ee_trajectories.hdf5 not found: {ee_path}\n"
                    f"Run convert_ee_trajectories.py first to generate it."
                )

        super().__init__(*args, **kwargs)

        # Override to 64D after base init uses raw dims
        self._state_dim = 64
        self._action_dim = 64

        # Rebuild metadata with 64D shapes (base used raw dims like 52)
        # Re-run _compute_statistics to get proper 64D stats
        stats = self._compute_statistics()
        self._xe_task_stats = stats
        self._metadata = self._build_metadata(stats)
        self._transforms.set_metadata(self._metadata)

        if self._xe_stats_only:
            print(
                f"[GR00T_XE] xe_stats_only=True: this dataset was NOT given per-robot "
                f"normalization from {ARTIFACT_NAME}. It exists to be measured, not to "
                f"be trained on.",
                flush=True,
            )
            return

        # Normalization does NOT come from `stats`.  dims 0:12 are arm EE in the
        # shared world frame, so they use statistics pooled over the whole corpus;
        # dims 12:56 are hand slots and use THIS robot's own statistics; dims
        # 56:64 are pad and stay 0.  `stats` is kept only to check that the
        # artifact still matches this task's data (a stale artifact -- e.g. one
        # built before the EE trajectories were re-exported -- raises here).
        _art = load_artifact()
        check_task_against_artifact(
            stats, self._xe_robot_key, self._input_dir, _art,
            context=f"dataset {self._input_dir}",
        )
        apply_to_transform(
            self._transforms, self._metadata, self._xe_robot_key, artifact=_art,
            context=f"dataset {self._input_dir}",
        )

    def set_transforms_metadata(self, metadata) -> None:
        """Re-apply the artifact after ``LeRobotMixtureDataset``'s merge.

        ``update_metadata`` merges every member dataset's statistics with
        ``np.min`` / ``np.max`` and pushes the result back into each dataset --
        exactly the normalization this dataset must not use (every robot that
        does not author a slot reports 0/0, which drags that slot's range toward
        zero).  The merged metadata still carries the modality configs, so keep
        those and replace only the statistics.
        """
        # No getattr default: __init__ always sets this, so a missing attribute
        # is a real bug and should raise rather than read as "False".
        if self._xe_stats_only:
            raise RuntimeError(
                "[GR00T_XE] a xe_stats_only dataset reached training "
                "(set_transforms_metadata was called on it). It has no per-robot "
                f"normalization from {ARTIFACT_NAME} and would train with the "
                f"mixture-merged statistics."
            )
        self._metadata = metadata
        apply_to_transform(
            self._transforms, metadata, self._xe_robot_key,
            context=f"dataset {self._input_dir} (post-merge)",
        )

    def _compute_statistics(self) -> dict[str, dict[str, list[float]]]:
        """Compute 64D statistics on the unified state/action space.

        Instead of reading raw HDF5 (which has per-robot dims like 54), we
        iterate over all episodes and build the 64D unified representation,
        then compute per-dim min/max/mean/std for the normalizer.
        """
        if not self._episode_paths:
            return super()._compute_statistics()

        all_state: list[np.ndarray] = []
        all_action: list[np.ndarray] = []

        ee_path = Path(self._input_dir).parent / "arm_ee_trajectories.hdf5"
        ee_f = h5py.File(ee_path, "r")

        for ep_idx, (p, valid) in enumerate(zip(self._episode_paths, self._episode_valid_indices)):
            ep_stem = self._get_ee_ep_stem(p)
            with h5py.File(p, "r") as ep:
                raw_qpos_all = np.asarray(ep["robot"]["qpos"][valid], dtype=np.float32)
                raw_action_all = np.asarray(ep["action"]["commanded"][valid], dtype=np.float32)
                joint_names = [n.decode() if isinstance(n, bytes) else str(n)
                               for n in ep["robot/joint_names"][:]]
                if self._first_joint_names is not None and joint_names != self._first_joint_names:
                    raise ValueError(
                        f"[GR00T_XE] episode {p.name} has joint_names that differ from the first "
                        f"episode in {self._input_dir}. A replay directory must contain a single "
                        f"robot (this would otherwise silently scramble the unified hand mapping)."
                    )

            if self._adi is not None:
                raw_qpos_all = select_active(raw_qpos_all, self._adi)
                raw_action_all = select_active(raw_action_all, self._adi)

            ee_qpos_all = {side: np.asarray(ee_f[f"{ep_stem}/{side}_ee_qpos"][valid], dtype=np.float32)
                           for side in ("right", "left")}
            ee_action_all = {side: np.asarray(ee_f[f"{ep_stem}/{side}_ee_action"][valid], dtype=np.float32)
                             for side in ("right", "left")}

            for t in range(len(valid)):
                s, a, _ = self._build_unified(
                    {side: arr[t] for side, arr in ee_qpos_all.items()},
                    {side: arr[t] for side, arr in ee_action_all.items()},
                    raw_qpos_all[t], raw_action_all[t], joint_names,
                )
                all_state.append(s)
                all_action.append(a)

        ee_f.close()
        cat_state = np.stack(all_state, axis=0)
        cat_action = np.stack(all_action, axis=0)

        def _per_dim(arr: np.ndarray) -> dict[str, list[float]]:
            return {
                "mean": arr.mean(axis=0).astype(float).tolist(),
                "std": arr.std(axis=0).astype(float).tolist(),
                "min": arr.min(axis=0).astype(float).tolist(),
                "max": arr.max(axis=0).astype(float).tolist(),
                "q01": np.quantile(arr, 0.01, axis=0).astype(float).tolist(),
                "q99": np.quantile(arr, 0.99, axis=0).astype(float).tolist(),
            }

        return {
            "state": {"qpos": _per_dim(cat_state)},
            "action": {"qpos": _per_dim(cat_action)},
        }

    def _build_hand_mapping(self, joint_names: list[str]) -> None:
        """Build + validate the hand joint→slot mapping from the single source.

        Joint-name→slot tables come from xe_hand_mapping (single source shared
        with ik_arm_converter so train/inference cannot drift).  Slots are the
        YAML slots +12 → unified layout (right 12-33, left 34-55).  Coverage is
        validated against the real episode joint names and RAISES on any gap, so
        we never train with hand columns silently stuck at 0.
        """
        hand_name = robot_key_to_hand_name(self._xe_robot_key)
        right_map, left_map = build_name_to_slot_maps(hand_name)

        stats = validate_hand_coverage(self._xe_robot_key, joint_names, context="dataset")
        print(
            f"[GR00T_XE] robot={self._xe_robot_key} hand={hand_name} "
            f"hand_joints right={stats['right']}/{stats['right_total']} "
            f"left={stats['left']}/{stats['left_total']} "
            f"-> {stats['hand_slots_filled']} unified hand slots filled"
        )

        self._hand_joint_to_slot = right_map
        self._left_hand_joint_to_slot = left_map

        mask = np.zeros(64, dtype=np.float32)
        mask[0:12] = 1.0  # arm EE
        for slot in right_map.values():
            mask[slot] = 1.0
        for slot in left_map.values():
            mask[slot] = 1.0
        self._unified_hand_mask = mask

    def _init_ee_file(self) -> None:
        """Open arm_ee_trajectories.hdf5 (must exist)."""
        ee_path = Path(self._input_dir).parent / "arm_ee_trajectories.hdf5"
        if not ee_path.is_file():
            raise FileNotFoundError(
                f"[GR00T_XE] arm_ee_trajectories.hdf5 not found: {ee_path}\n"
                f"Run convert_ee_trajectories.py first to generate it."
            )
        self._ee_file = h5py.File(ee_path, "r")
        print(f"[GR00T_XE] Loaded EE trajectories from {ee_path}")
        print(f"[GR00T_XE] robot={self._xe_robot_key}, hand={robot_key_to_hand_name(self._xe_robot_key)}, "
              f"right_slots={len(self._hand_joint_to_slot)}, left_slots={len(self._left_hand_joint_to_slot)}")

    def _build_unified(self, ee_qpos: np.ndarray, ee_action: np.ndarray,
                       hand_qpos: np.ndarray, hand_action: np.ndarray,
                       joint_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """构建统一 64D state, action, mask."""
        state = np.zeros(64, dtype=np.float32)
        action = np.zeros(64, dtype=np.float32)

        # arm ee (前 12 维)
        state[0:6] = ee_qpos["right"]
        state[6:12] = ee_qpos["left"]
        action[0:6] = ee_action["right"]
        action[6:12] = ee_action["left"]

        # hand: 映射到语义槽
        for i, name in enumerate(joint_names):
            if name in self._hand_joint_to_slot:
                slot = self._hand_joint_to_slot[name]
                state[slot] = hand_qpos[i]
                action[slot] = hand_action[i]
            elif name in self._left_hand_joint_to_slot:
                slot = self._left_hand_joint_to_slot[name]
                state[slot] = hand_qpos[i]
                action[slot] = hand_action[i]

        return state, action, self._unified_hand_mask.copy()

    def _get_ee_data(self, ep_stem: str, raw_idx: int) -> dict:
        """从 arm_ee_trajectories.hdf5 读取指定帧的 EE 数据."""
        ep_grp = self._ee_file[ep_stem]
        ee = {}
        for side in ("right", "left"):
            key = f"{side}_ee_qpos"
            ee[side] = np.asarray(ep_grp[key][raw_idx], dtype=np.float32)
        return ee

    def _get_ee_action_data(self, ep_stem: str, raw_idx: int) -> dict:
        """从 arm_ee_trajectories.hdf5 读取 EE action."""
        ep_grp = self._ee_file[ep_stem]
        ee = {}
        for side in ("right", "left"):
            key = f"{side}_ee_action"
            ee[side] = np.asarray(ep_grp[key][raw_idx], dtype=np.float32)
        return ee

    def _get_ee_frames(self, ep_stem: str,
                       frame_idxs: np.ndarray) -> tuple[list[dict], list[dict]]:
        """Batch form of _get_ee_data/_get_ee_action_data over several frames.

        Reading the 16 horizon frames individually costs 32 small h5py reads per
        sample; one slice per array is 4 and keeps the dataloader from being
        dominated by per-row reads.
        """
        ep_grp = self._ee_file[ep_stem]
        qpos = {side: _take_rows(ep_grp[f"{side}_ee_qpos"], frame_idxs)
                for side in ("right", "left")}
        action = {side: _take_rows(ep_grp[f"{side}_ee_action"], frame_idxs)
                  for side in ("right", "left")}
        return (
            [{side: qpos[side][i] for side in qpos} for i in range(len(frame_idxs))],
            [{side: action[side][i] for side in action} for i in range(len(frame_idxs))],
        )

    def _get_ee_ep_stem(self, ep_path: Path) -> str:
        """Get the EE file group name for an episode path.

        arm_ee_trajectories.hdf5 stores one group per episode, named without
        the ``_1`` variant suffix that the replay-generalization directory uses
        for the second 25-episode half.  Strip the suffix for lookup.
        """
        stem = ep_path.stem  # e.g. episode_000014_1
        # Drop trailing _1 variant suffix
        if stem.endswith("_1") and stem.startswith("episode_"):
            stem = stem[:-2]
        return stem

    def get_step_data(self, ep_idx: int, local_valid_idx: int) -> dict[str, Any]:
        result = super().get_step_data(ep_idx, local_valid_idx)

        ep_path = self._episode_paths[ep_idx]
        ep_stem = self._get_ee_ep_stem(ep_path)  # strip _1 suffix

        # 获取原始帧索引
        valid_idxs = self._episode_valid_indices[ep_idx]
        raw_idx = int(valid_idxs[local_valid_idx])

        # Horizon frames for the action chunk.  The base class resolves the same
        # indices from the action modality's delta_indices, but builds them in
        # raw per-robot joint space; the unified 64D space is assembled here, so
        # the indices have to be resolved again.
        action_mcfg = self._modality_configs.get("action")
        if action_mcfg is None:
            raise RuntimeError(
                "[GR00T_XE] no 'action' modality config; cannot build the action chunk"
            )
        delta = np.asarray(action_mcfg.delta_indices, dtype=np.intp)
        if int(delta[0]) != 0:
            raise RuntimeError(
                f"[GR00T_XE] action delta_indices must start at 0 (chunk slot 0 is the "
                f"observation frame the state is read from), got {delta[:4].tolist()}"
            )
        last_valid = int(valid_idxs[-1])
        frame_idxs = np.clip(raw_idx + delta, 0, last_valid)

        # 读取原始关节数据 (整段 horizon 一起读)
        with h5py.File(ep_path, "r") as ep:
            qpos_h = _take_rows(ep["robot"]["qpos"], frame_idxs)
            action_h = _take_rows(ep["action"]["commanded"], frame_idxs)
            joint_names = [n.decode() if isinstance(n, bytes) else str(n)
                          for n in ep["robot/joint_names"][:]]

        # 统一 64D: state 取观测帧, action 逐帧构建 horizon (16 帧各一个不同的 64D 目标).
        # 这里曾经是 np.tile(单帧 action, (16,1)) —— 16 个 slot 的目标完全相同, 等于
        # 完全没有未来轨迹监督: flow-matching head 的 16 个 slot 之间学不到任何时间
        # 结构, 采样时各自独立去噪, 于是 chunk 内部互不相干 (部署只取 chunk[0] 时,
        # 那一步的方差也没有被其余 slot 约束住)。
        ee_qpos_h, ee_action_h = self._get_ee_frames(ep_stem, frame_idxs)
        unified_state = None
        unified_actions = np.zeros((len(frame_idxs), 64), dtype=np.float32)
        for h in range(len(frame_idxs)):
            s_h, a_h, _ = self._build_unified(
                ee_qpos_h[h], ee_action_h[h], qpos_h[h], action_h[h], joint_names
            )
            if h == 0:
                unified_state = s_h  # slot 0 == observation frame
            unified_actions[h] = a_h

        # 替换 result 中的 state 和 action
        # state: [1, 64] (observation_indices = [0])
        # action: [16, 64] (action_indices = list(range(16)))
        # GR00TTransform._prepare_state 要求 state.shape[0] == state_horizon (1)
        # GR00TTransform._prepare_action 要求 action.shape[0] == action_horizon (16)
        result["state.qpos"] = unified_state[None, :]          # [64] -> [1, 64]

        # Ablation switch, DEFAULT OFF.  GR00T_XE_TILE_ACTION=1 restores the
        # behaviour this repo shipped before 2026-09-13: every horizon slot
        # carries slot 0's action, so the chunk gets no future-trajectory
        # supervision at all.  Slot 0 is the same frame (_build_unified on
        # frame_idxs[0] == raw_idx, the observation frame) the old single-frame
        # call used, so this reproduces that target exactly.
        # Purpose: separate "the chunk fix" from "the from-base recipe" as the
        # cause of the online result.
        if os.environ.get("GR00T_XE_TILE_ACTION", "0") == "1":
            global _TILE_MODE_ANNOUNCED
            if not _TILE_MODE_ANNOUNCED:
                _TILE_MODE_ANNOUNCED = True
                print("[GR00T_XE][ABLATION] GR00T_XE_TILE_ACTION=1 -- action target "
                      "is np.tile(slot0, horizon); the future trajectory is NOT "
                      "supervised. This reproduces the pre-2026-09-13 target.",
                      flush=True)
            result["action.qpos"] = np.tile(unified_actions[0][None, :],
                                            (len(frame_idxs), 1))
        else:
            result["action.qpos"] = unified_actions            # [16, 64]

        # Per-robot authored-dims mask (0:12 arms + this robot's hand slots).
        # GR00TTransform would otherwise re-derive action_mask from the raw
        # action width (always 64 here) -> all-ones -> the 8 global pads
        # (56-63) and every un-authored hand slot would get a constant-zero
        # target in the loss.  Carrying the mask lets the flow-matching head
        # exclude exactly the non-authored dims (per-embodiment).
        am = self._unified_hand_mask
        if am is None:
            raise RuntimeError(
                f"[GR00T_XE] _unified_hand_mask not built for robot {self._xe_robot_key!r}; "
                "hand mapping was never validated."
            )
        result["action_mask"] = np.tile((am > 0).astype(bool), (16, 1))  # [16, 64]

        return result

    @property
    def robot_key(self) -> str | None:
        return self._xe_robot_key

    def __del__(self):
        if self._ee_file is not None:
            try:
                self._ee_file.close()
            except Exception:
                pass
        super().__del__()


def build_hdf5_dataset(
    input_dir: Path | str,
    camera_map: dict[str, str],
    modality_configs: dict,
    transforms,
    embodiment_tag: str = "new_embodiment",
    prompt: str | None = None,
    use_active_dof: bool = False,
    robot_key: str | None = None,
    state_dim: int | None = None,
    action_dim: int | None = None,
    truncate_at_homing: bool = False,
    xe_stats_only: bool = False,
) -> CrossEmbodimentHDF5Dataset:
    """``xe_stats_only`` is for build_xe_norm_stats.py alone -- see the class
    docstring for _xe_stats_only."""
    from robots.active_dof_utils import ActiveDofInfo, get_active_dof_info, get_active_dof_info_for_hdf5

    input_dir = Path(input_dir)
    hdf5_files = sorted(input_dir.glob("episode_*.hdf5"))

    adi: ActiveDofInfo | None = None
    if use_active_dof:
        if hdf5_files:
            adi = get_active_dof_info_for_hdf5(str(hdf5_files[0]))
        if adi is None and robot_key is not None:
            adi = get_active_dof_info(robot_key)

    return CrossEmbodimentHDF5Dataset(
        input_dir=input_dir,
        camera_map=camera_map,
        modality_configs=modality_configs,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        prompt=prompt,
        # active_dof_info/state_dim/action_dim not passed — XE dataset
        # handles them internally (uses 64D unified space, not raw per-robot dims)
        truncate_at_homing=truncate_at_homing,
        xe_stats_only=xe_stats_only,
    )