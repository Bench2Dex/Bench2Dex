"""PyTorch Dataset that reads dex2bench HDF5 episodes directly, bypassing LeRobot conversion.

Produces the same raw-data dict format as ``LeRobotSingleDataset.get_step_data``,
so the existing GR00T transform pipeline (VideoToTensor, StateActionTransform, etc.)
applies unchanged.

Usage::

    from gr00t_hdf5_dataset import Dex2BenchHDF5Dataset

    ds = Dex2BenchHDF5Dataset(
        input_dir="/path/to/replay-generalization",
        camera_map={"stereo_left": "cam_stereo_left", ...},
        modality_configs=modality_configs,
        transforms=transforms,
        embodiment_tag="new_embodiment",
        prompt="perform task ...",
    )
    sample = ds[0]  # fully transformed model-ready dict
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch

_DEX2BENCH_ROOT = Path(__file__).resolve().parents[2]
if str(_DEX2BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEX2BENCH_ROOT))

from gr00t.data.dataset import ModalityConfig
from gr00t.data.schema import (
    DatasetMetadata,
    DatasetModalities,
    DatasetStatisticalValues,
    DatasetStatistics,
    EmbodimentTag,
    StateActionMetadata,
    VideoMetadata,
)
from gr00t.data.transform.base import ComposedModalityTransform
from policy.GR00T_n15_Tactile.tactile_io import (
    TactileSchema,
    read_tactile_frame,
    read_tactile_schema,
    validate_tactile_schema,
)
from robots.active_dof_utils import (
    ActiveDofInfo,
    get_active_dof_info,
    get_active_dof_info_for_hdf5,
    robot_key_from_hdf5,
    select_active,
)

# ---------------------------------------------------------------------------
# HDF5 I/O helpers (shared with convert_dex2bench_to_gr00t.py)
# ---------------------------------------------------------------------------

ANNOTATION_KEY = "annotation.human.action.task_description"


def _decode_hdf5_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_hdf5_string(value.item())
    return str(value)


def _decode_rgb_frame(frame: Any) -> np.ndarray:
    """Decode a single RGB frame from an HDF5 dataset cell (raw bytes or JPEG)."""
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


def _valid_indices(ep: h5py.File) -> np.ndarray:
    """Return sorted indices of frames that have valid qpos, action, and frame_valid."""
    if "robot" not in ep or "qpos" not in ep["robot"]:
        raise KeyError("Episode is missing /robot/qpos")
    frame_count = int(ep["robot"]["qpos"].shape[0])
    frame_valid = (
        np.asarray(ep["frame_valid"][:], dtype=bool)
        if "frame_valid" in ep
        else np.ones(frame_count, dtype=bool)
    )
    action_valid = (
        np.asarray(ep["action"]["action_valid"][:], dtype=bool)
        if "action" in ep and "action_valid" in ep["action"]
        else np.ones(frame_count, dtype=bool)
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
    # searchsorted returns first i where time_sim_step[i] >= homing_val
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
    # Count valid frames whose absolute index is < cutoff_full.
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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class Dex2BenchHDF5Dataset(torch.utils.data.Dataset):
    """Read dex2bench HDF5 episodes and apply GR00T transforms directly.

    Parameters
    ----------
    input_dir : Path or str
        Directory containing ``episode_*.hdf5`` files.
    camera_map : dict[str, str]
        Mapping from GR00T video sub-keys (e.g. ``"stereo_left"``) to HDF5
        camera IDs (e.g. ``"cam_stereo_left"``).
    modality_configs : dict[str, ModalityConfig]
        Modality config from the DataConfig (keys: ``"video"``, ``"state"``,
        ``"action"``, ``"language"``).
    transforms : ComposedModalityTransform
        Transform pipeline from the DataConfig.
    embodiment_tag : str or EmbodimentTag
        Tag forwarded to ``DatasetMetadata``.
    prompt : str or None
        Language prompt stored in the annotation for every sample.
    active_dof_info : ActiveDofInfo or None
        If given, state/action arrays are sliced to active DOF only.
    state_dim : int or None
        Expected state dimension (after active-DOF slicing).  Inferred if omitted.
    action_dim : int or None
        Expected action dimension (after active-DOF slicing).  Inferred if omitted.
    seed : int
        Seed for epoch-based shuffling (currently unused, kept for interface).
    """

    def __init__(
        self,
        input_dir: Path | str,
        camera_map: dict[str, str],
        modality_configs: dict[str, ModalityConfig],
        transforms: ComposedModalityTransform,
        embodiment_tag: str | EmbodimentTag = "new_embodiment",
        prompt: str | None = None,
        active_dof_info: ActiveDofInfo | None = None,
        state_dim: int | None = None,
        action_dim: int | None = None,
        seed: int = 0,
        truncate_at_homing: bool = False,
        tactile_height: int = 120,
        tactile_width: int = 120,
    ):
        input_dir = Path(input_dir).expanduser().resolve()
        self._input_dir = input_dir
        self._camera_map = dict(camera_map)
        self._modality_configs = modality_configs
        self._transforms = transforms
        if isinstance(embodiment_tag, EmbodimentTag):
            self._tag = embodiment_tag.value
        else:
            self._tag = str(embodiment_tag)
        self._prompt = prompt
        self._adi = active_dof_info
        self._seed = seed
        self._dataset_name = input_dir.name
        self._tactile_output_shape = (int(tactile_height), int(tactile_width))
        if min(self._tactile_output_shape) <= 0:
            raise ValueError(f"tactile dimensions must be positive, got {self._tactile_output_shape}")
        self._tactile_schema: TactileSchema | None = None

        # -- scan episodes ---------------------------------------------------
        hdf5_paths = sorted(input_dir.glob("episode_*.hdf5"))
        if not hdf5_paths:
            raise FileNotFoundError(f"No episode_*.hdf5 files in {input_dir}")

        self._episode_paths: list[Path] = []
        self._episode_valid_indices: list[np.ndarray] = []  # per-episode valid local frame idxs
        self._episode_lengths: list[int] = []  # number of *valid* frames per episode

        _n_truncated = 0
        _total_trunc_removed = 0
        for p in hdf5_paths:
            with h5py.File(p, "r") as ep:
                tactile_schema = read_tactile_schema(ep, source=str(p))
                if self._tactile_schema is None:
                    self._tactile_schema = tactile_schema
                else:
                    validate_tactile_schema(self._tactile_schema, tactile_schema, source=str(p))
                valid = _valid_indices(ep)
                if len(valid) == 0:
                    continue

                # -- truncate at homing frame ---------------------------------
                if truncate_at_homing:
                    ep_len_full = int(ep["robot"]["qpos"].shape[0])
                    cutoff, _trunc_info = _find_homing_cutoff(
                        ep, ep_len_full, valid,
                        truncate=True,
                    )
                    if cutoff is not None:
                        _n_truncated += 1
                        _total_trunc_removed += len(valid) - cutoff
                        valid = valid[:cutoff]
                        if len(valid) == 0:
                            continue

                self._episode_paths.append(p)
                self._episode_valid_indices.append(valid)
                self._episode_lengths.append(len(valid))

        if not self._episode_paths:
            raise ValueError(f"No valid episodes found in {input_dir}")
        assert self._tactile_schema is not None

        # -- infer dimensions ------------------------------------------------
        with h5py.File(self._episode_paths[0], "r") as ep:
            first_state = np.asarray(ep["robot"]["qpos"][:], dtype=np.float32)
            first_action = np.asarray(ep["action"]["commanded"][:], dtype=np.float32)

        self._raw_state_dim = int(first_state.shape[1])
        self._raw_action_dim = int(first_action.shape[1])

        if self._adi is not None:
            self._state_dim = self._adi.active_dof
            self._action_dim = self._adi.active_dof
        else:
            self._state_dim = state_dim or self._raw_state_dim
            self._action_dim = action_dim or self._raw_action_dim

        if state_dim is not None and self._state_dim != state_dim:
            raise ValueError(f"State dim mismatch: expected {state_dim}, got {self._state_dim}")
        if action_dim is not None and self._action_dim != action_dim:
            raise ValueError(f"Action dim mismatch: expected {action_dim}, got {self._action_dim}")

        # -- pre-compute statistics (streaming pass) -------------------------
        stats = self._compute_statistics()

        # -- build GR00T DatasetMetadata ------------------------------------
        self._metadata = self._build_metadata(stats)

        # -- notify transforms of metadata -----------------------------------
        self._transforms.set_metadata(self._metadata)

        # -- build global index ----------------------------------------------
        # Global index: one entry per sample (frame that has enough future
        # frames for the action horizon).
        self._max_delta: int = 0
        for mcfg in self._modality_configs.values():
            for di in mcfg.delta_indices:
                self._max_delta = max(self._max_delta, di)
        self._max_delta = max(self._max_delta, 0)

        self._global_index: list[tuple[int, int]] = []  # (ep_idx, local_valid_idx)
        for ep_idx, length in enumerate(self._episode_lengths):
            # Need max_delta future frames available
            usable = max(0, length - self._max_delta)
            for local_idx in range(usable):
                self._global_index.append((ep_idx, local_idx))

        # -- episode HDF5 cache (per-worker, safe because each worker process
        #    opens its own independent file handles after fork) --
        self._cached_ep_idx: int | None = None
        self._cached_h5: h5py.File | None = None

        _trunc_parts: list[str] = []
        if truncate_at_homing:
            _trunc_parts.append(
                f"{_n_truncated} truncated, {_total_trunc_removed} frames removed"
            )
        _trunc_suffix = f"  [{', '.join(_trunc_parts)}]" if _trunc_parts else ""
        print(
            f"[Dex2BenchHDF5Dataset] {len(self._episode_paths)} episodes, "
            f"{len(self._global_index)} samples, "
            f"state_dim={self._state_dim}, action_dim={self._action_dim}"
            f"{_trunc_suffix}"
        )

    # ------------------------------------------------------------------
    # Public properties (required by TrainRunner / DualBrainTrainer)
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> DatasetMetadata:
        return self._metadata

    @property
    def tag(self) -> str:
        return self._tag

    @property
    def modality_configs(self) -> dict[str, ModalityConfig]:
        return self._modality_configs

    @property
    def transforms(self) -> ComposedModalityTransform:
        return self._transforms

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def tactile_schema(self) -> TactileSchema:
        assert self._tactile_schema is not None
        return self._tactile_schema

    @property
    def tactile_site_names(self) -> tuple[str, ...]:
        return self.tactile_schema.site_names

    @property
    def tactile_output_shape(self) -> tuple[int, int]:
        return self._tactile_output_shape

    @property
    def trajectory_ids(self) -> list[int]:
        """Episode indices treated as trajectory ids."""
        return list(range(len(self._episode_paths)))

    @property
    def trajectory_lengths(self) -> list[int]:
        """Number of valid frames per episode."""
        return self._episode_lengths

    @property
    def all_steps(self) -> list[tuple[int, int]]:
        """Global index as (ep_idx, local_valid_idx) pairs."""
        return self._global_index

    @property
    def max_delta(self) -> int:
        return self._max_delta

    def set_epoch(self, epoch: int) -> None:
        """Compatibility with LeRobot epoch-based sampling (no-op)."""
        pass

    # ------------------------------------------------------------------
    # Statistics computation
    # ------------------------------------------------------------------

    def _compute_statistics(self) -> dict[str, dict[str, list[float]]]:
        """Streaming pass over all episodes to compute per-dim stats.

        Uses the already-filtered (and optionally truncated) per-episode valid
        indices so that statistics reflect exactly the data the model sees.
        """
        all_state: list[np.ndarray] = []
        all_action: list[np.ndarray] = []

        for p, valid in zip(self._episode_paths, self._episode_valid_indices):
            with h5py.File(p, "r") as ep:
                s = np.asarray(ep["robot"]["qpos"][:], dtype=np.float32)
                a = np.asarray(ep["action"]["commanded"][:], dtype=np.float32)
                s = s[valid]
                a = a[valid]

            if self._adi is not None:
                s = select_active(s, self._adi)
                a = select_active(a, self._adi)

            all_state.append(s)
            all_action.append(a)

        cat_state = np.concatenate(all_state, axis=0)
        cat_action = np.concatenate(all_action, axis=0)

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

    # ------------------------------------------------------------------
    # Metadata construction
    # ------------------------------------------------------------------

    def _build_metadata(
        self, stats: dict[str, dict[str, list[float]]]
    ) -> DatasetMetadata:
        """Build a ``DatasetMetadata`` that matches what the transform pipeline expects."""

        # -- infer video resolution from first frame of first episode ---------
        with h5py.File(self._episode_paths[0], "r") as ep:
            first_cam = next(iter(self._camera_map.values()))
            first_frame = _decode_rgb_frame(ep[f"cameras/{first_cam}/rgb"][0])
            height, width = int(first_frame.shape[0]), int(first_frame.shape[1])

        fps = 20.0  # default; individual episodes may differ but this is fine for metadata

        # -- video metadata ---------------------------------------------------
        # Keys must be sub-keys (without the "video." prefix) — this is what
        # VideoTransform.set_metadata expects.
        video_meta: dict[str, VideoMetadata] = {}
        video_mcfg = self._modality_configs.get("video")
        if video_mcfg is not None:
            for gr00t_key in video_mcfg.modality_keys:
                # gr00t_key e.g. "video.stereo_left"
                sub_key = gr00t_key.replace("video.", "")
                video_meta[sub_key] = VideoMetadata(
                    resolution=(width, height),
                    channels=3,
                    fps=fps,
                )

        # -- state / action metadata ------------------------------------------
        state_meta: dict[str, StateActionMetadata] = {}
        for key in self._modality_configs.get("state", ModalityConfig(delta_indices=[], modality_keys=[])).modality_keys:
            sub_key = key.replace("state.", "")
            state_meta[sub_key] = StateActionMetadata(
                absolute=True,
                rotation_type=None,
                shape=(self._state_dim,),
                continuous=True,
            )

        action_meta: dict[str, StateActionMetadata] = {}
        for key in self._modality_configs.get("action", ModalityConfig(delta_indices=[], modality_keys=[])).modality_keys:
            sub_key = key.replace("action.", "")
            action_meta[sub_key] = StateActionMetadata(
                absolute=True,
                rotation_type=None,
                shape=(self._action_dim,),
                continuous=True,
            )

        # -- statistics -------------------------------------------------------
        # stats has format: {"state": {"qpos": {...}}, "action": {"qpos": {...}}}
        stat_state: dict[str, DatasetStatisticalValues] = {}
        stat_action: dict[str, DatasetStatisticalValues] = {}
        for modality, sub_dict in stats.items():
            for sub_key, vals in sub_dict.items():
                dsv = DatasetStatisticalValues(
                    max=np.array(vals["max"], dtype=np.float64),
                    min=np.array(vals["min"], dtype=np.float64),
                    mean=np.array(vals["mean"], dtype=np.float64),
                    std=np.array(vals["std"], dtype=np.float64),
                    q01=np.array(vals["q01"], dtype=np.float64),
                    q99=np.array(vals["q99"], dtype=np.float64),
                )
                if modality == "state":
                    stat_state[sub_key] = dsv
                elif modality == "action":
                    stat_action[sub_key] = dsv

        return DatasetMetadata(
            statistics=DatasetStatistics(state=stat_state, action=stat_action),
            modalities=DatasetModalities(
                video=video_meta,
                state=state_meta,
                action=action_meta,
            ),
            embodiment_tag=EmbodimentTag(self._tag),
        )

    # ------------------------------------------------------------------
    # Raw data loading (mirrors get_step_data)
    # ------------------------------------------------------------------

    def _get_episode(self, ep_idx: int) -> h5py.File:
        """Cache and return an open HDF5 handle for *ep_idx*.

        Each DataLoader worker process has its own copy of this dataset (via
        fork), so each worker independently opens its own file handles.  This is
        safe as long as no HDF5 handles are held open *before* fork — which we
        guarantee by closing all handles after ``__init__`` stats pass.
        """
        if self._cached_ep_idx != ep_idx:
            if self._cached_h5 is not None:
                self._cached_h5.close()
            self._cached_h5 = h5py.File(self._episode_paths[ep_idx], "r")
            self._cached_ep_idx = ep_idx
        assert self._cached_h5 is not None
        return self._cached_h5

    def get_step_data(self, ep_idx: int, local_valid_idx: int) -> dict[str, Any]:
        """Return raw (pre-transform) data for a single step."""
        ep = self._get_episode(ep_idx)
        valid_idxs = self._episode_valid_indices[ep_idx]
        base = valid_idxs[local_valid_idx]

        result: dict[str, Any] = {}

        video_mcfg = self._modality_configs.get("video")
        if video_mcfg is not None:
            delta = np.asarray(video_mcfg.delta_indices, dtype=np.intp)
            frame_idxs = np.clip(base + delta, 0, valid_idxs[-1])
            for gr00t_key in video_mcfg.modality_keys:
                sub_key = gr00t_key.replace("video.", "")
                hdf5_cam = self._camera_map[sub_key]
                path = f"cameras/{hdf5_cam}/rgb"
                frames = [_decode_rgb_frame(ep[path][int(i)]) for i in frame_idxs]
                result[gr00t_key] = np.stack(frames, axis=0)

        state_mcfg = self._modality_configs.get("state")
        if state_mcfg is not None:
            delta = np.asarray(state_mcfg.delta_indices, dtype=np.intp)
            frame_idxs = np.clip(base + delta, 0, valid_idxs[-1])
            raw_state = np.asarray(ep["robot"]["qpos"][:], dtype=np.float32)
            if self._adi is not None:
                raw_state = select_active(raw_state, self._adi)
            for gr00t_key in state_mcfg.modality_keys:
                result[gr00t_key] = raw_state[frame_idxs].copy()

        action_mcfg = self._modality_configs.get("action")
        if action_mcfg is not None:
            delta = np.asarray(action_mcfg.delta_indices, dtype=np.intp)
            frame_idxs = np.clip(base + delta, 0, valid_idxs[-1])
            raw_action = np.asarray(ep["action"]["commanded"][:], dtype=np.float32)
            if self._adi is not None:
                raw_action = select_active(raw_action, self._adi)
            for gr00t_key in action_mcfg.modality_keys:
                result[gr00t_key] = raw_action[frame_idxs].copy()

        lang_mcfg = self._modality_configs.get("language")
        if lang_mcfg is not None:
            prompt_text = self._prompt or "perform task"
            for gr00t_key in lang_mcfg.modality_keys:
                result[gr00t_key] = [prompt_text]

        result["tactile"] = read_tactile_frame(
            ep,
            int(base),
            self.tactile_schema,
            output_shape=self._tactile_output_shape,
        )

        return result

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._global_index)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, local_idx = self._global_index[idx]
        raw = self.get_step_data(ep_idx, local_idx)
        return self._transforms(raw)

    def __del__(self):
        if hasattr(self, "_cached_h5") and self._cached_h5 is not None:
            self._cached_h5.close()
            self._cached_h5 = None


# ---------------------------------------------------------------------------
# Factory helper (used by gr00t_finetune.py)
# ---------------------------------------------------------------------------


def build_hdf5_dataset(
    input_dir: Path | str,
    camera_map: dict[str, str],
    modality_configs: dict[str, ModalityConfig],
    transforms: ComposedModalityTransform,
    embodiment_tag: str = "new_embodiment",
    prompt: str | None = None,
    use_active_dof: bool = False,
    robot_key: str | None = None,
    state_dim: int | None = None,
    action_dim: int | None = None,
    truncate_at_homing: bool = False,
    tactile_height: int = 120,
    tactile_width: int = 120,
) -> Dex2BenchHDF5Dataset:
    """Convenience factory that resolves active-DOF info and builds the dataset."""

    adi: ActiveDofInfo | None = None
    input_dir = Path(input_dir)
    hdf5_files = sorted(input_dir.glob("episode_*.hdf5"))

    if use_active_dof:
        if hdf5_files:
            adi = get_active_dof_info_for_hdf5(str(hdf5_files[0]))
        if adi is None and robot_key is not None:
            adi = get_active_dof_info(robot_key)
        elif adi is None:
            rk = robot_key_from_hdf5(str(hdf5_files[0])) if hdf5_files else None
            if rk is not None:
                adi = get_active_dof_info(rk)

    return Dex2BenchHDF5Dataset(
        input_dir=input_dir,
        camera_map=camera_map,
        modality_configs=modality_configs,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        prompt=prompt,
        active_dof_info=adi,
        state_dim=state_dim,
        action_dim=action_dim,
        truncate_at_homing=truncate_at_homing,
        tactile_height=tactile_height,
        tactile_width=tactile_width,
    )
