"""Dex2Bench HDF5 loading for the tactile-aware ACT policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from robots.active_dof_utils import get_active_dof_info_for_runtime


DEFAULT_CAMERA_NAMES = (
    "cam_right_wrist",
    "cam_left_wrist",
    "cam_stereo_left",
    "cam_stereo_right",
)

_CAMERA_TO_HDF5 = {
    "cam_high": "cam_overhead",
    "cam_right_wrist": "cam_wrist_right",
    "cam_left_wrist": "cam_wrist_left",
    "cam_chest": "cam_chest",
    "cam_stereo_left": "cam_stereo_left",
    "cam_stereo_right": "cam_stereo_right",
}


class TactileSchemaError(ValueError):
    """Raised when an episode cannot satisfy the tactile policy data contract."""


@dataclass(frozen=True)
class EpisodeSchema:
    """Validated metadata needed to load one HDF5 episode."""

    path: Path
    robot_key: str
    camera_names: tuple[str, ...]
    camera_keys: tuple[str, ...]
    site_names: tuple[str, ...]
    tactile_shape: tuple[int, int]
    frame_count: int
    effective_length: int
    full_state_dim: int
    state_dim: int
    active_indices: tuple[int, ...]
    joint_names: tuple[str, ...]
    success: bool | None
    demo_eligible: bool | None


@dataclass(frozen=True)
class FrameRef:
    """A frame inside an inspected episode."""

    episode_index: int
    frame_index: int


@dataclass(frozen=True)
class NormalizationStats:
    """Per-dimension qpos and action normalization statistics."""

    qpos_mean: np.ndarray
    qpos_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    @property
    def state_dim(self) -> int:
        return int(self.qpos_mean.shape[0])

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "qpos_mean": self.qpos_mean.copy(),
            "qpos_std": self.qpos_std.copy(),
            "action_mean": self.action_mean.copy(),
            "action_std": self.action_std.copy(),
        }

    @classmethod
    def from_dict(cls, values: dict) -> "NormalizationStats":
        return cls(
            qpos_mean=np.asarray(values["qpos_mean"], dtype=np.float32),
            qpos_std=np.asarray(values["qpos_std"], dtype=np.float32),
            action_mean=np.asarray(values["action_mean"], dtype=np.float32),
            action_std=np.asarray(values["action_std"], dtype=np.float32),
        )


def _decode_string(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_optional_bool(h5: h5py.File, key: str) -> bool | None:
    dataset = h5.get(key)
    return None if dataset is None else bool(dataset[()])


def _require_dataset(h5: h5py.File, key: str, path: Path) -> h5py.Dataset:
    dataset = h5.get(key)
    if not isinstance(dataset, h5py.Dataset):
        raise TactileSchemaError(f"{path}: missing dataset '{key}'")
    return dataset


def _effective_length(h5: h5py.File, frame_count: int) -> int:
    homing = h5.get("meta/homing_start_sim_step")
    sim_steps = h5.get("time/sim_step")
    if isinstance(homing, h5py.Dataset) and isinstance(sim_steps, h5py.Dataset):
        index = int(np.searchsorted(np.asarray(sim_steps[:]), homing[()]))
        if 0 < index < frame_count:
            return index
    return frame_count


def inspect_episode(
    path: str | Path,
    *,
    camera_names: Sequence[str] = DEFAULT_CAMERA_NAMES,
    use_active_dof: bool = True,
) -> EpisodeSchema:
    """Validate one episode and return the metadata needed by the dataset."""

    episode_path = Path(path).expanduser().resolve()
    if not episode_path.is_file():
        raise FileNotFoundError(episode_path)

    with h5py.File(episode_path, "r") as h5:
        qpos = _require_dataset(h5, "robot/qpos", episode_path)
        commanded = _require_dataset(h5, "action/commanded", episode_path)
        if qpos.ndim != 2:
            raise TactileSchemaError(f"{episode_path}: robot/qpos must be rank 2, got {qpos.shape}")
        if commanded.shape != qpos.shape:
            raise TactileSchemaError(
                f"{episode_path}: action/commanded shape {commanded.shape} does not match robot/qpos {qpos.shape}"
            )
        frame_count, full_state_dim = map(int, qpos.shape)

        robot_key_ds = _require_dataset(h5, "meta/robot_key", episode_path)
        robot_key = _decode_string(robot_key_ds[()])
        joint_names_ds = _require_dataset(h5, "robot/joint_names", episode_path)
        joint_names = tuple(_decode_string(value) for value in joint_names_ds[:])
        if len(joint_names) != full_state_dim:
            raise TactileSchemaError(
                f"{episode_path}: robot/joint_names has {len(joint_names)} names for {full_state_dim} qpos columns"
            )

        if use_active_dof:
            try:
                active_info = get_active_dof_info_for_runtime(robot_key, list(joint_names))
            except Exception as exc:
                raise TactileSchemaError(
                    f"{episode_path}: cannot resolve active DOF for robot '{robot_key}': {exc}"
                ) from exc
            active_indices = tuple(int(index) for index in active_info.active_indices)
        else:
            active_indices = tuple(range(full_state_dim))
        state_dim = len(active_indices)

        resolved_camera_names = tuple(camera_names)
        camera_keys = tuple(_CAMERA_TO_HDF5.get(name, name) for name in resolved_camera_names)
        for camera_name, camera_key in zip(resolved_camera_names, camera_keys):
            rgb = _require_dataset(h5, f"cameras/{camera_key}/rgb", episode_path)
            if len(rgb) != frame_count:
                raise TactileSchemaError(
                    f"{episode_path}: camera '{camera_name}' has {len(rgb)} frames, expected {frame_count}"
                )

        site_names_ds = _require_dataset(h5, "robot/tactile/meta/site_names", episode_path)
        site_names = tuple(_decode_string(value) for value in site_names_ds[:])
        if not site_names:
            raise TactileSchemaError(f"{episode_path}: expected at least one tactile site")
        if len(set(site_names)) != len(site_names):
            raise TactileSchemaError(f"{episode_path}: tactile site_names contains duplicates")

        tactile_shape: tuple[int, int] | None = None
        for site_name in site_names:
            dataset = _require_dataset(h5, f"robot/tactile/tacmap/{site_name}", episode_path)
            if dataset.ndim != 3 or dataset.shape[0] != frame_count:
                raise TactileSchemaError(
                    f"{episode_path}: tactile site '{site_name}' must have shape [T,H,W], got {dataset.shape}"
                )
            if dataset.dtype != np.dtype(np.uint8):
                raise TactileSchemaError(
                    f"{episode_path}: tactile site '{site_name}' must be uint8, got {dataset.dtype}"
                )
            current_shape = (int(dataset.shape[1]), int(dataset.shape[2]))
            if tactile_shape is None:
                tactile_shape = current_shape
            elif current_shape != tactile_shape:
                raise TactileSchemaError(
                    f"{episode_path}: tactile site '{site_name}' has shape {current_shape}, expected {tactile_shape}"
                )

        effective_length = _effective_length(h5, frame_count)
        if effective_length <= 0:
            raise TactileSchemaError(f"{episode_path}: episode has no pre-homing frames")

        return EpisodeSchema(
            path=episode_path,
            robot_key=robot_key,
            camera_names=resolved_camera_names,
            camera_keys=camera_keys,
            site_names=site_names,
            tactile_shape=tactile_shape or (0, 0),
            frame_count=frame_count,
            effective_length=effective_length,
            full_state_dim=full_state_dim,
            state_dim=state_dim,
            active_indices=active_indices,
            joint_names=joint_names,
            success=_read_optional_bool(h5, "meta/success"),
            demo_eligible=_read_optional_bool(h5, "meta/demo_eligible"),
        )


def make_frame_refs(episodes: Sequence[EpisodeSchema]) -> list[FrameRef]:
    """Return every pre-homing frame in episode order."""

    return [
        FrameRef(episode_index=episode_index, frame_index=frame_index)
        for episode_index, episode in enumerate(episodes)
        for frame_index in range(episode.effective_length)
    ]


def split_frame_refs(
    episodes: Sequence[EpisodeSchema],
    *,
    val_ratio: float,
    chunk_size: int,
    seed: int,
) -> tuple[list[FrameRef], list[FrameRef]]:
    """Split by episode, or temporally with a horizon gap for one episode."""

    if not 0 <= val_ratio < 1:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    if val_ratio == 0:
        return make_frame_refs(episodes), []
    if len(episodes) == 1:
        length = episodes[0].effective_length
        val_start = min(max(1, int(round(length * (1 - val_ratio)))), length - 1)
        train_end = max(1, val_start - chunk_size)
        train = [FrameRef(0, index) for index in range(train_end)]
        val = [FrameRef(0, index) for index in range(val_start, length)]
        return train, val

    rng = np.random.default_rng(seed)
    episode_ids = np.arange(len(episodes))
    rng.shuffle(episode_ids)
    val_count = min(max(1, int(round(len(episodes) * val_ratio))), len(episodes) - 1)
    val_ids = set(int(index) for index in episode_ids[:val_count])
    train: list[FrameRef] = []
    val: list[FrameRef] = []
    for episode_index, episode in enumerate(episodes):
        destination = val if episode_index in val_ids else train
        destination.extend(FrameRef(episode_index, index) for index in range(episode.effective_length))
    return train, val


def _select_columns(array: np.ndarray, schema: EpisodeSchema) -> np.ndarray:
    return array[..., np.asarray(schema.active_indices, dtype=np.intp)]


def compute_normalization_stats(
    episodes: Sequence[EpisodeSchema],
    frame_refs: Sequence[FrameRef],
    *,
    std_floor: float = 1e-2,
) -> NormalizationStats:
    """Compute qpos/action statistics from training frame anchors only."""

    if not frame_refs:
        raise ValueError("cannot compute normalization statistics without frames")
    grouped: dict[int, list[int]] = {}
    for ref in frame_refs:
        grouped.setdefault(ref.episode_index, []).append(ref.frame_index)

    qpos_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    for episode_index, indices in grouped.items():
        schema = episodes[episode_index]
        unique_indices = np.asarray(sorted(set(indices)), dtype=np.intp)
        with h5py.File(schema.path, "r") as h5:
            qpos = _select_columns(h5["robot/qpos"][unique_indices].astype(np.float32), schema)
            actions = _select_columns(h5["action/commanded"][unique_indices].astype(np.float32), schema)
            valid_ds = h5.get("action/action_valid")
            valid = (
                np.asarray(valid_ds[unique_indices], dtype=bool)
                if isinstance(valid_ds, h5py.Dataset)
                else np.ones(len(unique_indices), dtype=bool)
            )
            valid &= np.isfinite(actions).all(axis=1)
        qpos_parts.append(qpos)
        if valid.any():
            action_parts.append(actions[valid])

    if not action_parts:
        raise TactileSchemaError("training frames contain no valid commanded actions")
    qpos_all = np.concatenate(qpos_parts, axis=0).astype(np.float64)
    action_all = np.concatenate(action_parts, axis=0).astype(np.float64)
    qpos_std = np.maximum(qpos_all.std(axis=0), std_floor)
    action_std = np.maximum(action_all.std(axis=0), std_floor)
    return NormalizationStats(
        qpos_mean=qpos_all.mean(axis=0).astype(np.float32),
        qpos_std=qpos_std.astype(np.float32),
        action_mean=action_all.mean(axis=0).astype(np.float32),
        action_std=action_std.astype(np.float32),
    )


def _decode_jpeg(raw, *, path: Path, camera_key: str, frame_index: int) -> np.ndarray:
    if isinstance(raw, np.ndarray):
        buffer = np.asarray(raw, dtype=np.uint8).reshape(-1)
    else:
        buffer = np.frombuffer(bytes(raw), dtype=np.uint8)
    bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if bgr is None:
        raise TactileSchemaError(
            f"{path}: failed to decode cameras/{camera_key}/rgb[{frame_index}]"
        )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class TactileEpisodeDataset(Dataset):
    """Frame-indexed RGB, qpos, TacMap, and action-chunk dataset."""

    def __init__(
        self,
        episodes: Sequence[EpisodeSchema],
        frame_refs: Sequence[FrameRef],
        stats: NormalizationStats,
        *,
        chunk_size: int,
        tactile_mode: str = "normal",
        shuffle_seed: int = 0,
        image_size: tuple[int, int] = (480, 640),
        tactile_size: tuple[int, int] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if tactile_mode not in {"normal", "zero", "shuffle"}:
            raise ValueError(f"unsupported tactile_mode '{tactile_mode}'")
        if not frame_refs:
            raise ValueError("frame_refs must not be empty")
        self.episodes = tuple(episodes)
        self.frame_refs = tuple(frame_refs)
        self.stats = stats
        self.chunk_size = int(chunk_size)
        self.tactile_mode = tactile_mode
        self.image_size = tuple(int(value) for value in image_size)
        self.tactile_size = None if tactile_size is None else tuple(int(value) for value in tactile_size)
        self._tactile_refs = self._build_tactile_refs(shuffle_seed)

    def _build_tactile_refs(self, seed: int) -> tuple[FrameRef, ...]:
        if self.tactile_mode != "shuffle":
            return self.frame_refs
        rng = np.random.default_rng(seed)
        mapped = list(self.frame_refs)
        by_episode: dict[int, list[int]] = {}
        for position, ref in enumerate(self.frame_refs):
            by_episode.setdefault(ref.episode_index, []).append(position)
        for positions in by_episode.values():
            if len(positions) < 2:
                continue
            source_positions = np.asarray(positions, dtype=np.intp)
            permutation = rng.permutation(source_positions)
            if np.array_equal(permutation, source_positions):
                permutation = np.roll(permutation, 1)
            for destination, source in zip(positions, permutation):
                mapped[destination] = self.frame_refs[int(source)]
        return tuple(mapped)

    def __len__(self) -> int:
        return len(self.frame_refs)

    def _load_images(self, h5: h5py.File, schema: EpisodeSchema, frame_index: int) -> np.ndarray:
        images: list[np.ndarray] = []
        output_h, output_w = self.image_size
        for camera_key in schema.camera_keys:
            image = _decode_jpeg(
                h5[f"cameras/{camera_key}/rgb"][frame_index],
                path=schema.path,
                camera_key=camera_key,
                frame_index=frame_index,
            )
            if image.shape[:2] != (output_h, output_w):
                image = cv2.resize(image, (output_w, output_h), interpolation=cv2.INTER_LINEAR)
            images.append(np.moveaxis(image, -1, 0))
        return np.stack(images, axis=0)

    def _load_tactile(self, ref: FrameRef) -> np.ndarray:
        schema = self.episodes[ref.episode_index]
        if self.tactile_mode == "zero":
            height, width = self.tactile_size or schema.tactile_shape
            return np.zeros((len(schema.site_names), 1, height, width), dtype=np.float32)
        maps: list[np.ndarray] = []
        with h5py.File(schema.path, "r") as h5:
            for site_name in schema.site_names:
                tactile = h5[f"robot/tactile/tacmap/{site_name}"][ref.frame_index]
                if self.tactile_size is not None and tactile.shape != self.tactile_size:
                    height, width = self.tactile_size
                    tactile = cv2.resize(tactile, (width, height), interpolation=cv2.INTER_AREA)
                maps.append(tactile[None, ...].astype(np.float32) / 255.0)
        return np.stack(maps, axis=0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ref = self.frame_refs[index]
        tactile_ref = self._tactile_refs[index]
        schema = self.episodes[ref.episode_index]
        if tactile_ref.episode_index != ref.episode_index:
            raise RuntimeError("tactile shuffle crossed episode boundaries")

        with h5py.File(schema.path, "r") as h5:
            images = self._load_images(h5, schema, ref.frame_index)
            qpos = _select_columns(
                h5["robot/qpos"][ref.frame_index].astype(np.float32), schema
            )
            actions = np.zeros((self.chunk_size, schema.state_dim), dtype=np.float32)
            is_pad = np.ones(self.chunk_size, dtype=bool)
            target_end = min(ref.frame_index + self.chunk_size, schema.effective_length)
            if target_end > ref.frame_index:
                raw_actions = _select_columns(
                    h5["action/commanded"][ref.frame_index:target_end].astype(np.float32), schema
                )
                valid_ds = h5.get("action/action_valid")
                valid = (
                    np.asarray(valid_ds[ref.frame_index:target_end], dtype=bool)
                    if isinstance(valid_ds, h5py.Dataset)
                    else np.ones(target_end - ref.frame_index, dtype=bool)
                )
                valid &= np.isfinite(raw_actions).all(axis=1)
                normalized = (raw_actions - self.stats.action_mean) / self.stats.action_std
                valid_rows = np.flatnonzero(valid)
                actions[valid_rows] = normalized[valid_rows]
                is_pad[: len(raw_actions)] = ~valid

        qpos = (qpos - self.stats.qpos_mean) / self.stats.qpos_std
        tactile = self._load_tactile(tactile_ref)
        return {
            "images": torch.from_numpy(np.ascontiguousarray(images)),
            "tactile": torch.from_numpy(np.ascontiguousarray(tactile)),
            "qpos": torch.from_numpy(np.ascontiguousarray(qpos)).float(),
            "actions": torch.from_numpy(actions).float(),
            "is_pad": torch.from_numpy(is_pad),
            "episode_index": torch.tensor(ref.episode_index, dtype=torch.long),
            "frame_index": torch.tensor(ref.frame_index, dtype=torch.long),
            "tactile_source_index": torch.tensor(tactile_ref.frame_index, dtype=torch.long),
        }


def inspect_dataset_directory(
    dataset_dir: str | Path,
    *,
    camera_names: Sequence[str] = DEFAULT_CAMERA_NAMES,
    use_active_dof: bool = True,
    robot_key: str | None = None,
) -> list[EpisodeSchema]:
    """Inspect every HDF5 file in a directory and require one compatible schema."""

    directory = Path(dataset_dir).expanduser().resolve()
    paths = sorted(directory.glob("*.hdf5"))
    if not paths:
        raise FileNotFoundError(f"no HDF5 files found in {directory}")
    episodes = [
        inspect_episode(path, camera_names=camera_names, use_active_dof=use_active_dof)
        for path in paths
    ]
    for episode in episodes:
        if robot_key is not None and episode.robot_key != robot_key:
            raise TactileSchemaError(
                f"{episode.path}: expected robot '{robot_key}', "
                f"recorded '{episode.robot_key}'"
            )
    reference = episodes[0]
    for episode in episodes[1:]:
        if episode.robot_key != reference.robot_key:
            raise TactileSchemaError(
                f"{episode.path}: robot '{episode.robot_key}' does not match '{reference.robot_key}'"
            )
        if episode.site_names != reference.site_names:
            raise TactileSchemaError(f"{episode.path}: tactile site order does not match {reference.path}")
        if episode.camera_names != reference.camera_names or episode.state_dim != reference.state_dim:
            raise TactileSchemaError(f"{episode.path}: policy input schema does not match {reference.path}")
    return episodes


def failed_demo_warnings(episodes: Iterable[EpisodeSchema]) -> list[str]:
    """Return explicit warnings for files unsuitable as successful demonstrations."""

    warnings: list[str] = []
    for episode in episodes:
        if episode.success is False:
            warnings.append(f"{episode.path.name}: meta/success=False")
        if episode.demo_eligible is False:
            warnings.append(f"{episode.path.name}: meta/demo_eligible=False")
    return warnings
