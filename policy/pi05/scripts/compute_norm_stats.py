"""Compute normalization statistics for a config.

This script computes mean/std and quantile stats for ``state`` and ``actions``.
For dex2bench data, ``--dataset-dir`` can point directly at a replay HDF5
folder, so no LeRobot conversion is required.
"""
# uv run scripts/compute_norm_stats.py \
#     --config-name pi05_base_dex2bench_lora \
#     --dataset-dir "$DATA_DIR" \
#     --output-dir "$OUT_DIR/_assets/pi05_base_dex2bench_lora/dex2bench"
import dataclasses
import os
import pathlib
import sys

_DEX2BENCH_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_DEX2BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEX2BENCH_ROOT))

# Normalization stats only need CPU-side HDF5 reads. Keep JAX from initializing
# CUDA/NCCL when this script imports the OpenPI config/data modules.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

print("[compute_norm_stats] Importing OpenPI modules...", flush=True)

import numpy as np
import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):

    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def _with_repo_id(config: _config.TrainConfig, repo_id: str | None) -> _config.TrainConfig:
    if repo_id is None:
        return config
    return dataclasses.replace(config, data=dataclasses.replace(config.data, repo_id=repo_id))


def _dex2bench_repo_id(dataset_dir: str | None, prompt: str) -> str | None:
    if dataset_dir is None:
        return None
    path = pathlib.Path(dataset_dir).expanduser().resolve()
    return f"dex2bench:{path}|{prompt}"


def create_dataset(config: _config.TrainConfig) -> tuple[_config.DataConfig, _data_loader.Dataset]:
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(
        data_config,
        config.model.action_horizon,
        config.model,
    )
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    return data_config, dataset


def _default_output_dir(config: _config.TrainConfig, data_config: _config.DataConfig) -> pathlib.Path:
    # Prefer asset_id (filesystem-safe) over repo_id, which may contain ':' or '|' for
    # custom data sources like dex2bench.
    asset_dir_name = data_config.asset_id or data_config.repo_id
    return config.assets_dirs / asset_dir_name


def _list_hdf5_episode_files(dataset_dir: str) -> list[str]:
    path = pathlib.Path(dataset_dir).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {path}")
    files = sorted(str(p) for p in path.iterdir() if p.name.endswith(".hdf5"))
    if not files:
        raise FileNotFoundError(f"No HDF5 files in {path}")
    return files


def _env_bool(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _pad_to_dim(values: np.ndarray, target_dim: int) -> np.ndarray:
    if values.shape[-1] == target_dim:
        return values
    if values.shape[-1] > target_dim:
        raise ValueError(f"Data dim {values.shape[-1]} exceeds configured action_dim {target_dim}")
    pad_width = [(0, 0)] * values.ndim
    pad_width[-1] = (0, target_dim - values.shape[-1])
    return np.pad(values, pad_width, mode="constant")


def _valid_indices(h5file) -> np.ndarray:
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
    h5file,
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


def _compute_dex2bench_hdf5_stats(
    *,
    config: _config.TrainConfig,
    dataset_dir: str,
    max_frames: int | None,
) -> dict[str, normalize.NormStats]:
    """Fast norm-stats path for dex2bench replay HDF5.

    This mirrors Dex2BenchPi0Dataset's state/action semantics but skips image
    reads, JPEG decode, Torch/JAX data loaders, and GPU initialization.
    """
    import h5py
    from robots.active_dof_utils import (
        get_active_dof_info,
        get_active_dof_info_for_hdf5,
        select_active,
    )

    source_paths = _list_hdf5_episode_files(dataset_dir)
    action_horizon = int(config.model.action_horizon)
    max_delta = max(action_horizon - 1, 0)

    raw_lengths: list[int] = []
    valid_lengths: list[int] = []
    sample_lengths: list[int] = []
    episode_valid_indices: list[np.ndarray] = []
    paths: list[str] = []
    for episode_path in source_paths:
        with h5py.File(episode_path, "r") as f:
            total = int(f["robot/qpos"].shape[0])
            raw_lengths.append(total)
            valid = _valid_indices(f)
            if valid.size == 0:
                continue
            valid = _truncate_valid_indices_at_homing(f, total, valid)
            if valid.size == 0:
                continue
            paths.append(episode_path)
            episode_valid_indices.append(valid)
            valid_lengths.append(int(valid.size))
            sample_lengths.append(max(int(valid.size) - max_delta, 0))
    if not paths:
        raise RuntimeError("No valid dex2bench frames after valid-frame filtering and homing truncation")

    raw_total_frames = int(sum(raw_lengths))
    total_frames = int(sum(valid_lengths))
    usable_sample_starts = int(sum(sample_lengths))
    frame_count = total_frames
    selected_by_episode: list[np.ndarray]
    if max_frames is not None and max_frames < total_frames:
        frame_count = int(max_frames)
        rng = np.random.default_rng(config.seed)
        selected = np.sort(rng.choice(total_frames, size=frame_count, replace=False))
        cum = np.cumsum(valid_lengths)
        selected_by_episode = []
        start = 0
        for stop in cum:
            mask = (selected >= start) & (selected < stop)
            selected_by_episode.append(selected[mask] - start)
            start = int(stop)
    else:
        selected_by_episode = [np.arange(length, dtype=np.int64) for length in valid_lengths]

    state_dim = int(getattr(config.data, "state_dim", None) or config.model.action_dim)
    action_dim = int(config.model.action_dim)
    env_state_dim = _env_int("PI05_DEX2BENCH_STATE_DIM")
    env_action_dim = _env_int("PI05_DEX2BENCH_ACTION_DIM")
    if env_state_dim is not None:
        state_dim = env_state_dim
    if env_action_dim is not None:
        action_dim = env_action_dim

    active_dof_info = None
    if _env_bool("PI05_DEX2BENCH_USE_ACTIVE_DOF", True):
        active_dof_info = get_active_dof_info_for_hdf5(paths[0])
        if active_dof_info is None:
            robot_key = os.environ.get("PI05_DEX2BENCH_ROBOT_KEY") or None
            if robot_key:
                active_dof_info = get_active_dof_info(robot_key)
        if active_dof_info is not None:
            print(
                f"Active DOF stats: robot={active_dof_info.robot_key} "
                f"full_dof={active_dof_info.full_dof} active_dof={active_dof_info.active_dof}"
            )
            if state_dim == active_dof_info.full_dof:
                state_dim = active_dof_info.active_dof
            elif state_dim != active_dof_info.active_dof:
                if env_state_dim is not None:
                    raise ValueError(
                        "Active DOF is enabled, but PI05_DEX2BENCH_STATE_DIM "
                        f"{state_dim} does not match full_dof {active_dof_info.full_dof} "
                        f"or active_dof {active_dof_info.active_dof}."
                    )
                print(
                    "Configured state_dim "
                    f"{state_dim} does not match robot full_dof {active_dof_info.full_dof} "
                    f"or active_dof {active_dof_info.active_dof}; using active_dof.",
                    flush=True,
                )
                state_dim = active_dof_info.active_dof
            if action_dim == active_dof_info.full_dof:
                action_dim = active_dof_info.active_dof
            elif action_dim != active_dof_info.active_dof:
                if env_action_dim is not None:
                    raise ValueError(
                        "Active DOF is enabled, but PI05_DEX2BENCH_ACTION_DIM "
                        f"{action_dim} does not match full_dof {active_dof_info.full_dof} "
                        f"or active_dof {active_dof_info.active_dof}."
                    )
                print(
                    "Configured action_dim "
                    f"{action_dim} does not match robot full_dof {active_dof_info.full_dof} "
                    f"or active_dof {active_dof_info.active_dof}; using active_dof.",
                    flush=True,
                )
                action_dim = active_dof_info.active_dof
            print(f"Effective stats dims: state_dim={state_dim} action_dim={action_dim}", flush=True)

    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}

    print(f"Reading dex2bench HDF5 state/actions from: {pathlib.Path(dataset_dir).expanduser().resolve()}")
    print("Using fast path: no image reads, no JPEG decode, JAX forced to CPU")
    print(
        f"Episodes: {len(paths)}/{len(source_paths)} | stat frames: {frame_count}/{total_frames} "
        f"| usable sample starts: {usable_sample_starts} "
        f"| raw frames: {raw_total_frames} | action_horizon: {action_horizon}"
    )

    with tqdm.tqdm(total=frame_count, desc="Computing stats") as progress:
        for episode_path, valid, selected_local in zip(paths, episode_valid_indices, selected_by_episode, strict=True):
            if selected_local.size == 0:
                continue
            frame_indices = valid[selected_local]
            with h5py.File(episode_path, "r") as f:
                qpos = np.asarray(f["robot/qpos"][:], dtype=np.float32)[frame_indices]
                commanded = np.asarray(f["action/commanded"][:], dtype=np.float32)[frame_indices]

                if active_dof_info is not None and qpos.shape[-1] == active_dof_info.full_dof:
                    qpos = select_active(qpos, active_dof_info)
                    commanded = select_active(commanded, active_dof_info)

                if qpos.shape[-1] != state_dim:
                    raise ValueError(
                        f"{episode_path} robot/qpos dim {qpos.shape[-1]} does not match configured state_dim {state_dim}."
                    )
                if commanded.shape[-1] != state_dim:
                    raise ValueError(
                        f"{episode_path} action/commanded dim {commanded.shape[-1]} "
                        f"does not match configured state_dim {state_dim}."
                    )

            chunk_size = 512
            for start in range(0, qpos.shape[0], chunk_size):
                state = _pad_to_dim(qpos[start: start + chunk_size], action_dim)
                actions = _pad_to_dim(commanded[start: start + chunk_size], action_dim)
                stats["state"].update(state.reshape(-1, state.shape[-1]))
                stats["actions"].update(actions.reshape(-1, actions.shape[-1]))
                progress.update(int(state.shape[0]))

    return {key: value.get_statistics() for key, value in stats.items()}

def main(
    config_name: str,
    dataset_dir: str | None = None,
    output_dir: str | None = None,
    prompt: str = "",
    max_frames: int | None = None,
    seed: int | None = None,
):
    """Compute and save normalization stats.

    Args:
        config_name: Name registered in openpi.training.config, e.g. pi05_base_dex2bench_lora.
        dataset_dir: Optional dex2bench replay directory containing episode_*.hdf5 files.
        output_dir: Optional directory where norm_stats.json will be written.
        prompt: Prompt used when dataset_dir is provided. It is ignored by stat computation.
        max_frames: Optional maximum number of frames to sample.
        seed: Optional random seed for frame sampling.
    """
    config = _config.get_config(config_name)
    if seed is not None:
        config = dataclasses.replace(config, seed=seed)
    repo_id = _dex2bench_repo_id(dataset_dir, prompt)
    config = _with_repo_id(config, repo_id)

    if dataset_dir is not None:
        norm_stats = _compute_dex2bench_hdf5_stats(
            config=config,
            dataset_dir=dataset_dir,
            max_frames=max_frames,
        )
        if output_dir is not None:
            output_path = pathlib.Path(output_dir).expanduser().resolve()
        else:
            data_config = config.data.create(config.assets_dirs, config.model)
            output_path = _default_output_dir(config, data_config)
        print(f"Writing stats to: {output_path}")
        normalize.save(output_path, norm_stats)
        return

    data_config, dataset = create_dataset(config)

    num_frames = len(dataset)
    shuffle = False

    if max_frames is not None and max_frames < num_frames:
        num_frames = max_frames
        shuffle = True

    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=8,
        num_workers=8,
        shuffle=shuffle,
        num_batches=num_frames,
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    print(f"Reading data from: {data_config.repo_id}")
    for batch in tqdm.tqdm(data_loader, total=num_frames, desc="Computing stats"):
        for key in keys:
            values = np.asarray(batch[key][0])
            stats[key].update(values.reshape(-1, values.shape[-1]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = pathlib.Path(output_dir).expanduser().resolve() if output_dir else _default_output_dir(config, data_config)
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
