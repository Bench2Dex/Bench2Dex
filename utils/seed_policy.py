"""Deterministic seed policy helpers for collection and evaluation."""

from __future__ import annotations

import os
import random
import random as _random
import re
import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np


SEED_POLICY_ONE_INDEXED = "episode_seed = task_base_seed + episode_index - 1"
COLLECTION_SEED_NAMESPACE = "collection"
EVAL_SEED_NAMESPACE = "eval"
COLLECT_BASE_SEED_DEFAULT = 10_000
EVAL_BASE_SEED_DEFAULT = 100_000_000
TASK_SEED_STRIDE = 100_000
BENCHMARK_HASH_TASK_ID_OFFSET = 10_000
BENCHMARK_HASH_TASK_ID_BUCKETS = 10_000
_MAX_NUMPY_SEED = 2**32
_EPISODE_PATTERN = re.compile(r"^episode_(\d{6})\.hdf5$")
_TASK_PREFIX_PATTERN = re.compile(r"^(\d+)[_-]")

# ---- explicit RNG instances (replaced by seed_everything) ----
_module_py_rng: _random.Random | None = None
_module_np_rng: "np.random.Generator | None" = None


def get_py_rng() -> _random.Random:
    """Return the current module-level Python RNG.

    Returns a :class:`random.Random` instance created by the most recent
    :func:`seed_everything` call, or a fresh unseeded instance if
    :func:`seed_everything` has never been called.

    All scene-building code under ``build/`` consumes randomness through
    this accessor so that every random decision is traceable to a single,
    explicitly-seeded generator.
    """
    global _module_py_rng
    if _module_py_rng is None:
        _module_py_rng = _random.Random()
    return _module_py_rng


def get_np_rng() -> "np.random.Generator":
    """Return the current module-level NumPy Generator.

    Returns a :class:`numpy.random.Generator` instance created by the most
    recent :func:`seed_everything` call, or a fresh unseeded instance if
    :func:`seed_everything` has never been called.
    """
    global _module_np_rng
    if _module_np_rng is None:
        _module_np_rng = np.random.default_rng()
    return _module_np_rng


@dataclass(frozen=True)
class EpisodeSeedContext:
    seed_namespace: str
    base_seed: int
    task_seed_id: int
    task_base_seed: int
    episode_index: int
    episode_seed: int
    seed_policy: str = SEED_POLICY_ONE_INDEXED


def validate_seed(seed: int, label: str = "seed") -> int:
    seed_int = int(seed)
    if seed_int < 0 or seed_int >= _MAX_NUMPY_SEED:
        raise ValueError(f"{label} must be within [0, 2**32).")
    return seed_int


def seed_everything(seed: int, *, torch_enabled: bool = True) -> None:
    seed_int = validate_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed_int)

    # -- explicit RNG instances (primary path for build/* and tools code) --
    global _module_py_rng, _module_np_rng
    _module_py_rng = _random.Random(seed_int)
    _module_np_rng = np.random.default_rng(seed_int)

    # -- legacy global seeds (compatibility shim for policy/ and third-party libs) --
    random.seed(seed_int)
    np.random.seed(seed_int)
    if not torch_enabled:
        return
    try:
        import torch

        torch.manual_seed(seed_int)
        if hasattr(torch, "cuda"):
            torch.cuda.manual_seed_all(seed_int)
        if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        return


def resolve_task_seed_id(task_path: str, explicit_task_seed_id: int | None = None) -> int:
    if explicit_task_seed_id is not None:
        parsed = int(explicit_task_seed_id)
        if parsed < 0:
            raise ValueError("--task-seed-id must be non-negative.")
        return parsed
    stem = os.path.splitext(os.path.basename(str(task_path)))[0]
    match = _TASK_PREFIX_PATTERN.match(stem)
    if match:
        return int(match.group(1))
    raise ValueError(
        f"Cannot infer task seed id from task filename '{stem}'. "
        "Use --task-seed-id for tasks without a numeric prefix."
    )


def resolve_benchmark_task_seed_id(scene_name: str) -> int:
    """Return a stable task seed partition id for benchmark scene names.

    Standard Dex2Bench task names use a numeric prefix and keep the same
    partition as collection/eval. External benchmark scene names may not have a
    prefix, so they get a deterministic hash bucket instead of sharing one
    global seed stream.
    """
    try:
        return resolve_task_seed_id(scene_name)
    except ValueError:
        digest = hashlib.sha1(str(scene_name).encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % BENCHMARK_HASH_TASK_ID_BUCKETS
        return BENCHMARK_HASH_TASK_ID_OFFSET + bucket


def task_base_seed(base_seed: int, task_seed_id: int, *, task_seed_stride: int = TASK_SEED_STRIDE) -> int:
    base = validate_seed(base_seed, "base_seed")
    task_id = int(task_seed_id)
    if task_id < 0:
        raise ValueError("task_seed_id must be non-negative.")
    stride = int(task_seed_stride)
    if stride <= 0:
        raise ValueError("task_seed_stride must be positive.")
    return validate_seed(base + task_id * stride, "task_base_seed")


def episode_seed(task_base_seed_value: int, episode_index: int) -> int:
    index = int(episode_index)
    if index < 1:
        raise ValueError("episode_index must be 1-indexed and >= 1.")
    return validate_seed(int(task_base_seed_value) + index - 1, "episode_seed")


def episode_seed_context(
    *,
    seed_namespace: str,
    base_seed: int,
    task_seed_id: int,
    episode_index: int,
) -> EpisodeSeedContext:
    task_base = task_base_seed(base_seed, task_seed_id)
    return EpisodeSeedContext(
        seed_namespace=str(seed_namespace),
        base_seed=validate_seed(base_seed, "base_seed"),
        task_seed_id=int(task_seed_id),
        task_base_seed=task_base,
        episode_index=int(episode_index),
        episode_seed=episode_seed(task_base, episode_index),
    )


def collection_output_dir(dataset_root: str, dataset_name: str, task_path: str) -> str:
    task_name = os.path.basename(os.path.dirname(os.path.abspath(task_path))) or "task"
    scene_name = os.path.splitext(os.path.basename(task_path))[0]
    return os.path.abspath(os.path.join(dataset_root, dataset_name, task_name, scene_name))


def _read_hdf5_scalar(path: str, dataset_path: str) -> Any:
    try:
        import h5py
    except Exception:
        return None
    try:
        with h5py.File(path, "r") as f:
            if dataset_path not in f:
                return None
            value = f[dataset_path][()]
    except Exception:
        return None
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def scan_collection_seed_state(output_dir: str, task_base_seed_value: int) -> tuple[int, int, int, int]:
    """Return ``(next_episode_index, next_episode_seed, file_count, legacy_count)``."""
    if not os.path.isdir(output_dir):
        return 1, episode_seed(task_base_seed_value, 1), 0, 0

    file_count = 0
    legacy_count = 0
    max_episode_file_id: int | None = None
    max_episode_seed: int | None = None
    for filename in sorted(os.listdir(output_dir)):
        match = _EPISODE_PATTERN.match(filename)
        if not match:
            continue
        episode_file_id = int(match.group(1))
        max_episode_file_id = episode_file_id if max_episode_file_id is None else max(max_episode_file_id, episode_file_id)
        file_count += 1
        path = os.path.join(output_dir, filename)
        raw_seed = _read_hdf5_scalar(path, "meta/episode_seed")
        if raw_seed is None:
            legacy_count += 1
            continue
        try:
            parsed_seed = validate_seed(int(raw_seed), "meta/episode_seed")
        except (TypeError, ValueError):
            legacy_count += 1
            continue
        max_episode_seed = parsed_seed if max_episode_seed is None else max(max_episode_seed, parsed_seed)

    if max_episode_seed is not None:
        next_seed = validate_seed(max_episode_seed + 1, "next_episode_seed")
        if next_seed >= task_base_seed_value:
            next_index = next_seed - int(task_base_seed_value) + 1
        else:
            next_index = file_count + 1
        return int(next_index), next_seed, file_count, legacy_count

    if max_episode_file_id is not None:
        # Episode filenames are zero-indexed while episode_index is one-indexed.
        # Use the next writer id instead of file_count so sparse legacy dirs
        # continue with seeds aligned to episode_<id>.hdf5.
        next_index = int(max_episode_file_id) + 2
    else:
        next_index = file_count + 1
    return next_index, episode_seed(task_base_seed_value, next_index), file_count, legacy_count


class CollectionSeedAllocator:
    def __init__(
        self,
        *,
        base_seed: int,
        task_seed_id: int,
        output_dir: str,
    ):
        self.base_seed = validate_seed(base_seed, "base_seed")
        self.task_seed_id = int(task_seed_id)
        self.task_base_seed = task_base_seed(self.base_seed, self.task_seed_id)
        self.output_dir = os.path.abspath(output_dir)
        next_index, _next_seed, file_count, legacy_count = scan_collection_seed_state(
            self.output_dir,
            self.task_base_seed,
        )
        self.next_episode_index = int(next_index)
        self.existing_file_count = int(file_count)
        self.legacy_file_count = int(legacy_count)

    def next(self) -> EpisodeSeedContext:
        context = EpisodeSeedContext(
            seed_namespace=COLLECTION_SEED_NAMESPACE,
            base_seed=self.base_seed,
            task_seed_id=self.task_seed_id,
            task_base_seed=self.task_base_seed,
            episode_index=self.next_episode_index,
            episode_seed=episode_seed(self.task_base_seed, self.next_episode_index),
        )
        self.next_episode_index += 1
        return context
