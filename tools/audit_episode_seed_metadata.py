"""Audit per-episode seed metadata in a replay HDF5 directory."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from collections import Counter
from typing import Any


def _read_scalar(h5, path: str) -> Any:
    if path not in h5:
        return None
    value = h5[path][()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def audit_replay_dir(replay_dir: str) -> dict[str, Any]:
    import h5py

    files = sorted(glob.glob(os.path.join(os.path.abspath(replay_dir), "episode_*.hdf5")))
    seed_counts: Counter[int | None] = Counter()
    legacy_files: list[str] = []
    duplicate_seeds: dict[int, int] = {}
    sample_hash_counts: Counter[str] = Counter()
    base_seed_counts: Counter[int | None] = Counter()
    seed_policy_counts: Counter[str | None] = Counter()

    for path in files:
        with h5py.File(path, "r") as h5:
            episode_seed = _read_scalar(h5, "meta/episode_seed")
            base_seed = _read_scalar(h5, "meta/base_seed")
            seed_policy = _read_scalar(h5, "meta/seed_policy")
            sample = _read_scalar(h5, "meta/scene_generalization_sample")

        try:
            parsed_episode_seed = int(episode_seed) if episode_seed is not None else None
        except (TypeError, ValueError):
            parsed_episode_seed = None
        try:
            parsed_base_seed = int(base_seed) if base_seed is not None else None
        except (TypeError, ValueError):
            parsed_base_seed = None

        if parsed_episode_seed is None:
            legacy_files.append(os.path.basename(path))
        seed_counts[parsed_episode_seed] += 1
        base_seed_counts[parsed_base_seed] += 1
        seed_policy_counts[str(seed_policy) if seed_policy is not None else None] += 1
        if isinstance(sample, str) and sample:
            sample_hash_counts[hashlib.sha1(sample.encode("utf-8")).hexdigest()[:12]] += 1

    duplicate_seeds = {
        int(seed): count
        for seed, count in seed_counts.items()
        if seed is not None and count > 1
    }
    duplicate_sample_hashes = {
        sample_hash: count
        for sample_hash, count in sample_hash_counts.items()
        if count > 1
    }
    return {
        "replay_dir": os.path.abspath(replay_dir),
        "episode_file_count": len(files),
        "legacy_missing_episode_seed_count": len(legacy_files),
        "legacy_missing_episode_seed_examples": legacy_files[:10],
        "unique_episode_seed_count": len([seed for seed in seed_counts if seed is not None]),
        "duplicate_episode_seeds": duplicate_seeds,
        "base_seed_counts": {str(key): value for key, value in base_seed_counts.items()},
        "seed_policy_counts": {str(key): value for key, value in seed_policy_counts.items()},
        "unique_scene_generalization_sample_hash_count": len(sample_hash_counts),
        "duplicate_scene_generalization_sample_hashes": duplicate_sample_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit replay HDF5 episode seed metadata.")
    parser.add_argument("replay_dir", help="Directory containing episode_*.hdf5 files.")
    args = parser.parse_args()
    print(json.dumps(audit_replay_dir(args.replay_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
