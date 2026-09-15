#!/usr/bin/env python3
"""Merge eval output directories and recompute benchmark summaries.

This is intended for historical runs that were split across multiple output
directories. New chunked eval should write every chunk for a channel into the
same output directory with run_policy.py append mode.

Usage:
  python tools/merge_eval_chunks.py <dir1> <dir2> ... -o <output_dir> [--force]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.results import episode_result_from_dict, write_summary_files


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not path.is_file():
        return entries
    with open(path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse {path}:{line_no}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_no}")
            entries.append(payload)
    return entries


def write_jsonl(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_entries(src_dirs: list[Path]) -> tuple[list[dict[str, Any]], dict[int, Path]]:
    by_episode: dict[int, dict[str, Any]] = {}
    src_map: dict[int, Path] = {}

    for src_dir in src_dirs:
        entries = read_jsonl(src_dir / "per_episode.jsonl")
        if not entries:
            print(f"[WARN] No per_episode.jsonl entries in {src_dir}")
            continue

        kept = 0
        for entry in entries:
            raw_idx = entry.get("episode_index")
            if raw_idx is None:
                print(f"[WARN] Skipping row without episode_index from {src_dir}")
                continue
            episode_idx = int(raw_idx)
            if episode_idx in by_episode:
                print(f"[WARN] Duplicate episode_index={episode_idx}; keeping first from {src_map[episode_idx]}")
                continue

            normalized = dict(entry)
            if "safety_hard_violation" not in normalized and "hard_violation" in normalized:
                normalized["safety_hard_violation"] = bool(normalized.get("hard_violation"))
            by_episode[episode_idx] = normalized
            src_map[episode_idx] = src_dir
            kept += 1

        print(f"  Loaded {kept}/{len(entries)} unique episodes from {src_dir.name}")

    merged = [by_episode[idx] for idx in sorted(by_episode)]
    return merged, src_map


def episode_record_dirs(src_base: Path, success: bool) -> list[Path]:
    subdir = "success" if success else "failure"
    dirs = [src_base / "episodes" / subdir]
    dirs.extend(sorted(src_base.glob(f"chunks/*/episodes/{subdir}")))
    return [path for path in dirs if path.is_dir()]


def find_hdf5_for_episode(src_base: Path, episode_index: int, success: bool) -> Path | None:
    """Find a recorded HDF5 by absolute episode index.

    Supported names:
    - episode_000041.hdf5              current absolute episode naming
    - episode_000041_dup001.hdf5       current conflict fallback
    - episode_000000_41.hdf5           legacy sequential-plus-episode suffix
    """
    exact = f"episode_{episode_index:06d}.hdf5"
    dup_prefix = f"episode_{episode_index:06d}_dup"
    legacy_suffix = re.compile(rf"^episode_\d{{6}}_{episode_index}\.hdf5$")
    legacy_padded_suffix = re.compile(rf"^episode_\d{{6}}_{episode_index:06d}\.hdf5$")

    for ep_dir in episode_record_dirs(src_base, success):
        exact_path = ep_dir / exact
        if exact_path.is_file():
            return exact_path

        dup_matches = sorted(path for path in ep_dir.glob(f"{dup_prefix}*.hdf5") if path.is_file())
        if dup_matches:
            return dup_matches[0]

        for path in sorted(ep_dir.glob("episode_*.hdf5")):
            if legacy_suffix.match(path.name) or legacy_padded_suffix.match(path.name):
                return path

    return None


def unique_episode_path(target_dir: Path, episode_index: int) -> Path:
    stem = f"episode_{episode_index:06d}"
    path = target_dir / f"{stem}.hdf5"
    if not path.exists():
        return path

    suffix = 1
    while True:
        candidate = target_dir / f"{stem}_dup{suffix:03d}.hdf5"
        if not candidate.exists():
            return candidate
        suffix += 1


def copy_hdf5_files(entries: list[dict[str, Any]], src_map: dict[int, Path], output_dir: Path) -> tuple[int, int]:
    success_dir = output_dir / "episodes" / "success"
    failure_dir = output_dir / "episodes" / "failure"
    success_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)

    n_success = 0
    n_failure = 0
    for entry in entries:
        episode_index = int(entry["episode_index"])
        success = bool(entry.get("success"))
        src_base = src_map[episode_index]
        hdf5_path = find_hdf5_for_episode(src_base, episode_index, success)
        if hdf5_path is None:
            print(f"  [WARN] HDF5 not found for episode {episode_index} (success={success})")
            continue

        target_dir = success_dir if success else failure_dir
        dest = unique_episode_path(target_dir, episode_index)
        shutil.copy2(hdf5_path, dest)
        if success:
            n_success += 1
        else:
            n_failure += 1

    return n_success, n_failure


def infer_policy_name(src_dirs: list[Path], override: str | None) -> str:
    if override:
        return override

    for src_dir in src_dirs:
        summary_path = src_dir / "summary.json"
        if not summary_path.is_file():
            continue
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            continue
        policy_name = (summary.get("metadata") or {}).get("policy_name")
        if policy_name:
            return str(policy_name)

    return "merged"


def recompute_summaries(output_dir: Path, policy_name: str, entries: list[dict[str, Any]]) -> None:
    results = []
    for idx, entry in enumerate(entries, start=1):
        try:
            results.append(episode_result_from_dict(entry))
        except Exception as exc:
            raise ValueError(f"Failed to convert merged episode row #{idx}: {exc}") from exc
    write_summary_files(output_dir, policy_name, results)


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge chunked eval output directories.")
    parser.add_argument("dirs", nargs="+", help="Source eval directories to merge.")
    parser.add_argument("-o", "--output", required=True, help="Output directory.")
    parser.add_argument("--policy-name", default=None, help="Policy name for recomputed summary metadata.")
    parser.add_argument("--force", action="store_true", help="Replace output directory if it already exists.")
    args = parser.parse_args()

    src_dirs = [Path(path).resolve() for path in args.dirs]
    output_dir = Path(args.output).resolve()

    for src_dir in src_dirs:
        if not src_dir.is_dir():
            print(f"[ERROR] Not a directory: {src_dir}", file=sys.stderr)
            return 1

    entries, src_map = load_entries(src_dirs)
    if not entries:
        print("[ERROR] No episodes to merge.", file=sys.stderr)
        return 1

    first_ep = int(entries[0]["episode_index"])
    last_ep = int(entries[-1]["episode_index"])
    print(f"\n  Total episodes: {len(entries)}")
    print(f"  Range: {first_ep} - {last_ep}")

    if output_dir.exists():
        if not args.force:
            print(f"[ERROR] Output exists: {output_dir}. Use --force to replace it.", file=sys.stderr)
            return 1
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n  Merging HDF5 files...")
    n_success, n_failure = copy_hdf5_files(entries, src_map, output_dir)
    print(f"  success: {n_success} HDF5 files")
    print(f"  failure: {n_failure} HDF5 files")

    write_jsonl(output_dir / "per_episode.jsonl", entries)
    print(f"\n  Wrote per_episode.jsonl ({len(entries)} lines)")

    policy_name = infer_policy_name(src_dirs, args.policy_name)
    recompute_summaries(output_dir, policy_name, entries)
    print(f"  Recomputed per_task.json and summary.json (policy_name={policy_name})")

    print(f"\n  Done. Merged output: {output_dir}")
    print(f"  Success HDF5: {n_success} | Failure HDF5: {n_failure} | Episodes: {len(entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
