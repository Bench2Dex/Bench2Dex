#!/usr/bin/env python3
"""Fix corrupted per_episode.jsonl files with duplicate episode_index values.

After a broken resume, per_episode.jsonl may contain duplicate episode_index rows.
This script:
1. Deduplicates per_episode.jsonl (keeps first occurrence of each episode_index)
2. Regenerates summary.json and per_task.json
3. Removes duplicate HDF5 recordings (_dup001 suffix)
4. Regenerates success_rates.tsv

Usage:
    python tools/fix_resume_duplicates.py <eval_root_dir> [--dry-run] [--channels ch1 ch2 ...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmark.results import (
    EpisodeResult,
    load_episode_results,
    write_summary_files,
    write_results,
)


def deduplicate_per_episode(jsonl_path: Path, dry_run: bool = False) -> dict:
    """Deduplicate per_episode.jsonl by episode_index. Returns stats."""
    if not jsonl_path.exists():
        return {"status": "missing", "original_lines": 0, "unique": 0, "duplicates": 0}

    # Read all lines
    with open(jsonl_path, "r") as f:
        lines = f.readlines()

    seen_indices: set[int] = set()
    unique_lines: list[str] = []
    dup_count = 0
    index_counts: Counter = Counter()

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        try:
            payload = json.loads(line_stripped)
            ep_idx = int(payload.get("episode_index", -1))
            index_counts[ep_idx] += 1
            if ep_idx in seen_indices:
                dup_count += 1
                continue
            seen_indices.add(ep_idx)
            unique_lines.append(line_stripped)
        except (json.JSONDecodeError, TypeError, ValueError):
            # Keep malformed lines (shouldn't happen but be safe)
            unique_lines.append(line_stripped)

    # Sort by episode_index if possible
    def _extract_index(line_str: str) -> int:
        try:
            return int(json.loads(line_str).get("episode_index", 0))
        except Exception:
            return 0

    unique_lines.sort(key=_extract_index)

    stats = {
        "status": "fixed" if dup_count > 0 else "clean",
        "original_lines": len([l for l in lines if l.strip()]),
        "unique": len(unique_lines),
        "duplicates": dup_count,
        "duplicate_indices": sorted(
            [idx for idx, cnt in index_counts.items() if cnt > 1]
        ),
    }

    if dry_run:
        return stats

    # Backup original
    backup_path = jsonl_path.with_suffix(".jsonl.bak")
    with open(backup_path, "w") as f:
        f.writelines(lines)
    print(f"  Backup saved: {backup_path}")

    # Write deduplicated
    with open(jsonl_path, "w") as f:
        for line in unique_lines:
            f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())

    print(f"  Deduplicated: {jsonl_path} ({stats['original_lines']} -> {stats['unique']} lines)")
    return stats


def regenerate_summaries(channel_dir: Path, policy_name: str = "DP") -> None:
    """Regenerate per_task.json and summary.json from per_episode.jsonl."""
    results = load_episode_results(channel_dir)
    if not results:
        print(f"  WARNING: No results loaded from {channel_dir}, skipping summary regeneration")
        return
    write_summary_files(channel_dir, policy_name, results)
    print(f"  Regenerated summaries: {channel_dir} ({len(results)} episodes)")


def cleanup_duplicate_recordings(channel_dir: Path, dry_run: bool = False) -> int:
    """Remove HDF5 files with _dup001 suffix (duplicate recordings)."""
    episodes_dir = channel_dir / "episodes"
    if not episodes_dir.exists():
        return 0

    removed = 0
    for subdir in ["success", "failure"]:
        sd = episodes_dir / subdir
        if not sd.exists():
            continue
        for hdf5_file in sorted(sd.glob("*.hdf5")):
            if "_dup" in hdf5_file.name:
                if dry_run:
                    print(f"  [DRY-RUN] Would remove: {hdf5_file}")
                else:
                    hdf5_file.unlink()
                    print(f"  Removed: {hdf5_file}")
                removed += 1

    return removed


def regenerate_success_rates(eval_root: Path, channels: list[str]) -> None:
    """Regenerate success_rates.tsv."""
    tsv_path = eval_root / "success_rates.tsv"

    lines = ["channel\tepisodes\tsuccesses\tsuccess_rate\tpct\toutput_dir"]
    overall_total = 0
    overall_success = 0

    for ch in channels:
        ch_dir = eval_root / ch
        jsonl_path = ch_dir / "per_episode.jsonl"
        if not jsonl_path.exists():
            lines.append(f"{ch}\t-1\tn/a\tn/a\tn/a\t{ch_dir}")
            continue

        total = 0
        successes = 0
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    total += 1
                    if row.get("success"):
                        successes += 1
                except json.JSONDecodeError:
                    pass

        if total > 0:
            rate = successes / total
            lines.append(
                f"{ch}\t{total}\t{successes}\t{rate:.6f}\t{rate*100:.2f}%\t{ch_dir}"
            )
            overall_total += total
            overall_success += successes
        else:
            lines.append(f"{ch}\t0\t0\t0.000000\t0.00%\t{ch_dir}")

    if overall_total > 0:
        overall_rate = overall_success / overall_total
        lines.append(
            f"OVERALL\t{overall_total}\t{overall_success}\t{overall_rate:.6f}\t{overall_rate*100:.2f}%\t"
        )
    else:
        lines.append("OVERALL\t0\t0\tn/a\tn/a\t")

    with open(tsv_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Regenerated: {tsv_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_root", type=Path, help="Path to the eval root directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument(
        "--channels",
        nargs="+",
        default=["none", "cov_only", "inv_only", "inv_cov"],
        help="Channel subdirectories to process (default: all 4)",
    )
    parser.add_argument(
        "--policy-name",
        default=None,
        help="Policy display name for summary.json (auto-detected if not set)",
    )
    args = parser.parse_args()

    root = args.eval_root.expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: Directory not found: {root}")
        sys.exit(1)

    # Auto-detect channels that actually exist
    existing_channels = [ch for ch in args.channels if (root / ch).is_dir()]
    if not existing_channels:
        print(f"ERROR: No channel directories found in {root}")
        print(f"  Expected: {args.channels}")
        sys.exit(1)

    print(f"Eval root: {root}")
    print(f"Channels: {existing_channels}")
    if args.dry_run:
        print("*** DRY RUN MODE ***")
    print()

    total_dups = 0
    total_recording_dups = 0

    for ch in existing_channels:
        ch_dir = root / ch
        print(f"=== Channel: {ch} ===")

        # Step 1: Deduplicate per_episode.jsonl
        jsonl_path = ch_dir / "per_episode.jsonl"
        stats = deduplicate_per_episode(jsonl_path, dry_run=args.dry_run)
        print(f"  per_episode.jsonl: {stats}")
        total_dups += stats["duplicates"]

        # Step 2: Remove duplicate recordings
        removed = cleanup_duplicate_recordings(ch_dir, dry_run=args.dry_run)
        if removed > 0:
            total_recording_dups += removed
            print(f"  Removed {removed} duplicate recording(s)")

        # Step 3: Regenerate summaries
        if not args.dry_run:
            # Auto-detect policy name from existing summary if available
            policy_name = args.policy_name
            if policy_name is None:
                summary_path = ch_dir / "summary.json"
                if summary_path.exists():
                    try:
                        with open(summary_path) as f:
                            old_summary = json.load(f)
                        policy_name = (
                            old_summary.get("metadata", {}).get("policy_name", "DP")
                        )
                    except Exception:
                        policy_name = "DP"
                else:
                    policy_name = "DP"

            regenerate_summaries(ch_dir, policy_name)

        print()

    # Step 4: Regenerate success_rates.tsv
    if not args.dry_run:
        regenerate_success_rates(root, existing_channels)
    else:
        print("[DRY-RUN] Would regenerate success_rates.tsv")

    # Summary
    print()
    print("=" * 60)
    print(f"Total per_episode duplicates removed: {total_dups}")
    print(f"Total recording duplicates removed: {total_recording_dups}")
    if args.dry_run:
        print("DRY RUN — no changes made. Remove --dry-run to apply fixes.")
    else:
        print("Done. Backup files saved as *.jsonl.bak")


if __name__ == "__main__":
    main()
