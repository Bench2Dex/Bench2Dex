"""Batch retarget HDF5 episodes from old HOME joint positions to new HOME.

Usage:
    # Dry-run: show what would change (no files modified)
    python tools/replay/retarget_home.py \
        --data-dir /path/to/episodes \
        --old-home 1.57,-1.57 \
        --new-home -1.57,1.57 \
        --joints shoulder_pan_joint,L_arm_shoulder_pan_joint \
        --dry-run

    # Apply the transformation
    python tools/replay/retarget_home.py \
        --data-dir /path/to/episodes \
        --old-home 1.57,-1.57 \
        --new-home -1.57,1.57 \
        --joints shoulder_pan_joint,L_arm_shoulder_pan_joint
"""

import argparse
import os
import shutil
from pathlib import Path

import h5py
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch retarget HDF5 joint data to a new HOME position."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing episode_*.hdf5 files.",
    )
    parser.add_argument(
        "--old-home",
        type=str,
        required=True,
        help="Comma-separated old HOME joint values (rad), order must match --joints.",
    )
    parser.add_argument(
        "--new-home",
        type=str,
        required=True,
        help="Comma-separated new HOME joint values (rad), order must match --joints.",
    )
    parser.add_argument(
        "--joints",
        type=str,
        required=True,
        help="Comma-separated joint names to retarget.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without modifying files.",
    )
    parser.add_argument(
        "--backup-suffix",
        type=str,
        default=".old_home",
        help="Suffix for backup files. Set to empty to disable backup.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"[ERROR] Directory not found: {data_dir}")
        return 1

    old_home = [float(x.strip()) for x in args.old_home.split(",")]
    new_home = [float(x.strip()) for x in args.new_home.split(",")]
    joint_names = [x.strip() for x in args.joints.split(",")]

    if len(old_home) != len(new_home) or len(old_home) != len(joint_names):
        print(
            "[ERROR] --old-home, --new-home, and --joints must have the same length: "
            f"{len(old_home)} vs {len(new_home)} vs {len(joint_names)}"
        )
        return 1

    offsets = {name: new - old for name, old, new in zip(joint_names, old_home, new_home)}
    print("Retarget offsets:")
    for name, offset in offsets.items():
        print(f"  {name}: {offset:+.4f} rad ({np.rad2deg(offset):+.0f}°)")

    episodes = sorted(data_dir.glob("episode_*.hdf5"))
    if not episodes:
        print(f"[ERROR] No episode_*.hdf5 files found in {data_dir}")
        return 1

    print(f"\nEpisodes: {len(episodes)}")
    print(f"Mode: {'DRY-RUN (no changes)' if args.dry_run else 'APPLY'}")
    print()

    changed_count = 0
    skipped_count = 0

    for idx, hdf5_path in enumerate(episodes, start=1):
        name = hdf5_path.name
        print(f"[{idx:3d}/{len(episodes)}] {name} ...", end=" ", flush=True)

        with h5py.File(hdf5_path, "r") as f:
            hdf5_joint_names = [
                n.decode() if isinstance(n, bytes) else str(n)
                for n in f["robot/joint_names"][:]
            ]
            qpos = f["robot/qpos"][:]  # (N, n_joints)

        # Build index map
        joint_indices = {}
        for target_name in joint_names:
            if target_name in hdf5_joint_names:
                joint_indices[target_name] = hdf5_joint_names.index(target_name)
            else:
                print(f"WARN: joint '{target_name}' not found in HDF5, skipping")

        if not joint_indices:
            print("no matching joints, skipped")
            skipped_count += 1
            continue

        # Compute per-joint offset and check if change is needed
        frame0_before = {n: qpos[0, i] for n, i in joint_indices.items()}
        frame0_after = {n: qpos[0, i] + offsets[n] for n, i in joint_indices.items()}

        # Show before/after for first episode
        if idx == 1:
            for n, i in sorted(joint_indices.items(), key=lambda x: x[1]):
                before = qpos[0, i]
                after = before + offsets[n]
                print(f"\n      {n}: {before:+.4f} → {after:+.4f}", end="")
            print()

        if args.dry_run:
            print("dry-run OK")
            changed_count += 1
            continue

        # Apply offset
        qpos_new = qpos.copy()
        for name, i in joint_indices.items():
            qpos_new[:, i] += offsets[name]

        # Backup
        if args.backup_suffix:
            backup_path = hdf5_path.with_suffix(hdf5_path.suffix + args.backup_suffix)
            if not backup_path.exists():
                shutil.copy2(hdf5_path, backup_path)

        # Write back
        with h5py.File(hdf5_path, "r+") as f:
            del f["robot/qpos"]
            f.create_dataset("robot/qpos", data=qpos_new, dtype="float32")

        print("OK")
        changed_count += 1

    print(f"\nDone: {changed_count} changed, {skipped_count} skipped")
    if args.dry_run:
        print("DRY-RUN — no files were modified. Remove --dry-run to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
