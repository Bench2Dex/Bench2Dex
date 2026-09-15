#!/usr/bin/env python3
"""
Record Manus hand skeleton data from shared memory for offline testing.

Usage:
    conda activate manus
    python teleop/tests/record_manus.py --duration 10 --out teleop/tests/data/manus_right.npz

Requires the Manus SDK MinimalClient to be running (writing to SHM).
"""

import argparse
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from teleop.shm_reader import ShmReader


def main():
    parser = argparse.ArgumentParser(description="Record Manus SHM data")
    parser.add_argument("--duration", type=float, default=10.0, help="Recording duration (seconds)")
    parser.add_argument("--side", choices=["right", "left", "both"], default="right")
    parser.add_argument("--out", type=str, default="teleop/tests/data/manus_recording.npz")
    parser.add_argument("--hz", type=float, default=60.0, help="Target sample rate")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    reader = ShmReader()
    reader.open()

    if not reader.wait_for_data(timeout=5.0):
        print("ERROR: No Manus data in SHM. Is the SDK client running?")
        reader.close()
        return

    dt = 1.0 / args.hz
    frames_right = []
    frames_left = []
    timestamps = []

    print(f"Recording {args.side} hand(s) for {args.duration}s at ~{args.hz}Hz...")
    t_start = time.monotonic()
    while time.monotonic() - t_start < args.duration:
        frame = reader.read_if_new()
        if frame is None:
            time.sleep(dt / 4)
            continue

        t = time.monotonic() - t_start
        timestamps.append(t)

        if args.side in ("right", "both"):
            pd = reader.get_pos_dict(frame.right)
            if pd:
                arr = np.zeros((25, 3), dtype=np.float32)
                for k, v in pd.items():
                    arr[k] = v
                frames_right.append(arr)
            else:
                frames_right.append(np.full((25, 3), np.nan, dtype=np.float32))

        if args.side in ("left", "both"):
            pd = reader.get_pos_dict(frame.left)
            if pd:
                arr = np.zeros((25, 3), dtype=np.float32)
                for k, v in pd.items():
                    arr[k] = v
                frames_left.append(arr)
            else:
                frames_left.append(np.full((25, 3), np.nan, dtype=np.float32))

        time.sleep(dt)

    reader.close()

    save_dict = {"timestamps": np.array(timestamps)}
    if frames_right:
        save_dict["right"] = np.stack(frames_right)
    if frames_left:
        save_dict["left"] = np.stack(frames_left)

    np.savez_compressed(args.out, **save_dict)
    n = len(timestamps)
    print(f"Saved {n} frames ({n / args.duration:.1f} fps) to {args.out}")


if __name__ == "__main__":
    main()
