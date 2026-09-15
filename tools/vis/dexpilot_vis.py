#!/usr/bin/env python3
"""
dexpilot_vis.py -- Hand keypoint + DexPilot robot FK visualizer.

Reads live Manus SHM data, converts to MediaPipe 21 keypoints,
and draws the hand skeleton (blue) alongside the DexPilot retarget
result via pinocchio FK (red) in real-time.

Usage:
    cd .
    python tools/vis/dexpilot_vis.py --hand allegro
    python tools/vis/dexpilot_vis.py --hand allegro --raw   # human only, no transform
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from teleop.shm_reader import ShmReader
from teleop.retarget_bridge import (
    RetargetBridge, manus_nodes_to_mediapipe, _wrist_frame_from_quat,
    _HAND_TYPE_TO_CONFIG,
)

# MediaPipe hand skeleton connections (parent -> child)
BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
]

FINGER_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
FINGER_NAMES = ['thumb', 'index', 'middle', 'ring', 'pinky']
TIP_INDICES = [4, 8, 12, 16, 20]


def bone_color(bi):
    return FINGER_COLORS[bi // 4]


def build_robot_bones(model):
    """Build bone connections from pinocchio model BODY frames via previousFrame chain."""
    body_ids = [i for i in range(model.nframes)
                if 'BODY' in str(model.frames[i].type)]
    body_set = set(body_ids)
    bones = []
    for bid in body_ids:
        prev = model.frames[bid].previousFrame
        visited = set()
        while prev > 0 and prev not in visited:
            visited.add(prev)
            if prev in body_set:
                bones.append((prev, bid))
                break
            prev = model.frames[prev].previousFrame
    return bones, body_ids


def main():
    parser = argparse.ArgumentParser(description="Hand keypoint live visualizer")
    parser.add_argument("--hand", type=str, default="allegro",
                        choices=list(_HAND_TYPE_TO_CONFIG.keys()))
    parser.add_argument("--side", type=str, default="right", choices=["left", "right"])
    parser.add_argument("--fps", type=float, default=10.0, help="Update rate Hz")
    parser.add_argument("--raw", action="store_true",
                        help="Show raw keypoints only (no transform, no DexPilot)")
    args = parser.parse_args()

    print(f"[Vis] hand={args.hand}, side={args.side}, raw={args.raw}")

    reader = ShmReader()
    if not reader.open():
        print("[Vis] ERROR: Cannot open Manus SHM")
        sys.exit(1)

    bridge = RetargetBridge(hand_type=args.hand, hand_side=args.side)

    # Setup robot FK info — differs between DexPilot and wuji retarget

    opt = bridge._retargeting.optimizer
    n_fingers = opt.num_fingers
    num_pairs = n_fingers * (n_fingers - 1) // 2

    # Get fingertip link indices in computed_link_indices
    tip_computed_indices = []
    for fi in range(n_fingers):
        vi = num_pairs + fi
        tip_computed_indices.append(int(opt.task_link_indices[vi]))

    robot_model = opt.robot.model
    n_frames = robot_model.nframes
    robot_bones, body_frame_ids = build_robot_bones(robot_model)

    # Get frame names for debugging
    frame_names = [robot_model.frames[i].name for i in range(n_frames)]
    print(f"[Vis] Robot model: {n_frames} frames, {len(robot_bones)} bones, "
          f"{n_fingers} fingers")

    plt.ion()
    # Fix Tk scaling bug: system reports ~0.02 instead of ~1.0,
    # causing PIL toolbar button resize to fail (height=0).
    import tkinter as tk
    _tk_root = tk._default_root or tk.Tk()
    _tk_root.withdraw()
    _tk_root.tk.call('tk', 'scaling', 1.0)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    dt = 1.0 / args.fps
    frame_count = 0
    last_reload_check = 0.0

    print(f"[Vis] Running at {args.fps} Hz. Close window or Ctrl+C to exit.")

    try:
        while plt.fignum_exists(fig.number):
            t0 = time.time()

            # Hot-reload tunable yaml params every ~1s.
            if t0 - last_reload_check > 1.0:
                last_reload_check = t0
                if hasattr(bridge, "reload_tunables_if_changed"):
                    bridge.reload_tunables_if_changed()

            shm_frame = reader.read()
            if shm_frame is None:
                plt.pause(dt)
                continue

            hand_data = shm_frame.right if args.side == "right" else shm_frame.left
            if not hand_data.valid or hand_data.node_count < 21:
                plt.pause(dt)
                continue

            pos_dict = {}
            for nid in range(hand_data.node_count):
                pos_dict[nid] = hand_data.positions[nid].copy()
            wrist_quat = hand_data.quaternions[0]  # node 0 = wrist (qx,qy,qz,qw)

            keypoints = manus_nodes_to_mediapipe(pos_dict)
            if keypoints is None:
                plt.pause(dt)
                continue

            # Center on wrist
            keypoints = keypoints - keypoints[0:1, :]

            if args.raw:
                pts = keypoints
            else:
                wrist_frame = _wrist_frame_from_quat(wrist_quat)
                if bridge._quat_correction_R is not None:
                    wrist_frame = wrist_frame @ bridge._quat_correction_R
                pts = (keypoints @ wrist_frame @ bridge._operator2mano).astype(np.float32)
                # Apply mano_to_pinocchio correction (matches retarget_bridge ref_value transform)
                if bridge._mano_to_pin_R is not None:
                    pts = (pts @ bridge._mano_to_pin_R.T).astype(np.float32)
                # Apply hand proportion correction (offset + scale from YAML config)
                # Offset only on non-wrist (index 1+) so wrist→tip vectors change,
                # matching the behavior in retarget_bridge.retarget().
                if bridge._hand_scale is not None:
                    pts = pts * bridge._hand_scale
                if bridge._hand_offset is not None:
                    pts[1:] = pts[1:] + bridge._hand_offset

            # ---- Robot FK (retarget result) ----
            robot_pts = None
            robot_tip_pts = None
            if not args.raw:
                qpos = bridge.retarget(pos_dict, wrist_quat=wrist_quat)
                if qpos is not None:
                    opt.robot.compute_forward_kinematics(qpos)
                    robot_all = np.zeros((n_frames, 3), dtype=np.float32)
                    for fid in body_frame_ids:
                        pose = opt.robot.get_link_pose(fid)
                        robot_all[fid] = pose[:3, 3]
                    wrist_ci = int(opt.origin_link_indices[num_pairs])
                    wrist_fid = opt.computed_link_indices[wrist_ci]
                    robot_wrist_pos = robot_all[wrist_fid].copy()
                    robot_pts = robot_all - robot_wrist_pos
                    robot_tip_pts = np.zeros((n_fingers, 3), dtype=np.float32)
                    for fi in range(n_fingers):
                        ci = tip_computed_indices[fi]
                        fid = opt.computed_link_indices[ci]
                        robot_tip_pts[fi] = robot_pts[fid]

            # ---- Draw ----
            ax.cla()
            mode = "RAW" if args.raw else "Blue=Human  Red=Robot"
            ax.set_title(f"{mode} | frame {frame_count}", fontsize=11)

            # Draw human bones (blue)
            for bi, (p, c) in enumerate(BONES):
                ax.plot([pts[p, 0], pts[c, 0]],
                        [pts[p, 1], pts[c, 1]],
                        [pts[p, 2], pts[c, 2]],
                        color=bone_color(bi), linewidth=2.5, alpha=0.8)

            # Draw human joints
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       c='tab:blue', s=40, alpha=0.9, depthshade=True)

            # Label human fingertips
            for fi, ti in enumerate(TIP_INDICES):
                ax.text(pts[ti, 0], pts[ti, 1], pts[ti, 2],
                        f"  {FINGER_NAMES[fi]}", fontsize=10, fontweight='bold',
                        color=FINGER_COLORS[fi])

            # Draw robot skeleton (red) if available
            if robot_pts is not None:
                # Draw robot bones
                for pfid, cfid in robot_bones:
                    ax.plot([robot_pts[pfid, 0], robot_pts[cfid, 0]],
                            [robot_pts[pfid, 1], robot_pts[cfid, 1]],
                            [robot_pts[pfid, 2], robot_pts[cfid, 2]],
                            color='tab:red', linewidth=2.0, alpha=0.6)

                # Draw robot joints (only BODY frames)
                rpts = robot_pts[body_frame_ids]
                ax.scatter(rpts[:, 0], rpts[:, 1], rpts[:, 2],
                           c='tab:red', s=25, alpha=0.7, depthshade=True)

                # Label robot fingertips
                if robot_tip_pts is not None:
                    for fi in range(n_fingers):
                        ax.text(robot_tip_pts[fi, 0], robot_tip_pts[fi, 1],
                                robot_tip_pts[fi, 2],
                                f"  R_{FINGER_NAMES[fi]}", fontsize=9,
                                color='tab:red')

            # Wrist marker
            ax.scatter(0, 0, 0, c='black', s=80, marker='o', zorder=10)

            # Axis limits (include both human and robot)
            all_data = [pts]
            if robot_pts is not None:
                all_data.append(robot_pts)
            all_pts = np.concatenate(all_data, axis=0)
            lim = max(np.abs(all_pts).max() * 1.2, 0.05)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_zlim(-lim, lim)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            frame_count += 1
            elapsed = time.time() - t0
            if dt - elapsed > 0:
                plt.pause(dt - elapsed)
            else:
                plt.pause(0.001)

    except KeyboardInterrupt:
        print("\n[Vis] Interrupted.")

    plt.close('all')
    reader.close()
    print(f"[Vis] Done. {frame_count} frames.")


if __name__ == "__main__":
    main()
