#!/usr/bin/env python3
"""离线推理脚本: 用训练数据轨迹对比 GR00T XE 模型输出 vs 记录动作.

用法:
  python offline_inference.py \
    --model_path /path/to/checkpoint \
    --robot_key multi_iiwa7_with_sharpa \
    --hdf5 /path/to/episode_000000.hdf5 \
    --max_frames 100 \
    --output /tmp/offline_compare.npz
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

# Add repo root to path
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent  # policy/GR00T_XE -> repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Add GR00T src to path
_GR00T_SRC = _SCRIPT_DIR.parent / "GR00T_n15" / "src"
if str(_GR00T_SRC) not in sys.path:
    sys.path.insert(0, str(_GR00T_SRC))

# Add GR00T_XE src to path (for gr00t package)
_GR00T_XE_SRC = _SCRIPT_DIR / "src"
if str(_GR00T_XE_SRC) not in sys.path:
    sys.path.insert(0, str(_GR00T_XE_SRC))


def parse_args():
    p = argparse.ArgumentParser(description="GR00T XE offline inference comparison")
    p.add_argument("--model_path", required=True, help="Path to GR00T XE checkpoint")
    p.add_argument("--robot_key", required=True, help="Robot key, e.g. multi_iiwa7_with_sharpa")
    p.add_argument("--hdf5", required=True, help="Path to training HDF5 episode")
    p.add_argument("--max_frames", type=int, default=100, help="Max frames to compare (default 100)")
    p.add_argument("--output", default="/tmp/offline_compare.npz", help="Output npz path")
    p.add_argument("--skip", type=int, default=0, help="Skip first N frames")
    return p.parse_args()


def build_hdf5_to_isaac_reindex(hdf5_joint_names: list[str], isaac_joint_names: list[str]) -> np.ndarray:
    """Build reindex array: HDF5-order qpos → Isaac-order qpos.

    For each position in Isaac order, finds the corresponding position in HDF5 order.
    isaac_qpos[i] = hdf5_qpos[reindex[i]]
    """
    hdf5_name_to_idx = {name: i for i, name in enumerate(hdf5_joint_names)}
    reindex = np.zeros(len(isaac_joint_names), dtype=int)
    for i, name in enumerate(isaac_joint_names):
        if name in hdf5_name_to_idx:
            reindex[i] = hdf5_name_to_idx[name]
        else:
            # Check for "multi_" prefix variant (HDF5 may have prefix, Isaac may not or vice versa)
            alt_name = name.replace("multi_", "") if name.startswith("multi_") else "multi_" + name
            if alt_name in hdf5_name_to_idx:
                reindex[i] = hdf5_name_to_idx[alt_name]
            else:
                raise KeyError(f"Joint '{name}' not found in HDF5 joint names (tried '{alt_name}' too)")
    return reindex


def load_hdf5_frame_remote(hdf5_path: str, frame_idx: int, robot_key: str,
                           reindex: np.ndarray | None = None) -> dict:
    """从训练 HDF5 加载单帧, 构造 REMOTE 格式观测 (与 eval_policy_client 一致).

    qpos is reordered from HDF5 order to Isaac full_dof order.
    """
    with h5py.File(hdf5_path, "r") as f:
        # Camera images are stored as serialized PNG bytes
        observation = {}
        for cam_name in ["cam_stereo_left", "cam_stereo_right", "cam_wrist_right", "cam_wrist_left"]:
            raw = f[f"cameras/{cam_name}/rgb"][frame_idx]
            if isinstance(raw, np.ndarray) and raw.ndim == 1:
                img = np.array(Image.open(io.BytesIO(raw.tobytes())))
            elif isinstance(raw, bytes):
                img = np.array(Image.open(io.BytesIO(raw)))
            else:
                img = np.asarray(raw)
            observation[cam_name] = {"rgb": np.asarray(img, dtype=np.uint8)}

        qpos_hdf5 = np.asarray(f["robot/qpos"][frame_idx], dtype=np.float64)

    if reindex is not None:
        qpos_isaac = qpos_hdf5[reindex]
    else:
        qpos_isaac = qpos_hdf5

    return {
        "robot_key": robot_key,
        "joint_action": {"qpos": qpos_isaac},
        "observation": observation,
    }


def main():
    global args
    args = parse_args()

    os.environ.setdefault("GR00T_MAX_STATE_DIM", "64")
    os.environ.setdefault("GR00T_MAX_ACTION_DIM", "64")
    os.environ.setdefault("GR00T_CAMERA_MODE", "4cam")
    os.environ.setdefault("GR00T_VIDEO_KEYS", "video.stereo_left,video.stereo_right,video.right_wrist,video.left_wrist")

    # Import deploy_policy (must be after path setup)
    from policy.GR00T_XE.deploy_policy import GR00TXEPolicy, encode_obs
    from robots.active_dof_utils import get_active_dof_info

    # Load model
    model = GR00TXEPolicy(model_path=args.model_path, robot_key=args.robot_key)

    # Build reindex: HDF5 order → Isaac full_dof order
    adi = get_active_dof_info(args.robot_key)
    with h5py.File(args.hdf5, "r") as f:
        hdf5_joint_names = [jn.decode() if isinstance(jn, bytes) else jn for jn in f["robot/joint_names"]]
        n_frames = len(f["robot/qpos"])
    reindex = build_hdf5_to_isaac_reindex(hdf5_joint_names, adi.full_joint_names)
    print(f"Built reindex: HDF5({len(hdf5_joint_names)}) → Isaac({len(adi.full_joint_names)})")
    print(f"  HDF5 sample: {hdf5_joint_names[:5]}")
    print(f"  Isaac sample: {adi.full_joint_names[:5]}")
    print(f"  Reindex sample: {reindex[:5]}")

    n_compare = min(args.max_frames, n_frames - args.skip)
    print(f"HDF5 frames: {n_frames}, comparing {n_compare} frames (skip={args.skip})")

    model_actions = []
    recorded_qpos = []
    frame_errors = []

    for i in range(args.skip, args.skip + n_compare):
        raw_obs = load_hdf5_frame_remote(args.hdf5, i, args.robot_key, reindex)

        # encode_obs -> model.get_action
        encoded = encode_obs(raw_obs)
        action = model.get_action(encoded)

        # Compare with recorded NEXT frame qpos (reordered to Isaac order)
        if i + 1 < n_frames:
            next_raw = load_hdf5_frame_remote(args.hdf5, i + 1, args.robot_key, reindex)
            next_qpos = next_raw["joint_action"]["qpos"]
        else:
            next_qpos = raw_obs["joint_action"]["qpos"]

        model_actions.append(action)
        recorded_qpos.append(next_qpos)

        # Error metrics
        err = np.abs(action - next_qpos)
        frame_errors.append({
            "frame": i,
            "mean_abs_error": float(np.mean(err)),
            "max_abs_error": float(np.max(err)),
            "arm_mae": float(np.mean(err[:14])),  # 双臂 7+7
            "hand_mae": float(np.mean(err[14:])),  # 双手
        })

        if i == args.skip:
            print(f"\nFrame {i} first comparison:")
            print(f"  Model action:  {np.array2string(action[:14], precision=3, max_line_width=120)}")
            print(f"  Next qpos:     {np.array2string(next_qpos[:14], precision=3, max_line_width=120)}")
            print(f"  Arm MAE: {frame_errors[-1]['arm_mae']:.6f}")
            print(f"  Hand MAE: {frame_errors[-1]['hand_mae']:.6f}")

        if (i - args.skip + 1) % 20 == 0:
            mean_mae = np.mean([e["mean_abs_error"] for e in frame_errors])
            print(f"  Frame {i}: mean_abs_error={frame_errors[-1]['mean_abs_error']:.6f}, running_avg={mean_mae:.6f}")

    # Summary
    model_actions = np.array(model_actions)
    recorded_qpos = np.array(recorded_qpos)
    all_mae = np.mean(np.abs(model_actions - recorded_qpos), axis=1)

    print(f"\n{'='*60}")
    print(f"Offline comparison summary ({n_compare} frames):")
    print(f"  Overall MAE:  {np.mean(all_mae):.6f} ± {np.std(all_mae):.6f}")
    print(f"  Min MAE:      {np.min(all_mae):.6f}")
    print(f"  Max MAE:      {np.max(all_mae):.6f}")

    # Per-joint stats
    per_joint_mae = np.mean(np.abs(model_actions - recorded_qpos), axis=0)
    print(f"\n  Per-joint MAE (top 10 worst):")
    worst_idx = np.argsort(per_joint_mae)[::-1][:10]
    for idx in worst_idx:
        jn = adi.full_joint_names[idx] if idx < len(adi.full_joint_names) else f"idx_{idx}"
        print(f"    {jn:30s}: {per_joint_mae[idx]:.6f}")

    # Save results
    np.savez(args.output,
             model_actions=model_actions,
             recorded_qpos=recorded_qpos,
             per_joint_mae=per_joint_mae,
             joint_names=np.array(adi.full_joint_names),
             all_mae=all_mae,
             frame_errors=frame_errors)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()