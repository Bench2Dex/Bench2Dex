#!/usr/bin/env python3
"""Generate arm_ee_trajectories.hdf5 alongside the original HDF5 episodes.

Reads robot/qpos and action/commanded from each episode, splits arm and hand
joints, and writes a separate HDF5 file with FK-computed 6-DOF ee poses.

This is a preprocessing step that runs once per task dataset.  The resulting
file is read by :class:`CrossEmbodimentHDF5Dataset` during training.

The EE convention (URDF, ``pin_ee_frame``, ``ee_offset``) MUST match the
eval-time converter ``ik_arm_converter._ROBOT_KEY_TO_ARM``: joint values are
placed into the Pinocchio model by joint NAME (``getJointId(...).idx_q``), the
same way the converter builds ``qpos_to_unified``, so qpos -> EE -> IK is an
identity and eval reproduces the teleop joint trajectory.  Robot-specific arm
configs are used (shadow -> ``ur5_shadow.yml``, leap -> ``xarm7_leap.yml``);
the generic RH56DFX ``ur5.yml`` / ``xarm7_ability.yml`` would silently put the
EE in the wrong wrist frame (measured ~0.4 m position bias for shadow).

Usage::

    python policy/GR00T_XE/convert_ee_trajectories.py \\
        --dataset-dir /data/73_jigsaw/replay-generalization \\
        --robot-key multi_iiwa7_with_sharpa
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

# Pinocchio imports (available in groot env)
import pinocchio as pin

# Local imports
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_arm_cfg(robot_key: str) -> dict:
    """Load the arm config (URDF path, joint names, ee body) for a robot."""
    arm_cfg_dir = _REPO_ROOT / "teleop" / "arm_configs"
    # Map robot_key -> arm config file (kept identical to
    # ik_arm_converter._ROBOT_KEY_TO_ARM so the training EE convention cannot
    # drift from the eval-time converter).
    key_to_cfg = {
        "multi_ur5_rh56dfx_with_flange": "ur5.yml",
        "multi_ur5_rh5dg2_with_flange": "ur5.yml",
        "multi_ur5_schunk_hand_with_flange": "ur5.yml",
        # shadow -> ur5_shadow.yml and leap -> xarm7_leap.yml: each robot must
        # use ITS OWN arm URDF/ee frame.  The generic RH56DFX ur5.yml / the
        # xarm7_ability.yml would silently produce an EE in the wrong wrist
        # frame (shadow measured ~0.4 m position / ~300 deg orientation bias),
        # so eval could not reproduce the teleop joint trajectories.
        "multi_ur5_shadow_hand_with_flange": "ur5_shadow.yml",
        "multi_ur5_wuji_with_flange": "ur5.yml",
        "multi_iiwa7_with_sharpa": "iiwa7.yml",
        "multi_xarm7_with_ability": "xarm7_ability.yml",
        "multi_xarm7_with_leap": "xarm7_leap.yml",
        "multi_rm_65_with_revo2": "rm65_revo2.yml",
        "multi_panda_with_allegro": "panda_allegro.yml",
        "multi_panda_with_orca": "panda_orca.yml",
        "multi_jaka_zu7_dexhand021_with_flange": "jaka_zu7.yml",
    }
    cfg_file = key_to_cfg.get(robot_key)
    if cfg_file is None:
        raise ValueError(f"Unknown robot_key: {robot_key}")
    import yaml
    with open(arm_cfg_dir / cfg_file) as f:
        return yaml.safe_load(f)


def _build_fk_side(model: pin.Model, data, arm_cfg: dict, robot_key: str,
                   side_joint_names: list[str], episode_joint_names: list[str],
                   ee_frame: str, ee_offset: np.ndarray):
    """Return a callable that FK-computes one side's EE for a full-q frame.

    ``episode_joint_names`` is the HDF5 joint-name order of the episode; each
    arm joint's value is read from that order and placed into the model at its
    own ``idx_q`` (name-based, identical to ik_arm_converter.qpos_to_unified).
    """
    # name -> (position in episode array, idx_q in model)
    positions = []
    for jn in side_joint_names:
        try:
            pos = episode_joint_names.index(jn)
        except ValueError:
            # Never silently skip: a missing arm joint would silently scramble
            # the EE (memory: no-silent-error-skipping).
            raise ValueError(
                f"[convert_ee] robot_key '{robot_key}' arm joint '{jn}' not found in "
                f"episode joint_names (n={len(episode_joint_names)}). Refusing to "
                f"produce a silently-wrong EE trajectory."
            )
        jid = model.getJointId(jn)
        positions.append((pos, model.joints[jid].idx_q))
    ee_id = model.getFrameId(ee_frame)

    def fk(qframe: np.ndarray) -> np.ndarray:
        q = np.zeros(model.nq)
        for pos, iq in positions:
            q[iq] = qframe[pos]
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        T = data.oMf[ee_id].copy()
        T.translation = T.translation + T.rotation @ ee_offset
        rpy = pin.rpy.matrixToRpy(T.rotation)
        return np.array(
            [T.translation[0], T.translation[1], T.translation[2],
             rpy[0], rpy[1], rpy[2]], dtype=np.float32
        )
    return fk


def convert_dataset(dataset_dir: str, robot_key: str, output: str | None = None) -> str:
    """Convert one dataset directory.  Writes arm_ee_trajectories.hdf5 alongside."""
    input_dir = Path(dataset_dir).expanduser().resolve()
    if output is None:
        output = str(input_dir.parent / "arm_ee_trajectories.hdf5")
    print(f"[convert_ee] {input_dir} → {output}")

    arm_cfg = _load_arm_cfg(robot_key)
    arm = arm_cfg["arm"]

    # URDF of the full robot (both arms + hand) — placed by joint name below.
    import os
    urdf_zoo = _REPO_ROOT / "dex2bench_dataset" / "URDF-Zoo"
    if not urdf_zoo.exists():
        urdf_zoo = _REPO_ROOT.parent / "dex2bench_dataset" / "URDF-Zoo"
    urdf_path = urdf_zoo / arm["urdf_path"]
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()

    files = sorted(input_dir.glob("episode_*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No HDF5 files in {input_dir}")

    # Peek the first episode to learn the HDF5 joint-name order up front.
    with h5py.File(files[0], "r") as probe:
        ep_joint_names = [n.decode() if isinstance(n, bytes) else str(n)
                          for n in probe["robot/joint_names"][:]]
    fk_sides = {}
    for side in ("right", "left"):
        sc = arm[side]
        fk_sides[side] = _build_fk_side(
            model, data, arm_cfg, robot_key,
            sc["joint_names"], ep_joint_names,
            sc["pin_ee_frame"], np.asarray(sc.get("ee_offset", [0, 0, 0]), dtype=np.float64),
        )

    with h5py.File(output, "w") as out:
        for ep_file in files:
            stem = ep_file.stem
            ep_grp = out.create_group(stem)
            with h5py.File(ep_file, "r") as ep:
                jn = [n.decode() if isinstance(n, bytes) else str(n)
                      for n in ep["robot/joint_names"][:]]
                if jn != ep_joint_names:
                    raise ValueError(
                        f"[convert_ee] {ep_file.name} joint_names differ from the first "
                        f"episode — refuse to mix orderings."
                    )
                qpos = ep["robot/qpos"][:].astype(np.float32)
                action = ep["action/commanded"][:].astype(np.float32)
                n_frames = qpos.shape[0]

                # NaN rows in action/commanded = uncommanded frames; fall back
                # to the qpos joint values for FK (same behavior as the
                # reference tools/convert_arm_joints_to_ee.py generator), so the
                # stored EE stays finite for every frame.
                action_safe = action.copy()
                nan_rows = np.isnan(action_safe).any(axis=1)
                if nan_rows.any():
                    action_safe[nan_rows] = qpos[nan_rows]

                for qname, arr in (("qpos", qpos), ("action", action_safe)):
                    for side, fk in fk_sides.items():
                        out_arr = np.zeros((n_frames, 6), dtype=np.float32)
                        for t in range(n_frames):
                            try:
                                out_arr[t] = fk(arr[t])
                            except Exception as e:
                                raise RuntimeError(
                                    f"[convert_ee] FK failed at frame {t} of "
                                    f"{ep_file.name}: {e}"
                                )
                        ep_grp.create_dataset(f"{side}_ee_{qname}", data=out_arr)

            print(f"  {stem}: {n_frames} frames")

    print(f"[convert_ee] Done: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Convert arm joints to EE trajectories")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--robot-key", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    convert_dataset(args.dataset_dir, args.robot_key, args.output)


if __name__ == "__main__":
    main()
