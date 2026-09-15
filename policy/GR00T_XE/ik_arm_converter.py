"""Cross-embodiment 状态/动作转换器: FK/IK + 手部映射。

核心功能:
  1. qpos_to_unified():  原始关节角 → 统一 64D state (arm ee_pose + hand slots)
  2. unified_to_joint_action(): 统一 64D action → 实际关节角 (IK arm + reverse hand)
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

# xe_hand_mapping is the single source of truth shared with dataset.py so
# the train/inference joint→slot mapping cannot drift (memory: no-silent-error-skipping).
_GR00T_XE_DIR = Path(__file__).resolve().parent
if str(_GR00T_XE_DIR) not in sys.path:
    sys.path.insert(0, str(_GR00T_XE_DIR))
from xe_hand_mapping import build_name_to_slot_maps, robot_key_to_hand_name, validate_hand_coverage

# 延迟导入 Pinocchio (只在需要时加载)
_pin = None
_pin_models: Dict[str, object] = {}  # robot_key → (pin_model, pin_data)


def _get_pin():
    global _pin
    if _pin is None:
        import pinocchio as _p
        _pin = _p
    return _pin


# CLIK solves that hit the iteration cap, keyed by "robot/side".
_IK_FAILURES: Dict[str, int] = {}
_IK_WARN_EVERY = 100


def _note_ik_failure(robot_key: str, side: str, residual: float) -> None:
    """Count and (sparsely) report a CLIK solve that did not converge.

    Deliberately does not substitute a fallback pose.  The last iterate is a
    valid configuration that started from the robot's *actual* current qpos, so
    it is already the best "move as close as you can" answer; swapping in "hold
    the last action" or "hold current qpos" would discard that progress and turn
    a visible failure into an invisible freeze.  Count it instead, so the rate
    is measurable rather than guessed at.
    """
    key = f"{robot_key}/{side}"
    n = _IK_FAILURES[key] = _IK_FAILURES.get(key, 0) + 1
    if n == 1 or n % _IK_WARN_EVERY == 0:
        print(
            f"[GR00T_XE][IK] {key}: CLIK did not converge after 30 iters "
            f"(residual {residual:.4f}; {n} occurrence(s) so far) -- "
            f"using the best iterate, clamped to joint limits",
            file=sys.stderr, flush=True,
        )


def get_ik_failure_stats() -> Dict[str, int]:
    """Non-convergent CLIK counts by 'robot/side'. Empty dict means none."""
    return dict(_IK_FAILURES)


# ---------------------------------------------------------------------------
# Arm config 路径
# ---------------------------------------------------------------------------

_ARM_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "teleop" / "arm_configs"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_URDF_ZOO = Path(os.environ.get("URDF_ZOO_DIR", str(
    _REPO_ROOT.parent / "dex2bench_dataset" / "URDF-Zoo"
)))

# robot_key → arm_config name
_ROBOT_KEY_TO_ARM = {
    "multi_iiwa7_with_sharpa": "iiwa7",
    "multi_ur5_rh56dfx_with_flange": "ur5",
    "multi_ur5_rh5dg2_with_flange": "ur5",
    "multi_ur5_schunk_hand_with_flange": "ur5",
    "multi_ur5_shadow_hand_with_flange": "ur5_shadow",
    "multi_ur5_wuji_with_flange": "ur5",
    "multi_rm_65_with_revo2": "rm65_revo2",
    "multi_panda_with_allegro": "panda_allegro",
    "multi_panda_with_orca": "panda_orca",
    "multi_xarm7_with_ability": "xarm7_ability",
    "multi_xarm7_with_leap": "xarm7_leap",
    "multi_jaka_zu7_dexhand021_with_flange": "jaka_zu7",
}

# arm_config → joint_names (right, left)
_ARM_JOINTS: Dict[str, Dict[str, List[str]]] = {}


def _load_arm_config(arm_name: str) -> dict:
    with open(_ARM_CONFIGS_DIR / f"{arm_name}.yml") as f:
        return yaml.safe_load(f)["arm"]


def _ensure_arm_joints(arm_name: str):
    if arm_name not in _ARM_JOINTS:
        cfg = _load_arm_config(arm_name)
        _ARM_JOINTS[arm_name] = {
            "right": cfg["right"]["joint_names"],
            "left": cfg["left"]["joint_names"],
            "right_ee": cfg["right"]["pin_ee_frame"],
            "left_ee": cfg["left"]["pin_ee_frame"],
            "right_offset": cfg["right"].get("ee_offset", [0, 0, 0]),
            "left_offset": cfg["left"].get("ee_offset", [0, 0, 0]),
            "urdf_path": str(_URDF_ZOO / cfg["urdf_path"]),
        }


def _load_pin_model(robot_key: str) -> Tuple[object, object]:
    if robot_key not in _pin_models:
        pin = _get_pin()
        arm_name = _ROBOT_KEY_TO_ARM.get(robot_key, "")
        urdf_path = _URDF_ZOO / _load_arm_config(arm_name)["urdf_path"]
        if not Path(urdf_path).exists():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        model = pin.buildModelFromUrdf(str(urdf_path))
        data = model.createData()
        _pin_models[robot_key] = (model, data)
    return _pin_models[robot_key]


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 手部映射 (从 embodiment_mapping.yml → 统一 slot) — single source: xe_hand_mapping
# ---------------------------------------------------------------------------
#
# DO NOT re-implement hand→slot parsing here.  dataset.py (training) and this
# converter (inference) both call xe_hand_mapping so the two paths cannot drift
# (memory: no-silent-error-skipping).  Coverage is validated against the actual
# joint names once per robot_key and RAISES on any gap instead of silently
# leaving hand columns at 0.

_COVERAGE_CHECKED: set = set()


def _hand_names_to_slots(robot_key: str, joint_names: List[str]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Return (right_joint_to_slot, left_joint_to_slot): joint_name→unified slot.

    right values are unified 12-33, left 34-55.  Validates that every authored
    hand joint resolves against ``joint_names`` (once per robot_key).
    """
    hand_name = robot_key_to_hand_name(robot_key)
    right_joint_to_slot, left_joint_to_slot = build_name_to_slot_maps(hand_name)
    if robot_key not in _COVERAGE_CHECKED:
        stats = validate_hand_coverage(robot_key, list(joint_names), context="ik_arm_converter")
        print(
            f"[GR00T_XE] robot={robot_key} hand={hand_name} "
            f"hand_joints right={stats['right']}/{stats['right_total']} "
            f"left={stats['left']}/{stats['left_total']}"
        )
        _COVERAGE_CHECKED.add(robot_key)
    return right_joint_to_slot, left_joint_to_slot



# 主转换器
# ---------------------------------------------------------------------------

class XEStateActionConverter:
    """Cross-embodiment state/action converter.

    用法:
        conv = XEStateActionConverter()
        unified_state = conv.qpos_to_unified(qpos_raw, robot_key)
        joint_action = conv.unified_to_joint_action(unified_action, robot_key)
    """

    def __init__(self):
        self._arm_joints_cache: Dict[str, Tuple[List[int], List[int]]] = {}

    def _get_arm_indices(self, robot_key: str, joint_names: List[str]) -> Tuple[List[int], List[int]]:
        """返回 (right_arm_indices, left_arm_indices) 在 HDF5/Isaac 关节顺序中的位置."""
        cache_key = (robot_key, tuple(joint_names))
        if cache_key in self._arm_joints_cache:
            return self._arm_joints_cache[cache_key]

        arm_name = _ROBOT_KEY_TO_ARM.get(robot_key, "")
        _ensure_arm_joints(arm_name)
        arm_jt = _ARM_JOINTS[arm_name]

        right_indices = []
        left_indices = []
        for name in arm_jt["right"]:
            try:
                right_indices.append(joint_names.index(name))
            except ValueError:
                pass
        for name in arm_jt["left"]:
            try:
                left_indices.append(joint_names.index(name))
            except ValueError:
                pass

        # 调用方（qpos_to_unified / unified_to_joint_action）是把 indices[k] 与
        # arm_jt[side][k] 按位置一一对应的：只缺一个名字，后面所有关节都会整体错位，
        # 于是 FK 读错关节、IK 把解写回错的关节，而且完全静默。整侧缺失（== 0）
        # 是既有的“该机器人没有这条臂”分支，只有“缺一部分”才是必须报错的错位。
        for side, found in (("right", right_indices), ("left", left_indices)):
            expected = len(arm_jt[side])
            if 0 < len(found) < expected:
                raise RuntimeError(
                    f"{robot_key} ({arm_name}) {side} arm: only {len(found)}/{expected} "
                    f"joint names found in the canonical joint list "
                    f"({ [n for n in arm_jt[side] if n not in joint_names] } missing) "
                    "-- indices would be positionally misaligned with the arm joint names"
                )

        result = (right_indices, left_indices)
        self._arm_joints_cache[cache_key] = result
        return result

    def _get_hand_maps(self, robot_key: str, joint_names: List[str]):
        """返回 (right_joint_to_slot, left_joint_to_slot, right_hand_indices, left_hand_indices).

        right_joint_to_slot: {joint_name → unified_slot(0-21)} for right hand
        left_joint_to_slot:  {joint_name → unified_slot(22-43)} for left hand
        right_hand_indices:  list of indices in full joint array
        """
        right_jt2slot, left_jt2slot = _hand_names_to_slots(robot_key, joint_names)
        right_hand_indices = []
        left_hand_indices = []
        for i, name in enumerate(joint_names):
            if name in right_jt2slot:
                right_hand_indices.append(i)
            elif name in left_jt2slot:
                left_hand_indices.append(i)
        return right_jt2slot, left_jt2slot, right_hand_indices, left_hand_indices

    def qpos_to_unified(self, qpos_full: np.ndarray, robot_key: str) -> np.ndarray:
        """原始关节角 → 统一 64D state.

        Args:
            qpos_full: [full_dof] 原始关节角
            robot_key: 机器人标识

        Returns:
            unified_state: [64] (arm_ee_pose 12 + hand_slots 44 + pad 8)
        """
        from robots.active_dof_utils import get_active_dof_info
        adi = get_active_dof_info(robot_key)
        joint_names = adi.full_joint_names

        right_arm_idx, left_arm_idx = self._get_arm_indices(robot_key, joint_names)
        right_jt2slot, left_jt2slot, right_hand_idx, left_hand_idx = self._get_hand_maps(robot_key, joint_names)

        pin = _get_pin()
        pin_model, pin_data = _load_pin_model(robot_key)
        arm_name = _ROBOT_KEY_TO_ARM.get(robot_key, "")
        _ensure_arm_joints(arm_name)
        arm_jt = _ARM_JOINTS[arm_name]

        unified = np.zeros(64, dtype=np.float64)

        # Arm: FK
        for side, indices, offset, ee_key, off_key in [
            ("right", right_arm_idx, 0, "right_ee", "right_offset"),
            ("left", left_arm_idx, 6, "left_ee", "left_offset"),
        ]:
            if len(indices) == 0:
                continue
            q_arm = np.array([qpos_full[i] for i in indices], dtype=np.float64)
            ee_offset = np.array(arm_jt[off_key], dtype=np.float64)

            # 构建 full q
            q_full = np.zeros(pin_model.nq)
            arm_jt_names = arm_jt[side]  # joint names for this arm side
            for i, jn in enumerate(arm_jt_names):
                jid = pin_model.getJointId(jn)
                if jid < pin_model.njoints and i < len(q_arm):
                    q_full[pin_model.joints[jid].idx_q] = q_arm[i]

            pin.forwardKinematics(pin_model, pin_data, q_full)
            pin.updateFramePlacements(pin_model, pin_data)
            ee_id = pin_model.getFrameId(arm_jt[ee_key])
            T = pin_data.oMf[ee_id].copy()
            T.translation += T.rotation @ ee_offset
            rpy = pin.rpy.matrixToRpy(T.rotation)

            unified[offset + 0] = T.translation[0]
            unified[offset + 1] = T.translation[1]
            unified[offset + 2] = T.translation[2]
            unified[offset + 3] = rpy[0]
            unified[offset + 4] = rpy[1]
            unified[offset + 5] = rpy[2]

        # Hand: 映射到 44 语义槽
        for indices, jt2slot, slot_offset in [
            (right_hand_idx, right_jt2slot, 12),
            (left_hand_idx, left_jt2slot, 34),
        ]:
            for i in indices:
                name = joint_names[i]
                if name in jt2slot:
                    unified[jt2slot[name]] = qpos_full[i]

        return unified

    def unified_to_joint_action(self, unified_action: np.ndarray, robot_key: str,
                                 current_qpos: np.ndarray) -> np.ndarray:
        """统一 64D action → 实际关节角 (IK arm + reverse hand + pad 忽略).

        Args:
            unified_action: [64] 模型输出
            robot_key: 机器人标识
            current_qpos: [full_dof] 当前关节角 (用于 IK 初始化)

        Returns:
            joint_action: [full_dof] 实际关节角
        """
        from robots.active_dof_utils import get_active_dof_info
        adi = get_active_dof_info(robot_key)
        joint_names = adi.full_joint_names

        right_arm_idx, left_arm_idx = self._get_arm_indices(robot_key, joint_names)
        right_jt2slot, left_jt2slot, right_hand_idx, left_hand_idx = self._get_hand_maps(robot_key, joint_names)

        pin = _get_pin()
        pin_model, pin_data = _load_pin_model(robot_key)
        arm_name = _ROBOT_KEY_TO_ARM.get(robot_key, "")
        _ensure_arm_joints(arm_name)
        arm_jt = _ARM_JOINTS[arm_name]

        joint_action = np.array(current_qpos, dtype=np.float64).copy()

        # Arm: IK
        for side, indices, offset, ee_key, off_key in [
            ("right", right_arm_idx, 0, "right_ee", "right_offset"),
            ("left", left_arm_idx, 6, "left_ee", "left_offset"),
        ]:
            if len(indices) == 0:
                continue
            ee_offset = np.array(arm_jt[off_key], dtype=np.float64)
            ee_pose = unified_action[offset:offset + 6]

            # 目标位姿
            T_target = pin.SE3(
                pin.rpy.rpyToMatrix(np.array(ee_pose[3:], dtype=np.float64)),
                np.array(ee_pose[:3], dtype=np.float64),
            )
            T_target.translation -= T_target.rotation @ ee_offset

            # 初始化 q (从 current_qpos)
            q_armed = np.array([current_qpos[i] for i in indices], dtype=np.float64)
            q_full = np.zeros(pin_model.nq)
            arm_jt_names = arm_jt[side]
            for i, jn in enumerate(arm_jt_names):
                jid = pin_model.getJointId(jn)
                if jid < pin_model.njoints and i < len(q_armed):
                    q_full[pin_model.joints[jid].idx_q] = q_armed[i]

            ee_id = pin_model.getFrameId(arm_jt[ee_key])

            # CLIK
            converged = False
            for _ in range(30):
                pin.forwardKinematics(pin_model, pin_data, q_full)
                pin.updateFramePlacements(pin_model, pin_data)
                pin.computeJointJacobians(pin_model, pin_data, q_full)
                T_cur = pin_data.oMf[ee_id]
                # err 的平移/旋转两部分必须与下面 Jacobian 的坐标系一致。
                # Jacobian 取 LOCAL_WORLD_ALIGNED（线速度、角速度都在 world 系），
                # 所以旋转误差必须是 world 系的 log3(R_tgt @ R_cur.T)，不能写成
                # body 系的 log3(R_cur.T @ R_tgt)（两者相差一个 R_cur），否则迭代方向
                # 是错的：起点不动时 err=0 直接退出（FK→IK 往返自检因此恒为 0，
                # 掩盖了这个问题），只要目标偏离当前位姿就会单调发散。
                err = np.concatenate([
                    T_target.translation - T_cur.translation,
                    pin.log3(T_target.rotation @ T_cur.rotation.T),
                ])
                if np.linalg.norm(err) < 1e-4:
                    converged = True
                    break
                J = pin.getFrameJacobian(pin_model, pin_data, ee_id, pin.LOCAL_WORLD_ALIGNED)
                J_arm = J[:, [pin_model.joints[pin_model.getJointId(jn)].idx_q for jn in arm_jt_names]]
                dq = J_arm.T @ np.linalg.solve(J_arm @ J_arm.T + 1e-3 * 1e-3 * np.eye(6), err)
                dq = np.clip(dq, -0.3, 0.3)
                for i, jn in enumerate(arm_jt_names):
                    if i < len(dq):
                        jid = pin_model.getJointId(jn)
                        if jid < pin_model.njoints:
                            q_full[pin_model.joints[jid].idx_q] += dq[i]

            # 残差检查: 30 次迭代没收敛时, q_full 只是"目前最好的一次迭代", 并不满足
            # 目标位姿。以前这里既不报残差也不查限位, 直接当解写回 —— 模型给出不可达
            # 位姿时会静默地把机械臂指到别处, 且没有任何信号说明它没解出来。
            # 注意残差要在循环之后重新做一次 FK 才是"真正被发出的那个解"的残差,
            # 循环里的 err 是最后一次更新之前的值。
            if not converged:
                pin.forwardKinematics(pin_model, pin_data, q_full)
                pin.updateFramePlacements(pin_model, pin_data)
                T_end = pin_data.oMf[ee_id]
                residual = float(np.linalg.norm(np.concatenate([
                    T_target.translation - T_end.translation,
                    pin.log3(T_target.rotation @ T_end.rotation.T),
                ])))
                _note_ik_failure(robot_key, side, residual)

            # 关节限位: 单步 dq 只被 clip(±0.3) 限住, 30 步累计仍可能离种子很远, 所以
            # 最终解按模型限位夹一次 (连续关节的限位是 ±inf, 不夹)。
            for i, jn in enumerate(arm_jt_names):
                jid = pin_model.getJointId(jn)
                if jid < pin_model.njoints:
                    iq = pin_model.joints[jid].idx_q
                    lo = float(pin_model.lowerPositionLimit[iq])
                    hi = float(pin_model.upperPositionLimit[iq])
                    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                        q_full[iq] = np.clip(q_full[iq], lo, hi)

            # 写回 arm 关节
            for i, jn in enumerate(arm_jt_names):
                jid = pin_model.getJointId(jn)
                if jid < pin_model.njoints and i < len(indices):
                    joint_action[indices[i]] = q_full[pin_model.joints[jid].idx_q]

        # Hand: 反向映射
        for indices, jt2slot, slot_offset in [
            (right_hand_idx, right_jt2slot, 12),
            (left_hand_idx, left_jt2slot, 34),
        ]:
            for i in indices:
                name = joint_names[i]
                if name in jt2slot:
                    joint_action[i] = unified_action[jt2slot[name]]

        return joint_action
