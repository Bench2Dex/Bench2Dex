"""
math_utils.py — Shared quaternion / rotation matrix utilities for the teleop module.

All functions are pure NumPy unless noted. The torch variant
(_quat_wxyz_to_rotmat_torch) requires torch but is only called at
runtime inside Isaac Sim where torch is always available.
"""

from __future__ import annotations

import numpy as np


# ── NumPy quaternion / rotation utilities ──────────────────────────────────

def quat_wxyz_to_rotmat(q) -> np.ndarray:
    """四元数 (w,x,y,z) np.array → (3,3) 旋转矩阵。"""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def quat_xyzw_to_rotmat(q) -> np.ndarray:
    """四元数 (qx,qy,qz,qw) np.array → (3,3) 旋转矩阵。"""
    return quat_wxyz_to_rotmat(np.array([q[3], q[0], q[1], q[2]]))


def rotmat_to_quat_wxyz(R) -> np.ndarray:
    """旋转矩阵 (3,3) np.array → 四元数 (w,x,y,z) np.float32。"""
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float32)


# ── Torch quaternion utility (only used in Isaac Sim runtime) ──────────────

def quat_wxyz_to_rotmat_torch(q):
    """四元数 (w,x,y,z) tensor (4,) → (3,3) 旋转矩阵 tensor。

    Requires torch; imported lazily to avoid hard dependency at module level.
    """
    import torch
    w, x, y, z = q[0], q[1], q[2], q[3]
    R = torch.zeros(3, 3, device=q.device, dtype=q.dtype)
    R[0, 0] = 1 - 2*(y*y + z*z); R[0, 1] = 2*(x*y - w*z); R[0, 2] = 2*(x*z + w*y)
    R[1, 0] = 2*(x*y + w*z);     R[1, 1] = 1 - 2*(x*x + z*z); R[1, 2] = 2*(y*z - w*x)
    R[2, 0] = 2*(x*z - w*y);     R[2, 1] = 2*(y*z + w*x);     R[2, 2] = 1 - 2*(x*x + y*y)
    return R


# ── ARKit → Sim coordinate mapping ────────────────────────────────────────

# AXIS_MAPPING=(0,2,1), AXIS_SIGN=(1,-1,1) → sim_x=+arkit_x, sim_y=-arkit_z, sim_z=+arkit_y
_ARKIT_AXIS_MAPPING = (0, 2, 1)
_ARKIT_AXIS_SIGN = (1.0, -1.0, 1.0)

# 手掌对齐修正 (体坐标系右乘)
_PALM_ALIGN_WXYZ = (-0.5, 0.5, 0.5, -0.5)


def arkit_quat_to_sim_quat(arkit_quat_xyzw: np.ndarray) -> np.ndarray:
    """ARKit 四元数 (qx,qy,qz,qw) → 仿真四元数 (w,x,y,z)。

    步骤:
      1. 轴映射 (和位置映射相同规则): sim_qx=+arkit_qx, sim_qy=-arkit_qz, sim_qz=+arkit_qy
      2. 手掌对齐修正 (体坐标系右乘 q_fix): 掌心 +X→+Y, 手指 +Z→-X
    """
    ax, ay, az, aw = (float(arkit_quat_xyzw[0]), float(arkit_quat_xyzw[1]),
                      float(arkit_quat_xyzw[2]), float(arkit_quat_xyzw[3]))
    a, b, c = _ARKIT_AXIS_MAPPING
    imag = (ax, ay, az)
    # 步骤1: 轴映射
    w1 = aw
    x1 = _ARKIT_AXIS_SIGN[0] * imag[a]
    y1 = _ARKIT_AXIS_SIGN[1] * imag[b]
    z1 = _ARKIT_AXIS_SIGN[2] * imag[c]
    # 步骤2: 右乘手掌修正
    w2, x2, y2, z2 = _PALM_ALIGN_WXYZ
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32)
