"""
arm_ik_controller.py — 基于 Pinocchio 的手臂 IK 遥操作控制器。

使用 Pinocchio 库进行真正的迭代逆运动学 (CLIK):
- 每帧多次迭代，每次重新计算正运动学 (FK) 和雅可比矩阵
- 比手算单次 DLS 收敛精度高得多
- 自动在首帧标定 Pinocchio 基座系到仿真世界系的变换
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pinocchio as pin
import torch
import yaml

from utils.seed_policy import get_np_rng

# Pinocchio IK 参数
# 每帧 CLIK 迭代次数 (每次迭代重新计算 FK + Jacobian)
NUM_IK_ITERS = 30
# DLS 阻尼 (UR5 6×6 方阵, 适中阻尼防止奇异点发散)
DLS_DAMP = 1e-3
# 收敛阈值: 6D误差 (线性+角度) 范数小于此值视为收敛
EPS_CONVERGE = 1e-4
# 关节角单步最大变化 (防跳跃, 保证遥操作平滑)
MAX_JOINT_DELTA = 0.8  # rad/frame (匹配执行器实际跟踪能力)
# 每迭代步长因子 (1.0=完整步, <1.0 更保守)
IK_DT = 1.0
# 每迭代最大关节变化 (防止奇异点附近发散)
MAX_DQ_PER_ITER = 0.3  # rad
# 6D 误差权重: [位置(3), 姿态(3)]  — 加大位置权重, 减小姿态权重
# 使 IK 优先追踪位置 (遥操作最重要), 姿态允许小偏差
POS_WEIGHT = 3.0
ORI_WEIGHT = 0.5
# 输出关节平滑系数 (0=完全用上帧, 1=完全用新值)
_SMOOTH_ALPHA = 0.7
# 近目标阈值 (位置误差小于此值时启用平滑)
_SMOOTH_NEAR_THRESH = 0.15  # m

# ── Random-restart 逃出局部最小 ──
# 6-DOF 臂在某些位姿下 CLIK 会陷入局部极小（特别是腕奇异点附近，elbow up/down 分支切换）。
# 当 warm-start 的解 best_err > IK_RESTART_THRESH 时，做 IK_RESTART_TRIES 次随机重启，
# 每次从关节限位内的均匀随机点出发，取所有尝试中误差最小的解。
IK_RESTART_THRESH = 0.05   # 6D log 误差范数; 经验值 (≈ 5cm 位置 + ~3° 姿态)
IK_RESTART_TRIES = 3        # 重启次数 (每次 NUM_IK_ITERS 迭代; 总开销 ~7x baseline)

def _default_urdf_zoo_dir() -> Path:
    env_path = os.environ.get("URDF_ZOO_DIR")
    if env_path:
        return Path(env_path)
    repo_parent = Path(__file__).resolve().parents[2]
    candidates = (
        repo_parent / "dex2bench_dataset" / "URDF-Zoo",
        repo_parent / "dex2bench_dataset" / "URDF_Zoo",
        repo_parent / "URDF-Zoo-main" / "URDF-Zoo",
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


_URDF_ZOO_DIR = str(_default_urdf_zoo_dir())

# 机械臂配置: 从 arm_configs/*.yml 自动加载
_ARM_CONFIGS_DIR = Path(__file__).resolve().parent / "arm_configs"


def _load_arm_configs() -> Dict[str, Dict]:
    """扫描 arm_configs/ 目录下所有 .yml 文件，返回 {arm_type: config_dict}。"""
    configs = {}
    if not _ARM_CONFIGS_DIR.is_dir():
        return configs
    for yml_path in sorted(_ARM_CONFIGS_DIR.glob("*.yml")):
        with open(yml_path) as f:
            raw = yaml.safe_load(f)
        arm_sec = raw.get("arm", {})
        # 解析 URDF 绝对路径
        rel_urdf = arm_sec.get("urdf_path", "")
        arm_sec["urdf_path"] = str(Path(_URDF_ZOO_DIR) / rel_urdf)
        configs[yml_path.stem] = arm_sec
    return configs


ARM_JOINT_CONFIGS: Dict[str, Dict] = _load_arm_configs()


def detect_arm_type(all_joint_names: List[str], side: str = "right") -> Optional[str]:
    """根据 Articulation 的关节名列表自动识别机械臂型号。

    匹配策略: 检查每种臂型配置中 per-side 关节名是否都存在于 Articulation 中。
    优先匹配关节数更多的配置 (如 panda_orca 8关节 vs panda_allegro 7关节)。
    """
    all_names_set = set(all_joint_names)
    sorted_configs = sorted(
        ARM_JOINT_CONFIGS.items(),
        key=lambda kv: kv[1].get("num_joints", 0),
        reverse=True,
    )
    for arm_key, cfg in sorted_configs:
        side_sec = cfg.get(side, {})
        joint_names = side_sec.get("joint_names", [])
        if joint_names and all(j in all_names_set for j in joint_names):
            return arm_key
    return None


def _parse_axes_mapping(mapping: List[str]) -> np.ndarray:
    """解析轴映射配置为 3×3 符号置换矩阵 M。

    mapping: 长度 3 的列表, 每项形如 "+x", "-z" 等。
    - mapping[j] 描述: B (EE body) 的第 j 列 = sign * A 的第 i 列
    - A 轴约定: +x=手指方向, +y=掌心方向, +z=拇指侧方向

    使用: R_B = R_A @ M  将 ARKit 手掌姿态映射为 EE body 目标姿态。

    Examples:
        ["+x", "+y", "+z"] → 单位矩阵 (A 和 B 轴完全对齐)
        ["-z", "+y", "+x"] → Ry(90°) 等效
    """
    _axis_to_idx = {'x': 0, 'y': 1, 'z': 2}
    M = np.zeros((3, 3), dtype=np.float64)
    for j, spec in enumerate(mapping):
        spec = spec.strip().lower()
        if spec[0] in '+-':
            sign = 1.0 if spec[0] == '+' else -1.0
            axis_char = spec[1]
        else:
            sign = 1.0
            axis_char = spec[0]
        i = _axis_to_idx[axis_char]
        M[i, j] = sign
    return M


def _quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """四元数 (w,x,y,z) → 3×3 旋转矩阵 (numpy)。转发到 math_utils。"""
    from .math_utils import quat_wxyz_to_rotmat
    return quat_wxyz_to_rotmat(q)


class ArmIKController:
    """基于 Pinocchio CLIK 的机械臂逆运动学控制器。

    核心优势 (相比手算单次 DLS):
    - 每次迭代重新计算 FK 和 Jacobian → 二阶收敛 (Newton-Raphson)
    - 使用 pin.log (李代数) 精确计算 SE3 误差
    - 适配任意 6+ DOF 机械臂 (UR5/Panda/xArm7)
    """

    def __init__(self, arm_type: str, num_envs: int, device: str,
                 side: str = "right", hand_type: Optional[str] = None):
        cfg_dict = ARM_JOINT_CONFIGS[arm_type]
        side_cfg = cfg_dict.get(side, {})

        self.side = side
        self.num_joints = cfg_dict["num_joints"]
        self.device = device

        # 关节名: 从 per-side section 读取
        self.joint_names = list(side_cfg.get("joint_names", []))

        # EE body name (in Isaac Sim)
        self.ee_body = side_cfg.get("ee_body", "")

        # EE 轴映射矩阵 (per-side 优先, 否则用全局)
        axes_mapping = side_cfg.get("ee_axes_mapping", cfg_dict.get("ee_axes_mapping", ["-z", "+y", "+x"]))
        M = _parse_axes_mapping(axes_mapping)

        # 可选的额外 EE 朝向修正 (绕映射后的 EE 轴旋转, RPY 弧度)
        # 用于补偿左右臂 EE frame 的旋转差异
        ee_corr_rpy = side_cfg.get("ee_orientation_correction_rpy")
        if ee_corr_rpy is not None:
            from scipy.spatial.transform import Rotation as ScRot
            R_corr = ScRot.from_euler('xyz', ee_corr_rpy).as_matrix()
            M = M @ R_corr
        self.ee_axes_mapping_matrix = M

        # EE 末端偏置: palm 相对于 EE body 的位置, 在 EE body 局部系下表达 (m)
        # IK 目标 = 手部追踪位置 - R_target @ ee_offset
        # (因为 palm_world = ee_body_world + R_ee_body * offset, 所以让 palm
        #  到达目标位置等价于让 ee_body 到达 target_pos - R_target * offset)
        ee_offset = side_cfg.get("ee_offset")
        if ee_offset is None:
            ee_offset = cfg_dict.get("ee_offset")
        if ee_offset is not None:
            self._ee_offset = np.array(ee_offset, dtype=np.float64)
        else:
            self._ee_offset = None

        # 关节限位 (numpy, 含软边距): side-specific 优先, 否则回退到全局
        margin = side_cfg.get("joint_margin", cfg_dict.get("joint_margin", 0.1))
        lower = side_cfg.get("joint_lower", cfg_dict.get("joint_lower"))
        upper = side_cfg.get("joint_upper", cfg_dict.get("joint_upper"))
        if lower is not None and upper is not None:
            if len(lower) != self.num_joints or len(upper) != self.num_joints:
                raise ValueError(
                    f"{arm_type}:{side} joint limits length mismatch: "
                    f"expected {self.num_joints}, got lower={len(lower)}, upper={len(upper)}"
                )
            self._q_lower = np.array([l + margin for l in lower], dtype=np.float64)
            self._q_upper = np.array([u - margin for u in upper], dtype=np.float64)
        else:
            self._q_lower = None
            self._q_upper = None

        # torch 版本 (for debug info and clamping output)
        if self._q_lower is not None:
            self._joint_lower = torch.tensor(self._q_lower, dtype=torch.float32, device=device)
            self._joint_upper = torch.tensor(self._q_upper, dtype=torch.float32, device=device)
        else:
            self._joint_lower = None
            self._joint_upper = None

        # ── 加载 Pinocchio 模型 ──
        urdf_path = cfg_dict.get("urdf_path")
        pin_ee_frame = side_cfg.get("pin_ee_frame", "flange")
        pin_arm_q_idx = side_cfg.get("pin_arm_q_indices", list(range(self.num_joints)))
        if urdf_path and os.path.exists(urdf_path):
            actual_urdf = urdf_path
            if cfg_dict.get("urdf_needs_material_cleanup", False):
                actual_urdf = self._cleanup_urdf_materials(urdf_path)
            self._pin_model = pin.buildModelFromUrdf(actual_urdf)
            self._pin_data = self._pin_model.createData()
            self._ee_frame_id = self._pin_model.getFrameId(pin_ee_frame)
            assert self._ee_frame_id < self._pin_model.nframes, \
                f"Frame '{pin_ee_frame}' not found in URDF. Available: " \
                f"{[self._pin_model.frames[i].name for i in range(self._pin_model.nframes)]}"
            self._pin_nq = self._pin_model.nq
            if pin_arm_q_idx == "auto":
                pin_arm_q_idx = self._auto_detect_q_indices(self.joint_names)
            self._pin_arm_q_idx = np.array(pin_arm_q_idx, dtype=int)
            print(f"[ArmIK] Pinocchio model loaded: {urdf_path}")
            print(f"[ArmIK]   full_nq={self._pin_nq}, arm_q_idx={self._pin_arm_q_idx}, "
                  f"ee_frame='{pin_ee_frame}' (id={self._ee_frame_id})")
        else:
            self._pin_model = None
            self._pin_data = None
            self._ee_frame_id = None
            self._pin_nq = 0
            self._pin_arm_q_idx = np.array([], dtype=int)
            print(f"[ArmIK] WARNING: URDF not found at {urdf_path}, Pinocchio IK disabled")

        # 首帧标定: Pinocchio 基座系 → 仿真世界系
        self._T_base_to_world: Optional[pin.SE3] = None
        self._prev_q_result: Optional[np.ndarray] = None  # 平滑滤波用
        self._prev_ik_solution: Optional[np.ndarray] = None  # CLIK 暖启动用

    @staticmethod
    def _cleanup_urdf_materials(urdf_path: str) -> str:
        """清理 URDF 中重复的 material 定义 (如 panda URDF)。

        返回清理后的临时文件路径。
        """
        import re
        import tempfile
        with open(urdf_path) as f:
            content = f.read()
        # 移除所有 <material ...>...</material> 和 <material .../> 标签
        content = re.sub(r'<material\s+name="[^"]*"\s*/>', '', content)
        content = re.sub(r'<material\s+name="[^"]*">\s*</material>', '', content)
        content = re.sub(
            r'<material\s+name="[^"]*">\s*<color[^/]*/>\s*</material>', '', content
        )
        tmp = tempfile.NamedTemporaryFile(
            suffix='.urdf', mode='w', delete=False, prefix='pin_clean_'
        )
        tmp.write(content)
        tmp.close()
        return tmp.name

    def _auto_detect_q_indices(self, joint_names: List[str]) -> List[int]:
        """在 Pinocchio 模型中按关节名自动查找 q 索引。"""
        indices = []
        for jn in joint_names:
            jid = self._pin_model.getJointId(jn)
            if jid < self._pin_model.njoints:
                idx_q = self._pin_model.joints[jid].idx_q
                indices.append(idx_q)
            else:
                print(f"[ArmIK] WARNING: joint '{jn}' not found in Pinocchio model")
        return indices

    def reset(self, env_ids=None):
        """重置标定 (下次 compute() 会重新标定)。"""
        self._T_base_to_world = None

    def _run_clik(
        self,
        T_target_base: pin.SE3,
        q_init_arm: np.ndarray,
        arm_idx: np.ndarray,
        err_weight: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool]:
        """单次 CLIK 迭代求解。

        Args:
            T_target_base: 目标位姿 (Pinocchio 基座系)
            q_init_arm: 初始手臂关节角 (n_arm,)
            arm_idx: 手臂关节在完整 q 中的索引
            err_weight: 6D 误差权重 (pos×3, ori×3)

        Returns:
            (best_q_arm, best_err_norm, converged): 误差最小帧的 q + 误差范数 + 是否收敛
        """
        # 构建完整 config (手臂以外的关节 = 0)
        q_full = np.zeros(self._pin_nq, dtype=np.float64)
        q_full[arm_idx] = np.clip(q_init_arm, self._q_lower, self._q_upper) \
            if self._q_lower is not None else q_init_arm

        converged = False
        best_err_norm = float('inf')
        best_q_arm = q_full[arm_idx].copy()
        W = np.diag(err_weight)

        for _ in range(NUM_IK_ITERS):
            pin.forwardKinematics(self._pin_model, self._pin_data, q_full)
            pin.updateFramePlacements(self._pin_model, self._pin_data)
            oMf = self._pin_data.oMf[self._ee_frame_id]

            # SE3 误差 (local frame): iMd = oMf^{-1} @ oMdes
            iMd = oMf.inverse() * T_target_base
            err = pin.log(iMd).vector  # 6D: (v_linear, omega) in local frame
            err_norm = np.linalg.norm(err)

            # 在更新前记录此配置 (更新后误差可能变大)
            if err_norm < best_err_norm:
                best_err_norm = err_norm
                best_q_arm = q_full[arm_idx].copy()

            if err_norm < EPS_CONVERGE:
                converged = True
                break

            err_w = err * err_weight

            # 完整雅可比 (6 × nq_full), 提取手臂列
            J_full = pin.computeFrameJacobian(
                self._pin_model, self._pin_data, q_full,
                self._ee_frame_id, pin.LOCAL,
            )
            J_arm = J_full[:, arm_idx]

            # 加权 DLS 求解
            Jw = W @ J_arm
            dq_arm = Jw.T @ np.linalg.solve(Jw @ Jw.T + DLS_DAMP * np.eye(6), err_w)

            # 每迭代步幅裁剪 (防止奇异点附近大跳跃)
            dq_max = np.max(np.abs(dq_arm))
            if dq_max > MAX_DQ_PER_ITER:
                dq_arm *= MAX_DQ_PER_ITER / dq_max

            q_full[arm_idx] += dq_arm * IK_DT
            if self._q_lower is not None:
                q_full[arm_idx] = np.clip(q_full[arm_idx], self._q_lower, self._q_upper)

        return best_q_arm, best_err_norm, converged

    def _calibrate_base_to_world(
        self, q_arm: np.ndarray, ee_pos_w_np: np.ndarray, ee_quat_wxyz_np: np.ndarray,
    ):
        """首帧标定: 根据当前关节角 + Isaac EE 世界位姿, 求 Pinocchio 基座到世界的变换。

        T_base_world = T_ee_world @ T_ee_base^{-1}
        """
        # 构建完整 config (手臂关节填入, 其余为 0)
        q_full = np.zeros(self._pin_nq, dtype=np.float64)
        q_full[self._pin_arm_q_idx] = q_arm

        # Pinocchio FK → EE in base frame
        pin.forwardKinematics(self._pin_model, self._pin_data, q_full)
        pin.updateFramePlacements(self._pin_model, self._pin_data)
        T_ee_base = self._pin_data.oMf[self._ee_frame_id]  # SE3

        # Isaac EE 世界位姿 → SE3
        R_ee_w = _quat_wxyz_to_rotmat(ee_quat_wxyz_np)
        T_ee_world = pin.SE3(R_ee_w, ee_pos_w_np)

        # T_base_world = T_ee_world @ T_ee_base^{-1}
        self._T_base_to_world = T_ee_world * T_ee_base.inverse()
        print(f"[ArmIK] Base-to-world calibrated:")
        print(f"[ArmIK]   q_arm = {q_arm.round(4).tolist()}")
        print(f"[ArmIK]   T_ee_base.translation  = {T_ee_base.translation}")
        print(f"[ArmIK]   T_ee_world.translation = {T_ee_world.translation}")
        print(f"[ArmIK]   T_base_world.translation = {self._T_base_to_world.translation}")
        # Sanity check: re-compose. If isaac and pinocchio disagree on what
        # `ee_body / pin_ee_frame` means, future IK will be wrong by a constant
        # offset even though best_err converges.
        T_ee_world_check = self._T_base_to_world * T_ee_base
        delta = np.linalg.norm(T_ee_world_check.translation - ee_pos_w_np)
        print(f"[ArmIK]   recompose check: |isaac_ee - T_base*T_ee_base|={delta:.6f}")

    def compute(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        current_joint_pos: torch.Tensor,
        ee_pos_w: torch.Tensor,
        ee_quat_w: torch.Tensor,
        root_pos_w: torch.Tensor,
        root_quat_w: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算目标关节位置 (6-DOF, Pinocchio 迭代 CLIK)。

        Args:
            target_pos: (3,) 目标位置, 仿真世界系
            target_quat: (4,) 目标朝向 (w,x,y,z), 仿真世界系
            current_joint_pos: (num_envs, num_arm_joints) 当前关节角
            ee_pos_w, ee_quat_w: 当前 EE 世界位姿 (quat: w,x,y,z)
            root_pos_w, root_quat_w: 机器人基座世界位姿

        Returns:
            joint_pos_des: (num_envs, num_arm_joints)
        """
        n = current_joint_pos.shape[0]
        q_arm_np = current_joint_pos[0].cpu().numpy().astype(np.float64)

        # ── 首帧标定 ──
        if self._T_base_to_world is None:
            ee_pos_np = ee_pos_w[0].cpu().numpy().astype(np.float64)
            ee_quat_np = ee_quat_w[0].cpu().numpy().astype(np.float64)  # wxyz
            self._calibrate_base_to_world(q_arm_np, ee_pos_np, ee_quat_np)

        target_pos = np.array(target_pos, dtype=np.float64, copy=True)

        # ── EE 末端偏置: 将手部追踪目标位置转换为 EE body 目标位置 ──
        # palm_world = ee_body_world + R_ee_body @ offset
        # IK 目标是让 palm 到达 target_pos, 所以:
        #   ee_body_target = target_pos - R_target @ offset
        if self._ee_offset is not None:
            R_tgt_for_offset = _quat_wxyz_to_rotmat(target_quat)
            offset_world = R_tgt_for_offset.astype(np.float64) @ self._ee_offset
            target_pos = target_pos - offset_world

        # ── 目标位姿 → Pinocchio 基座系 ──
        R_tgt_w = _quat_wxyz_to_rotmat(target_quat)
        T_target_world = pin.SE3(R_tgt_w.astype(np.float64), target_pos.astype(np.float64))
        T_target_base = self._T_base_to_world.inverse() * T_target_world  # oMdes

        arm_idx = self._pin_arm_q_idx  # e.g. [0,1,2,3,4,5]

        # 6D 误差权重向量: [pos(3), ori(3)]
        err_weight = np.array([POS_WEIGHT]*3 + [ORI_WEIGHT]*3, dtype=np.float64)

        # ── 第 1 次尝试: warm-start (上帧 IK 解 或 当前关节角) ──
        if self._prev_ik_solution is not None:
            q_init_warm = self._prev_ik_solution.copy()
        else:
            q_init_warm = q_arm_np.copy()
        q_warm, err_warm, conv_warm = self._run_clik(
            T_target_base, q_init_warm, arm_idx, err_weight,
        )
        # 候选解列表 (q, err_norm, converged)
        candidates: List[Tuple[np.ndarray, float, bool]] = [(q_warm, err_warm, conv_warm)]

        # ── 若误差较大, 做 random restart 逃出局部最优 ──
        if err_warm > IK_RESTART_THRESH and self._q_lower is not None:
            rng = get_np_rng()
            for _ in range(IK_RESTART_TRIES):
                q_init_rand = rng.uniform(self._q_lower, self._q_upper)
                q_try, err_try, conv_try = self._run_clik(
                    T_target_base, q_init_rand, arm_idx, err_weight,
                )
                candidates.append((q_try, err_try, conv_try))

        # ── 综合评分: 优先选 误差小 且 离当前 isaac 状态近 的解 ──
        # 单纯按 err 选会导致 elbow flip / 分支跳变，每帧给 isaac 完全不同的命令，
        # 关节被命令反复反向运动，最终撞上限位卡死。加 |q - q_isaac| 惩罚抑制分支跳变。
        DQ_PENALTY_WEIGHT = 0.5   # err_norm 单位 ~ rad (log SE3); dq 单位 rad → 同量级
                                   # 压制偶发的 elbow/wrist 分支跳变 (单帧 |Δq|>5 rad)
        # 只在"足够好"的候选中挑（err < max(IK_RESTART_THRESH, best_err * 1.5)），
        # 避免为了更近就接受高误差解。
        min_err = min(c[1] for c in candidates)
        err_thresh = max(IK_RESTART_THRESH, min_err * 1.5)
        good = [c for c in candidates if c[1] <= err_thresh] or candidates
        def _score(c):
            q, err, _ = c
            dq = np.linalg.norm(q - q_arm_np)
            return err + DQ_PENALTY_WEIGHT * dq
        best_q_arm, best_err_norm, converged = min(good, key=_score)

        # ── 裁剪总关节变化量 (遥操作平滑) ──
        # 使用最优帧（非最后帧），防止不可达时迭代跑偏并最终输出错误配置
        q_arm_result = best_q_arm
        # 暖启动: 只在 IK 收敛（或接近收敛）时更新，防止不可达目标把 warm-start
        # 污染成"乱跑"的奇异点附近配置，导致下一帧从坏的出发点继续跑偏。
        if converged or best_err_norm < 0.3:
            self._prev_ik_solution = q_arm_result.copy()
        # else: 保持上一帧的 warm-start（靠近上一帧目标，比当前乱解更好）

        # delta 相对于 Isaac 实际关节角 (保证安全)
        delta_q = q_arm_result - q_arm_np

        # 记录是否被裁剪
        delta_q = np.clip(delta_q, -MAX_JOINT_DELTA, MAX_JOINT_DELTA)
        q_result = q_arm_np + delta_q

        # 最终关节限位
        if self._q_lower is not None:
            q_result = np.clip(q_result, self._q_lower, self._q_upper)

        # ── 近目标平滑 (抗抖动): 只在位置误差小时启用, 避免追踪大误差时引入延迟 ──
        pos_err_now = np.linalg.norm(target_pos - ee_pos_w[0].cpu().numpy())
        if self._prev_q_result is not None and pos_err_now < _SMOOTH_NEAR_THRESH:
            q_result = _SMOOTH_ALPHA * q_result + (1.0 - _SMOOTH_ALPHA) * self._prev_q_result
            if self._q_lower is not None:
                q_result = np.clip(q_result, self._q_lower, self._q_upper)

        self._prev_q_result = q_result.copy()

        # 转 torch (batch)
        q_result_t = torch.tensor(q_result, dtype=torch.float32, device=self.device).unsqueeze(0)
        if n > 1:
            q_result_t = q_result_t.expand(n, -1)

        return q_result_t
