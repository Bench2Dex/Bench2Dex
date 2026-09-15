"""
isaaclab_bridge.py — Bridge between TeleopController and IsaacLab Articulation.

Maps teleop joint targets to the correct indices in the full robot articulation
(which includes arm + hand joints).

Usage:
    from teleop import TeleopController
    from teleop.isaaclab_bridge import TeleopIsaacLabBridge

    ctrl = TeleopController(hand_type="rh56dfx", enable_right=True)
    bridge = TeleopIsaacLabBridge(ctrl, articulation)
    bridge.start()

    # In simulation loop:
    bridge.update(hold_targets)  # modifies hold_targets in-place
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .teleop_controller import TeleopController
from .arkit_wrist_reader import ARKitWristReader
from .arm_ik_controller import ArmIKController, detect_arm_type, ARM_JOINT_CONFIGS
from .math_utils import (
    quat_wxyz_to_rotmat as _quat_wxyz_to_rotmat_np,
    quat_xyzw_to_rotmat as _quat_xyzw_to_rotmat_np,
    rotmat_to_quat_wxyz,
    quat_wxyz_to_rotmat_torch as _quat_wxyz_to_rotmat,
    arkit_quat_to_sim_quat,
)

try:
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    _HAS_MARKERS = True
except ImportError:
    _HAS_MARKERS = False


# 可视化开关
_VIS_HAND_ROBOT_FK     = False  # 红色球: 机器人 FK 指尖 (5 pts)
_VIS_MANUS_SKELETON    = False  # 绿色球: Manus 原始骨架 (21 pts)
_VIS_ARM_TARGET_POS    = True  # 红/橙色球: 手臂 IK 目标位置
_VIS_ARM_EE_POS        = True  # 绿/青色球: 手臂实际 EE 位置
_VIS_ARM_AXES          = False  # RGB 坐标轴: EE 朝向 + 目标朝向

AXIS_VIS_LEN = 0.06  # 坐标轴球到原点距离 (m)


# ── Hand joint index resolution ────────────────────────────────────────────
# With dex_retargeting, RetargetBridge.joint_names gives the output joint
# names in pinocchio order.  We match these against the articulation's
# joint_names to build the index mapping.  This replaces the old
# HAND_JOINT_ORDERED / HAND_JOINT_PATTERNS dictionaries.


def _build_retarget_to_articulation_indices(
    retarget_joint_names: List[str],
    articulation_joint_names: List[str],
    skip_retarget_indices: set = None,
) -> Tuple[List[int], List[int]]:
    """Build mapping from retarget output indices → articulation joint indices.

    For each joint name in *retarget_joint_names*, find its index in
    *articulation_joint_names* by exact match.  Returns two parallel lists:

      - retarget_indices: positions in the retarget output array
      - articulation_indices: corresponding positions in the articulation

    Joint names that are not found in the articulation are skipped with a
    warning.  Joints whose retarget index is in *skip_retarget_indices* (e.g.
    fixed/mimic joints with value always 0) are also skipped — Isaac Sim
    handles them via USD mimic constraints.
    """
    if skip_retarget_indices is None:
        skip_retarget_indices = set()
    name_to_art_idx = {name: i for i, name in enumerate(articulation_joint_names)}
    ret_indices: List[int] = []
    art_indices: List[int] = []
    missing: List[str] = []
    skipped: List[str] = []

    for ret_i, name in enumerate(retarget_joint_names):
        if ret_i in skip_retarget_indices:
            skipped.append(name)
            continue
        art_i = name_to_art_idx.get(name)
        if art_i is not None:
            ret_indices.append(ret_i)
            art_indices.append(art_i)
        else:
            missing.append(name)

    if missing:
        print(f"[TeleopIsaacLabBridge] WARNING: {len(missing)} retarget joints not found "
              f"in articulation: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if skipped:
        print(f"[TeleopIsaacLabBridge] Skipping {len(skipped)} fixed/mimic joints "
              f"(handled by USD): {skipped}")

    return ret_indices, art_indices


class TeleopIsaacLabBridge:
    """
    Bridge between TeleopController and an IsaacLab Articulation object.

    Call update() each sim step to overwrite the hand joints in hold_targets.
    """

    def __init__(
        self,
        teleop: TeleopController,
        articulation,
        right_joint_indices: Optional[List[int]] = None,
        left_joint_indices: Optional[List[int]] = None,
        enable_arm_teleop: bool = False,
    ):
        """
        Args:
            teleop: TeleopController instance (already configured).
            articulation: IsaacLab Articulation object.
            right_joint_indices: Override auto-detected indices for right hand joints.
            left_joint_indices: Override auto-detected indices for left hand joints.
            enable_arm_teleop: 启用手臂 IK 遥操作 (iPhone 位置 + Manus 姿态).
        """
        self._teleop = teleop
        self._articulation = articulation

        # Get joint names from articulation
        self._joint_names: List[str] = list(articulation.data.joint_names)
        self._num_joints = len(self._joint_names)
        print(f"[TeleopIsaacLabBridge] Articulation has {self._num_joints} joints: {self._joint_names}")

        # ── Hand joint index mapping ──────────────────────────────────
        # Will be built in start() after teleop.start() creates RetargetBridge.
        # Store any explicit overrides for later.
        self._override_right_indices = right_joint_indices
        self._override_left_indices = left_joint_indices

        self._right_ret_indices: List[int] = []
        self._right_art_indices: List[int] = []
        self._left_ret_indices: List[int] = []
        self._left_art_indices: List[int] = []
        # Numpy/torch index arrays for vectorized assignment (built in _build_hand_joint_mapping)
        self._right_ret_indices_np: Optional[np.ndarray] = None
        self._right_art_indices_np: Optional[np.ndarray] = None
        self._left_ret_indices_np: Optional[np.ndarray] = None
        self._left_art_indices_np: Optional[np.ndarray] = None

        # ── 手臂 IK 遥操作 (iPhone 位置 + Manus 姿态) ──────────
        self._enable_arm_teleop = enable_arm_teleop
        self._arkit_right: Optional[ARKitWristReader] = None
        self._arkit_left: Optional[ARKitWristReader] = None
        self._arm_ik_right: Optional[ArmIKController] = None
        self._arm_ik_left: Optional[ArmIKController] = None
        self._arm_right_joint_indices: List[int] = []
        self._arm_left_joint_indices: List[int] = []
        # Cached body index for EE (avoids list().index() every frame)
        self._arm_right_ee_body_idx: Optional[int] = None
        self._arm_left_ee_body_idx: Optional[int] = None

        # 识别臂型号 (可能为 None)
        self._arm_type_right: Optional[str] = None
        self._arm_type_left: Optional[str] = None
        if self._enable_arm_teleop:
            self._arm_type_right = detect_arm_type(self._joint_names, side="right")
            self._arm_type_left = detect_arm_type(self._joint_names, side="left")

            if self._arm_type_right:
                self._arkit_right = ARKitWristReader(side="right")
                print(f"[TeleopIsaacLabBridge] Detected right arm: {self._arm_type_right}")

            if self._arm_type_left:
                self._arkit_left = ARKitWristReader(side="left")
                print(f"[TeleopIsaacLabBridge] Detected left arm: {self._arm_type_left}")

        # ── 可视化 markers (在 start() 的 _setup_visualization 中初始化) ──
        self._target_marker = None
        self._ee_marker = None
        self._ee_axes_marker = None
        self._tgt_axes_marker = None
        self._left_target_marker = None
        self._left_ee_marker = None
        self._left_ee_axes_marker = None
        self._left_tgt_axes_marker = None
        self._dexpilot_robot_marker = None
        self._manus_skeleton_marker = None
        self._right_wrist_body_idx = None
        self._right_tip_body_idxs: List[Optional[int]] = []
        self._right_tip_offsets: List[Optional[np.ndarray]] = []
        self._calibrated_pin_wrist_R = None
        self._vis_error_logged = False
        self._dexpilot_vis_count = 0

        self._started = False

    def start(self, wait_timeout: float = 5.0) -> bool:
        """Start the underlying teleop controller and arm IK (if enabled)."""
        result = self._teleop.start(wait_timeout=wait_timeout)
        self._started = result

        # ── Build hand joint index mapping (now that RetargetBridge exists) ──
        if result:
            self._build_hand_joint_mapping()

        # ── 初始化手臂 IK ──────────────────────────────────
        if self._enable_arm_teleop:
            self._init_arm_ik()

        # ── 可视化 markers ──
        if _HAS_MARKERS and result:
            self._setup_visualization()

        return result

    def _build_hand_joint_mapping(self):
        """Build retarget → articulation index mapping for left/right hands."""
        if self._override_right_indices is not None:
            self._right_art_indices = self._override_right_indices
            self._right_ret_indices = list(range(len(self._override_right_indices)))
        elif self._teleop._enable_right and self._teleop._bridge_right is not None:
            bridge_r = self._teleop._bridge_right
            match_names = bridge_r.articulation_joint_names or bridge_r.joint_names
            self._right_ret_indices, self._right_art_indices = \
                _build_retarget_to_articulation_indices(
                    match_names, self._joint_names,
                    skip_retarget_indices=bridge_r.fixed_joint_indices,
                )

        if self._override_left_indices is not None:
            self._left_art_indices = self._override_left_indices
            self._left_ret_indices = list(range(len(self._override_left_indices)))
        elif self._teleop._enable_left and self._teleop._bridge_left is not None:
            bridge_l = self._teleop._bridge_left
            match_names = bridge_l.articulation_joint_names or bridge_l.joint_names
            self._left_ret_indices, self._left_art_indices = \
                _build_retarget_to_articulation_indices(
                    match_names, self._joint_names,
                    skip_retarget_indices=bridge_l.fixed_joint_indices,
                )

        # Build numpy index arrays for vectorized assignment
        if self._right_ret_indices:
            self._right_ret_indices_np = np.array(self._right_ret_indices, dtype=np.intp)
            self._right_art_indices_np = np.array(self._right_art_indices, dtype=np.intp)
        if self._left_ret_indices:
            self._left_ret_indices_np = np.array(self._left_ret_indices, dtype=np.intp)
            self._left_art_indices_np = np.array(self._left_art_indices, dtype=np.intp)

        info = []
        if self._right_art_indices:
            rnames = [self._joint_names[i] for i in self._right_art_indices]
            info.append(f"right_hand={rnames} ({len(self._right_art_indices)}j)")
        if self._left_art_indices:
            lnames = [self._joint_names[i] for i in self._left_art_indices]
            info.append(f"left_hand={lnames} ({len(self._left_art_indices)}j)")
        print(f"[TeleopIsaacLabBridge] Mapped hand joints: {'; '.join(info)}")

    def _init_arm_ik_side(self, side: str):
        """Initialize arm IK controller for one side (right or left)."""
        arm_type = self._arm_type_right if side == "right" else self._arm_type_left
        arkit = self._arkit_right if side == "right" else self._arkit_left
        if arm_type is None or arkit is None:
            return

        device = str(self._articulation.device)
        num_envs = self._articulation.num_instances
        try:
            ik = ArmIKController(
                arm_type, num_envs=num_envs, device=device, side=side,
                hand_type=self._teleop._hand_type,
            )
            name_to_idx = {n: i for i, n in enumerate(self._joint_names)}
            joint_indices = [name_to_idx[j] for j in ik.joint_names if j in name_to_idx]
            arkit.start()
            # 验证 EE body 存在
            body_names = list(self._articulation.data.body_names)
            if ik.ee_body not in body_names:
                candidates = [n for n in body_names
                              if "tool" in n.lower() or "flange" in n.lower() or "ee" in n.lower()]
                raise ValueError(
                    f"EE body '{ik.ee_body}' not in body_names. "
                    f"All bodies: {body_names}. Candidates: {candidates}"
                )
            # Cache EE body index
            ee_body_idx = body_names.index(ik.ee_body)
            # 赋值
            if side == "right":
                self._arm_ik_right = ik
                self._arm_right_joint_indices = joint_indices
                self._arm_right_ee_body_idx = ee_body_idx
            else:
                self._arm_ik_left = ik
                self._arm_left_joint_indices = joint_indices
                self._arm_left_ee_body_idx = ee_body_idx
            print(f"[TeleopIsaacLabBridge] {side.title()} arm IK ready, "
                  f"joints={ik.joint_names}, ee={ik.ee_body}")
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"[TeleopIsaacLabBridge] {side.title()} arm IK init failed: {e}")

    def _init_arm_ik(self):
        """Initialize arm IK controllers for both sides."""
        self._init_arm_ik_side("right")
        self._init_arm_ik_side("left")

    def _setup_visualization(self):
        """Create all visualization markers (arm + hand). Called from start()."""
        # ── 手臂 IK markers ──
        if self._arm_ik_right or self._arm_ik_left:
            self._setup_arm_markers()

        # ── DexPilot hand keypoint markers ──
        self._setup_hand_markers()

    def _setup_arm_markers(self):
        """Create arm target/EE/axes markers."""
        if _VIS_ARM_TARGET_POS:
            self._target_marker = VisualizationMarkers(VisualizationMarkersCfg(
                prim_path="/World/Visuals/arm_target",
                markers={"target": sim_utils.SphereCfg(
                    radius=0.02,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                )},
            ))
        if _VIS_ARM_EE_POS:
            self._ee_marker = VisualizationMarkers(VisualizationMarkersCfg(
                prim_path="/World/Visuals/arm_ee",
                markers={"ee": sim_utils.SphereCfg(
                    radius=0.015,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                )},
            ))

        if _VIS_ARM_AXES:
            def _axes_cfg(path, r=0.008):
                return VisualizationMarkersCfg(
                    prim_path=path,
                    markers={
                        "x": sim_utils.SphereCfg(radius=r, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))),
                        "y": sim_utils.SphereCfg(radius=r, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))),
                        "z": sim_utils.SphereCfg(radius=r, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0))),
                    },
                )
            self._ee_axes_marker = VisualizationMarkers(_axes_cfg("/World/Visuals/ee_axes", 0.008))
            self._tgt_axes_marker = VisualizationMarkers(_axes_cfg("/World/Visuals/tgt_axes", 0.006))
            self._left_ee_axes_marker = VisualizationMarkers(_axes_cfg("/World/Visuals/left_ee_axes", 0.008))
            self._left_tgt_axes_marker = VisualizationMarkers(_axes_cfg("/World/Visuals/left_tgt_axes", 0.006))

        # 左臂专用 target/EE markers
        if self._arm_ik_left:
            if _VIS_ARM_TARGET_POS:
                self._left_target_marker = VisualizationMarkers(VisualizationMarkersCfg(
                    prim_path="/World/Visuals/left_arm_target",
                    markers={"target": sim_utils.SphereCfg(
                        radius=0.02,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)),
                    )},
                ))
            if _VIS_ARM_EE_POS:
                self._left_ee_marker = VisualizationMarkers(VisualizationMarkersCfg(
                    prim_path="/World/Visuals/left_arm_ee",
                    markers={"ee": sim_utils.SphereCfg(
                        radius=0.015,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.5)),
                    )},
                ))

    def _setup_hand_markers(self):
        """Locate wrist/fingertip body indices and create DexPilot markers."""
        bridge_r = getattr(self._teleop, '_bridge_right', None)
        if bridge_r is None or not bridge_r.wrist_link_name:
            return

        body_names = list(self._articulation.data.body_names)
        wname = bridge_r.wrist_link_name

        # ── 查找手腕 body index ──
        if wname in body_names:
            self._right_wrist_body_idx = body_names.index(wname)
        else:
            self._right_wrist_body_idx = self._find_wrist_fallback(wname, body_names, bridge_r)

        # ── 查找指尖 body indices ──
        self._right_tip_body_idxs = []
        self._right_tip_offsets = []
        for tname in bridge_r.finger_tip_link_names:
            idx, offset = self._find_tip_body(tname, body_names)
            self._right_tip_body_idxs.append(idx)
            self._right_tip_offsets.append(offset)

        # ── 创建 markers ──
        n_tips = len(bridge_r.finger_tip_link_names)
        if self._right_wrist_body_idx is not None and n_tips > 0:
            if _VIS_HAND_ROBOT_FK:
                self._dexpilot_robot_marker = VisualizationMarkers(VisualizationMarkersCfg(
                    prim_path="/World/Visuals/dexpilot_robot_tips",
                    markers={"tip": sim_utils.SphereCfg(
                        radius=0.008,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2)),
                    )},
                ))
            if _VIS_MANUS_SKELETON:
                self._manus_skeleton_marker = VisualizationMarkers(VisualizationMarkersCfg(
                    prim_path="/World/Visuals/manus_skeleton",
                    markers={"node": sim_utils.SphereCfg(
                        radius=0.006,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 1.0, 0.3)),
                    )},
                ))
            print(f"[DexPilotVis] Markers ready: wrist={wname}, tips={bridge_r.finger_tip_link_names}")

    @staticmethod
    def _find_wrist_fallback(wname: str, body_names: List[str], bridge_r) -> Optional[int]:
        """Fallback search for wrist body index when exact name not found."""
        arm_ee_hints = [
            'Link_6', 'r_Link_6', 'l_Link_6',
            'wrist_3_link', 'L_arm_wrist_3_link',
            'r_arm_flange', 'l_arm_flange',
            'xarm_r_link7', 'xarm_l_link7',
            'panda_link8', 'multi_panda_link8',
            'panda_link7', 'multi_panda_link7',
            'link_tcp', 'tool0',
        ]
        # Prefer candidates matching the wrist name prefix
        wname_prefix = ''
        if wname.startswith('multi_'):
            wname_prefix = 'multi_'
        elif wname.startswith('r_') or wname.startswith('l_'):
            wname_prefix = wname[:2]
        if wname_prefix:
            arm_ee_hints = sorted(arm_ee_hints,
                                  key=lambda c: 0 if c.startswith(wname_prefix) else 1)
        for cand in arm_ee_hints:
            if cand in body_names:
                print(f"[DexPilotVis] wrist '{wname}' not found, using '{cand}'")
                bridge_r._pin_wrist_offset = np.zeros(3, dtype=np.float32)
                return body_names.index(cand)
        # Last resort: fuzzy match
        for bname in body_names:
            if 'flange' in bname.lower() or 'wrist' in bname.lower():
                print(f"[DexPilotVis] wrist '{wname}' fuzzy match: '{bname}'")
                bridge_r._pin_wrist_offset = np.zeros(3, dtype=np.float32)
                return body_names.index(bname)
        print(f"[DexPilotVis] wrist '{wname}' not found in bodies")
        return None

    @staticmethod
    def _find_tip_body(tname: str, body_names: List[str]) -> Tuple[Optional[int], Optional[np.ndarray]]:
        """Find body index for a fingertip link, with fallback strategies."""
        if tname in body_names:
            return body_names.index(tname), None
        # Strategy 1: suffix replacement (dexhand021 pattern)
        for suffix, repl in [('_tip', '_4'), ('_tip', '_3'), ('_tip', '_pad'), ('_tip', '')]:
            cand = tname.replace(suffix, repl) if suffix in tname else None
            if cand and cand in body_names:
                offset = np.array([0.035, -0.004, 0.0], dtype=np.float32) if repl == '_4' else None
                print(f"[DexPilotVis] tip '{tname}' -> '{cand}'")
                return body_names.index(cand), offset
        # Strategy 2: substring match
        for bname in body_names:
            if tname in bname or bname.endswith(tname):
                print(f"[DexPilotVis] tip '{tname}' -> substr match '{bname}'")
                return body_names.index(bname), None
        print(f"[DexPilotVis] tip '{tname}' not found in bodies")
        return None, None

    def stop(self):
        """Stop the underlying teleop controller and arm readers."""
        self._teleop.stop()
        if self._arkit_right:
            self._arkit_right.stop()
        if self._arkit_left:
            self._arkit_left.stop()
        self._started = False

    def reset_articulation(self, new_articulation) -> None:
        """Update robot articulation reference after a scene respawn.

        Call this after build_scene() creates new Articulation objects
        (existing_robot_runtime=None) to keep teleop working. The joint/body
        name order is assumed unchanged (same robot USD).
        """
        self._articulation = new_articulation
        # Reinitialize arm IK with new articulation context (device/num_envs may differ)
        if self._enable_arm_teleop:
            self._arm_ik_right = None
            self._arm_ik_left = None
            self._arm_right_joint_indices = []
            self._arm_left_joint_indices = []
            self._arm_right_ee_body_idx = None
            self._arm_left_ee_body_idx = None
            self._init_arm_ik()
        # Refresh visualization body indices
        if self._right_wrist_body_idx is not None or self._right_tip_body_idxs:
            self._setup_hand_markers()
        print("[TeleopIsaacLabBridge] Articulation reference updated after scene respawn.")

    def update(self, hold_targets: dict) -> bool:
        """Read teleop SHM → apply hand joint targets → compute arm IK.

        Args:
            hold_targets: {id(articulation): joint_pos_tensor}
                          as used by write_articulation_targets() in utils/runtime_helpers.py

        Returns:
            True if any data was applied this frame.
        """
        if not self._started:
            return False

        target_key = id(self._articulation)
        target = hold_targets.get(target_key)
        if target is None:
            return False

        applied = False

        result = self._teleop.step()
        if result is None:
            result = {}

        # Apply right hand (vectorized)
        right_q = result.get("right")
        if right_q is not None and self._right_ret_indices_np is not None:
            vals = np.asarray(right_q, dtype=np.float32)[self._right_ret_indices_np]
            target[0, self._right_art_indices_np] = torch.from_numpy(vals).to(
                dtype=target.dtype, device=target.device)
            applied = True

        # Apply left hand (vectorized)
        left_q = result.get("left")
        if left_q is not None and self._left_ret_indices_np is not None:
            vals = np.asarray(left_q, dtype=np.float32)[self._left_ret_indices_np]
            target[0, self._left_art_indices_np] = torch.from_numpy(vals).to(
                dtype=target.dtype, device=target.device)
            applied = True

        # ── Hand visualization ──────────────────────────────
        self._update_hand_visualization(target)

        # ── Arm IK ─────────────────────────────────────────
        if self._enable_arm_teleop:
            applied = self._update_arm_ik(target) or applied

        return applied

    def _update_hand_visualization(self, target):
        """Update DexPilot hand markers (robot FK + Manus skeleton)."""
        if self._right_wrist_body_idx is None:
            return
        if self._dexpilot_robot_marker is None and self._manus_skeleton_marker is None:
            return

        bridge_r = self._teleop._bridge_right
        if bridge_r is None:
            return

        art = self._articulation
        wrist_pos_w = art.data.body_pos_w[0, self._right_wrist_body_idx]
        wrist_quat_w = art.data.body_quat_w[0, self._right_wrist_body_idx]
        wrist_R_sim = _quat_wxyz_to_rotmat(wrist_quat_w)

        # Auto-calibrate pin_wrist_R via Procrustes on first valid frame.
        if self._calibrated_pin_wrist_R is None:
            self._try_procrustes_calibration(bridge_r, art, wrist_pos_w, wrist_R_sim)

        if self._calibrated_pin_wrist_R is not None:
            pin_wrist_R = self._calibrated_pin_wrist_R
        else:
            pin_wrist_R = torch.tensor(bridge_r._pin_wrist_R, dtype=torch.float32,
                                       device=art.device)

        R_transform = wrist_R_sim @ pin_wrist_R.T

        # Robot FK wrist→tip vectors
        if self._dexpilot_robot_marker is not None:
            r_vecs = torch.tensor(bridge_r.last_robot_wrist_to_tips,
                                  dtype=torch.float32, device=art.device)
            r_world = (r_vecs @ R_transform.T) + wrist_pos_w.unsqueeze(0)
            all_robot = torch.cat([wrist_pos_w.unsqueeze(0), r_world], dim=0)
            self._dexpilot_robot_marker.visualize(translations=all_robot)

        # Manus raw skeleton
        if self._manus_skeleton_marker is not None:
            manus_kp = bridge_r.last_manus_keypoints_scaled
            if manus_kp is not None and manus_kp.shape[0] == 21:
                m_vecs = torch.tensor(manus_kp, dtype=torch.float32, device=art.device)
                m_world = (m_vecs @ R_transform.T) + wrist_pos_w.unsqueeze(0)
                self._manus_skeleton_marker.visualize(translations=m_world)

    def _try_procrustes_calibration(self, bridge_r, art, wrist_pos_w, wrist_R_sim):
        """One-shot Procrustes calibration of pin_wrist_R from sim tip positions."""
        r_tips = bridge_r.last_robot_wrist_to_tips
        if not np.any(np.abs(r_tips) > 1e-4):
            return  # FK not ready yet

        sim_tip_vecs = []
        for fi, tidx in enumerate(self._right_tip_body_idxs):
            if tidx is None:
                continue
            tip_w = art.data.body_pos_w[0, tidx]
            offset = self._right_tip_offsets[fi] if fi < len(self._right_tip_offsets) else None
            if offset is not None:
                tip_R = _quat_wxyz_to_rotmat(art.data.body_quat_w[0, tidx])
                off_t = torch.tensor(offset, dtype=torch.float32, device=art.device)
                tip_w = tip_w + (tip_R @ off_t)
            sim_tip_vecs.append((tip_w - wrist_pos_w).cpu().numpy())

        n = min(len(sim_tip_vecs), len(r_tips))
        if n < 3:
            return

        A = np.array(r_tips[:n])
        B = np.array(sim_tip_vecs[:n])
        H = A.T @ B
        U, _, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        S = np.diag([1, 1, d])
        R_transform_calib = (Vt.T @ S @ U.T).astype(np.float32)
        wrist_R_np = wrist_R_sim.cpu().numpy()
        calib_pin_R = (R_transform_calib.T @ wrist_R_np).astype(np.float32)
        self._calibrated_pin_wrist_R = torch.tensor(
            calib_pin_R, dtype=torch.float32, device=art.device)
        print(f"[DexPilotVis] Auto-calibrated pin_wrist_R via Procrustes ({n} tips)")

    def _update_arm_ik(self, target) -> bool:
        """位置=iPhone ARKit, 姿态=Manus SHM → IK 计算 → 写入 target。

        """
        applied = False
        art = self._articulation

        # 右臂
        if self._arm_ik_right and self._arkit_right and self._arkit_right.is_receiving():
            pos = self._arkit_right.get_position()
            quat = self._teleop.get_wrist_quaternion("right")
            if pos is not None and quat is not None:
                # 获取 IK 所需数据 (ee_idx cached at init)
                ee_idx = self._arm_right_ee_body_idx
                arm_col_indices = self._arm_right_joint_indices
                cur_joint_pos = art.data.joint_pos[:, arm_col_indices]
                ee_pos_w = art.data.body_pos_w[:, ee_idx]
                ee_quat_w = art.data.body_quat_w[:, ee_idx]
                root_pos_w = art.data.root_pos_w
                root_quat_w = art.data.root_quat_w

                # 朝向: 从 iPhone ARKit 获取四元数 (替代 Manus node0)
                arkit_quat = self._arkit_right.get_quaternion()  # (qx,qy,qz,qw) ARKit Y-up

                if arkit_quat is not None:
                    # A: 可视化目标四元数 (含手掌对齐修正)
                    target_quat_wxyz = arkit_quat_to_sim_quat(arkit_quat)
                    # IK 目标: 通过轴映射矩阵将 A 姿态转换为 EE body 目标姿态
                    # R_B = R_A @ M, 其中 M 由 ARM_JOINT_CONFIGS["ee_axes_mapping"] 定义
                    R_A = _quat_wxyz_to_rotmat_np(target_quat_wxyz)
                    R_B = R_A @ self._arm_ik_right.ee_axes_mapping_matrix
                    ik_quat_wxyz = rotmat_to_quat_wxyz(R_B)
                else:
                    # 回退: 仅位置, 用当前 EE 朝向作为目标
                    target_quat_wxyz = ee_quat_w[0].cpu().numpy()
                    ik_quat_wxyz = target_quat_wxyz

                joint_pos_des = self._arm_ik_right.compute(
                    target_pos=pos, target_quat=ik_quat_wxyz,
                    current_joint_pos=cur_joint_pos,
                    ee_pos_w=ee_pos_w, ee_quat_w=ee_quat_w,
                    root_pos_w=root_pos_w, root_quat_w=root_quat_w,
                )
                # 写入 hold_targets (vectorized)
                target[0, arm_col_indices] = joint_pos_des[0, :len(arm_col_indices)]
                applied = True

                # 可视化标记
                if self._target_marker is not None:
                    tgt_t = torch.tensor(pos, dtype=torch.float32, device=art.device).unsqueeze(0)
                    self._target_marker.visualize(translations=tgt_t)
                if self._ee_marker is not None:
                    self._ee_marker.visualize(translations=ee_pos_w)

                # 朝向可视化: EE 坐标轴 (RGB = XYZ)
                if self._ee_axes_marker is not None:
                    ee_R = _quat_wxyz_to_rotmat(ee_quat_w[0])  # (3, 3)
                    ee_p = ee_pos_w[0]  # (3,)
                    ee_axes_t = torch.stack([
                        ee_p + AXIS_VIS_LEN * ee_R[:, 0],
                        ee_p + AXIS_VIS_LEN * ee_R[:, 1],
                        ee_p + AXIS_VIS_LEN * ee_R[:, 2],
                    ])  # (3, 3)
                    self._ee_axes_marker.visualize(
                        translations=ee_axes_t,
                        marker_indices=torch.tensor([0, 1, 2], device=art.device),
                    )

                # 朝向可视化: 目标坐标轴 (RGB = XYZ, ARKit 映射后)
                if self._tgt_axes_marker is not None and arkit_quat is not None:
                    tgt_quat_t = torch.tensor(target_quat_wxyz, dtype=torch.float32, device=art.device)
                    mR_w = _quat_wxyz_to_rotmat(tgt_quat_t)
                    tp = torch.tensor(pos, dtype=torch.float32, device=art.device)
                    tgt_axes_t = torch.stack([
                        tp + AXIS_VIS_LEN * mR_w[:, 0],
                        tp + AXIS_VIS_LEN * mR_w[:, 1],
                        tp + AXIS_VIS_LEN * mR_w[:, 2],
                    ])
                    self._tgt_axes_marker.visualize(
                        translations=tgt_axes_t,
                        marker_indices=torch.tensor([0, 1, 2], device=art.device),
                    )

        # 左臂
        if self._arm_ik_left and self._arkit_left and self._arkit_left.is_receiving():
            pos = self._arkit_left.get_position()
            quat = self._teleop.get_wrist_quaternion("left")
            if pos is not None and quat is not None:
                ee_idx = self._arm_left_ee_body_idx
                arm_col_indices = self._arm_left_joint_indices
                cur_joint_pos = art.data.joint_pos[:, arm_col_indices]
                ee_pos_w = art.data.body_pos_w[:, ee_idx]
                ee_quat_w = art.data.body_quat_w[:, ee_idx]
                root_pos_w = art.data.root_pos_w
                root_quat_w = art.data.root_quat_w

                # 朝向: 从 iPhone ARKit 获取四元数 (和右臂一致的变换流程)
                arkit_quat = self._arkit_left.get_quaternion()  # (qx,qy,qz,qw) ARKit Y-up

                if arkit_quat is not None:
                    # A: 可视化目标四元数 (含手掌对齐修正)
                    target_quat_wxyz = arkit_quat_to_sim_quat(arkit_quat)
                    # IK 目标: 通过轴映射矩阵将 A 姿态转换为 EE body 目标姿态
                    R_A = _quat_wxyz_to_rotmat_np(target_quat_wxyz)
                    R_B = R_A @ self._arm_ik_left.ee_axes_mapping_matrix
                    ik_quat_wxyz = rotmat_to_quat_wxyz(R_B)
                else:
                    # 回退: 仅位置, 用当前 EE 朝向作为目标
                    target_quat_wxyz = ee_quat_w[0].cpu().numpy()
                    ik_quat_wxyz = target_quat_wxyz

                joint_pos_des = self._arm_ik_left.compute(
                    target_pos=pos, target_quat=ik_quat_wxyz,
                    current_joint_pos=cur_joint_pos,
                    ee_pos_w=ee_pos_w, ee_quat_w=ee_quat_w,
                    root_pos_w=root_pos_w, root_quat_w=root_quat_w,
                )
                target[0, arm_col_indices] = joint_pos_des[0, :len(arm_col_indices)]
                applied = True

                # 可视化标记: 目标点 (橙) + EE 位置 (青绿)
                if self._left_target_marker is not None:
                    tgt_t = torch.tensor(pos, dtype=torch.float32, device=art.device).unsqueeze(0)
                    self._left_target_marker.visualize(translations=tgt_t)
                if self._left_ee_marker is not None:
                    self._left_ee_marker.visualize(translations=ee_pos_w)

                # 朝向可视化: EE 坐标轴 (RGB = XYZ)
                if self._left_ee_axes_marker is not None:
                    ee_R = _quat_wxyz_to_rotmat(ee_quat_w[0])
                    ee_p = ee_pos_w[0]
                    ee_axes_t = torch.stack([
                        ee_p + AXIS_VIS_LEN * ee_R[:, 0],
                        ee_p + AXIS_VIS_LEN * ee_R[:, 1],
                        ee_p + AXIS_VIS_LEN * ee_R[:, 2],
                    ])
                    self._left_ee_axes_marker.visualize(
                        translations=ee_axes_t,
                        marker_indices=torch.tensor([0, 1, 2], device=art.device),
                    )

                # 朝向可视化: 目标坐标轴 (RGB = XYZ, ARKit 映射后)
                if self._left_tgt_axes_marker is not None and arkit_quat is not None:
                    tgt_quat_t = torch.tensor(target_quat_wxyz, dtype=torch.float32, device=art.device)
                    mR_w = _quat_wxyz_to_rotmat(tgt_quat_t)
                    tp = torch.tensor(pos, dtype=torch.float32, device=art.device)
                    tgt_axes_t = torch.stack([
                        tp + AXIS_VIS_LEN * mR_w[:, 0],
                        tp + AXIS_VIS_LEN * mR_w[:, 1],
                        tp + AXIS_VIS_LEN * mR_w[:, 2],
                    ])
                    self._left_tgt_axes_marker.visualize(
                        translations=tgt_axes_t,
                        marker_indices=torch.tensor([0, 1, 2], device=art.device),
                    )

        return applied

    @property
    def right_joint_names(self) -> List[str]:
        return [self._joint_names[i] for i in self._right_art_indices]

    @property
    def left_joint_names(self) -> List[str]:
        return [self._joint_names[i] for i in self._left_art_indices]

    # ── 手臂 IK 运行时 API ──────────────────────────────────

    def reset_arm_anchor(self):
        """重置手臂位置锚点 (下一帧 ARKit 数据成为新的位置参考原点)。
        
        仿真锚点固定为 SIM_ANCHOR, 只需重置 ARKit 侧的参考原点。
        """
        if self._arkit_right:
            self._arkit_right.reset_anchor()
        if self._arkit_left:
            self._arkit_left.reset_anchor()

    def reset_input_state(self):
        """Reset teleop filters and arm anchors after discontinuities such as homing."""
        self._teleop.reset_filters()
        self.reset_arm_anchor()

    def __repr__(self) -> str:
        arm_info = ""
        if self._enable_arm_teleop:
            arm_r = f"right_arm={len(self._arm_right_joint_indices)}j" if self._arm_ik_right else "right_arm=OFF"
            arm_l = f"left_arm={len(self._arm_left_joint_indices)}j" if self._arm_ik_left else "left_arm=OFF"
            arm_info = f", {arm_r}, {arm_l}"
        return (
            f"TeleopIsaacLabBridge("
            f"hand_type={self._teleop.hand_type!r}, "
            f"right_hand={len(self._right_art_indices)}j, "
            f"left_hand={len(self._left_art_indices)}j"
            f"{arm_info}, "
            f"total_joints={self._num_joints})"
        )
