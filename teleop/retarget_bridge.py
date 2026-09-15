"""
retarget_bridge.py — Manus 25 skeleton nodes → target hand joint angles.

Pipeline: pos_dict → MediaPipe 21 keypoints → wrist frame → DexPilot → qpos.
Hand-side aware: one RetargetBridge instance per hand (left or right).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from dex_retargeting.constants import OPERATOR2MANO_RIGHT, OPERATOR2MANO_LEFT

# ── Monkey-patch: pinocchio model.nqs binding breaks under Isaac Sim ───────
from dex_retargeting.robot_wrapper import RobotWrapper as _RW

@property
def _patched_dof_joint_names(self):
    model = self.model
    return [
        model.names[i] for i in range(1, model.njoints)
        if model.joints[i].nq > 0
    ]

_RW.dof_joint_names = _patched_dof_joint_names

# ── Manus 25-node to MediaPipe 21-keypoint mapping ────────────────────────
# Manus has 25 nodes: wrist(0), then 5 fingers × {CMC, MCP, PIP, DIP, TIP}.
# MediaPipe uses 21 keypoints: wrist(0), then 5 fingers × 4 (no separate CMC for non-thumb).
# We skip nodes 5 (Index CMC), 10 (Middle CMC), 15 (Ring CMC), 20 (Pinky CMC).
MANUS_TO_MEDIAPIPE_MAP = [
    0,                 # MP[0]  = Wrist
    1, 2, 3, 4,        # MP[1-4]  = Thumb CMC, MCP, IP, TIP
    6, 7, 8, 9,        # MP[5-8]  = Index MCP, PIP, DIP, TIP (skip 5=CMC)
    11, 12, 13, 14,    # MP[9-12] = Middle (skip 10=CMC)
    16, 17, 18, 19,    # MP[13-16] = Ring (skip 15=CMC)
    21, 22, 23, 24,    # MP[17-20] = Pinky (skip 20=CMC)
]

# ── Default URDF and config directories ────────────────────────────────────
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


_URDF_ZOO_DIR = _default_urdf_zoo_dir()
_HAND_CFGS_DIR = Path(__file__).parent / "hand_cfgs"

# Map dex2bench hand_type → retarget config filename (single file per hand)
_HAND_TYPE_TO_CONFIG = {
    "rh56dfx":    "rh56dfx.yml",
    "rh5dg2":     "rh5dg2.yml",
    "wuji":       "wuji.yml",
    "shadow":     "shadow.yml",
    "schunk_svh": "schunk_svh.yml",
    "allegro":    "allegro.yml",
    "orca":       "orca.yml",
    "ability":    "ability.yml",
    "leap":       "leap.yml",
    "sharpa":     "sharpa.yml",
    "dexhand021": "dexhand021.yml",
    "revo2":      "revo2.yml",
}


def manus_nodes_to_mediapipe(pos_dict: dict) -> Optional[np.ndarray]:
    """
    Convert Manus 25-node position dict to MediaPipe 21-keypoint array.

    Args:
        pos_dict: {node_id (int): np.array([x, y, z])} from ShmReader.get_pos_dict()

    Returns:
        (21, 3) np.float32 array, or None if insufficient data
    """
    out = np.zeros((21, 3), dtype=np.float32)
    for mp_idx, manus_id in enumerate(MANUS_TO_MEDIAPIPE_MAP):
        if manus_id not in pos_dict:
            return None  # incomplete data
        out[mp_idx] = pos_dict[manus_id]
    return out


def _wrist_frame_from_quat(quat: np.ndarray) -> np.ndarray:
    """Compute wrist rotation frame from Manus node 0 quaternion (qx,qy,qz,qw).

    Manus local frame: -Y=palm normal, +Z=finger direction, -X=thumb side.
    We map to canonical: x=finger(+Z), y=palm outward(+Y), z=towards index(-X).
    """
    qx, qy, qz, qw = quat
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float32)

    R_local_to_canonical = np.array([
        [0., 0., -1.],
        [0., 1.,  0.],
        [1., 0.,  0.],
    ], dtype=np.float32)

    return R @ R_local_to_canonical


class RetargetBridge:
    """
    Bridge from Manus skeleton data to target robot hand joint angles.

    Uses dex_retargeting's DexPilot optimizer for all hand types.

    Usage:
        bridge = RetargetBridge("rh56dfx", hand_side="right")
        target_q = bridge.retarget(pos_dict)  # → (12,) for RH56DFX or None
    """

    def __init__(
        self,
        hand_type: str,
        hand_side: str = "right",
        yaml_path: Optional[str] = None,
        urdf_dir: Optional[str] = None,
    ):
        """
        Args:
            hand_type: Key in _HAND_TYPE_TO_CONFIG, e.g. "rh56dfx".
            hand_side: "left" or "right".
            yaml_path: Override path to retarget YAML config. If None, auto-detected.
            urdf_dir: Override path to the robot URDF root directory.
                      If None, uses URDF-Zoo.
        """
        self._hand_type = hand_type
        self._side = hand_side.lower()
        assert self._side in ("left", "right"), f"Invalid side: {hand_side}"

        # Resolve config path
        if yaml_path is None:
            cfg_name = _HAND_TYPE_TO_CONFIG.get(hand_type)
            if cfg_name is None:
                raise ValueError(
                    f"No retarget config for hand type '{hand_type}'. "
                    f"Available: {list(_HAND_TYPE_TO_CONFIG.keys())}"
                )
            yaml_path = str(_HAND_CFGS_DIR / cfg_name)

        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Retarget config not found: {yaml_path}")

        # Cache for hot-reload
        self._yaml_path = yaml_path
        self._yaml_mtime = os.path.getmtime(yaml_path)

        # Set URDF search directory
        if urdf_dir is None:
            urdf_dir = str(_URDF_ZOO_DIR)
        RetargetingConfig.set_default_urdf_dir(urdf_dir)

        # Read YAML and extract our custom fields before passing to RetargetingConfig
        import yaml
        with open(yaml_path, "r") as f:
            raw_cfg = yaml.safe_load(f)
        retarget_sec = raw_cfg.get("retargeting", {})

        # Merge side-specific overrides ("left:" / "right:" sections in YAML)
        side_overrides = retarget_sec.pop(self._side, {})
        other_side = "left" if self._side == "right" else "right"
        retarget_sec.pop(other_side, None)
        retarget_sec.update(side_overrides)

        # Pop custom fields that RetargetingConfig doesn't understand
        explicit_names = retarget_sec.pop("articulation_joint_names", None)
        art_prefix = retarget_sec.pop("articulation_prefix", None)
        coupling_cfg = retarget_sec.pop("joint_coupling", None)
        joint_limits_override = retarget_sec.pop("joint_limits_override", None)
        sim_wrist_rpy = retarget_sec.pop("sim_wrist_to_pin_root_rpy", None)
        mano_to_pin_rpy = retarget_sec.pop("mano_to_pinocchio_rpy", None)
        mano_corr_vis = retarget_sec.pop("mano_correction_vis", True)
        hand_offset = retarget_sec.pop("hand_offset_xyz", None)
        hand_scale = retarget_sec.pop("hand_scale_xyz", None)
        quat_corr_rpy = retarget_sec.pop("wrist_quat_correction_rpy", None)

        urdf_path_value = retarget_sec.get("urdf_path")
        if urdf_path_value:
            urdf_path = Path(str(urdf_path_value))
            local_urdf_path = Path(yaml_path).resolve().parent / urdf_path
            if not urdf_path.is_absolute() and local_urdf_path.exists():
                retarget_sec["urdf_path"] = str(local_urdf_path)

        # Store wrist and fingertip link names for debug visualization
        self.wrist_link_name: str = retarget_sec.get("wrist_link_name", "")
        self.finger_tip_link_names: List[str] = list(retarget_sec.get("finger_tip_link_names", []))

        # Build retargeting engine (from cleaned dict)
        config = RetargetingConfig.from_dict(retarget_sec)
        self._retargeting: SeqRetargeting = config.build()

        # ── Apply joint limit overrides from YAML ──
        # Allows locking specific joints (e.g. abduction) to a fixed range.
        if joint_limits_override:
            seq = self._retargeting
            opt = seq.optimizer
            target_names = opt.target_joint_names
            for jname, bounds in joint_limits_override.items():
                if jname in target_names:
                    ti = target_names.index(jname)
                    lo, hi = float(bounds[0]), float(bounds[1])
                    seq.joint_limits[ti, 0] = lo
                    seq.joint_limits[ti, 1] = hi
                    print(f"[RetargetBridge] Joint limit override: {jname} → [{lo}, {hi}]")
                else:
                    print(f"[RetargetBridge] WARNING: joint_limits_override key '{jname}' not in target joints")
            # Re-apply limits to the nlopt optimizer
            opt.set_joint_limit(seq.joint_limits)

        # Build articulation joint names mapping
        if explicit_names is not None:
            self._articulation_joint_names: Optional[List[str]] = list(explicit_names)
        elif art_prefix is not None:
            self._articulation_joint_names = [
                f"{art_prefix}{n}".replace(".", "_") for n in self._retargeting.joint_names
            ]
            # Also prefix wrist and tip names for articulation lookup
            self.wrist_link_name = f"{art_prefix}{self.wrist_link_name}".replace(".", "_")
            self.finger_tip_link_names = [f"{art_prefix}{n}".replace(".", "_") for n in self.finger_tip_link_names]
        else:
            self._articulation_joint_names = None

        # Cache joint names early (needed by _parse_urdf_mimic)
        self._joint_names: List[str] = list(self._retargeting.joint_names)
        self._n_dof = len(self._joint_names)

        # Cache fixed_qpos for non-target, non-mimic joints.
        self._fixed_joint_indices = set(int(i) for i in self._retargeting.optimizer.idx_pin2fixed)
        n_fixed = len(self._fixed_joint_indices)
        self._fixed_qpos = np.zeros(n_fixed, dtype=np.float32)

        # ── Parse URDF mimic relationships for manual post-processing ──
        self._mimic_rules: List[Tuple[int, int, float, float]] = []
        adaptor = getattr(self._retargeting.optimizer, 'adaptor', None)
        if adaptor is not None:
            mimic_idx_set = set(int(i) for i in adaptor.idx_pin2mimic)
            self._fixed_joint_indices -= mimic_idx_set
        elif self._fixed_joint_indices:
            self._mimic_rules = self._parse_urdf_mimic(config.urdf_path)
            # Mimic joints are now computed, not fixed at 0 → clear from skip set
            mimic_idx_set = set(r[0] for r in self._mimic_rules)
            self._fixed_joint_indices -= mimic_idx_set

        # Debug: adaptor status
        print(f"[RetargetBridge] {hand_type} ({self._side}) adaptor={type(adaptor).__name__ if adaptor else 'None'}, "
              f"fixed_idx={sorted(self._fixed_joint_indices)}, "
              f"mimic_rules={len(self._mimic_rules)}")

        # ── Joint coupling: PIP/DIP follow MCP for natural bending ──
        self._coupling_rules: List[Tuple[int, List[int], float]] = []
        if coupling_cfg:
            name_to_idx = {n: i for i, n in enumerate(self._joint_names)}
            for rule in coupling_cfg:
                src_name = rule["source"]
                tgt_names = rule["targets"]
                ratio = float(rule.get("ratio", 0.6))
                src_idx = name_to_idx.get(src_name)
                tgt_idxs = [name_to_idx.get(t) for t in tgt_names]
                if src_idx is not None and all(t is not None for t in tgt_idxs):
                    self._coupling_rules.append((src_idx, tgt_idxs, ratio))
            if self._coupling_rules:
                print(f"[RetargetBridge] Joint coupling: {len(self._coupling_rules)} rules")

        # Cache joint limits in full-qpos space for coupling
        if self._coupling_rules:
            robot = self._retargeting.optimizer.robot
            self._coupling_limits = {}
            # Target joints have explicit limits
            for ti, pi in enumerate(self._retargeting.optimizer.idx_pin2target):
                jname = self._joint_names[pi]
                idx = name_to_idx[jname]
                self._coupling_limits[idx] = (
                    float(self._retargeting.joint_limits[ti, 0]),
                    float(self._retargeting.joint_limits[ti, 1]),
                )

        # Cache ref_value indices
        self._indices = self._retargeting.optimizer.target_link_human_indices

        # Operator2mano transform (for Manus coordinate convention)
        self._operator2mano = (
            OPERATOR2MANO_RIGHT if self._side == "right" else OPERATOR2MANO_LEFT
        )
        # MANO → pinocchio rotation correction.
        self._mano_to_pin_R = None
        if mano_to_pin_rpy is not None:
            from scipy.spatial.transform import Rotation as ScipyRot
            R_corr = ScipyRot.from_euler('xyz', mano_to_pin_rpy).as_matrix().astype(np.float32)
            if mano_corr_vis:
                self._operator2mano = (self._operator2mano @ R_corr).astype(np.float32)
                print(f"[RetargetBridge] Applied mano_to_pinocchio_rpy={mano_to_pin_rpy}")
            else:
                self._mano_to_pin_R = R_corr
                print(f"[RetargetBridge] Applied mano_to_pinocchio_rpy={mano_to_pin_rpy} (optimizer-only)")

        # ── Hand offset and per-axis scale ──
        self._hand_offset = np.array(hand_offset, dtype=np.float32) if hand_offset is not None else None
        self._hand_scale = np.array(hand_scale, dtype=np.float32) if hand_scale is not None else None
        if self._hand_offset is not None:
            print(f"[RetargetBridge] hand_offset_xyz={hand_offset}")
        if self._hand_scale is not None:
            print(f"[RetargetBridge] hand_scale_xyz={hand_scale}")

        # ── Wrist quaternion correction (fixed calibration, replaces auto-calibration) ──
        self._quat_correction_R = None
        if quat_corr_rpy is not None:
            from scipy.spatial.transform import Rotation as ScipyRot
            self._quat_correction_R = ScipyRot.from_euler('xyz', quat_corr_rpy).as_matrix().astype(np.float32)
            print(f"[RetargetBridge] wrist_quat_correction_rpy={quat_corr_rpy}")

        # ── Initialize last_qpos to lower bounds (open-hand) ──
        seq = self._retargeting
        seq.last_qpos = seq.joint_limits[:, 0].astype(np.float32)
        if seq.filter is not None:
            seq.filter.reset()
        if hasattr(seq.optimizer, 'projected'):
            seq.optimizer.projected[:] = False

        # Debug visualization: wrist-to-tip vectors (pinocchio/operator frame)
        self.last_human_wrist_to_tips = np.zeros((5, 3), dtype=np.float32)
        self.last_robot_wrist_to_tips = np.zeros((5, 3), dtype=np.float32)
        self.last_manus_keypoints_scaled = np.zeros((21, 3), dtype=np.float32)
        self._vis_diag_count = 0
        self._retarget_count = 0

        # Compute pinocchio wrist rotation at rest pose for coordinate calibration.
        opt = self._retargeting.optimizer
        zero_qpos = np.zeros(opt.robot.dof)
        opt.robot.compute_forward_kinematics(zero_qpos)
        n_fingers = opt.num_fingers
        num_pairs = n_fingers * (n_fingers - 1) // 2
        wrist_ci = opt.origin_link_indices[num_pairs]
        wrist_link_idx = opt.computed_link_indices[wrist_ci]
        wrist_pose = opt.robot.get_link_pose(wrist_link_idx)  # (4, 4) SE3
        pin_root_R = wrist_pose[:3, :3].copy()
        wrist_pos = wrist_pose[:3, 3]

        if sim_wrist_rpy is not None:
            from scipy.spatial.transform import Rotation as ScipyRot
            R_fixed = ScipyRot.from_euler('xyz', sim_wrist_rpy).as_matrix()
            self._pin_wrist_R = R_fixed @ pin_root_R
            print(f"[RetargetBridge] Applied sim_wrist_to_pin_root_rpy={sim_wrist_rpy}")
        else:
            self._pin_wrist_R = pin_root_R

        self._pin_wrist_offset = np.zeros(3, dtype=np.float32)
        side_prefix = 'r_' if self._side == 'right' else 'l_'
        finger_base_candidates = [f'{side_prefix}f_link1_1']
        self._finger_base_offsets = {}
        for cand in finger_base_candidates:
            try:
                cand_frame_id = opt.robot.get_link_index(cand)
                cand_pose = opt.robot.get_link_pose(cand_frame_id)
                offset = (cand_pose[:3, 3] - wrist_pos).astype(np.float32)
                self._finger_base_offsets[cand] = offset
                print(f"[RetargetBridge] finger base offset ({cand}): "
                      f"norm={np.linalg.norm(offset):.4f}m")
            except Exception:
                pass

        print(f"[RetargetBridge] pinocchio wrist rotation at zero pose calibrated")

        print(f"[RetargetBridge] {hand_type} ({self._side}): "
              f"{self._n_dof} DOF, "
              f"{len(self._retargeting.optimizer.target_joint_names)} target joints, "
              f"{n_fixed} fixed joints")

    # ─────────────────────────────────────────────────────────────────────
    # Hot-reload of tunable yaml fields (no optimizer/URDF rebuild).
    # Only re-applies fields that don't change the kinematic structure:
    #   - mano_to_pinocchio_rpy (+ mano_correction_vis)
    #   - hand_offset_xyz / hand_scale_xyz
    #   - wrist_quat_correction_rpy
    # ─────────────────────────────────────────────────────────────────────
    def reload_tunables_if_changed(self) -> bool:
        """Returns True if the YAML mtime advanced and tunables were re-applied."""
        try:
            mt = os.path.getmtime(self._yaml_path)
        except OSError:
            return False
        if mt <= self._yaml_mtime:
            return False
        self._yaml_mtime = mt
        try:
            import yaml
            with open(self._yaml_path, "r") as f:
                raw_cfg = yaml.safe_load(f)
        except Exception as e:  # YAML parse error mid-edit, just skip
            print(f"[RetargetBridge] hot-reload skipped (YAML error): {e}")
            return False
        retarget_sec = raw_cfg.get("retargeting", {})
        side_overrides = retarget_sec.get(self._side, {}) or {}
        merged = {**retarget_sec, **side_overrides}

        from scipy.spatial.transform import Rotation as ScipyRot

        # mano → pin rpy
        mano_to_pin_rpy = merged.get("mano_to_pinocchio_rpy")
        mano_corr_vis = merged.get("mano_correction_vis", True)
        # Reset operator2mano to canonical, then re-apply if needed
        self._operator2mano = (
            OPERATOR2MANO_RIGHT if self._side == "right" else OPERATOR2MANO_LEFT
        )
        self._mano_to_pin_R = None
        if mano_to_pin_rpy is not None:
            R_corr = ScipyRot.from_euler('xyz', mano_to_pin_rpy).as_matrix().astype(np.float32)
            if mano_corr_vis:
                self._operator2mano = (self._operator2mano @ R_corr).astype(np.float32)
            else:
                self._mano_to_pin_R = R_corr

        # hand offset / scale
        ho = merged.get("hand_offset_xyz")
        hs = merged.get("hand_scale_xyz")
        self._hand_offset = np.array(ho, dtype=np.float32) if ho is not None else None
        self._hand_scale = np.array(hs, dtype=np.float32) if hs is not None else None

        # wrist quat correction
        qcr = merged.get("wrist_quat_correction_rpy")
        self._quat_correction_R = (
            ScipyRot.from_euler('xyz', qcr).as_matrix().astype(np.float32)
            if qcr is not None else None
        )

        print(f"[RetargetBridge] HOT-RELOAD ({Path(self._yaml_path).name}): "
              f"mano_to_pin_rpy={mano_to_pin_rpy}, hand_offset={ho}, hand_scale={hs}, "
              f"wrist_quat_corr={qcr}")
        return True

    @property
    def side(self) -> str:
        return self._side

    @property
    def hand_type(self) -> str:
        return self._hand_type

    @property
    def num_joints(self) -> int:
        """Total DOF output (including mimic joints expanded by LP filter)."""
        return self._n_dof

    @property
    def joint_names(self) -> List[str]:
        """All joint names in the retargeted output (pinocchio order)."""
        return self._joint_names

    @property
    def articulation_joint_names(self) -> Optional[List[str]]:
        """Joint names to use for matching against the articulation.

        When the multi-URDF uses prefixed joint names (e.g. ``multi_right_*``,
        ``l_*``), this list contains the articulation names in the same order
        as :attr:`joint_names`.  Returns *None* if not specified in the YAML
        config, in which case :attr:`joint_names` should be used directly.
        """
        return self._articulation_joint_names

    @property
    def target_joint_names(self) -> List[str]:
        """Only the optimized (non-fixed) joint names."""
        return list(self._retargeting.optimizer.target_joint_names)

    @property
    def fixed_joint_indices(self) -> set:
        """Indices in the retarget output that are 'fixed' (always 0).
        
        These correspond to mimic joints when ignore_mimic_joint=true.
        They should NOT be written to the articulation because Isaac Sim
        handles them via USD mimic constraints.
        """
        return self._fixed_joint_indices

    def _parse_urdf_mimic(self, urdf_path: str) -> List[Tuple[int, int, float, float]]:
        """Parse mimic joint relationships from URDF and build computation rules.

        Returns list of (mimic_idx, source_idx, multiplier, offset) tuples,
        topologically sorted so chain-mimic (A→B→C) is computed in correct order.
        """
        import xml.etree.ElementTree as ET
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        joint_names = self._joint_names
        name_to_idx = {n: i for i, n in enumerate(joint_names)}

        # Parse mimic tags
        raw_rules = {}  # {mimic_name: (source_name, mult, offset)}
        for joint in root.findall('.//joint'):
            mimic = joint.find('mimic')
            if mimic is not None:
                name = joint.get('name')
                src = mimic.get('joint')
                mult = float(mimic.get('multiplier', '1.0'))
                off = float(mimic.get('offset', '0.0'))
                if name in name_to_idx and src in name_to_idx:
                    raw_rules[name] = (src, mult, off)

        # Topological sort to handle chain mimic
        sorted_rules: List[Tuple[int, int, float, float]] = []
        resolved = set(name_to_idx.keys()) - set(raw_rules.keys())  # Non-mimic joints start resolved
        remaining = dict(raw_rules)
        max_iter = len(remaining) + 1
        while remaining and max_iter > 0:
            max_iter -= 1
            for name, (src, mult, off) in list(remaining.items()):
                if src in resolved:
                    sorted_rules.append((name_to_idx[name], name_to_idx[src], mult, off))
                    resolved.add(name)
                    del remaining[name]

        if remaining:
            print(f"[RetargetBridge] WARNING: unresolved mimic chains: {list(remaining.keys())}")

        if sorted_rules:
            desc = [(joint_names[mi], joint_names[si], m, o) for mi, si, m, o in sorted_rules]
            print(f"[RetargetBridge] Mimic rules (manual): {desc}")

        return sorted_rules

    def retarget(self, pos_dict: dict, wrist_quat: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Full pipeline: Manus pos_dict → MediaPipe → retarget → target hand qpos.

        Args:
            pos_dict: {node_id: np.array([x,y,z])} — Manus 25 node positions
            wrist_quat: Optional (4,) quaternion [qx,qy,qz,qw] from Manus node 0.

        Returns:
            (n_dof,) np.float32 joint angles (radians), or None if data incomplete
        """
        keypoints = manus_nodes_to_mediapipe(pos_dict)
        if keypoints is None:
            return None

        if self._retargeting is None:
            return None

        # 1) Center on wrist
        keypoints = keypoints - keypoints[0:1, :]

        # 2) Wrist rotation frame (from Manus node 0 quaternion)
        if wrist_quat is None:
            return None
        wrist_frame = _wrist_frame_from_quat(wrist_quat)
        if self._quat_correction_R is not None:
            wrist_frame = wrist_frame @ self._quat_correction_R
        joint_pos = (keypoints @ wrist_frame @ self._operator2mano).astype(np.float32)

        # 2b) Apply per-axis scale and offset
        #     Offset on non-wrist keypoints: negative z → shorter wrist→tip → more curl
        if self._hand_scale is not None:
            joint_pos = joint_pos * self._hand_scale
        if self._hand_offset is not None:
            joint_pos[1:] = joint_pos[1:] + self._hand_offset

        # 3) Compute ref_value: vectors between keypoints as specified by indices
        origin_idx = self._indices[0, :]
        task_idx = self._indices[1, :]
        ref_value = joint_pos[task_idx] - joint_pos[origin_idx]

        # Store human fingertip wrist→tip vectors for debug visualization.
        n_fingers = self._retargeting.optimizer.num_fingers
        num_pairs = n_fingers * (n_fingers - 1) // 2
        scaling = self._retargeting.optimizer.scaling
        self.last_human_wrist_to_tips = ref_value[num_pairs:] * scaling  # (n_fingers, 3)
        self.last_manus_keypoints_scaled = (joint_pos * scaling).astype(np.float32)

        # 4) Multi-start retarget
        seq = self._retargeting
        opt = seq.optimizer
        fixed_qpos = self._fixed_qpos

        # Apply MANO→pinocchio rotation if configured (optimizer-only path)
        if self._mano_to_pin_R is not None:
            ref_value = (ref_value @ self._mano_to_pin_R.T).astype(np.float32)
        ref_f32 = ref_value.astype(np.float32)
        fixed_f32 = fixed_qpos.astype(np.float32)

        # --- Run optimizer: warm-start from last_qpos ---
        last_qpos_clipped = np.clip(
            seq.last_qpos, seq.joint_limits[:, 0], seq.joint_limits[:, 1]
        )
        qpos_warm = opt.retarget(
            ref_value=ref_f32, fixed_qpos=fixed_f32, last_qpos=last_qpos_clipped,
        )
        best_qpos = qpos_warm

        # Update SeqRetargeting state with the chosen solution
        seq.last_qpos = best_qpos
        seq.num_retargeting += 1

        # Build full robot qpos (pinocchio order)
        robot_qpos = np.zeros(opt.robot.dof)
        robot_qpos[opt.idx_pin2fixed] = fixed_qpos
        robot_qpos[opt.idx_pin2target] = best_qpos

        if opt.adaptor is not None:
            robot_qpos = opt.adaptor.forward_qpos(robot_qpos)

        # Compute robot FK BEFORE coupling/LP for diagnostics
        robot_qpos_pre = robot_qpos.copy()

        # 4) Joint coupling: PIP/DIP follow MCP
        if self._coupling_rules:
            for src_idx, tgt_idxs, ratio in self._coupling_rules:
                src_lo = self._coupling_limits[src_idx][0]
                src_hi = self._coupling_limits[src_idx][1]
                src_range = src_hi - src_lo
                if src_range < 1e-6:
                    continue
                src_norm = (robot_qpos[src_idx] - src_lo) / src_range
                src_norm = np.clip(src_norm, 0, 1)

                for tgt_idx in tgt_idxs:
                    tgt_lo = self._coupling_limits[tgt_idx][0]
                    tgt_hi = self._coupling_limits[tgt_idx][1]
                    coupled = tgt_lo + src_norm * ratio * (tgt_hi - tgt_lo)
                    robot_qpos[tgt_idx] = coupled

        if seq.filter is not None:
            robot_qpos = seq.filter.next(robot_qpos)

        qpos = robot_qpos

        # 5) Compute mimic joints manually (when adaptor is None / ignore_mimic_joint=true)
        for mimic_idx, source_idx, mult, off in self._mimic_rules:
            qpos[mimic_idx] = qpos[source_idx] * mult + off

        # 6) Compute robot FK wrist→tip vectors (same pinocchio frame as human targets).
        #    Only every N frames to avoid performance impact.
        self._vis_diag_count += 1
        if self._vis_diag_count % 10 == 0:
            try:
                opt.robot.compute_forward_kinematics(qpos)
                poses = [opt.robot.get_link_pose(idx) for idx in opt.computed_link_indices]
                body_pos = np.array([p[:3, 3] for p in poses])
                robot_tips = np.empty((n_fingers, 3), dtype=np.float32)
                for fi in range(n_fingers):
                    vi = num_pairs + fi
                    oi = int(opt.origin_link_indices[vi])
                    ti = int(opt.task_link_indices[vi])
                    robot_tips[fi] = body_pos[ti] - body_pos[oi]
                self.last_robot_wrist_to_tips = robot_tips
            except RuntimeError:
                pass  # keep previous value

        self._retarget_count += 1

        return qpos.astype(np.float32)

    def reset(self):
        """Reset internal filter state and optimizer warm-start."""
        self._retargeting.reset()
        # Override midpoint reset to use lower bounds (open hand)
        seq = self._retargeting
        seq.last_qpos = seq.joint_limits[:, 0].astype(np.float32)
        # Clear projection flags to avoid persistent coupling
        if hasattr(seq.optimizer, 'projected'):
            seq.optimizer.projected[:] = False

    def __repr__(self) -> str:
        return (
            f"RetargetBridge(hand_type={self._hand_type!r}, "
            f"side={self._side!r}, dof={self._n_dof})"
        )
