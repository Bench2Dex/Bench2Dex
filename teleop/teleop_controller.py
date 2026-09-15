"""
teleop_controller.py — Top-level teleoperation controller.

Auto-detects left/right hand from shared memory, runs retarget pipeline
via dex_retargeting (DexPilot), outputs joint targets for the configured
target hand type *directly* — no intermediate Wuji 20DOF representation.

Design decisions:
  - Single hand default: right hand has priority
  - Dual hand: auto-detect from SHM data
  - No redundant config params — hand type string is all you need
  - Stateless per-frame: call step() in your loop
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .shm_reader import ShmReader, ShmFrame, HandFrame, SIDE_LEFT, SIDE_RIGHT
from .retarget_bridge import RetargetBridge, _HAND_TYPE_TO_CONFIG


class TeleopController:
    """
    Teleoperation controller: Manus glove → target hand joint angles.

    Usage (single hand, simplest):
        ctrl = TeleopController(hand_type="rh56dfx")
        ctrl.start()
        while running:
            result = ctrl.step()
            if result:
                joint_targets = result["right"]  # (12,) for RH56DFX
        ctrl.stop()

    Usage (dual hand):
        ctrl = TeleopController(hand_type="shadow", enable_left=True, enable_right=True)
        ctrl.start()
        result = ctrl.step()
        if result:
            right_q = result.get("right")  # (24,) or None
            left_q  = result.get("left")   # (24,) or None
    """

    def __init__(
        self,
        hand_type: str = "rh56dfx",
        enable_left: bool = False,
        enable_right: bool = True,
        shm_name: str = "/manus_hand_data",
        retarget_yaml_right: Optional[str] = None,
        retarget_yaml_left: Optional[str] = None,
    ):
        """
        Args:
            hand_type: Target hand type (key in HAND_CONFIGS).
            enable_left: Enable left hand tracking.
            enable_right: Enable right hand tracking (default True).
            shm_name: POSIX shared memory name.
            retarget_yaml_right: Custom dex_retargeting YAML config for right hand.
            retarget_yaml_left: Custom dex_retargeting YAML config for left hand.
        """
        if hand_type not in _HAND_TYPE_TO_CONFIG:
            raise ValueError(
                f"Unknown hand type '{hand_type}'. "
                f"Available: {list(_HAND_TYPE_TO_CONFIG.keys())}"
            )

        self._hand_type = hand_type
        self._enable_left = enable_left
        self._enable_right = enable_right

        # Shared memory reader
        self._reader = ShmReader(shm_name)

        # Retarget bridges (created lazily on start)
        self._bridge_right: Optional[RetargetBridge] = None
        self._bridge_left: Optional[RetargetBridge] = None
        self._yaml_right = retarget_yaml_right
        self._yaml_left = retarget_yaml_left

        # State
        self._started = False
        self._frame_count = 0
        self._last_frame: Optional[ShmFrame] = None

    @property
    def hand_type(self) -> str:
        return self._hand_type

    @property
    def dof(self) -> int:
        """Number of output DOFs. Available after start()."""
        if self._bridge_right is not None:
            return self._bridge_right.num_joints
        if self._bridge_left is not None:
            return self._bridge_left.num_joints
        return 0

    @property
    def is_started(self) -> bool:
        return self._started

    def start(self, wait_timeout: float = 5.0) -> bool:
        """
        Initialize shared memory reader and retarget bridges.

        Args:
            wait_timeout: Seconds to wait for SHM data. 0 = don't wait.

        Returns:
            True if SHM was opened (data may not be available yet).
        """
        if self._started:
            return True

        # Open shared memory (ManusSDK must be started manually in another terminal)
        if not self._reader.open():
            print(f"[TeleopController] WARNING: Could not open shared memory. ")
            print(f"[TeleopController] Please start ManusSDK manually:")
            print(f"[TeleopController]   cd teleop/manus_bin && ./SDKClient_Linux.out")
            return False

        # Create retarget bridges (dex_retargeting DexPilot)
        if self._enable_right:
            self._bridge_right = RetargetBridge(
                hand_type=self._hand_type,
                hand_side="right",
                yaml_path=self._yaml_right,
            )

        if self._enable_left:
            self._bridge_left = RetargetBridge(
                hand_type=self._hand_type,
                hand_side="left",
                yaml_path=self._yaml_left,
            )

        self._started = True

        # Optionally wait for data
        if wait_timeout > 0:
            print(f"[TeleopController] Waiting for Manus data (timeout={wait_timeout}s)...")
            if self._reader.wait_for_data(wait_timeout):
                print("[TeleopController] Manus data received!")
            else:
                print("[TeleopController] WARNING: No data received within timeout. "
                      "Will continue anyway.")

        print(f"[TeleopController] Started: hand_type={self._hand_type}, "
              f"dof={self.dof}, "
              f"right={'ON' if self._enable_right else 'OFF'}, "
              f"left={'ON' if self._enable_left else 'OFF'}")

        return True

    def stop(self):
        """Clean up resources."""
        self._reader.close()
        self._started = False
        self._bridge_right = None
        self._bridge_left = None

    def get_wrist_quaternion(self, side: str = "right") -> Optional[np.ndarray]:
        """获取 Manus SHM 中手腕朝向四元数 (qx,qy,qz,qw)。

        node 0 = 手腕节点, 位置始终为 (0,0,0) 但朝向有效。
        """
        if self._last_frame is None:
            return None
        hand = self._last_frame.right if side == "right" else self._last_frame.left
        if not hand.valid or hand.node_count == 0:
            return None
        return hand.quaternions[0].copy()  # (4,) = (qx, qy, qz, qw)

    def get_hand_frame(self, side: str = "right"):
        """获取最近一帧的原始 HandFrame 数据 (含 25 节点位置 + 朝向)。

        Returns:
            HandFrame or None (可通过 .positions, .node_count 访问)
        """
        if self._last_frame is None:
            return None
        hand = self._last_frame.right if side == "right" else self._last_frame.left
        return hand if hand.valid else None

    def step(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Read one frame from SHM, retarget, and map to target hand.

        Returns:
            Dict with keys "right" and/or "left", each mapping to
            (dof,) np.float32 joint angles. None if no new data.
        """
        if not self._started:
            return None

        frame = self._reader.read_if_new()
        if frame is None:
            return None

        self._last_frame = frame
        self._frame_count += 1
        result: Dict[str, np.ndarray] = {}

        # Process right hand
        if self._enable_right and frame.right.valid and self._bridge_right is not None:
            pos_dict = self._reader.get_pos_dict(frame.right)
            wrist_quat = frame.right.quaternions[0]  # node 0 = wrist
            target_q = self._bridge_right.retarget(pos_dict, wrist_quat=wrist_quat)
            if target_q is not None:
                result["right"] = target_q

        # Process left hand
        if self._enable_left and frame.left.valid and self._bridge_left is not None:
            pos_dict = self._reader.get_pos_dict(frame.left)
            wrist_quat = frame.left.quaternions[0]
            target_q = self._bridge_left.retarget(pos_dict, wrist_quat=wrist_quat)
            if target_q is not None:
                result["left"] = target_q

        return result if result else None

    def get_joint_names(self, side: str = "right") -> Optional[List[str]]:
        """
        Return the joint names output by the retarget bridge (pinocchio order).

        Available after start().
        """
        bridge = self._bridge_right if side == "right" else self._bridge_left
        if bridge is None:
            return None
        return bridge.joint_names

    def get_frame_info(self) -> Optional[dict]:
        """Get info about the last read frame (for debugging)."""
        if self._last_frame is None:
            return None
        f = self._last_frame
        return {
            "seq": f.seq,
            "timestamp_us": f.timestamp_us,
            "hand_count": f.hand_count,
            "right_valid": f.right.valid,
            "left_valid": f.left.valid,
            "frame_count": self._frame_count,
        }

    def reset_filters(self):
        """Reset retarget IK filters (call after discontinuity)."""
        if self._bridge_right:
            self._bridge_right.reset()
        if self._bridge_left:
            self._bridge_left.reset()

    @staticmethod
    def list_hand_types() -> List[str]:
        """Return all available hand type names."""
        return list(_HAND_TYPE_TO_CONFIG.keys())

    @staticmethod
    def auto_detect_hand_type(robot_name: str) -> Optional[str]:
        """
        Heuristic to detect hand type from dex2bench robot/scene name.

        Args:
            robot_name: Scene or robot config name, e.g.
                        "ur5_rh56dfx", "panda_allegro", "scene_001_ur5_rh56dfx"

        Returns:
            Hand type string or None if not detected.
        """
        lower = robot_name.lower()
        for hand_key in _HAND_TYPE_TO_CONFIG:
            if hand_key in lower:
                return hand_key
        # Fallback heuristics
        if "schunk" in lower:
            return "schunk_svh"
        if "shadow" in lower:
            return "shadow"
        if "svh" in lower:
            return "schunk_svh"
        if "inspire" in lower or "rh56" in lower:
            return "rh56dfx"
        if "rh5d" in lower:
            return "rh5dg2"
        return None

    def __repr__(self) -> str:
        return (
            f"TeleopController(hand_type={self._hand_type!r}, dof={self.dof}, "
            f"right={'ON' if self._enable_right else 'OFF'}, "
            f"left={'ON' if self._enable_left else 'OFF'}, "
            f"started={self._started})"
        )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def __del__(self):
        self.stop()
