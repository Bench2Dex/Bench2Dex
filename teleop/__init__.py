"""
dex2bench.teleop — Manus glove teleoperation via POSIX shared memory.

Data flow:
  ManusSDK C++ (ShmWriter) → /dev/shm/manus_hand_data
  → ShmReader (Python) → RetargetBridge (Manus 25→MediaPipe 21→dex_retargeting DexPilot→target hand N DOF)
  → TeleopController (auto-detect hands, output joint targets)
  → TeleopIsaacLabBridge (map to articulation joint indices, arm IK)

Supports all 11 hand types in dex2bench via dex_retargeting YAML configs.
"""

from .shm_reader import ShmReader
from .retarget_bridge import RetargetBridge
from .teleop_controller import TeleopController
from .isaaclab_bridge import TeleopIsaacLabBridge
from .arkit_wrist_reader import ARKitWristReader
from .arm_ik_controller import ArmIKController

__all__ = [
    "ShmReader",
    "RetargetBridge",
    "TeleopController",
    "TeleopIsaacLabBridge",
    "ARKitWristReader",
    "ArmIKController",
]
