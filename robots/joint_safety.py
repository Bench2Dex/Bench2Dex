"""Asset-backed joint roles for safety evaluation.

The role of a joint is part of the robot model, not a naming convention.  The
registry below is deliberately derived from ``active_dof_maps.yml``: that file
is generated from the combined URDFs and checked against the shipped USD
articulations.  It keeps evaluation code independent of Isaac while ensuring
that a name such as ``joint_1`` is interpreted in the context of its robot.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal


JointRole = Literal["arm", "hand_wrist", "finger", "finger_mimic", "unknown"]

_MAP_PATH = Path(__file__).with_name("active_dof_maps.yml")
_ARM_GROUP_MARKERS = ("arm", "panda", "xarm", "iiwa", "rm65", "rm75")
_HAND_GROUP_MARKERS = (
    "hand", "allegro", "orca", "leap", "ability", "sharpa", "dexhand",
    "revo", "wuji", "rh5", "rh56", "shadow",
)


def _group_role(group_name: str, joint_name: str, mimic_names: set[str]) -> JointRole:
    name = group_name.lower()
    if any(marker in name for marker in _ARM_GROUP_MARKERS):
        return "arm"
    if any(marker in name for marker in _HAND_GROUP_MARKERS):
        if joint_name in mimic_names:
            return "finger_mimic"
        # A wrist belonging to an articulated hand is not an arm wrist.  It
        # gets its own role because it has different range/velocity semantics
        # from both finger joints and the seven-DoF arm.
        if "wrist" in name or joint_name.lower().endswith("_wrist") or joint_name.lower().endswith("_wrj1") or joint_name.lower().endswith("_wrj2"):
            return "hand_wrist"
        return "finger"
    return "unknown"


@lru_cache(maxsize=1)
def _roles_by_robot() -> dict[str, dict[str, JointRole]]:
    import yaml

    data = yaml.safe_load(_MAP_PATH.read_text()) or {}
    output: dict[str, dict[str, JointRole]] = {}
    for robot_key, cfg in (data.get("robots") or {}).items():
        mimic_names = set(str(name) for name in cfg.get("mimic_joint_names") or [])
        roles: dict[str, JointRole] = {
            str(name): "unknown" for name in cfg.get("full_joint_names") or []
        }
        for group_name, group in (cfg.get("groups") or {}).items():
            for joint_name in group.get("matched_joint_names") or []:
                joint_name = str(joint_name)
                role = _group_role(str(group_name), joint_name, mimic_names)
                # An explicit arm actuator must override a broad hand marker.
                if role == "arm" or roles.get(joint_name, "unknown") == "unknown":
                    roles[joint_name] = role
        output[str(robot_key)] = roles
    return output


def joint_roles(robot_key: str | None) -> dict[str, JointRole] | None:
    """Return the exact URDF/USD-backed role map for *robot_key*.

    ``None`` means no manifest exists; callers must then report legacy
    name-based classification rather than silently claim asset validation.
    """
    if not robot_key:
        return None
    return _roles_by_robot().get(str(robot_key))


def role_for_joint(robot_key: str | None, joint_name: str) -> JointRole:
    roles = joint_roles(robot_key)
    if roles is None:
        return "unknown"
    return roles.get(str(joint_name), "unknown")


def validate_runtime_joint_names(robot_key: str | None, joint_names: list[str]) -> tuple[bool, list[str]]:
    """Check that the runtime articulation exactly matches its asset manifest."""
    roles = joint_roles(robot_key)
    if roles is None:
        return False, list(joint_names)
    unknown = sorted(set(map(str, joint_names)) - set(roles))
    missing = sorted(set(roles) - set(map(str, joint_names)))
    return not unknown and not missing and len(joint_names) == len(roles), unknown + missing
