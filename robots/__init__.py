"""Robot spawner registry for dex2scene.

Keep this module importable in non-Isaac training environments.  Spawner
modules import IsaacLab, so they are loaded lazily only when a robot is spawned.
"""

from importlib import import_module
from typing import Callable, Dict, Tuple

DEFAULT_ROBOT_KEY = "multi_ur5_wuji_with_flange"

RobotSpawner = Callable[[Tuple[float, float, float], float, str], Dict]

ROBOT_SPAWNERS: Dict[str, tuple[str, str]] = {
    "multi_iiwa7_with_sharpa": (
        "multi_iiwa7_with_sharpa",
        "spawn_multi_iiwa7_with_sharpa",
    ),
    "multi_jaka_zu7_dexhand021_with_flange": (
        "multi_jaka_zu7_dexhand021_with_flange",
        "spawn_multi_jaka_zu7_dexhand021_with_flange",
    ),
    "multi_panda_with_orca": (
        "multi_panda_with_orca",
        "spawn_multi_panda_with_orca",
    ),
    "multi_panda_with_allegro": (
        "multi_panda_with_allegro",
        "spawn_multi_panda_with_allegro",
    ),
    "multi_rm_65_with_revo2": (
        "multi_rm_65_with_revo2",
        "spawn_multi_rm_65_with_revo2",
    ),
    "multi_xarm7_with_ability": (
        "multi_xarm7_with_ability",
        "spawn_multi_xarm7_with_ability",
    ),
    "multi_xarm7_with_leap": (
        "multi_xarm7_with_leap",
        "spawn_multi_xarm7_with_leap",
    ),
    "multi_ur5_rh5dg2_with_flange": (
        "multi_ur5_rh5dg2_with_flange",
        "spawn_multi_ur5_rh5dg2_with_flange",
    ),
    "multi_ur5_rh56dfx_with_flange": (
        "multi_ur5_rh56dfx_with_flange",
        "spawn_multi_ur5_rh56dfx_with_flange",
    ),
    "multi_ur5_shadow_hand_with_flange": (
        "multi_ur5_shadow_hand_with_flange",
        "spawn_multi_ur5_shadow_hand_with_flange",
    ),
    "multi_ur5_schunk_hand_with_flange": (
        "multi_ur5_schunk_hand_with_flange",
        "spawn_multi_ur5_schunk_hand_with_flange",
    ),
    "multi_ur5_wuji_with_flange": (
        "multi_ur5_wuji_with_flange",
        "spawn_multi_ur5_wuji_with_flange",
    ),
}


def get_robot_keys() -> tuple[str, ...]:
    """Return stable robot registry keys for selection and validation."""
    return tuple(sorted(ROBOT_SPAWNERS.keys()))


def _load_spawner(robot_key: str) -> RobotSpawner:
    module_name, func_name = ROBOT_SPAWNERS[robot_key]
    module = import_module(f"{__name__}.{module_name}")
    return getattr(module, func_name)


def spawn_robot_by_key(robot_key: str, table_size: Tuple[float, float, float], table_height: float, task_dir: str) -> Dict:
    """Spawn robot by registry key and return runtime metadata.

    Return value may include:
      - interactive_objects: dict[str, object]
      - exclude_aabbs: list[[[x0,y0,z0],[x1,y1,z1]]]
      - robot_pose: dict metadata
    """
    if robot_key not in ROBOT_SPAWNERS:
        known = ", ".join(sorted(ROBOT_SPAWNERS.keys()))
        raise ValueError(f"Unknown robot key '{robot_key}'. Known keys: {known}")
    spawner = _load_spawner(robot_key)
    return spawner(table_size=table_size, table_height=table_height, task_dir=task_dir)
