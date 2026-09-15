"""Spawner for the RM65 + BrainCo Revo2 dual-arm robot."""

from pathlib import Path
from typing import Dict, Tuple

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg

from build.geometry import rpy_deg_to_quat
from .placement import robot_back_edge_y
from .hand_friction import set_dexterous_hand_friction
from .gravity_compensation import GravityCompensator

ROBOT_USD_PATH = Path("../dex2bench_dataset/Robots_p/rm_65+BrainCo/usd/muitl_rm_65_with_revo2_.usd")
ROBOT_PRIM_PATH = "/World/Objects/GlobalRobot"

BACK_EDGE_MARGIN = 0.2
BASE_X_OFFSET = 0.25
BASE_Z_OFFSET = 0.0
ROBOT_RPY_DEG = (0.0, 0.0, 90.0)

BASE_EXCLUDE_HALF_X = 0.90
BASE_EXCLUDE_FRONT_EXTENT = 0.30

HOME_JOINT_POS = {
    "joint1": 1.57,
    "joint2": 1.57,
    "joint3": -1.57,
    "joint4": 0.0,
    "joint5": 1.57,
    "joint6": 0.0,
    "l_joint1": -1.57,
    "l_joint2": -1.57,
    "l_joint3": 1.57,
    "l_joint4": 0.0,
    "l_joint5": -1.57,
    "l_joint6": 0.0,
    "right_thumb_.*_joint": 0.0,
    "right_(index|middle|ring|pinky)_(proximal|distal)_joint": 0.0,
    "left_thumb_.*_joint": 0.0,
    "left_(index|middle|ring|pinky)_(proximal|distal)_joint": 0.0,
}


def spawn_multi_rm_65_with_revo2(
    table_size: Tuple[float, float, float],
    table_height: float,
    task_dir: str,
) -> Dict:
    """Spawn the RM65 + Revo2 robot and return placement exclusion metadata."""

    robot_usd_path = ROBOT_USD_PATH.resolve()
    if not robot_usd_path.exists():
        raise FileNotFoundError(f"Robot USD not found: {robot_usd_path.as_posix()}")

    sx, sy, _sz = table_size
    robot_back_y = robot_back_edge_y(table_size)
    robot_pos = (BASE_X_OFFSET, robot_back_y + BACK_EDGE_MARGIN, table_height + BASE_Z_OFFSET)
    robot_quat = rpy_deg_to_quat(ROBOT_RPY_DEG)

    usd_cfg = sim_utils.UsdFileCfg(
        usd_path=robot_usd_path.as_posix(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            articulation_enabled=True,
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            fix_root_link=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        activate_contact_sensors=False,
    )

    robot_cfg = ArticulationCfg(
        prim_path=ROBOT_PRIM_PATH,
        spawn=usd_cfg,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=robot_pos,
            rot=robot_quat,
            joint_pos=HOME_JOINT_POS,
            joint_vel={".*": 0.0},
        ),
        actuators={
            "right_rm65_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-2]"],
                effort_limit_sim=500.0,
                velocity_limit_sim=1.0,
                stiffness=4000.0,
                damping=120.0,
                friction=0.0,
                armature=0.000005,
            ),
            "right_rm65_elbow_wrist": ImplicitActuatorCfg(
                joint_names_expr=["joint[3-6]"],
                effort_limit_sim=300.0,
                velocity_limit_sim=1.0,
                stiffness=2000.0,
                damping=80.0,
                friction=0.0,
                armature=0.000005,
            ),
            "left_rm65_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["l_joint[1-2]"],
                effort_limit_sim=500.0,
                velocity_limit_sim=1.0,
                stiffness=4000.0,
                damping=120.0,
                friction=0.0,
                armature=0.000005,
            ),
            "left_rm65_elbow_wrist": ImplicitActuatorCfg(
                joint_names_expr=["l_joint[3-6]"],
                effort_limit_sim=300.0,
                velocity_limit_sim=1.0,
                stiffness=2000.0,
                damping=80.0,
                friction=0.0,
                armature=0.000005,
            ),
            "right_revo2": ImplicitActuatorCfg(
                joint_names_expr=[
                    "right_thumb_.*_joint",
                    "right_(index|middle|ring|pinky)_(proximal|distal)_joint",
                ],
                effort_limit_sim=10.0,
                velocity_limit_sim=4.0,
                stiffness=80.0,
                damping=4.0,
                friction=0.0,
                armature=0.00001,
            ),
            "left_revo2": ImplicitActuatorCfg(
                joint_names_expr=[
                    "left_thumb_.*_joint",
                    "left_(index|middle|ring|pinky)_(proximal|distal)_joint",
                ],
                effort_limit_sim=10.0,
                velocity_limit_sim=4.0,
                stiffness=80.0,
                damping=4.0,
                friction=0.0,
                armature=0.00001,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )
    robot = Articulation(cfg=robot_cfg)

    set_dexterous_hand_friction(
        ROBOT_PRIM_PATH,
        static_friction=2.0,
        dynamic_friction=2.0,
        log_prefix="spawn_rm65_revo2",
    )

    exclusion = [
        [
            -BASE_EXCLUDE_HALF_X,
            robot_back_y,
            table_height,
        ],
        [
            BASE_EXCLUDE_HALF_X,
            robot_pos[1] + BASE_EXCLUDE_FRONT_EXTENT,
            table_height + 0.20,
        ],
    ]

    return {
        "interactive_objects": {"global_robot": robot},
        "pre_step_hooks": [GravityCompensator(robot).apply],
        "exclude_aabbs": [exclusion],
        "robot_pose": {
            "prim_path": ROBOT_PRIM_PATH,
            "pos": robot_pos,
            "rpy_deg": ROBOT_RPY_DEG,
            "table_size": (sx, sy),
            "usd_path": robot_usd_path.as_posix(),
        },
    }
