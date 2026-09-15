"""Spawner for the Panda + Orca dual-arm robot."""

from pathlib import Path
from typing import Dict, Tuple

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg

from build.geometry import rpy_deg_to_quat
from .placement import robot_back_edge_y
from .hand_friction import set_dexterous_hand_friction
from .gravity_compensation import GravityCompensator

ROBOT_USD_PATH = Path("../dex2bench_dataset/Robots_p/panda+orca/usd/multi_panda_with_orca.usd")
ROBOT_PRIM_PATH = "/World/Objects/GlobalRobot"

# The root Panda base sits slightly to +x so the second Panda, offset by the fixed joint,
# lands on the opposite side of the table centerline.
BACK_EDGE_MARGIN = 0.12
BASE_X_OFFSET = -0.35
BASE_Z_OFFSET = 0.0
ROBOT_RPY_DEG = (0.0, 0.0, 90.0)

# Exclude the rear strip occupied by the two Panda bases and their folded arm envelope.
BASE_EXCLUDE_HALF_X = 1.10
BASE_EXCLUDE_FRONT_EXTENT = 0.30

# Symmetric outward-folding pose to bias both hands away from the tabletop interior.
HOME_JOINT_POS = {
    "panda_joint1": -1.57,
    "panda_joint2": -1.57,
    "panda_joint3": 0.00,
    "panda_joint4": -1.57,
    "panda_joint5": 1.57,
    "panda_joint6": 6.25,
    "panda_joint7": -1.57,
    "multi_panda_joint1": 1.57,
    "multi_panda_joint2": -1.57,
    "multi_panda_joint3": 0.00,
    "multi_panda_joint4": -1.57,
    "multi_panda_joint5": 1.57,
    "multi_panda_joint6": 6.25,
    "multi_panda_joint7": -1.57,
    "left_wrist": 0.0,
    "left_thumb_(mcp|abd|pip|dip)": 0.0,
    "left_(index|middle|ring|pinky)_(abd|mcp|pip)": 0.0,
    "right_wrist": 0.0,
    "right_thumb_(mcp|abd|pip|dip)": 0.0,
    "right_(index|middle|ring|pinky)_(abd|mcp|pip)": 0.0,
}



def spawn_multi_panda_with_orca(
    table_size: Tuple[float, float, float],
    table_height: float,
    task_dir: str,
) -> Dict:
    """Spawn the panda+orca dual-arm robot and return placement exclusion metadata."""

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
            "left_panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit_sim=500.0,
                velocity_limit_sim=1.0,
                stiffness=4000.0,
                damping=120.0,
                friction=0.0,
                armature=0.0,
            ),
            "left_panda_wrist": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit_sim=100.0,
                velocity_limit_sim=1.0,
                stiffness=800.0,
                damping=50.0,
                friction=0.0,
                armature=0.0,
            ),
            "right_panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["multi_panda_joint[1-4]"],
                effort_limit_sim=500.0,
                velocity_limit_sim=1.0,
                stiffness=4000.0,
                damping=120.0,
                friction=0.0,
                armature=0.0,
            ),
            "right_panda_wrist": ImplicitActuatorCfg(
                joint_names_expr=["multi_panda_joint[5-7]"],
                effort_limit_sim=100.0,
                velocity_limit_sim=1.0,
                stiffness=800.0,
                damping=50.0,
                friction=0.0,
                armature=0.0,
            ),
            "left_orca": ImplicitActuatorCfg(
                joint_names_expr=[
                    "left_wrist",
                    "left_thumb_(mcp|abd|pip|dip)",
                    "left_(index|middle|ring|pinky)_(abd|mcp|pip)",
                ],
                effort_limit_sim=200.0,
                velocity_limit_sim=4.0,
                stiffness=800.0,
                damping=20.0,
                friction=0.0,
                armature=0.000005,
            ),
            "right_orca": ImplicitActuatorCfg(
                joint_names_expr=[
                    "right_wrist",
                    "right_thumb_(mcp|abd|pip|dip)",
                    "right_(index|middle|ring|pinky)_(abd|mcp|pip)",
                ],
                effort_limit_sim=200.0,
                velocity_limit_sim=4.0,
                stiffness=800.0,
                damping=20.0,
                friction=0.0,
                armature=0.000005,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )
    robot = Articulation(cfg=robot_cfg)

    set_dexterous_hand_friction(
        ROBOT_PRIM_PATH,
        static_friction=2.0,
        dynamic_friction=2.0,
        log_prefix="spawn_panda_orca",
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
