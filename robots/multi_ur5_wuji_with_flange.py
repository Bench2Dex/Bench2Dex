"""Spawner for the Multi_UR5_wuji dual-arm robot."""

from pathlib import Path
from typing import Dict, Tuple

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg

from build.geometry import rpy_deg_to_quat
from .gravity_compensation import GravityCompensator
from .hand_friction import set_dexterous_hand_friction
from .placement import robot_back_edge_y

ROBOT_USD_PATH = Path("../dex2bench_dataset/Robots_p/ur5+wuji/usd/Multi_UR5_wuji_with_flange.usd")
ROBOT_PRIM_PATH = "/World/Objects/GlobalRobot"

HAND_FRICTION_STATIC = 2.0
HAND_FRICTION_DYNAMIC = 2.0
HAND_FRICTION_LINK_KEYWORDS = ("finger", "palm")

# Pose policy: place robot at table back-center and face inward.
BACK_EDGE_MARGIN = 0.12
BASE_Z_OFFSET = 0.0
ROBOT_RPY_DEG = (0.0, 0.0, 90.0)

# Placement exclusion region on tabletop (keep random objects off robot base area).
BASE_EXCLUDE_HALF_X = 0.85
BASE_EXCLUDE_FRONT_EXTENT = 0.33

# Reuse the mirrored UR5 arm pose. Wuji hands need a legal neutral pose: finger1_joint1
# must start above its positive lower bound, while the remaining joints can stay at 0.
HOME_JOINT_POS = {
# right arm
    "shoulder_pan_joint": 1.57,
    "shoulder_lift_joint": -1.57,
    "elbow_joint": 1.57,
    "wrist_1_joint": 0.0,
    "wrist_2_joint": 1.57,
    "wrist_3_joint": 0.0,

    # left arm
    "L_arm_shoulder_pan_joint": -1.57,
    "L_arm_shoulder_lift_joint": 1.57,
    "L_arm_elbow_joint": -1.57,
    "L_arm_wrist_1_joint": 0.0,
    "L_arm_wrist_2_joint": -1.57,
    "L_arm_wrist_3_joint": 0.0,
    
    "right_finger1_joint1": 0.10,
    "left_finger1_joint1": 0.10,
    "right_finger[1-5]_joint[2-4]": 0.0,
    "right_finger[2-5]_joint1": 0.0,
    "left_finger[1-5]_joint[2-4]": 0.0,
    "left_finger[2-5]_joint1": 0.0,
}



def spawn_multi_ur5_wuji_with_flange(
    table_size: Tuple[float, float, float],
    table_height: float,
    task_dir: str,
) -> Dict:
    """Spawn the dual-arm UR5 + Wuji robot and return placement exclusion metadata."""

    robot_usd_path = ROBOT_USD_PATH.resolve()
    if not robot_usd_path.exists():
        raise FileNotFoundError(f"Robot USD not found: {robot_usd_path.as_posix()}")

    sx, sy, _sz = table_size
    robot_back_y = robot_back_edge_y(table_size)
    robot_pos = (0.5, robot_back_y + BACK_EDGE_MARGIN, table_height + BASE_Z_OFFSET)
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
            "arm_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_.*", "L_arm_shoulder_.*"],
                effort_limit_sim=500.0,
                velocity_limit_sim=1.0,
                stiffness=4000.0,
                damping=120.0,
                friction=0.0,
                armature=0.0,
            ),
            "arm_elbow": ImplicitActuatorCfg(
                joint_names_expr=["elbow_joint", "L_arm_elbow_joint"],
                effort_limit_sim=500.0,
                velocity_limit_sim=1.0,
                stiffness=2000.0,
                damping=80.0,
                friction=0.0,
                armature=0.0,
            ),
            "arm_wrist": ImplicitActuatorCfg(
                joint_names_expr=["wrist_[1-3]_joint", "L_arm_wrist_[1-3]_joint"],
                effort_limit_sim=100.0,
                velocity_limit_sim=1.0,
                stiffness=800.0,
                damping=50.0,
                friction=0.0,
                armature=0.0,
            ),
            "right_hand": ImplicitActuatorCfg(
                joint_names_expr=["right_finger[1-5]_joint[1-4]"],
                effort_limit_sim=800.0,
                velocity_limit_sim=4.0,
                stiffness=400.0,
                damping=20.0,
                friction=0.0,
                armature=0.000005,
            ),
            "left_hand": ImplicitActuatorCfg(
                joint_names_expr=["left_finger[1-5]_joint[1-4]"],
                effort_limit_sim=800.0,
                velocity_limit_sim=4.0,
                stiffness=400.0,
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
        static_friction=HAND_FRICTION_STATIC,
        dynamic_friction=HAND_FRICTION_DYNAMIC,
        hand_link_keywords=HAND_FRICTION_LINK_KEYWORDS,
        log_prefix="spawn_wuji",
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
