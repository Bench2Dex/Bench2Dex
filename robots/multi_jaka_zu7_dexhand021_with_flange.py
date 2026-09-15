"""Spawner for the JAKA ZU7 + DexHand021 dual-arm robot."""

from pathlib import Path
from typing import Dict, Tuple

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg

from build.geometry import rpy_deg_to_quat
from .placement import robot_back_edge_y
from .hand_friction import set_dexterous_hand_friction
from .gravity_compensation import GravityCompensator

ROBOT_USD_PATH = Path("../dex2bench_dataset/Robots_p/jaka_zu7+dexhand021/usd/Multi_jaka_zu7_dexhand021_with_flange_with_flange.usd")
ROBOT_PRIM_PATH = "/World/Objects/GlobalRobot"

# Place the two-arm assembly along the back edge and keep the inter-arm span centered on the table.
BACK_EDGE_MARGIN = 0.12
BASE_X_OFFSET = 0.65
BASE_Z_OFFSET = 0.0
ROBOT_RPY_DEG = (0.0, 0.0, 90.0)

# Keep random tabletop placement away from the robot base and folded arm envelope.
BASE_EXCLUDE_HALF_X = 1.15
BASE_EXCLUDE_FRONT_EXTENT = 0.28

# Keep both arms folded away from the tabletop interior on reset.
HOME_JOINT_POS = {
    # right arm
    "joint_1": 1.57,
    "joint_2": 1.57,
    "joint_3": -1.57,
    "joint_4": 0.00,
    "joint_5": 1.57,
    "joint_6": 0.00,

    # left arm
    "l_joint_1": -1.57,
    "l_joint_2": -1.57,
    "l_joint_3": 1.57,
    "l_joint_4": 0.00,
    "l_joint_5": -1.57,
    "l_joint_6": 0.00,

    # right hand
    "r_f_joint[1-5]_[1-4]": 0.0,

    # left hand
    "l_f_joint[1-5]_[1-4]": 0.0,
}



def spawn_multi_jaka_zu7_dexhand021_with_flange(
    table_size: Tuple[float, float, float],
    table_height: float,
    task_dir: str,
) -> Dict:
    """Spawn the JAKA + DexHand021 robot and return placement exclusion metadata."""

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
            "right_arm_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["joint_[1-2]"],
                effort_limit_sim=400.0,
                velocity_limit_sim=1.0,
                stiffness=1000.0,
                damping=30.0,
                friction=0.0,
                armature=0.0,
            ),
            "right_arm_elbow_wrist": ImplicitActuatorCfg(
                joint_names_expr=["joint_[3-6]"],
                effort_limit_sim=300.0,
                velocity_limit_sim=1.0,
                stiffness=500.0,
                damping=40.0,
                friction=0.0,
                armature=0.0,
            ),
            "left_arm_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["l_joint_[1-2]"],
                effort_limit_sim=400.0,
                velocity_limit_sim=1.0,
                stiffness=1000.0,
                damping=30.0,
                friction=0.0,
                armature=0.0,
            ),
            "left_arm_elbow_wrist": ImplicitActuatorCfg(
                joint_names_expr=["l_joint_[3-6]"],
                effort_limit_sim=300.0,
                velocity_limit_sim=1.0,
                stiffness=500.0,
                damping=40.0,
                friction=0.0,
                armature=0.0,
            ),
            "right_hand": ImplicitActuatorCfg(
                joint_names_expr=["r_f_joint[1-5]_[1-4]"],
                effort_limit_sim=200.0,
                velocity_limit_sim=4.0,
                stiffness=80.0,
                damping=4.0,
                friction=0.0,
                armature=0.00005,
            ),
            "left_hand": ImplicitActuatorCfg(
                joint_names_expr=["l_f_joint[1-5]_[1-4]"],
                effort_limit_sim=200.0,
                velocity_limit_sim=4.0,
                stiffness=80.0,
                damping=4.0,
                friction=0.0,
                armature=0.00005,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )
    robot = Articulation(cfg=robot_cfg)

    set_dexterous_hand_friction(
        ROBOT_PRIM_PATH,
        static_friction=2.0,
        dynamic_friction=2.0,
        hand_link_keywords=(
            "thumb", "index", "middle", "ring", "little", "finger",
            "palm", "plam", "f_link", "l_link", "r_link",
        ),
        log_prefix="spawn_jaka_dexhand021",
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
