"""Spawner for the Multi_UR5_RH5DG2 dual-arm robot."""

from pathlib import Path
from typing import Dict, Tuple

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg

from build.geometry import rpy_deg_to_quat
from .placement import robot_back_edge_y
from .hand_friction import set_dexterous_hand_friction
from .gravity_compensation import GravityCompensator

ROBOT_USD_PATH = Path("../dex2bench_dataset/Robots_p/ur5+RH5DG2/usd/Multi_UR5_RH5DG2_with_flange.usd")
ROBOT_PRIM_PATH = "/World/Objects/GlobalRobot"

# Pose policy: place robot at table back-center and face inward.
BACK_EDGE_MARGIN = 0.12
BASE_Z_OFFSET = 0.0
ROBOT_RPY_DEG = (0.0, 0.0, 90.0)

# Placement exclusion region on tabletop (keep random objects off robot base area).
BASE_EXCLUDE_HALF_X = 0.85
BASE_EXCLUDE_FRONT_EXTENT = 0.33

# Mirror UR5 arms so both hands fold outward, away from the tabletop interior.
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
    
    "right_thumb_(yaw|mcp|pip|dip)_joint": 0.0,
    "right_(index|middle)_(pitch|mcp|pip|dip)_joint": 0.0,
    "right_(ring|pinky)_(mcp|pip|dip)_joint": 0.0,
    "left_thumb_(yaw|mcp|pip|dip)_joint": 0.0,
    "left_(index|middle)_(pitch|mcp|pip|dip)_joint": 0.0,
    "left_(ring|pinky)_(mcp|pip|dip)_joint": 0.0,
}



def spawn_multi_ur5_rh5dg2_with_flange(
    table_size: Tuple[float, float, float],
    table_height: float,
    task_dir: str,
) -> Dict:
    """Spawn the dual-arm UR5 + RH5DG2 robot and return placement exclusion metadata."""

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
                joint_names_expr=[
                    "right_thumb_(yaw|mcp|pip|dip)_joint",
                    "right_(index|middle)_(pitch|mcp|pip|dip)_joint",
                    "right_(ring|pinky)_(mcp|pip|dip)_joint",
                ],
                effort_limit_sim=200.0,
                velocity_limit_sim=4.0,
                stiffness=400.0,
                damping=20.0,
                friction=0.0,
                armature=0.000005,
            ),
            "left_hand": ImplicitActuatorCfg(
                joint_names_expr=[
                    "left_thumb_(yaw|mcp|pip|dip)_joint",
                    "left_(index|middle)_(pitch|mcp|pip|dip)_joint",
                    "left_(ring|pinky)_(mcp|pip|dip)_joint",
                ],
                effort_limit_sim=200.0,
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
        static_friction=2.0,
        dynamic_friction=2.0,
        log_prefix="spawn_ur5_rh5dg2",
    )

    # Fix hand joint drive type: USD defaults to "acceleration" which multiplies
    # torque by tiny finger inertia, producing negligible force. Switch to "force"
    # mode so the PD controller can actually drive the fingers.
    try:
        import omni.usd
        from pxr import UsdPhysics

        _hand_patterns = ["thumb", "index", "middle", "ring", "pinky"]
        stage = omni.usd.get_context().get_stage()
        changed = 0
        skipped = 0
        for prim in stage.Traverse():
            pp = str(prim.GetPath())
            if not pp.startswith(ROBOT_PRIM_PATH):
                continue
            pn = prim.GetPath().name
            if not any(p in pn for p in _hand_patterns):
                continue
            if prim.GetTypeName() not in ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint"):
                continue
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if drive:
                ta = drive.GetTypeAttr()
                cur = ta.Get()
                if cur == "acceleration":
                    ta.Set("force")
                    changed += 1
                else:
                    skipped += 1
        print(f"[spawn_rh5dg2] drive-type fix: {changed} changed, {skipped} already force")
    except ImportError:
        print("[spawn_rh5dg2] WARNING: omni.usd/pxr not available, skipping drive-type fix")

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
