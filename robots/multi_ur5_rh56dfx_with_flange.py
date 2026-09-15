"""Spawner for Multi_UR5_RH56DFX dual-arm robot."""

from pathlib import Path
from typing import Dict, Tuple

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg

from build.geometry import rpy_deg_to_quat
from .hand_friction import set_dexterous_hand_friction
from .placement import robot_back_edge_y

ROBOT_USD_PATH = Path("../dex2bench_dataset/Robots_p/ur5+RH56DFX/usd/Multi_UR5_RH56DFX_with_flange.usd")
ROBOT_PRIM_PATH = "/World/Objects/GlobalRobot"

# Names (substrings) of hand joints that need drive-type correction
_HAND_JOINT_NAMES = [
    "thumb_1_joint", "thumb_2_joint", "thumb_3_joint", "thumb_4_joint",
    "index_1_joint", "index_2_joint",
    "middle_1_joint", "middle_2_joint",
    "ring_1_joint", "ring_2_joint",
    "little_1_joint", "little_2_joint",
]

HAND_FRICTION_STATIC = 2.0
HAND_FRICTION_DYNAMIC = 2.0
HAND_FRICTION_LINK_KEYWORDS = (
    "thumb",
    "index",
    "middle",
    "ring",
    "little",
    "palm",
    "plam",
)


def _fix_hand_joint_drive_type(robot_prim_path: str) -> None:
    """Change RH56DFX hand joint drives from 'acceleration' to 'force' mode.

    The USD ships with acceleration-mode drives (torque = stiffness × error × inertia).
    For distal joints (inertia < 5e-5 kg·m²) this produces < 5 mN·m, too weak to
    move against any friction.  Force mode (torque = stiffness × error) gives a
    predictable 100 N·m/rad with our actuator config, which is plenty.

    Must be called AFTER Articulation() spawns the prim but BEFORE sim.reset().
    """
    try:
        import omni.usd  # type: ignore
        from pxr import UsdPhysics  # type: ignore
    except ImportError:
        print("[spawn_rh56dfx] WARNING: omni.usd / pxr not available, skipping drive-type fix")
        return

    stage = omni.usd.get_context().get_stage()
    changed = 0
    skipped = 0

    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if not prim_path.startswith(robot_prim_path):
            continue
        prim_name = prim.GetPath().name
        if not any(p in prim_name for p in _HAND_JOINT_NAMES):
            continue
        if prim.GetTypeName() not in ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint"):
            continue

        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if drive:
            type_attr = drive.GetTypeAttr()
            current = type_attr.Get()
            if current == "acceleration":
                type_attr.Set("force")
                changed += 1
            else:
                skipped += 1

    print(f"[spawn_rh56dfx] drive-type fix: {changed} changed, {skipped} already force")


# Pose policy: place robot at table back-center and face inward.
BACK_EDGE_MARGIN = 0.12
BASE_Z_OFFSET = 0.0
ROBOT_RPY_DEG = (0.0, 0.0, 90.0)

# Placement exclusion region on tabletop (keep random objects off robot base area).
BASE_EXCLUDE_HALF_X = 0.85
BASE_EXCLUDE_FRONT_EXTENT = 0.33

# Arm posture is set to a raised "ready" pose. Fingers start open at 0.
HOME_JOINT_POS = {
    # right arm: shoulder turns right 90°
    "shoulder_pan_joint": 1.57,
    "shoulder_lift_joint": -1.57,
    "elbow_joint": 1.57,
    "wrist_1_joint": 0.0,
    "wrist_2_joint": 1.57,
    "wrist_3_joint": 0.0,

    # left arm: shoulder turns left 90°
    "L_arm_shoulder_pan_joint": -1.57,
    "L_arm_shoulder_lift_joint": 1.57,
    "L_arm_elbow_joint": -1.57,
    "L_arm_wrist_1_joint": 0.0,
    "L_arm_wrist_2_joint": -1.57,
    "L_arm_wrist_3_joint": 0.0,

    "right_thumb_[1-4]_joint": 0.0,
    "right_(index|middle|ring|little)_[1-2]_joint": 0.0,
    "left_thumb_[1-4]_joint": 0.0,
    "left_(index|middle|ring|little)_[1-2]_joint": 0.0,
}



def spawn_multi_ur5_rh56dfx_with_flange(
    table_size: Tuple[float, float, float],
    table_height: float,
    task_dir: str,
) -> Dict:
    """Spawn the dual-arm UR5 robot and return placement exclusion metadata."""
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
            solver_velocity_iteration_count=8,
            fix_root_link=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        activate_contact_sensors=True,
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
                armature=0.000005,
            ),
            "arm_elbow": ImplicitActuatorCfg(
                joint_names_expr=["elbow_joint", "L_arm_elbow_joint"],
                effort_limit_sim=500.0,
                velocity_limit_sim=1.0,
                stiffness=2000.0,
                damping=80.0,
                friction=0.0,
                armature=0.000005,
            ),
            "arm_wrist": ImplicitActuatorCfg(
                joint_names_expr=["wrist_[1-3]_joint", "L_arm_wrist_[1-3]_joint"],
                effort_limit_sim=100.0,
                velocity_limit_sim=4.0,
                stiffness=800.0,
                damping=50.0,
                friction=0.0,
                armature=0.000005,
            ),
            "right_hand": ImplicitActuatorCfg(
                joint_names_expr=["right_thumb_[1-4]_joint", "right_(index|middle|ring|little)_[1-2]_joint"],
                effort_limit_sim=200.0,
                velocity_limit_sim=4.0,
                stiffness=400.0,
                damping=20.0,
                friction=0.0,
                armature=0.000005,
            ),
            "left_hand": ImplicitActuatorCfg(
                joint_names_expr=["left_thumb_[1-4]_joint", "left_(index|middle|ring|little)_[1-2]_joint"],
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

    # ── Fix hand joint drive type (USD default is "acceleration", need "force") ──
    # In acceleration mode: torque = stiffness × error × inertia  (tiny for small joints)
    # In force mode:        torque = stiffness × error            (predictable, 100 N·m/rad)
    _fix_hand_joint_drive_type(ROBOT_PRIM_PATH)


    set_dexterous_hand_friction(
        ROBOT_PRIM_PATH,
        static_friction=HAND_FRICTION_STATIC,
        dynamic_friction=HAND_FRICTION_DYNAMIC,
        hand_link_keywords=HAND_FRICTION_LINK_KEYWORDS,
        log_prefix="spawn_rh56dfx",
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
        "exclude_aabbs": [exclusion],
        "robot_pose": {
            "prim_path": ROBOT_PRIM_PATH,
            "pos": robot_pos,
            "rpy_deg": ROBOT_RPY_DEG,
            "table_size": (sx, sy),
            "usd_path": robot_usd_path.as_posix(),
        },
    }
