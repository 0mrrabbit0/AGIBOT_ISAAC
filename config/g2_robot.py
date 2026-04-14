"""G2 dual-arm robot articulation config for Isaac Lab.

Robot: AgiBot G2 with OmniPicker grippers
- 5 DOF body/waist
- 7 DOF left arm + 7 DOF right arm
- OmniPicker gripper on each arm (angular, mimic-joint)
- 3 DOF head (not actuated in RL)

For RL sorting task, we fix the body joints and only control the right arm + right gripper.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# GenieSim assets directory
GENIESIM_ASSETS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "genie_sim", "source", "geniesim", "assets"
)
# Resolve to absolute path
GENIESIM_ASSETS_DIR = os.path.abspath(GENIESIM_ASSETS_DIR)

# G2 robot USD path
G2_USD_PATH = os.path.join(GENIESIM_ASSETS_DIR, "robot", "G2_omnipicker", "robot.usd")

# Default joint positions for G2 (from sorting_packages_continuous task template)
G2_DEFAULT_JOINT_POS = {
    # Body joints (fixed during RL)
    "idx01_body_joint1": -0.83423,
    "idx02_body_joint2": 1.2172,
    "idx03_body_joint3": 0.10025,
    "idx04_body_joint4": 0.0,
    "idx05_body_joint5": 0.0,
    # Head joints (fixed)
    "idx11_head_joint1": 0.0,
    "idx12_head_joint2": 0.0,
    "idx13_head_joint3": 0.11464,
    # Left arm (fixed at rest position for sorting task)
    "idx21_arm_l_joint1": 0.739033,
    "idx22_arm_l_joint2": -0.717023,
    "idx23_arm_l_joint3": -1.524419,
    "idx24_arm_l_joint4": -1.537612,
    "idx25_arm_l_joint5": 0.27811,
    "idx26_arm_l_joint6": -0.925845,
    "idx27_arm_l_joint7": -0.839257,
    # Right arm (active in RL)
    "idx61_arm_r_joint1": -0.739033,
    "idx62_arm_r_joint2": -0.717023,
    "idx63_arm_r_joint3": 1.524419,
    "idx64_arm_r_joint4": -1.537612,
    "idx65_arm_r_joint5": -0.27811,
    "idx66_arm_r_joint6": -0.925845,
    "idx67_arm_r_joint7": 0.839257,
    # Right gripper (open)
    "idx81_gripper_r_outer_joint1": 0.785,
}

# Right arm joint names (7-DOF, the actuated arm for sorting)
RIGHT_ARM_JOINT_NAMES = [
    "idx61_arm_r_joint1",
    "idx62_arm_r_joint2",
    "idx63_arm_r_joint3",
    "idx64_arm_r_joint4",
    "idx65_arm_r_joint5",
    "idx66_arm_r_joint6",
    "idx67_arm_r_joint7",
]

# Right gripper joint name
RIGHT_GRIPPER_JOINT_NAME = "idx81_gripper_r_outer_joint1"

# End-effector link for right arm
RIGHT_EE_LINK = "gripper_r_center_link"

# ---- Articulation Configs ----

G2_ARTICULATION = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=G2_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos=G2_DEFAULT_JOINT_POS,
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        # Body joints - high stiffness, effectively locked
        "body": ImplicitActuatorCfg(
            joint_names_expr=["idx0[1-5]_body_joint.*"],
            stiffness=1000.0,
            damping=100.0,
        ),
        # Head joints - locked
        "head": ImplicitActuatorCfg(
            joint_names_expr=["idx1[1-3]_head_joint.*"],
            stiffness=1000.0,
            damping=100.0,
        ),
        # Left arm - locked at rest
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["idx2[1-7]_arm_l_joint.*"],
            stiffness=500.0,
            damping=50.0,
        ),
        # Right arm - active for RL (softer for exploration)
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["idx6[1-7]_arm_r_joint.*"],
            stiffness=100.0,
            damping=80.0,
        ),
        # Right gripper - active
        "right_gripper": ImplicitActuatorCfg(
            joint_names_expr=["idx81_gripper_r_outer_joint1"],
            stiffness=17.0,
            damping=5.0,
        ),
        # Left gripper - locked
        "left_gripper": ImplicitActuatorCfg(
            joint_names_expr=["idx41_gripper_l_outer_joint1"],
            stiffness=17.0,
            damping=5.0,
        ),
    },
)
