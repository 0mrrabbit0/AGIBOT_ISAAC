"""Environment configuration for G2 Sorting Packages task.

Follows OmniReset/Isaac Lab ManagerBasedRLEnv patterns:
- Scene: robot + objects as RigidObjectCfg
- Actions: RelativeJointPositionActionCfg + BinaryJointPositionActionCfg
- Observations: ObsGroup with policy/critic separation
- Rewards: SortingStageTracker (ManagerTermBase) + shaped terms
- Terminations: stage completion, abnormal state, package fallen
- Events: reset randomization (package pose, material friction, mass)

Scene layout (top-down view, robot at origin facing +X):
    Workspace table (+0.5,  0.0) — packages start here
    Scanning table  (+0.3, -0.5) — barcode scan station
    Target box      (+0.3, +0.5) — final placement
"""

from __future__ import annotations

import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    RelativeJointPositionActionCfg,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .g2_robot import (
    G2_ARTICULATION,
    GENIESIM_ASSETS_DIR,
    RIGHT_ARM_JOINT_NAMES,
    RIGHT_EE_LINK,
    RIGHT_GRIPPER_JOINT_NAME,
)

# Import our MDP module
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import mdp as task_mdp


# ============================================================
# Asset paths
# ============================================================

CARTON_USD_DIR = os.path.join(GENIESIM_ASSETS_DIR, "objects", "benchmark", "carton")

CARTON_VARIANTS = {
    "carton_020": os.path.join(CARTON_USD_DIR, "benchmark_carton_020", "Aligned.usd"),
    "carton_028": os.path.join(CARTON_USD_DIR, "benchmark_carton_028", "Aligned.usd"),
    "carton_029": os.path.join(CARTON_USD_DIR, "benchmark_carton_029", "Aligned.usd"),
    "carton_030": os.path.join(CARTON_USD_DIR, "benchmark_carton_030", "Aligned.usd"),
}

DEFAULT_CARTON_USD = CARTON_VARIANTS["carton_020"]

# Scene entity config shortcuts
EE_CFG = SceneEntityCfg("robot", body_names=[RIGHT_EE_LINK])
RIGHT_ARM_CFG = SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINT_NAMES)
GRIPPER_CFG = SceneEntityCfg("robot", joint_names=[RIGHT_GRIPPER_JOINT_NAME])


# ============================================================
# Scene
# ============================================================

@configclass
class SortingPackagesSceneCfg(InteractiveSceneCfg):
    """Scene: G2 robot + workspace table + scanning table + target box + package."""

    robot = G2_ARTICULATION

    workspace_table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WorkspaceTable",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 1.0, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.5, 0.4)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 1.14), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    scanning_table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ScanningTable",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.4, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=30.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.35)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, -0.5, 1.14), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    target_box: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetBox",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.4, 0.3),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.3, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.5, 1.14), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    package: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Package",
        spawn=sim_utils.UsdFileCfg(
            usd_path=DEFAULT_CARTON_USD,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 1.20), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1500.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# ============================================================
# Carton variants (swappable via CLI: env.scene.package=carton_028)
# ============================================================

def make_package(usd_path: str) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Package",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 1.20), rot=(1.0, 0.0, 0.0, 0.0)),
    )


variants = {
    "scene.package": {k: make_package(v) for k, v in CARTON_VARIANTS.items()},
}


# ============================================================
# Actions
# ============================================================

@configclass
class G2RightArmActionCfg:
    """7-DOF right arm (relative joint position) + binary gripper."""

    right_arm = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=RIGHT_ARM_JOINT_NAMES,
        scale=0.05,
        use_zero_offset=True,
    )

    right_gripper = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=[RIGHT_GRIPPER_JOINT_NAME],
        open_command_expr={RIGHT_GRIPPER_JOINT_NAME: 0.785},
        close_command_expr={RIGHT_GRIPPER_JOINT_NAME: 0.0},
    )


# ============================================================
# Observations
# ============================================================

@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations: robot state + package state + targets + stage."""

        right_arm_joints = ObsTerm(
            func=task_mdp.right_arm_joint_pos,
            params={"asset_cfg": RIGHT_ARM_CFG},
        )
        gripper_pos = ObsTerm(
            func=task_mdp.right_gripper_pos,
            params={"asset_cfg": GRIPPER_CFG},
        )
        ee_position = ObsTerm(
            func=task_mdp.ee_pos_in_robot_frame,
            params={"asset_cfg": EE_CFG},
        )
        ee_orientation = ObsTerm(
            func=task_mdp.ee_quat_in_robot_frame,
            params={"asset_cfg": EE_CFG},
        )
        pkg_pos_rel_ee = ObsTerm(
            func=task_mdp.package_pos_rel_ee,
            params={"package_cfg": SceneEntityCfg("package"), "ee_cfg": EE_CFG},
        )
        pkg_orientation = ObsTerm(
            func=task_mdp.package_quat,
            params={"package_cfg": SceneEntityCfg("package")},
        )
        scan_table_rel_ee = ObsTerm(
            func=task_mdp.target_pos_rel_ee,
            params={"target_cfg": SceneEntityCfg("scanning_table"), "ee_cfg": EE_CFG},
        )
        box_rel_ee = ObsTerm(
            func=task_mdp.target_pos_rel_ee,
            params={"target_cfg": SceneEntityCfg("target_box"), "ee_cfg": EE_CFG},
        )
        pkg_rel_scan = ObsTerm(
            func=task_mdp.package_pos_rel_target,
            params={"package_cfg": SceneEntityCfg("package"), "target_cfg": SceneEntityCfg("scanning_table")},
        )
        pkg_rel_box = ObsTerm(
            func=task_mdp.package_pos_rel_target,
            params={"package_cfg": SceneEntityCfg("package"), "target_cfg": SceneEntityCfg("target_box")},
        )
        barcode_up = ObsTerm(
            func=task_mdp.package_barcode_up_score,
            params={"package_cfg": SceneEntityCfg("package")},
        )
        stage = ObsTerm(func=task_mdp.task_stage_onehot)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 3

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations: all policy obs + privileged (velocities)."""

        right_arm_joints = ObsTerm(
            func=task_mdp.right_arm_joint_pos,
            params={"asset_cfg": RIGHT_ARM_CFG},
        )
        right_arm_vel = ObsTerm(
            func=task_mdp.right_arm_joint_vel,
            params={"asset_cfg": RIGHT_ARM_CFG},
        )
        gripper_pos = ObsTerm(
            func=task_mdp.right_gripper_pos,
            params={"asset_cfg": GRIPPER_CFG},
        )
        ee_position = ObsTerm(
            func=task_mdp.ee_pos_in_robot_frame,
            params={"asset_cfg": EE_CFG},
        )
        ee_orientation = ObsTerm(
            func=task_mdp.ee_quat_in_robot_frame,
            params={"asset_cfg": EE_CFG},
        )
        pkg_pos_rel_ee = ObsTerm(
            func=task_mdp.package_pos_rel_ee,
            params={"package_cfg": SceneEntityCfg("package"), "ee_cfg": EE_CFG},
        )
        pkg_orientation = ObsTerm(
            func=task_mdp.package_quat,
            params={"package_cfg": SceneEntityCfg("package")},
        )
        scan_table_rel_ee = ObsTerm(
            func=task_mdp.target_pos_rel_ee,
            params={"target_cfg": SceneEntityCfg("scanning_table"), "ee_cfg": EE_CFG},
        )
        box_rel_ee = ObsTerm(
            func=task_mdp.target_pos_rel_ee,
            params={"target_cfg": SceneEntityCfg("target_box"), "ee_cfg": EE_CFG},
        )
        pkg_rel_scan = ObsTerm(
            func=task_mdp.package_pos_rel_target,
            params={"package_cfg": SceneEntityCfg("package"), "target_cfg": SceneEntityCfg("scanning_table")},
        )
        pkg_rel_box = ObsTerm(
            func=task_mdp.package_pos_rel_target,
            params={"package_cfg": SceneEntityCfg("package"), "target_cfg": SceneEntityCfg("target_box")},
        )
        barcode_up = ObsTerm(
            func=task_mdp.package_barcode_up_score,
            params={"package_cfg": SceneEntityCfg("package")},
        )
        stage = ObsTerm(func=task_mdp.task_stage_onehot)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ============================================================
# Rewards
# ============================================================

@configclass
class RewardsCfg:
    """Reward terms. stage_tracker MUST be first (other terms depend on it)."""

    # ---- Core: stateful stage tracker (weight=0, side-effect only) ----
    stage_tracker = RewTerm(
        func=task_mdp.SortingStageTracker,  # type: ignore
        weight=0.0,
        params={
            "ee_cfg": EE_CFG,
            "package_cfg": SceneEntityCfg("package"),
            "scan_table_cfg": SceneEntityCfg("scanning_table"),
            "box_cfg": SceneEntityCfg("target_box"),
            "gripper_cfg": GRIPPER_CFG,
        },
    )

    # ---- Sparse: stage completion ----
    stage_completion = RewTerm(func=task_mdp.stage_completion_reward, weight=10.0)

    # ---- Dense shaping ----
    ee_to_package = RewTerm(func=task_mdp.ee_to_package_distance, weight=0.5, params={"std": 0.1})
    package_to_scan = RewTerm(func=task_mdp.package_to_scan_table_distance, weight=0.5, params={"std": 0.1})
    package_to_box = RewTerm(func=task_mdp.package_to_box_distance, weight=0.5, params={"std": 0.1})
    barcode_orientation = RewTerm(func=task_mdp.barcode_orientation_reward, weight=0.3)
    grasp = RewTerm(func=task_mdp.grasp_reward, weight=1.0, params={"grasp_threshold": 0.05})

    # ---- Safety penalties ----
    action_l2 = RewTerm(func=task_mdp.action_l2_penalty, weight=-1e-4)
    action_rate = RewTerm(func=task_mdp.action_rate_l2_penalty, weight=-1e-3)
    joint_vel = RewTerm(
        func=task_mdp.joint_vel_l2_penalty,
        weight=-1e-3,
        params={"asset_cfg": RIGHT_ARM_CFG},
    )
    package_dropped = RewTerm(
        func=task_mdp.package_dropped_penalty,
        weight=-5.0,
        params={"package_cfg": SceneEntityCfg("package"), "min_height": 0.8},
    )


# ============================================================
# Terminations
# ============================================================

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)
    task_completed = DoneTerm(func=task_mdp.task_completed)
    package_fallen = DoneTerm(
        func=task_mdp.package_fallen,
        params={"package_cfg": SceneEntityCfg("package"), "min_height": 0.8},
    )
    robot_abnormal = DoneTerm(
        func=task_mdp.robot_abnormal,
        params={"asset_cfg": RIGHT_ARM_CFG, "max_joint_vel": 100.0},
    )


# ============================================================
# Events
# ============================================================

@configclass
class TrainEventCfg:
    """Training events: vanilla reset (package on table) + domain randomization."""

    # Reset scene to defaults first
    reset_scene = EventTerm(func=task_mdp.reset_scene_to_default, mode="reset")

    # Randomize package pose on table
    randomize_package = EventTerm(
        func=task_mdp.randomize_package_on_table,
        mode="reset",
        params={
            "package_cfg": SceneEntityCfg("package"),
            "x_range": (0.35, 0.65),
            "y_range": (-0.3, 0.3),
            "z_value": 1.20,
        },
    )

    # Material randomization (startup)
    robot_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "static_friction_range": (0.4, 1.0),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    package_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "static_friction_range": (0.5, 1.5),
            "dynamic_friction_range": (0.4, 1.2),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
            "asset_cfg": SceneEntityCfg("package"),
        },
    )

    # Mass randomization
    randomize_package_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("package"),
            "mass_distribution_params": (0.02, 0.15),
            "operation": "abs",
            "distribution": "uniform",
        },
    )


@configclass
class OmniResetTrainEventCfg:
    """OmniReset training events: diverse multi-stage resets + domain randomization.

    Key difference from vanilla: instead of always resetting package to the
    workspace table, resets to 6 diverse mid-task states, enabling the
    "backwards learning" phenomenon from OmniReset.
    """

    # Reset scene to defaults first
    reset_scene = EventTerm(func=task_mdp.reset_scene_to_default, mode="reset")

    # OmniReset: multi-stage reset manager (replaces randomize_package)
    multi_stage_reset = EventTerm(
        func=task_mdp.SortingMultiStageResetManager,
        mode="reset",
        params={
            "ee_cfg": EE_CFG,
            "package_cfg": SceneEntityCfg("package"),
            "scan_table_cfg": SceneEntityCfg("scanning_table"),
            "box_cfg": SceneEntityCfg("target_box"),
            "gripper_cfg": GRIPPER_CFG,
            # Uniform probability across 6 reset types
            "probs": [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
            # Optional: path to pre-generated reset states
            # "dataset_dir": os.path.join(os.path.dirname(__file__), "..", "data", "reset_states"),
            # Enable adaptive probabilities (lower success → higher prob)
            "adaptive_probs": True,
            "adaptive_alpha": 0.1,
        },
    )

    # Material randomization (startup) — same as vanilla
    robot_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "static_friction_range": (0.4, 1.0),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    package_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "static_friction_range": (0.5, 1.5),
            "dynamic_friction_range": (0.4, 1.2),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
            "asset_cfg": SceneEntityCfg("package"),
        },
    )

    # Mass randomization
    randomize_package_mass = EventTerm(
        func=task_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("package"),
            "mass_distribution_params": (0.02, 0.15),
            "operation": "abs",
            "distribution": "uniform",
        },
    )


@configclass
class EvalEventCfg:
    """Evaluation events: deterministic reset, no domain randomization."""

    reset_scene = EventTerm(func=task_mdp.reset_scene_to_default, mode="reset")

    randomize_package = EventTerm(
        func=task_mdp.randomize_package_on_table,
        mode="reset",
        params={
            "package_cfg": SceneEntityCfg("package"),
            "x_range": (0.45, 0.55),
            "y_range": (-0.1, 0.1),
            "z_value": 1.20,
        },
    )


# ============================================================
# Main Environment Configs
# ============================================================

@configclass
class G2SortingPackagesBaseCfg(ManagerBasedRLEnvCfg):
    """Base environment config."""

    scene: SortingPackagesSceneCfg = SortingPackagesSceneCfg(num_envs=64, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: G2RightArmActionCfg = G2RightArmActionCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: TrainEventCfg = MISSING
    viewer: ViewerCfg = ViewerCfg(
        eye=(2.0, 0.0, 2.0),
        origin_type="world",
        env_index=0,
        asset_name="robot",
    )
    variants = variants

    def __post_init__(self):
        self.decimation = 4  # Policy at 30Hz (120/4)
        self.episode_length_s = 30.0  # 30s per episode
        self.sim.dt = 1 / 120.0  # Physics at 120Hz

        # PhysX solver
        self.sim.physx.solver_type = 1  # TGS
        self.sim.physx.max_position_iteration_count = 64
        self.sim.physx.max_velocity_iteration_count = 4
        self.sim.physx.bounce_threshold_velocity = 0.02
        self.sim.physx.friction_offset_threshold = 0.01
        self.sim.physx.friction_correlation_distance = 0.0005

        # GPU memory
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 2
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**21
        self.sim.physx.gpu_max_rigid_contact_count = 2**21
        self.sim.physx.gpu_max_rigid_patch_count = 2**21

        # Render
        self.sim.render.enable_ambient_occlusion = True


@configclass
class G2SortingPackagesTrainCfg(G2SortingPackagesBaseCfg):
    """Training: 4096 envs, vanilla domain randomization."""
    events: TrainEventCfg = TrainEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4096


@configclass
class G2SortingPackagesOmniResetTrainCfg(G2SortingPackagesBaseCfg):
    """OmniReset Training: 4096 envs, diverse multi-stage resets.

    Uses SortingMultiStageResetManager for "backwards learning":
    - 6 reset distributions covering all task stages
    - Adaptive probabilities based on per-stage success rates
    - Domain randomization on top of diverse resets
    """
    events: OmniResetTrainEventCfg = OmniResetTrainEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4096
        # Longer episodes to accommodate mid-stage resets
        self.episode_length_s = 20.0


@configclass
class G2SortingPackagesEvalCfg(G2SortingPackagesBaseCfg):
    """Evaluation: 32 envs, minimal randomization."""
    events: EvalEventCfg = EvalEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
