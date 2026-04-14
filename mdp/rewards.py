"""Reward functions for the 6-stage sorting packages task.

Architecture follows OmniReset's ProgressContext pattern:
- SortingStageTracker (ManagerTermBase): stateful term called every step,
  tracks which stage each env is in and manages transitions.
- Other reward terms query the tracker via env.reward_manager.get_term_cfg().

Sorting Packages Task Stages (each worth 0.16 points):
0. Right EE follows target package
1. Pick up target package with right EE
2. Place package on scanning table
3. Package barcode facing up
4. Pick up package with right EE (again)
5. Place package into box
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ============================================================
# Core: Stateful Stage Tracker (called every step)
# ============================================================

class SortingStageTracker(ManagerTermBase):
    """Tracks 6-stage task progress. Registered as a reward term with weight=0.

    Other reward/observation/termination terms access stage info via:
        tracker = env.reward_manager.get_term_cfg("stage_tracker").func
        stage = tracker.stage  # (N,) int tensor [0..6]
        stage_just_completed = tracker.stage_just_completed  # (N,) bool
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.ee_cfg: SceneEntityCfg = cfg.params["ee_cfg"]
        self.package_cfg: SceneEntityCfg = cfg.params["package_cfg"]
        self.scan_table_cfg: SceneEntityCfg = cfg.params["scan_table_cfg"]
        self.box_cfg: SceneEntityCfg = cfg.params["box_cfg"]
        self.gripper_cfg: SceneEntityCfg = cfg.params["gripper_cfg"]

        # Resolve scene entities
        self.robot: Articulation = env.scene[self.ee_cfg.name]
        self.package: RigidObject = env.scene[self.package_cfg.name]
        self.scan_table: RigidObject = env.scene[self.scan_table_cfg.name]
        self.box: RigidObject = env.scene[self.box_cfg.name]

        # State buffers
        self.stage = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.stage_just_completed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.stages_completed_total = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

        # Cache for distance/orientation metrics (used by other reward terms)
        self.ee_pkg_dist = torch.zeros(env.num_envs, device=env.device)
        self.pkg_scan_dist = torch.zeros(env.num_envs, device=env.device)
        self.pkg_box_dist = torch.zeros(env.num_envs, device=env.device)
        self.barcode_up_score = torch.zeros(env.num_envs, device=env.device)
        self.gripper_closed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        # Cache stage reached before reset (used by adaptive reset probs)
        if env_ids is None:
            self.pre_reset_stage = self.stage.clone()
            self.stage[:] = 0
            self.stages_completed_total[:] = 0
        else:
            if not hasattr(self, "pre_reset_stage"):
                self.pre_reset_stage = torch.zeros_like(self.stage)
            self.pre_reset_stage[env_ids] = self.stage[env_ids].clone()
            self.stage[env_ids] = 0
            self.stages_completed_total[env_ids] = 0

    def infer_stage_from_state(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
        """Infer the correct starting stage from the current scene state.

        Called after OmniReset places the scene in a mid-task state.
        Without this, the tracker would start at stage 0 even when the
        package is already on the scanning table (stage 3-4).
        """
        body_idx = self.ee_cfg.body_ids[0]
        ee_pos = self.robot.data.body_pos_w[env_ids, body_idx, :]
        pkg_pos = self.package.data.root_pos_w[env_ids]
        pkg_quat = self.package.data.root_quat_w[env_ids]
        scan_pos = self.scan_table.data.root_pos_w[env_ids]
        box_pos = self.box.data.root_pos_w[env_ids]

        gripper_joint_ids = self.gripper_cfg.joint_ids
        gripper_pos = self.robot.data.joint_pos[env_ids][:, gripper_joint_ids].squeeze(-1)
        gripper_closed = gripper_pos < 0.3

        ee_pkg_dist = torch.norm(pkg_pos - ee_pos, dim=-1)
        pkg_scan_dist = torch.norm(pkg_pos[:, :2] - scan_pos[:, :2], dim=-1)
        pkg_box_dist = torch.norm(pkg_pos - box_pos, dim=-1)
        pkg_z = pkg_pos[:, 2]
        scan_z = scan_pos[:, 2]

        # Barcode up score: z-component of local Y-axis = R[2][1]
        w, x, y, z = pkg_quat[:, 0], pkg_quat[:, 1], pkg_quat[:, 2], pkg_quat[:, 3]
        barcode_up = 2 * (y * z + w * x)

        # Infer stage backwards from stage 5 to stage 0
        inferred = torch.zeros(len(env_ids), dtype=torch.long, device=self._env.device)

        # Near box with package grasped → stage 5
        near_box = (pkg_box_dist < 0.3) & gripper_closed & (pkg_z > scan_z + 0.05)
        inferred[near_box] = 5

        # On scan table with barcode up → stage 4
        on_scan_barcode = (~near_box) & (pkg_scan_dist < 0.2) & (barcode_up > 0.6) & (torch.abs(pkg_z - scan_z) < 0.15)
        inferred[on_scan_barcode] = 4

        # On scan table with random orientation → stage 3 (barcode not up yet)
        on_scan_random = (~near_box) & (~on_scan_barcode) & (pkg_scan_dist < 0.2) & (torch.abs(pkg_z - scan_z) < 0.15)
        inferred[on_scan_random] = 3

        # Package grasped and in air → stage 2
        grasped_air = (~near_box) & (~on_scan_barcode) & (~on_scan_random) & gripper_closed & (pkg_z > scan_z + 0.05)
        inferred[grasped_air] = 2

        # EE near package on table → stage 1
        ee_near = (~near_box) & (~on_scan_barcode) & (~on_scan_random) & (~grasped_air) & (ee_pkg_dist < 0.15)
        inferred[ee_near] = 1

        # Everything else → stage 0
        self.stage[env_ids] = inferred

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        ee_cfg: SceneEntityCfg,
        package_cfg: SceneEntityCfg,
        scan_table_cfg: SceneEntityCfg,
        box_cfg: SceneEntityCfg,
        gripper_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Called every step. Updates stage transitions and caches metrics."""
        body_idx = ee_cfg.body_ids[0]
        ee_pos = self.robot.data.body_pos_w[:, body_idx, :]
        pkg_pos = self.package.data.root_pos_w
        pkg_quat = self.package.data.root_quat_w
        scan_pos = self.scan_table.data.root_pos_w
        box_pos = self.box.data.root_pos_w

        gripper_joint_ids = gripper_cfg.joint_ids
        gripper_pos = self.robot.data.joint_pos[:, gripper_joint_ids].squeeze(-1)

        # ---- Cache metrics ----
        self.ee_pkg_dist[:] = torch.norm(pkg_pos - ee_pos, dim=-1)
        self.pkg_scan_dist[:] = torch.norm(pkg_pos[:, :2] - scan_pos[:, :2], dim=-1)
        self.pkg_box_dist[:] = torch.norm(pkg_pos - box_pos, dim=-1)
        self.gripper_closed[:] = gripper_pos < 0.3

        # Barcode up: z-component of local Y-axis in world frame = R[2][1]
        w, x, y, z = pkg_quat[:, 0], pkg_quat[:, 1], pkg_quat[:, 2], pkg_quat[:, 3]
        self.barcode_up_score[:] = 2 * (y * z + w * x)

        # ---- Stage transitions ----
        self.stage_just_completed[:] = False
        stage = self.stage
        pkg_z = pkg_pos[:, 2]
        scan_z = scan_pos[:, 2]

        # Stage 0 → 1: EE approaches package (within 8cm)
        s0 = (stage == 0) & (self.ee_pkg_dist < 0.08)
        # Stage 1 → 2: Package picked up (lifted above table + gripper closed)
        s1 = (stage == 1) & (pkg_z > scan_z + 0.1) & self.gripper_closed
        # Stage 2 → 3: Package placed on scanning table (XY close + Z close)
        s2 = (stage == 2) & (self.pkg_scan_dist < 0.1) & (torch.abs(pkg_z - scan_z) < 0.08)
        # Stage 3 → 4: Barcode facing up on scanning table
        s3 = (stage == 3) & (self.barcode_up_score > 0.8) & (self.pkg_scan_dist < 0.15)
        # Stage 4 → 5: Package picked up again
        s4 = (stage == 4) & (pkg_z > scan_z + 0.1) & self.gripper_closed
        # Stage 5 → 6: Package placed in box
        s5 = (stage == 5) & (self.pkg_box_dist < 0.12)

        for mask, next_stage in [(s0, 1), (s1, 2), (s2, 3), (s3, 4), (s4, 5), (s5, 6)]:
            self.stage[mask] = next_stage
            self.stage_just_completed[mask] = True
            self.stages_completed_total[mask] += 1

        # Return 0 — this term only tracks state, reward comes from other terms
        return torch.zeros(env.num_envs, device=env.device)


# ============================================================
# Helper to get tracker from other reward terms
# ============================================================

def _get_tracker(env: ManagerBasedRLEnv, context: str = "stage_tracker") -> SortingStageTracker:
    return env.reward_manager.get_term_cfg(context).func  # type: ignore


# ============================================================
# Sparse stage completion reward
# ============================================================

def stage_completion_reward(env: ManagerBasedRLEnv, context: str = "stage_tracker") -> torch.Tensor:
    """Sparse +1.0 reward each time a stage is completed."""
    tracker = _get_tracker(env, context)
    return tracker.stage_just_completed.float()


# ============================================================
# Dense distance shaping
# ============================================================

def ee_to_package_distance(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    context: str = "stage_tracker",
) -> torch.Tensor:
    """Tanh-shaped reward for EE approaching package. Active in stages 0, 1, 4."""
    tracker = _get_tracker(env, context)
    reward = 1.0 - torch.tanh(tracker.ee_pkg_dist / std)
    mask = (tracker.stage == 0) | (tracker.stage == 1) | (tracker.stage == 4)
    return reward * mask.float()


def package_to_scan_table_distance(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    context: str = "stage_tracker",
) -> torch.Tensor:
    """Dense reward for moving package toward scanning table. Active in stages 2, 3."""
    tracker = _get_tracker(env, context)
    reward = 1.0 - torch.tanh(tracker.pkg_scan_dist / std)
    mask = (tracker.stage == 2) | (tracker.stage == 3)
    return reward * mask.float()


def package_to_box_distance(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    context: str = "stage_tracker",
) -> torch.Tensor:
    """Dense reward for moving package toward box. Active in stage 5."""
    tracker = _get_tracker(env, context)
    reward = 1.0 - torch.tanh(tracker.pkg_box_dist / std)
    mask = tracker.stage == 5
    return reward * mask.float()


def barcode_orientation_reward(
    env: ManagerBasedRLEnv,
    context: str = "stage_tracker",
) -> torch.Tensor:
    """Reward for barcode facing up. Active in stages 3, 4, 5."""
    tracker = _get_tracker(env, context)
    reward = (tracker.barcode_up_score + 1.0) / 2.0
    mask = (tracker.stage == 3) | (tracker.stage == 4) | (tracker.stage == 5)
    return reward * mask.float()


def grasp_reward(
    env: ManagerBasedRLEnv,
    grasp_threshold: float = 0.05,
    context: str = "stage_tracker",
) -> torch.Tensor:
    """Reward for grasping (EE close + gripper closed). Active in stages 1, 4."""
    tracker = _get_tracker(env, context)
    grasped = (tracker.ee_pkg_dist < grasp_threshold) & tracker.gripper_closed
    mask = (tracker.stage == 1) | (tracker.stage == 4)
    return grasped.float() * mask.float()


# ============================================================
# Safety penalties (standard Isaac Lab patterns)
# ============================================================

def action_l2_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """L2 penalty on action magnitude."""
    return torch.clamp(torch.sum(torch.square(env.action_manager.action), dim=-1), 0, 1e4)


def action_rate_l2_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """L2 penalty on action rate."""
    return torch.clamp(
        torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=-1),
        0, 1e4,
    )


def joint_vel_l2_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for high joint velocities."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.clamp(torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=-1), 0, 1e4)


def package_dropped_penalty(
    env: ManagerBasedRLEnv,
    package_cfg: SceneEntityCfg,
    min_height: float = 0.5,
) -> torch.Tensor:
    """Penalty when package falls below threshold."""
    package: RigidObject = env.scene[package_cfg.name]
    return (package.data.root_pos_w[:, 2] < min_height).float()
