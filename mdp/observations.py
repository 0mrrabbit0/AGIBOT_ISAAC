"""Observation functions for sorting packages task.

All observations that need stage info access the SortingStageTracker
via env.reward_manager.get_term_cfg("stage_tracker").func.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def right_arm_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Right arm joint positions (7-DOF)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]


def right_arm_joint_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Right arm joint velocities (7-DOF)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids]


def right_gripper_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Right gripper joint position (1-DOF)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]


def ee_pos_in_robot_frame(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """End-effector position relative to robot root (3D)."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_idx = asset_cfg.body_ids[0]
    ee_pos_w = asset.data.body_pos_w[:, body_idx, :]
    root_pos_w = asset.data.root_pos_w
    return ee_pos_w - root_pos_w


def ee_quat_in_robot_frame(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """End-effector quaternion in world frame (4D)."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_idx = asset_cfg.body_ids[0]
    return asset.data.body_quat_w[:, body_idx, :]


def package_pos_rel_ee(
    env: ManagerBasedRLEnv,
    package_cfg: SceneEntityCfg,
    ee_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Package position relative to end-effector (3D)."""
    package: RigidObject = env.scene[package_cfg.name]
    robot: Articulation = env.scene[ee_cfg.name]
    body_idx = ee_cfg.body_ids[0]
    return package.data.root_pos_w - robot.data.body_pos_w[:, body_idx, :]


def package_quat(env: ManagerBasedRLEnv, package_cfg: SceneEntityCfg) -> torch.Tensor:
    """Package orientation quaternion (4D)."""
    package: RigidObject = env.scene[package_cfg.name]
    return package.data.root_quat_w


def target_pos_rel_ee(
    env: ManagerBasedRLEnv,
    target_cfg: SceneEntityCfg,
    ee_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Target position relative to end-effector (3D)."""
    target: RigidObject = env.scene[target_cfg.name]
    robot: Articulation = env.scene[ee_cfg.name]
    body_idx = ee_cfg.body_ids[0]
    return target.data.root_pos_w - robot.data.body_pos_w[:, body_idx, :]


def package_pos_rel_target(
    env: ManagerBasedRLEnv,
    package_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Package position relative to target (3D)."""
    package: RigidObject = env.scene[package_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    return package.data.root_pos_w - target.data.root_pos_w


def package_barcode_up_score(env: ManagerBasedRLEnv, package_cfg: SceneEntityCfg) -> torch.Tensor:
    """How much the barcode face (+Y local) is pointing up (1D). Range [-1, 1]."""
    package: RigidObject = env.scene[package_cfg.name]
    quat = package.data.root_quat_w  # (N, 4) wxyz
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    local_y_z = 2 * (y * z + w * x)
    return local_y_z.unsqueeze(-1)


def task_stage_onehot(env: ManagerBasedRLEnv, context: str = "stage_tracker") -> torch.Tensor:
    """Current task stage as one-hot encoding (6D).

    Reads from SortingStageTracker registered as reward term "stage_tracker".
    Returns zeros during shape inference (before reward_manager exists).
    """
    if not hasattr(env, "reward_manager") or env.reward_manager is None:
        return torch.zeros(env.num_envs, 6, device=env.device)
    tracker: ManagerTermBase = env.reward_manager.get_term_cfg(context).func  # type: ignore
    stage = getattr(tracker, "stage")
    clamped = torch.clamp(stage, 0, 5)
    onehot = torch.zeros(env.num_envs, 6, device=env.device)
    onehot.scatter_(1, clamped.unsqueeze(1), 1.0)
    done_mask = stage >= 6
    onehot[done_mask] = 0.0
    return onehot
