"""Termination conditions for the sorting packages task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def task_completed(env: ManagerBasedRLEnv, context: str = "stage_tracker") -> torch.Tensor:
    """Episode ends when all 6 stages are completed."""
    tracker: ManagerTermBase = env.reward_manager.get_term_cfg(context).func  # type: ignore
    return getattr(tracker, "stage") >= 6


def package_fallen(
    env: ManagerBasedRLEnv,
    package_cfg: SceneEntityCfg,
    min_height: float = 0.5,
) -> torch.Tensor:
    """Episode ends if package falls below threshold height."""
    package: RigidObject = env.scene[package_cfg.name]
    return package.data.root_pos_w[:, 2] < min_height


def robot_abnormal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    max_joint_vel: float = 10.0,
) -> torch.Tensor:
    """Episode ends if robot enters abnormal state."""
    asset: Articulation = env.scene[asset_cfg.name]
    vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    pos = asset.data.joint_pos[:, asset_cfg.joint_ids]

    has_nan = torch.any(torch.isnan(pos), dim=-1) | torch.any(torch.isnan(vel), dim=-1)
    excessive_vel = torch.any(torch.abs(vel) > max_joint_vel, dim=-1)
    return has_nan | excessive_vel
