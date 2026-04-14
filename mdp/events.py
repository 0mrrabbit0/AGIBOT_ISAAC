"""Event functions for sorting packages task.

Reset events that work with Isaac Lab's EventTerm manager.
Stage tracking is handled by SortingStageTracker (in rewards.py),
NOT by event functions.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def infer_stage_after_reset(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Post-reset hook: infer correct starting stage from scene state.

    Must be called after OmniReset places the scene in a mid-task state,
    so the SortingStageTracker starts tracking from the correct stage.
    """
    tracker = env.reward_manager.get_term_cfg("stage_tracker").func
    if hasattr(tracker, "infer_stage_from_state"):
        tracker.infer_stage_from_state(env, env_ids)


def randomize_package_on_table(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    package_cfg: SceneEntityCfg,
    x_range: tuple[float, float] = (0.35, 0.65),
    y_range: tuple[float, float] = (-0.3, 0.3),
    z_value: float = 1.20,
):
    """Randomize package position on workspace table at reset."""
    package: RigidObject = env.scene[package_cfg.name]
    num = len(env_ids)
    device = env.device

    pos = torch.zeros(num, 3, device=device)
    pos[:, 0] = torch.empty(num, device=device).uniform_(*x_range)
    pos[:, 1] = torch.empty(num, device=device).uniform_(*y_range)
    pos[:, 2] = z_value

    # Random yaw rotation
    yaw = torch.empty(num, device=device).uniform_(-3.14159, 3.14159)
    quat = torch.zeros(num, 4, device=device)
    quat[:, 0] = torch.cos(yaw / 2)  # w
    quat[:, 3] = torch.sin(yaw / 2)  # z (rotation around Z-axis)

    package.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    package.write_root_velocity_to_sim(torch.zeros(num, 6, device=device), env_ids=env_ids)
