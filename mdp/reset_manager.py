"""Multi-stage reset manager for Sorting Packages, inspired by OmniReset.

OmniReset's key insight: diverse simulator resets — not reward engineering
or curricula — are the key to solving long-horizon manipulation tasks.
Instead of always resetting to stage 0, we reset to diverse mid-task states
so the policy can practice every stage efficiently.

For the 6-stage Sorting Packages task, we define 6 reset distributions:

| Reset Type                     | Stages Practiced | Description                          |
|-------------------------------|-----------------|--------------------------------------|
| PackageOnTable_EEFar          | 0→1             | Package on workspace, EE at home     |
| PackageOnTable_EENear         | 1→2             | Package on workspace, EE near pkg    |
| PackageGrasped_AboveWorkspace | 2→3             | Package in gripper, heading to scan  |
| PackageOnScanTable_RandomOri  | 3→4             | Package on scan table, random orient |
| PackageOnScanTable_BarcodeUp  | 4→5             | Package on scan table, barcode up    |
| PackageGrasped_NearBox        | 5→6             | Package in gripper, heading to box   |

The "backwards learning" effect: PPO naturally learns the final stages
first (where success is close), then propagates value backward.
"""

from __future__ import annotations

import os
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class SortingMultiStageResetManager(ManagerTermBase):
    """Resets environments to diverse mid-task states for the sorting task.

    Unlike the vanilla approach (always reset to stage 0), this manager
    samples from 6 reset distributions — one per task stage — enabling
    efficient practice of all stages.

    Two modes:
    1. **Dataset mode**: Load pre-recorded states from .pt files
       (high quality, requires running record_sorting_reset_states.py first)
    2. **Procedural mode**: Generate reset states on-the-fly via heuristics
       (no pre-computation needed, works immediately)
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
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

        # Probabilities for each reset type (uniform by default)
        self.probs = torch.tensor(
            cfg.params.get("probs", [1 / 6] * 6),
            device=env.device,
        )
        self.probs = self.probs / self.probs.sum()

        # Track which reset type was used per env (for logging)
        self.reset_type_ids = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )

        # Success tracking per reset type
        self.success_counts = torch.zeros(6, device=env.device)
        self.total_counts = torch.zeros(6, device=env.device)

        # Adaptive probability settings
        self.adaptive = cfg.params.get("adaptive_probs", False)
        self.adaptive_alpha = cfg.params.get("adaptive_alpha", 0.1)

        # Dataset mode: try loading pre-recorded states
        dataset_dir = cfg.params.get("dataset_dir", None)
        self.datasets = None
        if dataset_dir and os.path.isdir(dataset_dir):
            self._load_datasets(dataset_dir, env.device)

        # Cache the stage each env reached before reset (for adaptive probs)
        # Updated every step via the post_physics_step hook
        self.last_stage = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

        # Cache joint indices for right arm
        self.right_arm_joint_ids = None

    def _load_datasets(self, dataset_dir: str, device: torch.device) -> None:
        """Load pre-recorded reset state datasets."""
        reset_names = [
            "PackageOnTable_EEFar",
            "PackageOnTable_EENear",
            "PackageGrasped_AboveWorkspace",
            "PackageOnScanTable_RandomOri",
            "PackageOnScanTable_BarcodeUp",
            "PackageGrasped_NearBox",
        ]
        self.datasets = []
        for name in reset_names:
            path = os.path.join(dataset_dir, f"resets_{name}.pt")
            if os.path.exists(path):
                data = torch.load(path, map_location=device)
                self.datasets.append(data)
            else:
                self.datasets.append(None)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        ee_cfg: SceneEntityCfg,
        package_cfg: SceneEntityCfg,
        scan_table_cfg: SceneEntityCfg,
        box_cfg: SceneEntityCfg,
        gripper_cfg: SceneEntityCfg,
        probs: list[float] | None = None,
        dataset_dir: str | None = None,
        adaptive_probs: bool = False,
        adaptive_alpha: float = 0.1,
    ) -> None:
        """Reset environments to diverse mid-task states."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        num_envs = len(env_ids)
        if num_envs == 0:
            return

        # Update success tracking and adaptive probabilities
        if self.adaptive:
            self._update_adaptive_probs(env, env_ids)

        # Sample reset type for each environment
        reset_types = torch.multinomial(
            self.probs.unsqueeze(0).expand(num_envs, -1),
            num_samples=1,
        ).squeeze(-1)
        self.reset_type_ids[env_ids] = reset_types

        # Update counts
        for rt in range(6):
            mask = reset_types == rt
            self.total_counts[rt] += mask.sum()

        # Dispatch to per-type reset logic
        for rt in range(6):
            mask = reset_types == rt
            if not mask.any():
                continue
            current_env_ids = env_ids[mask]

            if self.datasets and self.datasets[rt] is not None:
                self._reset_from_dataset(rt, current_env_ids)
            else:
                self._reset_procedural(rt, current_env_ids, env)

        # Infer correct starting stage from the reset state
        tracker = env.reward_manager.get_term_cfg("stage_tracker").func
        if hasattr(tracker, "infer_stage_from_state"):
            tracker.infer_stage_from_state(env, env_ids)

    def _update_adaptive_probs(
        self, env: ManagerBasedEnv, env_ids: torch.Tensor
    ) -> None:
        """Adapt reset probabilities based on per-stage success rates.

        Stages with lower success get higher reset probability.
        Uses self.last_stage (cached before tracker.reset) to determine
        whether each env advanced beyond its starting stage.
        """
        tracker = env.reward_manager.get_term_cfg("stage_tracker").func
        pre_stage = getattr(tracker, "pre_reset_stage", self.last_stage)
        for idx in env_ids:
            rt = self.reset_type_ids[idx].item()
            stage = pre_stage[idx].item()
            # Success = advanced beyond the starting stage of this reset type
            if stage > rt:
                self.success_counts[rt] += 1

        # Compute success rates
        rates = self.success_counts / (self.total_counts + 1)
        # Lower success → higher probability (inverse weighting)
        weights = 1.0 - rates + 0.01  # +0.01 to prevent zero probability
        self.probs = weights / weights.sum()

    def _reset_from_dataset(
        self, reset_type: int, env_ids: torch.Tensor
    ) -> None:
        """Reset from pre-recorded dataset states."""
        dataset = self.datasets[reset_type]
        num = len(env_ids)
        num_states = dataset["robot_joint_pos"].shape[0]

        # Sample random indices from dataset
        indices = torch.randint(0, num_states, (num,), device=self.device)

        # Reset robot joints
        joint_pos = dataset["robot_joint_pos"][indices]
        self.robot.write_joint_state_to_sim(
            position=joint_pos,
            velocity=torch.zeros_like(joint_pos),
            env_ids=env_ids,
        )

        # Reset package pose
        pkg_pose = dataset["package_pose"][indices]  # (num, 7) = pos + quat
        self.package.write_root_pose_to_sim(pkg_pose, env_ids=env_ids)
        self.package.write_root_velocity_to_sim(
            torch.zeros(num, 6, device=self.device), env_ids=env_ids
        )

    def _reset_procedural(
        self,
        reset_type: int,
        env_ids: torch.Tensor,
        env: ManagerBasedEnv,
    ) -> None:
        """Generate reset states procedurally (no dataset needed)."""
        num = len(env_ids)
        device = self.device

        # Get reference positions
        scan_pos = self.scan_table.data.root_pos_w[env_ids[0]]  # (3,)
        box_pos = self.box.data.root_pos_w[env_ids[0]]  # (3,)

        if reset_type == 0:
            # PackageOnTable_EEFar: package on workspace table, EE at home
            self._place_package_on_workspace(env_ids, num, device)
            self._reset_arm_to_home(env_ids)

        elif reset_type == 1:
            # PackageOnTable_EENear: package on table, EE nearby
            pkg_pos = self._place_package_on_workspace(env_ids, num, device)
            self._move_arm_near_target(env_ids, pkg_pos, offset_range=0.15)

        elif reset_type == 2:
            # PackageGrasped_AboveWorkspace: package in gripper, above table
            # Place package between workspace and scan table
            mid_pos = torch.zeros(num, 3, device=device)
            mid_pos[:, 0] = torch.empty(num, device=device).uniform_(0.3, 0.5)
            mid_pos[:, 1] = torch.empty(num, device=device).uniform_(-0.3, -0.1)
            mid_pos[:, 2] = torch.empty(num, device=device).uniform_(1.3, 1.5)
            self._place_package_grasped(env_ids, num, device, mid_pos)

        elif reset_type == 3:
            # PackageOnScanTable_RandomOri: package on scan table, random orient
            self._place_package_on_scan_table(
                env_ids, num, device, scan_pos, random_orientation=True
            )
            self._move_arm_near_target(
                env_ids,
                scan_pos.unsqueeze(0).expand(num, -1),
                offset_range=0.2,
            )

        elif reset_type == 4:
            # PackageOnScanTable_BarcodeUp: package on scan, barcode up
            self._place_package_on_scan_table(
                env_ids, num, device, scan_pos, random_orientation=False
            )
            self._move_arm_near_target(
                env_ids,
                scan_pos.unsqueeze(0).expand(num, -1),
                offset_range=0.15,
            )

        elif reset_type == 5:
            # PackageGrasped_NearBox: package grasped, near box
            near_box_pos = torch.zeros(num, 3, device=device)
            near_box_pos[:, 0] = box_pos[0] + torch.empty(num, device=device).uniform_(-0.1, 0.1)
            near_box_pos[:, 1] = box_pos[1] + torch.empty(num, device=device).uniform_(-0.1, 0.1)
            near_box_pos[:, 2] = box_pos[2] + torch.empty(num, device=device).uniform_(0.1, 0.3)
            self._place_package_grasped(env_ids, num, device, near_box_pos)

        # Log metrics
        if "log" not in env.extras:
            env.extras["log"] = {}
        for rt in range(6):
            rate = (
                self.success_counts[rt] / (self.total_counts[rt] + 1)
            ).item()
            env.extras["log"][f"Metrics/reset_type_{rt}_success_rate"] = rate
            env.extras["log"][f"Metrics/reset_type_{rt}_prob"] = self.probs[rt].item()

    # ----------------------------------------------------------------
    # Procedural reset helpers
    # ----------------------------------------------------------------

    def _place_package_on_workspace(
        self,
        env_ids: torch.Tensor,
        num: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Place package randomly on the workspace table."""
        pos = torch.zeros(num, 3, device=device)
        pos[:, 0] = torch.empty(num, device=device).uniform_(0.35, 0.65)
        pos[:, 1] = torch.empty(num, device=device).uniform_(-0.3, 0.3)
        pos[:, 2] = 1.20

        yaw = torch.empty(num, device=device).uniform_(-3.14159, 3.14159)
        quat = torch.zeros(num, 4, device=device)
        quat[:, 0] = torch.cos(yaw / 2)
        quat[:, 3] = torch.sin(yaw / 2)

        self.package.write_root_pose_to_sim(
            torch.cat([pos, quat], dim=-1), env_ids=env_ids
        )
        self.package.write_root_velocity_to_sim(
            torch.zeros(num, 6, device=device), env_ids=env_ids
        )
        return pos

    def _place_package_on_scan_table(
        self,
        env_ids: torch.Tensor,
        num: int,
        device: torch.device,
        scan_pos: torch.Tensor,
        random_orientation: bool = True,
    ) -> None:
        """Place package on scanning table with optional orientation control."""
        pos = torch.zeros(num, 3, device=device)
        pos[:, 0] = scan_pos[0] + torch.empty(num, device=device).uniform_(-0.08, 0.08)
        pos[:, 1] = scan_pos[1] + torch.empty(num, device=device).uniform_(-0.08, 0.08)
        pos[:, 2] = scan_pos[2] + 0.05  # Slightly above table surface

        if random_orientation:
            # Random rotation around all axes
            roll = torch.empty(num, device=device).uniform_(-1.57, 1.57)
            pitch = torch.empty(num, device=device).uniform_(-1.57, 1.57)
            yaw = torch.empty(num, device=device).uniform_(-3.14159, 3.14159)
        else:
            # Barcode up: minimal roll/pitch, random yaw
            roll = torch.empty(num, device=device).uniform_(-0.2, 0.2)
            pitch = torch.empty(num, device=device).uniform_(-0.2, 0.2)
            yaw = torch.empty(num, device=device).uniform_(-3.14159, 3.14159)

        # Euler to quaternion (ZYX convention)
        quat = _euler_to_quat(roll, pitch, yaw, device)

        self.package.write_root_pose_to_sim(
            torch.cat([pos, quat], dim=-1), env_ids=env_ids
        )
        self.package.write_root_velocity_to_sim(
            torch.zeros(num, 6, device=device), env_ids=env_ids
        )

    def _place_package_grasped(
        self,
        env_ids: torch.Tensor,
        num: int,
        device: torch.device,
        target_pos: torch.Tensor,
    ) -> None:
        """Place package at target position as if grasped (close gripper)."""
        # Place package at target position with slight random orientation
        yaw = torch.empty(num, device=device).uniform_(-0.5, 0.5)
        quat = torch.zeros(num, 4, device=device)
        quat[:, 0] = torch.cos(yaw / 2)
        quat[:, 3] = torch.sin(yaw / 2)

        self.package.write_root_pose_to_sim(
            torch.cat([target_pos, quat], dim=-1), env_ids=env_ids
        )
        self.package.write_root_velocity_to_sim(
            torch.zeros(num, 6, device=device), env_ids=env_ids
        )

        # Close gripper
        gripper_joint_ids = self.gripper_cfg.joint_ids
        current_joint_pos = self.robot.data.joint_pos[env_ids].clone()
        current_joint_pos[:, gripper_joint_ids] = 0.0  # closed
        self.robot.write_joint_state_to_sim(
            position=current_joint_pos,
            velocity=torch.zeros_like(current_joint_pos),
            env_ids=env_ids,
        )

    def _reset_arm_to_home(self, env_ids: torch.Tensor) -> None:
        """Reset right arm to default home position."""
        self.robot.write_joint_state_to_sim(
            position=self.robot.data.default_joint_pos[env_ids].clone(),
            velocity=torch.zeros(
                len(env_ids),
                self.robot.data.joint_pos.shape[1],
                device=self.device,
            ),
            env_ids=env_ids,
        )

    def _move_arm_near_target(
        self,
        env_ids: torch.Tensor,
        target_pos: torch.Tensor,
        offset_range: float = 0.15,
    ) -> None:
        """Move arm near a target position by adding noise to default joints.

        In procedural mode, we perturb default joint positions slightly.
        Dataset mode should be used for precise IK-solved arm configurations.
        """
        num = len(env_ids)
        default_pos = self.robot.data.default_joint_pos[env_ids].clone()

        # Add small random perturbation to right arm joints (indices 12-18)
        # This is a rough approximation; dataset mode gives better results
        noise = torch.empty(num, 7, device=self.device).uniform_(
            -0.1, 0.1
        )
        # Right arm joint indices in G2 (idx61-idx67)
        if self.right_arm_joint_ids is None:
            joint_names = self.robot.data.joint_names
            self.right_arm_joint_ids = [
                i
                for i, name in enumerate(joint_names)
                if "arm_r_joint" in name
            ]

        for i, joint_idx in enumerate(self.right_arm_joint_ids):
            if i < 7:
                default_pos[:, joint_idx] += noise[:, i]

        # Clamp to joint limits to avoid invalid states
        lower = self.robot.data.soft_joint_pos_limits[0, :, 0]
        upper = self.robot.data.soft_joint_pos_limits[0, :, 1]
        default_pos = default_pos.clamp(min=lower, max=upper)

        self.robot.write_joint_state_to_sim(
            position=default_pos,
            velocity=torch.zeros_like(default_pos),
            env_ids=env_ids,
        )


def _euler_to_quat(
    roll: torch.Tensor,
    pitch: torch.Tensor,
    yaw: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Convert Euler angles (ZYX) to quaternion (wxyz)."""
    cr = torch.cos(roll / 2)
    sr = torch.sin(roll / 2)
    cp = torch.cos(pitch / 2)
    sp = torch.sin(pitch / 2)
    cy = torch.cos(yaw / 2)
    sy = torch.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return torch.stack([w, x, y, z], dim=-1)
