"""Generate diverse reset state datasets for Sorting Packages OmniReset training.

Produces 6 .pt files (one per reset type) containing robot joint states
and package poses. These are consumed by SortingMultiStageResetManager
in dataset mode for higher-quality resets than procedural generation.

Usage:
    # Generate all reset types (recommended):
    ./uwlab.sh -p scripts/record_sorting_reset_states.py \
        --num_envs 4096 --num_states 10000 --headless

    # Generate specific type only:
    ./uwlab.sh -p scripts/record_sorting_reset_states.py \
        --num_envs 2048 --num_states 5000 --reset_type 3 --headless

Output directory: PROJECT_ROOT/data/reset_states/
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

parser = argparse.ArgumentParser(description="Generate reset states for sorting packages")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--num_states", type=int, default=10000, help="States to collect per reset type")
parser.add_argument("--reset_type", type=int, default=-1, help="-1 for all, 0-5 for specific type")
parser.add_argument("--output_dir", type=str, default=os.path.join(PROJECT_ROOT, "data", "reset_states"))
parser.add_argument("--headless", action="store_true")
parser.add_argument("--settle_steps", type=int, default=50, help="Physics steps to let scene settle")
args, unknown = parser.parse_known_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import config  # noqa: F401


RESET_TYPE_NAMES = [
    "PackageOnTable_EEFar",
    "PackageOnTable_EENear",
    "PackageGrasped_AboveWorkspace",
    "PackageOnScanTable_RandomOri",
    "PackageOnScanTable_BarcodeUp",
    "PackageGrasped_NearBox",
]


def generate_reset_states(
    env,
    reset_type: int,
    num_states: int,
    settle_steps: int,
) -> dict[str, torch.Tensor]:
    """Generate reset states for a given type by randomizing and settling."""
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    robot = env.unwrapped.scene["robot"]
    package = env.unwrapped.scene["package"]
    scan_table = env.unwrapped.scene["scanning_table"]
    box = env.unwrapped.scene["target_box"]

    scan_pos = scan_table.data.root_pos_w[0]
    box_pos = box.data.root_pos_w[0]

    collected_joint_pos = []
    collected_pkg_pose = []
    collected_stages = []

    print(f"\n  Generating reset type {reset_type}: {RESET_TYPE_NAMES[reset_type]}")
    print(f"  Target: {num_states} states, batch size: {num_envs}")

    while len(collected_joint_pos) * num_envs < num_states:
        all_ids = torch.arange(num_envs, device=device)

        # Reset to default first
        robot.write_joint_state_to_sim(
            position=robot.data.default_joint_pos.clone(),
            velocity=torch.zeros_like(robot.data.joint_vel),
            env_ids=all_ids,
        )

        # Apply reset-type-specific randomization
        _apply_reset_randomization(
            reset_type, all_ids, num_envs, device,
            robot, package, scan_pos, box_pos,
        )

        # Settle physics
        for _ in range(settle_steps):
            env.unwrapped.sim.step(render=False)

        # Validate and collect stable states
        pkg_pos = package.data.root_pos_w
        pkg_z = pkg_pos[:, 2]
        joint_vel = robot.data.joint_vel

        # Filter: package above ground, no excessive velocities, no NaN
        valid = (
            (pkg_z > 0.5)
            & (~torch.isnan(pkg_pos).any(dim=-1))
            & (~torch.isnan(robot.data.joint_pos).any(dim=-1))
            & (joint_vel.abs().max(dim=-1).values < 5.0)
        )

        if valid.any():
            valid_ids = valid.nonzero(as_tuple=False).squeeze(-1)
            collected_joint_pos.append(robot.data.joint_pos[valid_ids].cpu())
            collected_pkg_pose.append(
                torch.cat([
                    package.data.root_pos_w[valid_ids],
                    package.data.root_quat_w[valid_ids],
                ], dim=-1).cpu()
            )
            collected_stages.append(
                torch.full((len(valid_ids),), reset_type, dtype=torch.long)
            )

        total = sum(t.shape[0] for t in collected_joint_pos)
        print(f"    Collected {total}/{num_states} valid states", end="\r")

    # Concatenate and trim
    joint_pos = torch.cat(collected_joint_pos, dim=0)[:num_states]
    pkg_pose = torch.cat(collected_pkg_pose, dim=0)[:num_states]
    stages = torch.cat(collected_stages, dim=0)[:num_states]

    print(f"    Collected {num_states}/{num_states} valid states  [DONE]")

    return {
        "robot_joint_pos": joint_pos,
        "package_pose": pkg_pose,
        "reset_type": reset_type,
        "reset_type_name": RESET_TYPE_NAMES[reset_type],
        "starting_stage": stages,
    }


def _apply_reset_randomization(
    reset_type: int,
    env_ids: torch.Tensor,
    num: int,
    device: torch.device,
    robot,
    package,
    scan_pos: torch.Tensor,
    box_pos: torch.Tensor,
) -> None:
    """Apply reset-type-specific randomization to the scene."""

    if reset_type == 0:
        # PackageOnTable_EEFar: package random on workspace, arm at home
        _random_package_on_workspace(package, env_ids, num, device)

    elif reset_type == 1:
        # PackageOnTable_EENear: package on workspace, arm slightly perturbed
        _random_package_on_workspace(package, env_ids, num, device)
        _perturb_right_arm(robot, env_ids, num, device, magnitude=0.3)

    elif reset_type == 2:
        # PackageGrasped_AboveWorkspace: package in air between tables
        pos = torch.zeros(num, 3, device=device)
        pos[:, 0] = torch.empty(num, device=device).uniform_(0.3, 0.5)
        pos[:, 1] = torch.empty(num, device=device).uniform_(-0.35, -0.05)
        pos[:, 2] = torch.empty(num, device=device).uniform_(1.3, 1.5)
        _place_package_at(package, env_ids, num, device, pos, yaw_range=(-0.5, 0.5))
        _perturb_right_arm(robot, env_ids, num, device, magnitude=0.5)
        _close_gripper(robot, env_ids, device)

    elif reset_type == 3:
        # PackageOnScanTable_RandomOri: on scan table, random orientation
        pos = torch.zeros(num, 3, device=device)
        pos[:, 0] = scan_pos[0] + torch.empty(num, device=device).uniform_(-0.08, 0.08)
        pos[:, 1] = scan_pos[1] + torch.empty(num, device=device).uniform_(-0.08, 0.08)
        pos[:, 2] = scan_pos[2] + 0.06
        _place_package_at(
            package, env_ids, num, device, pos,
            roll_range=(-1.5, 1.5), pitch_range=(-1.5, 1.5),
        )
        _perturb_right_arm(robot, env_ids, num, device, magnitude=0.3)

    elif reset_type == 4:
        # PackageOnScanTable_BarcodeUp: on scan table, near upright
        pos = torch.zeros(num, 3, device=device)
        pos[:, 0] = scan_pos[0] + torch.empty(num, device=device).uniform_(-0.08, 0.08)
        pos[:, 1] = scan_pos[1] + torch.empty(num, device=device).uniform_(-0.08, 0.08)
        pos[:, 2] = scan_pos[2] + 0.06
        _place_package_at(
            package, env_ids, num, device, pos,
            roll_range=(-0.2, 0.2), pitch_range=(-0.2, 0.2),
        )
        _perturb_right_arm(robot, env_ids, num, device, magnitude=0.3)

    elif reset_type == 5:
        # PackageGrasped_NearBox: grasped, near box
        pos = torch.zeros(num, 3, device=device)
        pos[:, 0] = box_pos[0] + torch.empty(num, device=device).uniform_(-0.1, 0.1)
        pos[:, 1] = box_pos[1] + torch.empty(num, device=device).uniform_(-0.1, 0.1)
        pos[:, 2] = box_pos[2] + torch.empty(num, device=device).uniform_(0.1, 0.3)
        _place_package_at(package, env_ids, num, device, pos, yaw_range=(-0.5, 0.5))
        _perturb_right_arm(robot, env_ids, num, device, magnitude=0.5)
        _close_gripper(robot, env_ids, device)


# ----------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------

def _random_package_on_workspace(package, env_ids, num, device):
    pos = torch.zeros(num, 3, device=device)
    pos[:, 0] = torch.empty(num, device=device).uniform_(0.35, 0.65)
    pos[:, 1] = torch.empty(num, device=device).uniform_(-0.3, 0.3)
    pos[:, 2] = 1.20
    yaw = torch.empty(num, device=device).uniform_(-3.14159, 3.14159)
    quat = torch.zeros(num, 4, device=device)
    quat[:, 0] = torch.cos(yaw / 2)
    quat[:, 3] = torch.sin(yaw / 2)
    package.write_root_pose_to_sim(
        torch.cat([pos, quat], dim=-1), env_ids=env_ids
    )
    package.write_root_velocity_to_sim(
        torch.zeros(num, 6, device=device), env_ids=env_ids
    )


def _place_package_at(
    package, env_ids, num, device, pos,
    roll_range=(-0.0, 0.0), pitch_range=(-0.0, 0.0), yaw_range=(-3.14159, 3.14159),
):
    roll = torch.empty(num, device=device).uniform_(*roll_range)
    pitch = torch.empty(num, device=device).uniform_(*pitch_range)
    yaw = torch.empty(num, device=device).uniform_(*yaw_range)

    cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
    cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
    cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    quat = torch.stack([w, x, y, z], dim=-1)

    package.write_root_pose_to_sim(
        torch.cat([pos, quat], dim=-1), env_ids=env_ids
    )
    package.write_root_velocity_to_sim(
        torch.zeros(num, 6, device=device), env_ids=env_ids
    )


def _perturb_right_arm(robot, env_ids, num, device, magnitude=0.3):
    """Add random noise to right arm joints."""
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_names = robot.data.joint_names
    right_arm_ids = [
        i for i, name in enumerate(joint_names) if "arm_r_joint" in name
    ]
    noise = torch.empty(num, len(right_arm_ids), device=device).uniform_(
        -magnitude, magnitude
    )
    for i, jid in enumerate(right_arm_ids):
        joint_pos[:, jid] += noise[:, i]

    robot.write_joint_state_to_sim(
        position=joint_pos,
        velocity=torch.zeros_like(joint_pos),
        env_ids=env_ids,
    )


def _close_gripper(robot, env_ids, device):
    """Set gripper to closed position."""
    joint_names = robot.data.joint_names
    gripper_ids = [
        i for i, name in enumerate(joint_names)
        if "gripper_r_outer_joint" in name
    ]
    joint_pos = robot.data.joint_pos[env_ids].clone()
    for gid in gripper_ids:
        joint_pos[:, gid] = 0.0
    robot.write_joint_state_to_sim(
        position=joint_pos,
        velocity=torch.zeros_like(joint_pos),
        env_ids=env_ids,
    )


def main():
    os.makedirs(args.output_dir, exist_ok=True)

    env = gym.make(
        "GenieSim-G2-SortingPackages-State-Train-v0",
        num_envs=args.num_envs,
    )

    types_to_generate = (
        range(6) if args.reset_type == -1 else [args.reset_type]
    )

    print(f"\n{'='*60}")
    print(f"  Generating Sorting Packages Reset States")
    print(f"  Output dir:    {args.output_dir}")
    print(f"  Num envs:      {args.num_envs}")
    print(f"  States/type:   {args.num_states}")
    print(f"  Settle steps:  {args.settle_steps}")
    print(f"  Reset types:   {list(types_to_generate)}")
    print(f"{'='*60}")

    for rt in types_to_generate:
        states = generate_reset_states(
            env, rt, args.num_states, args.settle_steps
        )
        output_path = os.path.join(
            args.output_dir, f"resets_{RESET_TYPE_NAMES[rt]}.pt"
        )
        torch.save(states, output_path)
        print(f"  Saved: {output_path} ({args.num_states} states)")

    print(f"\n{'='*60}")
    print("  All reset states generated successfully!")
    print(f"{'='*60}\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
