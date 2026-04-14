"""Evaluate a trained sorting packages policy.

Usage:
    ./uwlab.sh -p /home/zlj/AgiBot/genie_sim_RL/scripts/play.py \
        --task GenieSim-G2-SortingPackages-State-Eval-v0 \
        --checkpoint logs/g2_sorting_packages_XXXX/model_XXXX.pt \
        --num_envs 4
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

parser = argparse.ArgumentParser(description="Evaluate G2 Sorting Packages Policy")
parser.add_argument("--task", type=str, default="GenieSim-G2-SortingPackages-State-Eval-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_episodes", type=int, default=100)
parser.add_argument("--headless", action="store_true")
args, unknown = parser.parse_known_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import importlib

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import importlib
importlib.import_module("config")  # avoids cv2/config.py clash


def main():
    env = gym.make(args.task, num_envs=args.num_envs)
    env = RslRlVecEnvWrapper(env)

    agent_cfg_entry = gym.spec(args.task).kwargs["rsl_rl_cfg_entry_point"]
    module_path, class_name = agent_cfg_entry.rsplit(":", 1)
    module = importlib.import_module(module_path)
    agent_cfg = getattr(module, class_name)()

    runner = OnPolicyRunner(env, agent_cfg, log_dir="/tmp/play_logs", device="cuda:0")
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device="cuda:0")

    obs, _ = env.get_observations()
    total_episodes = 0
    total_stages = 0
    total_success = 0

    print(f"\nEvaluating for {args.num_episodes} episodes...")

    while total_episodes < args.num_episodes:
        with torch.no_grad():
            actions = policy(obs)
        obs, rewards, dones, infos = env.step(actions)

        if dones.any():
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            for idx in done_ids:
                tracker = env.unwrapped.reward_manager.get_term_cfg("stage_tracker").func
                stages = tracker.stage[idx].item()
                total_stages += stages
                total_success += int(stages >= 6)
                total_episodes += 1

                if total_episodes % 10 == 0:
                    print(f"  [{total_episodes}/{args.num_episodes}] "
                          f"Avg stages: {total_stages/total_episodes:.2f}/6, "
                          f"Success: {total_success/total_episodes*100:.1f}%")

                if total_episodes >= args.num_episodes:
                    break

    avg = total_stages / max(total_episodes, 1)
    rate = total_success / max(total_episodes, 1) * 100

    print(f"\n{'='*60}")
    print(f"  Results ({total_episodes} episodes)")
    print(f"  Avg stages: {avg:.2f} / 6")
    print(f"  Avg score:  {avg * 0.16:.3f} / 1.0")
    print(f"  Success:    {rate:.1f}%")
    print(f"{'='*60}\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
