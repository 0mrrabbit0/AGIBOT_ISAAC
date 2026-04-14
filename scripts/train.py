"""Training script for G2 Sorting Packages RL task."""

from __future__ import annotations

import argparse
import os
import sys

# Ensure our project is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ---- Parse args before Isaac Sim launch ----
parser = argparse.ArgumentParser(description="Train G2 Sorting Packages")
parser.add_argument("--task", type=str, default="GenieSim-G2-SortingPackages-State-Train-v0")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=50000)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--distributed", action="store_true")
parser.add_argument("--resume_path", type=str, default=None)
parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"])
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--omnireset", action="store_true",
                    help="Shortcut for OmniReset task (overrides --task)")
args, unknown = parser.parse_known_args()

if args.omnireset:
    args.task = "GenieSim-G2-SortingPackages-OmniReset-Train-v0"

# ---- Launch Isaac Sim ----
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=args.headless, distributed=args.distributed)
simulation_app = app_launcher.app

# ---- Post-launch imports ----
import gymnasium as gym
import torch
from datetime import datetime

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner

# Register our environments - use explicit path-based import to avoid cv2/config clash
import importlib.util
_config_spec = importlib.util.spec_from_file_location(
    "config", os.path.join(PROJECT_ROOT, "config", "__init__.py"),
    submodule_search_locations=[os.path.join(PROJECT_ROOT, "config")]
)
_config_mod = importlib.util.module_from_spec(_config_spec)
sys.modules["config"] = _config_mod
_config_spec.loader.exec_module(_config_mod)


def main():
    # ---- Load env config from registry ----
    env_cfg_entry = gym.spec(args.task).kwargs["env_cfg_entry_point"]
    module_path, class_name = env_cfg_entry.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    env_cfg = getattr(mod, class_name)()
    env_cfg.scene.num_envs = args.num_envs

    # ---- Build env ----
    env = gym.make(args.task, cfg=env_cfg)

    # ---- Agent config ----
    agent_cfg_entry = gym.spec(args.task).kwargs["rsl_rl_cfg_entry_point"]
    module_path, class_name = agent_cfg_entry.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    agent_cfg = getattr(mod, class_name)()
    agent_cfg.max_iterations = args.max_iterations

    if args.resume_path:
        agent_cfg.resume = True

    # ---- Wrap environment ----
    env = RslRlVecEnvWrapper(env)

    # ---- Log directory ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(PROJECT_ROOT, "logs", f"{agent_cfg.experiment_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    # ---- Runner ----
    # Handle deprecated config (converts policy -> actor/critic for rsl_rl >= 4.0)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, "5.0.1")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    if args.resume_path:
        runner.load(args.resume_path)
        print(f"[INFO] Resumed from {args.resume_path}")

    # ---- Train ----
    sep = "=" * 60
    print()
    print(sep)
    print(f"  Task:           {args.task}")
    print(f"  Num envs:       {args.num_envs}")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  Log dir:        {log_dir}")
    print(f"  Distributed:    {args.distributed}")
    print(sep)
    print()

    runner.learn(num_learning_iterations=args.max_iterations)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
