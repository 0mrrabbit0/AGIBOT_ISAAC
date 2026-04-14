# GenieSim RL - Sorting Packages Task
# G2 dual-arm robot with OmniPicker gripper

import gymnasium as gym

from . import agents

# Training environment (vanilla resets)
gym.register(
    id="GenieSim-G2-SortingPackages-State-Train-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G2SortingPackagesTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:SortingPackagesPPORunnerCfg",
    },
)

# Training environment (OmniReset: diverse multi-stage resets)
gym.register(
    id="GenieSim-G2-SortingPackages-OmniReset-Train-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G2SortingPackagesOmniResetTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:SortingPackagesOmniResetPPORunnerCfg",
    },
)

# Evaluation environment
gym.register(
    id="GenieSim-G2-SortingPackages-State-Eval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G2SortingPackagesEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:SortingPackagesPPORunnerCfg",
    },
)
