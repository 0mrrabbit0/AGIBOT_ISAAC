"""RSL-RL PPO agent configuration for sorting packages task.

Uses rsl_rl >= 4.0.0 config style with separate actor/critic models.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlMLPModelCfg


@configclass
class SortingPackagesPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48
    max_iterations = 50000
    save_interval = 200
    resume = False
    experiment_name = "g2_sorting_packages"

    obs_groups = {
        "actor": ["policy"],
        "critic": ["critic"],
    }

    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128, 64],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        noise_std_type="log",
        state_dependent_std=False,
    )

    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128, 64],
        activation="elu",
        obs_normalization=True,
        stochastic=False,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        normalize_advantage_per_mini_batch=False,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class SortingPackagesOmniResetPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config tuned for OmniReset diverse-reset training."""
    num_steps_per_env = 32
    max_iterations = 40000
    save_interval = 200
    resume = False
    experiment_name = "g2_sorting_packages_omnireset"

    obs_groups = {
        "actor": ["policy"],
        "critic": ["critic"],
    }

    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128, 64],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        noise_std_type="log",
        state_dependent_std=False,
    )

    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128, 64],
        activation="elu",
        obs_normalization=True,
        stochastic=False,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        normalize_advantage_per_mini_batch=False,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=1e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
