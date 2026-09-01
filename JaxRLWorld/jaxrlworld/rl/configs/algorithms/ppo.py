from dataclasses import dataclass, field

from ..base_config import BaseConfig


@dataclass
class SymmetryConfig(BaseConfig):
    """Left/right mirror-symmetry auxiliary loss (option A: mirror loss).

    Adds ``mirror_loss_coeff * MSE(pi(mirror(o)).mean, mirror(pi(o).mean))`` to
    the PPO loss, enforcing left/right equivariance of the policy. The mirror
    operator is built automatically from the observation layout + joint names
    (see rl.algorithms.ppo.symmetry). No obs/action space change.
    """

    use_mirror_loss: bool = False
    mirror_loss_coeff: float = 1.0


@dataclass
class PPOConfig(BaseConfig):
    algorithm_name: str = "PPO"
    clip_param: float = 0.2
    use_early_stop: bool = False
    desired_kl: float = 0.01
    entropy_coef: float = 0.01
    gamma: float = 0.99
    lam: float = 0.95
    actor_lr: float = 5e-4
    critic_lr: float = 5e-4
    estimator_learning_rate: float = 5e-4
    max_grad_norm: float = 0.5
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    schedule: str = "adaptive"
    # Adam epsilon. SB3's PPO pins 1e-5; optax's default is 1e-8.
    optimizer_eps: float = 1e-8
    use_clipped_value_loss: bool = False
    # Value-target normalization (skrl-style). When True, the critic learns
    # in normalized return space and outputs are inverse-normalized for GAE
    # / storage / bootstrap. When False (default), behavior is identical
    # to a pure PPO with no value normalization.
    use_value_normalization: bool = False
    # When True, normalize advantages within each minibatch (default, current behavior).
    # When False, normalize once per rollout in compute_returns (rsl_rl default).
    normalize_advantage_per_minibatch: bool = True
    value_loss_coef: float = 1.0
    use_truth_value_for_actor: bool = False
    use_truth_value_for_critic: bool = True
    use_barrier_style: bool = False
    use_sde: bool = True
    sde_sample_freq: int = 100
    learning_starts: int = 10_000
    num_steps_per_env: int = 24
    obs_normalization: bool = False
    # Default instance (not None) so from_dict/update_from_dict recurses into it
    # and restores a real SymmetryConfig on checkpoint load; use_mirror_loss=False
    # keeps the mirror loss off unless a preset turns it on.
    symmetry_cfg: SymmetryConfig = field(default_factory=SymmetryConfig)
