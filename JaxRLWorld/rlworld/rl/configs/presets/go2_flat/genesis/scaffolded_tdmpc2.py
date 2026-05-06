from dataclasses import dataclass, field
from typing import Any, Dict, List

from rlworld.rl.configs.observations import ObservationTermConfig
from rlworld.rl.configs.presets.go2_flat.base import Go2FlatConfig
from rlworld.rl.envs.mdp.observations.genesis import proprioception, state


@dataclass
class Go2ScaffoldedTDMPC2Config(Go2FlatConfig):
    sim_type: str = "genesis"
    """
    Go2 flat terrain config for Scaffolded TD-MPC2 + ABD-Net.

    Differences from base:
      - algorithm_name = "ScaffoldedTDMPC2"
      - obs_group includes "privileged" (noise-free sim-only signals)
      - Algorithm config uses ScaffoldedTDMPC2Config fields
    """

    algorithm_name: str = "ScaffoldedTDMPC2"
    run_name: str = "Go2_ScaffoldedTDMPC2"
    max_iterations: int = 3000

    num_envs: int = 1

    # ---- ABD-Net ----
    latent_dim: int = 512
    mlp_dim: int = 512
    num_q: int = 5
    num_bins: int = 101
    link_channels: int = 8
    spatial_dim: int = 6
    learnable_contribution_weight: bool = False
    use_positive_constraint: bool = True
    residual_scale_init: float = 0.1

    # ---- Scaffolding ----
    explore_ratio: float = 0.5
    ortho_coef: float = 0.01

    # ---- TD-MPC2 training ----
    batch_size: int = 256
    buffer_size: int = 1_000_000
    learning_starts: int = 256
    num_gradient_steps: int = 1
    lr: float = 3e-4
    pi_lr: float = 3e-4
    tau: float = 0.01
    horizon: int = 3
    grad_clip_norm: float = 20.0
    max_grad_norm: float = 20.0

    # ---- Privileged obs settings ----
    privileged_feet_links: List[str] = field(default_factory=lambda: ["FR_foot", "FL_foot", "RR_foot", "RL_foot"])

    def __post_init__(self):
        super().__post_init__()

    def _build_privileged_terms(self) -> list[ObservationTermConfig]:
        """Privileged observations: noise-free sim-only ground truth."""
        return [
            # Ground truth base linear velocity (no noise)
            ObservationTermConfig(
                state.base_lin_vel,
                scale=2.0,
            ),
            # Ground truth base height
            ObservationTermConfig(
                state.base_height,
                scale=1.0,
            ),
            # Foot contact indicators
            ObservationTermConfig(
                state.contact_indicator,
                scale=1.0,
                params={
                    "entity_name": "robot",
                    "links": tuple(self.privileged_feet_links),
                },
            ),
            ObservationTermConfig(
                state.contact_force,
                scale=0.05,
                params={
                    "entity_name": "robot",
                    "links": tuple(self.privileged_feet_links),
                },
            ),
            ObservationTermConfig(state.links_acc, scale=0.005),
            ObservationTermConfig(state.actuated_dof_force, scale=0.01),
            # Ground truth DOF velocities (no noise)
            ObservationTermConfig(
                proprioception.dof_vel,
                scale=0.05,
            ),
            ObservationTermConfig(
                state.base_quat,
                scale=1.0,
            ),
            ObservationTermConfig(state.base_euler, scale=1.0),
        ]

    def _build_observation_config(self) -> Dict[str, Any]:
        return {
            "obs_group": {
                "actor": self.observations.to_terms(),
                "critic": self.observations.to_critic_terms(),
                "privileged": self._build_privileged_terms(),
            },
        }

    def _build_algorithm_config(self) -> Dict[str, Any]:
        return {
            "algorithm_name": self.algorithm_name,
            # Discount
            "gamma": 0.99,
            "episode_length": int(self.episode_length_s / (self.sim_dt * self.decimation)),
            "discount_min": 0.95,
            "discount_max": 0.995,
            "discount_denom": 5.0,
            # Learning rates
            "lr": self.lr,
            "pi_lr": self.pi_lr,
            # Target network
            "tau": self.tau,
            # Planning (MPPI)
            "mpc": True,
            "horizon": self.horizon,
            "num_samples": 512,
            "num_pi_trajs": 24,
            "num_elites": 64,
            "num_iterations": 6,
            "temperature": 0.5,
            "min_std": 0.05,
            "max_std": 0.8,
            # Loss coefficients
            "consistency_coef": 1.0,
            "reward_coef": 0.1,
            "value_coef": 0.1,
            "entropy_coef": 1e-4,
            "rho": 0.5,
            # Discrete regression
            "num_bins": self.num_bins,
            "vmin": -5.0,
            "vmax": 5.0,
            # World model architecture
            "latent_dim": self.latent_dim,
            "mlp_dim": self.mlp_dim,
            "num_q": self.num_q,
            "dropout": 0.01,
            # ABD-Net
            "link_channels": self.link_channels,
            "spatial_dim": self.spatial_dim,
            "learnable_contribution_weight": self.learnable_contribution_weight,
            "use_positive_constraint": self.use_positive_constraint,
            "residual_scale_init": self.residual_scale_init,
            # Scaffolding
            "explore_ratio": self.explore_ratio,
            "ortho_coef": self.ortho_coef,
            # Training
            "batch_size": self.batch_size,
            "buffer_size": self.buffer_size,
            "grad_clip_norm": self.grad_clip_norm,
            "max_grad_norm": self.max_grad_norm,
            "learning_starts": self.learning_starts,
            "num_gradient_steps": self.num_gradient_steps,
            "num_steps_per_env": 1,
        }

    def _build_nn_config(self) -> Dict[str, Any]:
        """Not used for TD-MPC2 (world model is built in algorithm)."""
        return {}

    def _build_runner_config(self) -> Dict[str, Any]:
        base = super()._build_runner_config()
        base.update(
            {
                "algorithm_class_name": self.algorithm_name,
                "run_name": self.run_name,
            }
        )
        return base


def get_config():
    return Go2ScaffoldedTDMPC2Config().to_dict()
