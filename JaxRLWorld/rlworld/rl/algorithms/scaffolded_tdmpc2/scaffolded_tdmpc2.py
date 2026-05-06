"""
Scaffolded TD-MPC2 with ABD-Net dynamics.

Follows rlworld.rl.algorithms.tdmpc2.tdmpc2 exactly:
- Extends TDMPC2
- update() returns ScaffoldedTDMPC2Metrics | None
- Same RunningScale, TDMPC2TrainState pattern
- Same checkpointing interface
"""

from typing import TYPE_CHECKING, Any, Dict, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from rlworld.rl.algorithms.base import copy_params
from rlworld.rl.algorithms.metrics import BatchMetrics
from rlworld.rl.algorithms.tdmpc2 import TDMPC2
from rlworld.rl.algorithms.tdmpc2.tdmpc2 import TDMPC2TrainState

if TYPE_CHECKING:
    from rlworld.rl.configs.robots.kinematic_tree import KinematicTree

from rlworld.rl.algorithms.scaffolded_tdmpc2.metrics import (
    ExplorationPolicyMetrics,
    QMetrics,
    ScaffoldedPolicyMetrics,
    ScaffoldedTDMPC2Metrics,
    ScaffoldedWorldModelMetrics,
)
from rlworld.rl.algorithms.scaffolded_tdmpc2.scaffolded_world_model import (
    ScaffoldedWorldModel,
)
from rlworld.rl.algorithms.scaffolded_tdmpc2.update import (
    scaffolded_unified_update,
)
from rlworld.rl.modules.policies.abd_world_model import ABDNetWorldModel
from rlworld.rl.storages.scaffolded_replay_buffer import (
    ScaffoldedSequenceBatch,
    ScaffoldedSequenceReplayBuffer,
)

# ==================== Train State ====================


class ScaffoldedTrainState(NamedTuple):
    """Full training state for scaffolded TD-MPC2."""

    # Target model
    target_model: ABDNetWorldModel
    target_q_params: Any
    target_wm_opt_state: optax.OptState
    pi_opt_state: optax.OptState
    # Scaffolded model
    scaffolded_model: ScaffoldedWorldModel
    scaff_target_q_params: Any
    scaff_wm_opt_state: optax.OptState
    explore_pi_opt_state: optax.OptState
    # Key
    key: jax.Array


# ==================== Algorithm ====================


class ScaffoldedTDMPC2(TDMPC2):
    """
    TD-MPC2 with ABD-Net dynamics and sensory scaffolding.

    Target: ABDNetWorldModel (ABD-Net dynamics on s-, no encoder)
    Scaffolded: ScaffoldedWorldModel (ABD-Net dynamics on s+, no encoder)
    Policy: trained with scaffolded critic (asymmetric actor-critic)
    Exploration: operates in s+ space for data collection
    """

    def __init__(
        self,
        kinematic_tree: "KinematicTree",
        obs_dim: int,
        action_dim: int,
        privileged_obs_dim: int,
        num_envs: int,
        # Base TD-MPC2 params
        gamma: float = 0.99,
        episode_length: int = 1000,
        discount_min: float = 0.95,
        discount_max: float = 0.995,
        discount_denom: float = 5.0,
        lr: float = 3e-4,
        pi_lr: float = 3e-4,
        tau: float = 0.01,
        mpc: bool = True,
        horizon: int = 3,
        num_samples: int = 512,
        num_pi_trajs: int = 24,
        num_elites: int = 64,
        num_iterations: int = 6,
        temperature: float = 0.5,
        min_std: float = 0.05,
        max_std: float = 2.0,
        consistency_coef: float = 2.0,
        reward_coef: float = 0.5,
        value_coef: float = 0.1,
        entropy_coef: float = 1e-4,
        rho: float = 0.5,
        num_bins: int = 101,
        vmin: float = -10.0,
        vmax: float = 10.0,
        batch_size: int = 256,
        grad_clip_norm: float = 20.0,
        max_grad_norm: float = 20.0,
        # Shared MLP params
        mlp_dim: int = 512,
        num_q: int = 5,
        # ABD-Net params
        link_channels: int = 8,
        spatial_dim: int = 6,
        learnable_contribution_weight: bool = False,
        use_positive_constraint: bool = True,
        residual_scale_init: float = 0.1,
        simnorm_dim: int = 8,
        ortho_coef: float = 0.01,
        # Scaffolding params
        explore_ratio: float = 0.5,
        key: jax.Array = None,
        **kwargs,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)

        self.kinematic_tree = kinematic_tree
        self.privileged_obs_dim = privileged_obs_dim
        self.explore_ratio = explore_ratio
        self.ortho_coef = ortho_coef
        self.simnorm_dim = simnorm_dim

        # Build target ABDNetWorldModel
        key, target_key, scaff_key = jax.random.split(key, 3)

        target_world_model = ABDNetWorldModel(
            kinematic_tree=kinematic_tree,
            obs_dim=obs_dim,
            action_dim=action_dim,
            mlp_dim=mlp_dim,
            num_q=num_q,
            num_bins=num_bins,
            dropout=0.01,
            link_channels=link_channels,
            spatial_dim=spatial_dim,
            learnable_contribution_weight=learnable_contribution_weight,
            use_positive_constraint=use_positive_constraint,
            residual_scale_init=residual_scale_init,
            simnorm_dim=simnorm_dim,
            key=target_key,
        )

        # Initialize base TDMPC2 with target model
        # (sets up self.train_state, self.wm_optimizer, self.pi_optimizer, etc.)
        super().__init__(
            world_model=target_world_model,
            num_envs=num_envs,
            gamma=gamma,
            episode_length=episode_length,
            discount_min=discount_min,
            discount_max=discount_max,
            discount_denom=discount_denom,
            lr=lr,
            pi_lr=pi_lr,
            tau=tau,
            mpc=mpc,
            horizon=horizon,
            num_samples=num_samples,
            num_pi_trajs=num_pi_trajs,
            num_elites=num_elites,
            num_iterations=num_iterations,
            temperature=temperature,
            min_std=min_std,
            max_std=max_std,
            consistency_coef=consistency_coef,
            reward_coef=reward_coef,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            rho=rho,
            num_bins=num_bins,
            vmin=vmin,
            vmax=vmax,
            batch_size=batch_size,
            grad_clip_norm=grad_clip_norm,
            max_grad_norm=max_grad_norm,
            key=key,
            **kwargs,
        )

        # Build scaffolded world model
        self.scaffolded_world_model = ScaffoldedWorldModel(
            kinematic_tree=kinematic_tree,
            target_obs_dim=obs_dim,
            privileged_obs_dim=privileged_obs_dim,
            action_dim=action_dim,
            mlp_dim=mlp_dim,
            num_q=num_q,
            num_bins=num_bins,
            dropout=0.01,
            link_channels=link_channels,
            spatial_dim=spatial_dim,
            learnable_contribution_weight=learnable_contribution_weight,
            use_positive_constraint=use_positive_constraint,
            residual_scale_init=residual_scale_init,
            simnorm_dim=simnorm_dim,
            key=scaff_key,
        )

        # Create scaffolded optimizers and train state
        self._create_scaffolded_optimizers()
        self._init_scaffolded_train_state(self.train_state.key)

        print("\n  Scaffolded TD-MPC2 + ABD-Net:")
        print(f"  Target obs dim: {obs_dim}")
        print(f"  Privileged obs dim: {privileged_obs_dim}")
        print(f"  Scaffolded dim: {obs_dim + privileged_obs_dim}")
        print(f"  ABD-Net bodies: {kinematic_tree.num_bodies}")
        print(f"  Link channels: {link_channels}, Spatial dim: {spatial_dim}")
        print(f"  Explore ratio: {explore_ratio}")
        print(f"  Ortho coef: {ortho_coef}")

    # ==================== Optimizer Overrides ====================

    def _create_optimizers(self, model):
        """
        Override: target WM optimizer without encoder lr group.
        ABDNetWorldModel has no encoder, so no "encoder" label needed.
        """
        param_filter = eqx.is_inexact_array

        all_params = eqx.filter(model, param_filter)
        labels = jax.tree.map(lambda _: "rest", all_params)

        # Freeze policy in WM optimizer (same as base)
        policy_labels = jax.tree.map(
            lambda _: "frozen",
            eqx.filter(model.policy, param_filter),
        )
        labels = eqx.tree_at(lambda m: m.policy, labels, policy_labels)

        self.wm_optimizer = optax.chain(
            optax.clip_by_global_norm(self.grad_clip_norm),
            optax.multi_transform(
                transforms={
                    "rest": optax.adam(learning_rate=self.lr),
                    "frozen": optax.set_to_zero(),
                },
                param_labels=labels,
            ),
        )

        # Policy optimizer (same as base)
        self.pi_optimizer = optax.chain(
            optax.clip_by_global_norm(self.grad_clip_norm),
            optax.adam(learning_rate=self.pi_lr, eps=1e-5),
        )

    def _create_scaffolded_optimizers(self):
        """Create optimizers for scaffolded model."""
        param_filter = eqx.is_inexact_array

        # Scaffolded WM optimizer (freeze exploration policy)
        all_params = eqx.filter(self.scaffolded_world_model, param_filter)
        labels = jax.tree.map(lambda _: "rest", all_params)

        explore_labels = jax.tree.map(
            lambda _: "frozen",
            eqx.filter(self.scaffolded_world_model.exploration_policy, param_filter),
        )
        labels = eqx.tree_at(lambda m: m.exploration_policy, labels, explore_labels)

        self.scaff_wm_optimizer = optax.chain(
            optax.clip_by_global_norm(self.grad_clip_norm),
            optax.multi_transform(
                transforms={
                    "rest": optax.adam(learning_rate=self.lr),
                    "frozen": optax.set_to_zero(),
                },
                param_labels=labels,
            ),
        )

        # Exploration policy optimizer
        self.explore_pi_optimizer = optax.chain(
            optax.clip_by_global_norm(self.grad_clip_norm),
            optax.adam(learning_rate=self.pi_lr, eps=1e-5),
        )

    def _init_scaffolded_train_state(self, key: jax.Array):
        """Initialize full scaffolded train state."""
        key, subkey = jax.random.split(key)

        # Scaffolded target Q params
        scaff_q_params, _ = eqx.partition(self.scaffolded_world_model.q_ensemble, eqx.is_inexact_array)
        scaff_target_q_params = copy_params(scaff_q_params)

        # Scaffolded WM optimizer state
        scaff_wm_opt_state = self.scaff_wm_optimizer.init(eqx.filter(self.scaffolded_world_model, eqx.is_inexact_array))

        # Exploration policy optimizer state
        explore_params, _ = eqx.partition(
            self.scaffolded_world_model.exploration_policy,
            eqx.is_inexact_array,
        )
        explore_pi_opt_state = self.explore_pi_optimizer.init(explore_params)

        self.scaffolded_train_state = ScaffoldedTrainState(
            target_model=self.train_state.model,
            target_q_params=self.train_state.target_q_params,
            target_wm_opt_state=self.train_state.wm_opt_state,
            pi_opt_state=self.train_state.pi_opt_state,
            scaffolded_model=self.scaffolded_world_model,
            scaff_target_q_params=scaff_target_q_params,
            scaff_wm_opt_state=scaff_wm_opt_state,
            explore_pi_opt_state=explore_pi_opt_state,
            key=subkey,
        )

    # ==================== Properties ====================

    @property
    def target_model(self) -> ABDNetWorldModel:
        return self.scaffolded_train_state.target_model

    @property
    def model(self) -> ABDNetWorldModel:
        return self.scaffolded_train_state.target_model

    # ==================== Storage ====================

    def init_storage(self, cfg: Dict[str, Any]) -> None:
        self.replay_buffer = ScaffoldedSequenceReplayBuffer(
            num_envs=cfg["num_envs"],
            obs_dim=cfg["obs_dim"],
            action_dim=cfg["action_dim"],
            privileged_obs_dim=cfg["privileged_obs_dim"],
            size_per_env=cfg["size_per_env"],
            horizon=self.horizon,
        )

    def store_transition(
        self,
        obs,
        action,
        reward,
        next_obs,
        terminated,
        truncated,
        privileged_obs=None,
        next_privileged_obs=None,
    ) -> None:
        self.replay_buffer.store_parallel(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            terminated=terminated,
            truncated=truncated,
            privileged_obs=privileged_obs,
            next_privileged_obs=next_privileged_obs,
        )

    # ==================== Acting ====================

    def act_explore(
        self,
        scaffolded_obs: jax.Array,
        *,
        key: jax.Array,
    ) -> jax.Array:
        """Exploration policy on s+. Training-time only."""
        scaff_model = self.scaffolded_train_state.scaffolded_model
        action, _ = scaff_model.pi_explore(scaffolded_obs, key=key)
        return action

    # ==================== Update ====================

    def update(
        self,
        batch: ScaffoldedSequenceBatch,
        build_metrics: bool = False,
    ) -> ScaffoldedTDMPC2Metrics | None:
        """
        Full scaffolded update. Returns ScaffoldedTDMPC2Metrics (dataclass).
        Follows TDMPC2.update() pattern exactly.
        """
        self.total_it += 1
        key = self.scaffolded_train_state.key
        key, update_key = jax.random.split(key)

        results = scaffolded_unified_update(
            target_model=self.scaffolded_train_state.target_model,
            target_wm_opt_state=self.scaffolded_train_state.target_wm_opt_state,
            target_wm_optimizer=self.wm_optimizer,
            target_q_params=self.scaffolded_train_state.target_q_params,
            scaffolded_model=self.scaffolded_train_state.scaffolded_model,
            scaff_wm_opt_state=self.scaffolded_train_state.scaff_wm_opt_state,
            scaff_wm_optimizer=self.scaff_wm_optimizer,
            scaff_target_q_params=self.scaffolded_train_state.scaff_target_q_params,
            pi_opt_state=self.scaffolded_train_state.pi_opt_state,
            pi_optimizer=self.pi_optimizer,
            explore_pi_opt_state=self.scaffolded_train_state.explore_pi_opt_state,
            explore_pi_optimizer=self.explore_pi_optimizer,
            tau=self.tau,
            batch=batch,
            two_hot_cfg=self.two_hot_cfg,
            discount=self.discount,
            rho=self.rho,
            entropy_coef=self.entropy_coef,
            consistency_coef=self.consistency_coef,
            reward_coef=self.reward_coef,
            value_coef=self.value_coef,
            ortho_coef=self.ortho_coef,
            scale_value=jnp.array(self.scale.value),
            scale_tau=self.scale.tau,
            key=update_key,
        )

        (
            new_target_model,
            new_twm_opt,
            new_target_q,
            new_scaff_model,
            new_swm_opt,
            new_scaff_target_q,
            new_pi_opt,
            new_explore_opt,
            info,
        ) = results

        # Update running scale (same as TDMPC2.update)
        self.scale._value = float(info.updated_scale_value)

        # Update scaffolded train state
        self.scaffolded_train_state = ScaffoldedTrainState(
            target_model=new_target_model,
            target_q_params=new_target_q,
            target_wm_opt_state=new_twm_opt,
            pi_opt_state=new_pi_opt,
            scaffolded_model=new_scaff_model,
            scaff_target_q_params=new_scaff_target_q,
            scaff_wm_opt_state=new_swm_opt,
            explore_pi_opt_state=new_explore_opt,
            key=key,
        )

        # Update base train state for compatibility (act_with_t0, checkpoint)
        self.train_state = TDMPC2TrainState(
            model=new_target_model,
            target_q_params=new_target_q,
            wm_opt_state=new_twm_opt,
            pi_opt_state=new_pi_opt,
            key=key,
        )

        if not build_metrics:
            return None

        return ScaffoldedTDMPC2Metrics(
            target_wm=ScaffoldedWorldModelMetrics(
                consistency_loss=float(info.target_consistency_loss),
                reward_loss=float(info.target_reward_loss),
                value_loss=float(info.target_value_loss),
                ortho_loss=float(info.target_ortho_loss),
                total_loss=float(info.target_total_loss),
                grad_norm=float(info.target_wm_grad_norm),
            ),
            scaff_wm=ScaffoldedWorldModelMetrics(
                consistency_loss=float(info.scaff_consistency_loss),
                reward_loss=float(info.scaff_reward_loss),
                value_loss=float(info.scaff_value_loss),
                ortho_loss=float(info.scaff_ortho_loss),
                total_loss=float(info.scaff_total_loss),
                grad_norm=float(info.scaff_wm_grad_norm),
            ),
            policy=ScaffoldedPolicyMetrics(
                pi_loss=float(info.pi_loss),
                pi_entropy=float(info.pi_entropy),
                pi_scaled_entropy=float(info.pi_scaled_entropy),
                pi_grad_norm=float(info.pi_grad_norm),
                pi_scale=self.scale.value,
            ),
            explore=ExplorationPolicyMetrics(
                pi_loss=float(info.explore_pi_loss),
                entropy=float(info.explore_entropy),
            ),
            q=QMetrics(
                mean=float(info.q_mean),
                std=float(info.q_std),
                p05=float(info.q_p05),
                p95=float(info.q_p95),
            ),
            target_q=QMetrics(
                mean=float(info.target_q_mean),
                std=float(info.target_q_std),
                p05=float(info.target_q_p05),
                p95=float(info.target_q_p95),
            ),
            batch=BatchMetrics(
                return_mean=float(batch.rewards.mean()),
                return_std=float(batch.rewards.std()),
                return_min=float(batch.rewards.min()),
                return_max=float(batch.rewards.max()),
            ),
            total_updates=self.total_it,
        )

    # ==================== Checkpointing ====================

    def save_train_state(self, checkpoint_dir: str) -> Dict[str, Any]:
        import os

        eqx.tree_serialise_leaves(
            os.path.join(checkpoint_dir, "model.eqx"),
            self.scaffolded_train_state.target_model,
        )
        eqx.tree_serialise_leaves(
            os.path.join(checkpoint_dir, "target_q.eqx"),
            self.scaffolded_train_state.target_q_params,
        )
        eqx.tree_serialise_leaves(
            os.path.join(checkpoint_dir, "scaffolded_model.eqx"),
            self.scaffolded_train_state.scaffolded_model,
        )
        eqx.tree_serialise_leaves(
            os.path.join(checkpoint_dir, "scaff_target_q.eqx"),
            self.scaffolded_train_state.scaff_target_q_params,
        )

        return {
            "alg_class": self.__class__.__name__,
            "alg_key": np.array(self.scaffolded_train_state.key),
            "lr": self.lr,
            "pi_lr": self.pi_lr,
            "total_it": self.total_it,
            "scale_value": self.scale.value,
            "prev_mean": self._prev_mean.tolist(),
            "privileged_obs_dim": self.privileged_obs_dim,
            "ortho_coef": self.ortho_coef,
            "explore_ratio": self.explore_ratio,
        }
