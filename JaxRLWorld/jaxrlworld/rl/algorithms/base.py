from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict

import equinox as eqx
import jax
import optax

# ==================== Common Data Structures ====================


@dataclass
class ActInput:
    """Standard observation input format for act methods."""

    actor_obs: jax.Array
    critic_obs: jax.Array


# ==================== Optimizer Utilities ====================


def create_optimizer_with_labels(
    model: eqx.Module,
    label_fn: Callable[[Any], str],
    lr_config: Dict[str, float],
    max_grad_norm: float = 1.0,
    optimizer_class: Callable = optax.adamw,
) -> tuple[optax.GradientTransformation, Any]:
    """
    Create optimizer with separate learning rates for different parameter groups.

    Args:
        model: Equinox model
        label_fn: Function that maps parameter path to label string
        lr_config: Dict mapping label -> learning rate
        max_grad_norm: Maximum gradient norm for clipping
        optimizer_class: Optax optimizer class (default: adamw)

    Returns:
        Tuple of (optimizer, param_labels)
    """
    params, _ = eqx.partition(model, eqx.is_inexact_array)

    def get_labels(tree):
        flat, treedef = jax.tree_util.tree_flatten_with_path(tree)
        labels = [label_fn(path) for path, _ in flat]
        return jax.tree_util.tree_unflatten(treedef, labels)

    param_labels = get_labels(params)

    # ``inject_hyperparams`` puts the learning rate INSIDE the optimizer
    # state instead of baking it into the transformation. Changing it then
    # costs one field write, where rebuilding the optimizer would cost the
    # whole Adam moment history — ``init`` returns zeroed moments, so an
    # adaptive-KL schedule that rebuilds on every adjustment restarts the
    # optimizer from scratch each time, most often early in training when
    # the KL swings hardest and the rate changes most. rsl_rl assigns to
    # ``param_group["lr"]`` for the same reason; this is the equivalent.
    transforms = {label: optax.inject_hyperparams(optimizer_class)(learning_rate=lr) for label, lr in lr_config.items()}

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.multi_transform(transforms, param_labels),
    )

    return optimizer, param_labels


def create_simple_optimizer(
    learning_rate: float,
    max_grad_norm: float = 1.0,
) -> optax.GradientTransformation:
    """
    Create simple optimizer with gradient clipping.

    Args:
        learning_rate: Learning rate
        max_grad_norm: Maximum gradient norm for clipping

    Returns:
        Optax optimizer
    """
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(learning_rate=learning_rate),
    )


# ==================== Target Network Utilities ====================


@jax.jit
def polyak_update(
    params: Any,
    target_params: Any,
    tau: float,
) -> Any:
    """
    Polyak averaging for target network update.

    target = tau * params + (1 - tau) * target

    Compiled, because the tree it walks is the whole target network. Run
    eagerly, each leaf costs its own handful of dispatches — a pair of
    [256, 256] critics is a couple of dozen arrays, so a few dozen
    launches — and an off-policy algorithm runs this once per
    environment step. Under ``jit`` the whole tree is one program.

    Args:
        params: Current network parameters
        target_params: Target network parameters
        tau: Interpolation factor (0 < tau <= 1)

    Returns:
        Updated target parameters
    """
    return jax.tree.map(lambda p, tp: tau * p + (1 - tau) * tp, params, target_params)


def copy_params(params: Any) -> Any:
    """
    Create a copy of parameters for target network initialization.

    Args:
        params: Source parameters

    Returns:
        Copied parameters
    """
    return jax.tree.map(lambda x: x, params)


# ==================== Base Algorithm Class ====================


class RLAlgorithm(ABC):
    """
    Abstract base class for JAX-based RL algorithms.

    Provides common interface for:
    - PPO (on-policy)
    - SAC, TD3 (off-policy)
    """

    def __init__(
        self,
        actor_critic: eqx.Module,
        gamma: float,
        key: jax.Array,
    ):
        """
        Initialize base RL algorithm.

        Args:
            actor_critic: Actor-critic network (Equinox module)
            gamma: Discount factor
            key: JAX random key
        """
        self.gamma = gamma

        if key is None:
            key = jax.random.PRNGKey(0)

        # Partition model into params and static
        params, static = eqx.partition(actor_critic, eqx.is_inexact_array)
        self._static = static

        # Subclass should initialize train_state (type varies by algorithm)
        self.train_state: Any | None = None

        # Storage (initialized via init_storage)
        self.storage = None

    @property
    def model(self) -> eqx.Module:
        """Get current model."""
        return self.train_state.model

    @abstractmethod
    def _create_optimizers(self, model: eqx.Module) -> optax.GradientTransformation:
        """Create optimizer for the algorithm."""
        pass

    @abstractmethod
    def init_storage(self, cfg: Dict[str, Any]) -> None:
        """Initialize storage/replay buffer."""
        pass

    @abstractmethod
    def act(self, obs: ActInput, deterministic: bool = False) -> jax.Array:
        """
        Select action given observation.

        Args:
            obs: Current observation (ActInput)
            deterministic: Whether to use deterministic policy

        Returns:
            Selected action
        """
        pass

    @abstractmethod
    def process_env_step(
        self,
        rewards: jax.Array,
        terminated: jax.Array,
        truncated: jax.Array,
        infos: Dict[str, Any],
    ) -> None:
        """
        Process environment step and store transition.

        Args:
            rewards: Reward signals
            terminated: True termination flags (fall, goal, failure)
            truncated: Truncation flags (time limit reached)
            infos: Additional information
        """
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Update policy and value networks.

        Returns:
            Dictionary containing training metrics
        """
        pass

    @abstractmethod
    def save_train_state(self, checkpoint_dir: str) -> None:
        """
        Save algorithm-specific training state.

        Args:
            checkpoint_dir: Directory to save state files
        """
        pass

    @abstractmethod
    def load_train_state(self, checkpoint_dir: str, metadata: Dict[str, Any]) -> None:
        """
        Load algorithm-specific training state.

        Args:
            checkpoint_dir: Directory containing state files
            metadata: Metadata dictionary from checkpoint
        """
        pass

    def train_mode(self) -> None:
        """Set to training mode (no-op for JAX)."""
        pass

    def test_mode(self) -> None:
        """Set to evaluation mode (no-op for JAX)."""
        pass


class OnPolicyAlgorithm(RLAlgorithm):
    """
    Base class for on-policy algorithms (e.g., PPO).

    On-policy algorithms:
    - Collect trajectories with current policy
    - Update policy using collected data
    - Discard data after update
    """

    def __init__(
        self,
        actor_critic: eqx.Module,
        gamma: float,
        gae_lambda: float,
        num_learning_epochs: int,
        num_mini_batches: int,
        key: jax.Array,
    ):
        """
        Initialize on-policy algorithm.

        Args:
            actor_critic: Actor-critic network
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            num_learning_epochs: Number of epochs per update
            num_mini_batches: Number of minibatches per epoch
            key: JAX random key
        """
        super().__init__(actor_critic=actor_critic, gamma=gamma, key=key)
        self.gae_lambda = gae_lambda
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches

    @abstractmethod
    def compute_returns(self, last_critic_obs: jax.Array) -> None:
        """
        Compute returns and advantages using GAE.

        Args:
            last_critic_obs: Critic observations for the last state
        """
        pass


class OffPolicyAlgorithm(RLAlgorithm):
    """
    Base class for off-policy algorithms (e.g., SAC, TD3).

    Off-policy algorithms:
    - Store transitions in replay buffer
    - Sample random batches for updates
    - Use target networks for stable learning
    """

    def __init__(
        self,
        actor_critic: eqx.Module,
        gamma: float,
        tau: float,
        key: jax.Array,
    ):
        """
        Initialize off-policy algorithm.

        Args:
            actor_critic: Actor-critic network
            gamma: Discount factor
            tau: Target network soft update rate (Polyak averaging)
            key: JAX random key
        """
        super().__init__(actor_critic=actor_critic, gamma=gamma, key=key)
        self.tau = tau

        # Target network params (initialized by subclass)
        self._target_params: Any | None = None

    def _init_target_params(self, params: Any) -> Any:
        """
        Initialize target network parameters as copy of main params.

        Args:
            params: Main network parameters

        Returns:
            Target network parameters
        """
        return copy_params(params)

    def _update_target_params(self, params: Any) -> Any:
        """
        Update target parameters with Polyak averaging.

        Args:
            params: Current network parameters

        Returns:
            Updated target parameters
        """
        self._target_params = polyak_update(params, self._target_params, self.tau)
        return self._target_params

    def update_many(self, num_updates: int, batch_size: int, build_metrics: bool = True) -> Any:
        """Run ``num_updates`` gradient steps, sampling a batch for each.

        Driven from Python here: sample, update, sample, update. That is
        three dispatches a pass and a gap between each while Python
        decides what to do next, which is most of the cost when the
        update is small. Subclasses whose buffer can be sampled inside a
        trace override this with a single ``lax.scan`` — the shape PPO's
        update has always had — and only the last metric set is built,
        since only the last is ever read.
        """
        metrics = None
        for step in range(num_updates):
            metrics = self.update(
                self.sample_batch(batch_size), build_metrics=build_metrics and step == num_updates - 1
            )
        return metrics

    @abstractmethod
    def sample_batch(self, batch_size: int, *args: Any) -> Any:
        """
        Sample batch from replay buffer.

        A buffer that keeps its storage in host memory owns its index
        sampler and takes only ``batch_size`` — drawing indices on the
        accelerator would add a blocking transfer per batch, which for an
        off-policy algorithm means one per environment step. Sequence
        buffers that build their batches on device still take a JAX key
        as the second argument.

        Args:
            batch_size: Number of transitions to sample
            *args: Implementation-specific; a JAX key for device-sampled
                buffers, nothing for host-sampled ones

        Returns:
            Batch of transitions
        """
        pass
