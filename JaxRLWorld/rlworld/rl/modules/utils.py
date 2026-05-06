import math
from typing import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp

from rlworld.rl.utils.console import (
    BLUE,
    BOLD,
    CYAN,
    DIM,
    GREEN,
    MAGENTA,
    RESET,
    YELLOW,
)


def count_parameters(model: eqx.Module) -> int:
    """Count total parameters in model."""
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    return sum(x.size for x in leaves)


def print_model_summary(model: eqx.Module, name: str = "Model", max_depth: int = 2) -> int:
    """
    Print complete model summary for any Equinox module.

    Args:
        model: Equinox module to summarize
        name: Display name for the model
        max_depth: Maximum depth to display (default: 2)
    """
    params = eqx.filter(model, eqx.is_array)
    flat, _ = jax.tree_util.tree_flatten_with_path(params)

    # Build hierarchy
    hierarchy = {}
    for path, leaf in flat:
        if leaf is None:
            continue

        parts = []
        for p in path:
            key = str(p.key) if hasattr(p, "key") else str(p)
            key = key.strip(".")
            if key.startswith("[") and key.endswith("]"):
                key = key[1:-1]
            parts.append(key)

        current = hierarchy
        for part in parts[:-1]:
            if part not in current:
                current[part] = {"_count": 0, "_children": {}}
            current[part]["_count"] += leaf.size
            current = current[part]["_children"]

        leaf_name = parts[-1] if parts else "root"
        if leaf_name not in current:
            current[leaf_name] = {"_count": 0, "_children": {}, "_shape": None}
        current[leaf_name]["_count"] += leaf.size
        current[leaf_name]["_shape"] = leaf.shape

    # Print
    print(f"\n{BOLD}{BLUE}{'═' * 80}{RESET}")
    print(f"{BOLD}{BLUE}  {name} Parameter Summary{RESET}")
    print(f"{BOLD}{BLUE}{'═' * 80}{RESET}")
    print(f"{DIM}{'Name':<50} {'Shape':<20} {'Params':>10}{RESET}")
    print(f"{DIM}{'-' * 80}{RESET}")

    total_params = 0
    colors = [CYAN, GREEN, YELLOW, MAGENTA]

    def print_node(node_name: str, node: dict, depth: int = 0, is_last: bool = True):
        nonlocal total_params
        prefix = "  " * depth
        connector = "└── " if is_last else "├── "
        if depth == 0:
            connector = ""

        count = node.get("_count", 0)
        shape = node.get("_shape", None)
        children = node.get("_children", {})
        color = colors[depth % len(colors)]
        shape_str = str(shape) if shape else ""

        if children:
            print(
                f"{prefix}{connector}{BOLD}{color}{node_name:<45}{RESET} {DIM}{shape_str:<20}{RESET} {BOLD}{count:>10,}{RESET}"
            )
        else:
            print(f"{prefix}{connector}{color}{node_name:<45}{RESET} {DIM}{shape_str:<20}{RESET} {count:>10,}")

        # Stop recursion at max_depth
        if depth >= max_depth:
            return

        child_names = [k for k in children.keys() if not k.startswith("_")]
        for i, child in enumerate(sorted(child_names)):
            print_node(child, children[child], depth + 1, i == len(child_names) - 1)

    top_level = [k for k in hierarchy.keys() if not k.startswith("_")]
    for i, node_name in enumerate(sorted(top_level)):
        print_node(node_name, hierarchy[node_name], 0, i == len(top_level) - 1)
        total_params += hierarchy[node_name].get("_count", 0)

    print(f"{DIM}{'-' * 80}{RESET}")
    print(f"{BOLD}{GREEN}{'TOTAL':<50} {'':<20} {total_params:>10,}{RESET}")
    print(f"{BOLD}{BLUE}{'═' * 80}{RESET}\n")

    return total_params


# ==================== Activation Functions ====================


def get_activation(act_name: str) -> Callable:
    """Get activation function by name."""
    activations = {
        "relu": jax.nn.relu,
        "elu": jax.nn.elu,
        "selu": jax.nn.selu,
        "lrelu": lambda x: jax.nn.leaky_relu(x, negative_slope=0.01),
        "tanh": jnp.tanh,
        "sigmoid": jax.nn.sigmoid,
        "softplus": jax.nn.softplus,
        "swish": jax.nn.swish,
        "gelu": jax.nn.gelu,
    }
    if act_name not in activations:
        raise ValueError(f"Unknown activation: {act_name}")
    return activations[act_name]


# ==================== MLP ====================


class MLP(eqx.Module):
    """
    Multi-layer perceptron with optional LayerNorm.

    Equivalent to create_mlp() in PyTorch version.
    Automatically handles batched inputs via vmap.
    """

    linears: tuple  # Tuple of Linear layers
    layer_norms: tuple  # Tuple of LayerNorm or None
    activation: Callable = eqx.field(static=True)
    output_activation: Callable | None = eqx.field(static=True)
    use_layer_norm: bool = eqx.field(static=True)
    num_hidden: int = eqx.field(static=True)  # Number of hidden layers

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation: str = "relu",
        output_activation: str | None = None,
        use_layer_norm: bool = False,
        *,
        key: jax.Array,
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            activation: Activation function name
            output_activation: Output activation (None for linear)
            use_layer_norm: Whether to apply LayerNorm after each hidden layer
            key: JAX random key
        """
        self.activation = get_activation(activation)
        self.output_activation = get_activation(output_activation) if output_activation else None
        self.use_layer_norm = use_layer_norm
        self.num_hidden = len(hidden_dims)

        # Build layers
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        keys = jax.random.split(key, len(dims) - 1)

        linears = []
        layer_norms = []

        for i, (in_d, out_d, k) in enumerate(zip(dims[:-1], dims[1:], keys)):
            linears.append(eqx.nn.Linear(in_d, out_d, key=k))

            # Add LayerNorm for hidden layers only
            is_hidden = i < len(hidden_dims)
            if use_layer_norm and is_hidden:
                layer_norms.append(eqx.nn.LayerNorm(out_d))
            else:
                layer_norms.append(None)

        self.linears = tuple(linears)
        self.layer_norms = tuple(layer_norms)

    def _forward_single(self, x: jax.Array) -> jax.Array:
        """Forward pass for a single sample [input_dim]."""
        for i, (linear, ln) in enumerate(zip(self.linears, self.layer_norms)):
            x = linear(x)

            # Apply LayerNorm and activation for hidden layers
            is_hidden = i < self.num_hidden
            if is_hidden:
                if ln is not None:
                    x = ln(x)
                x = self.activation(x)

        # Output activation (if any)
        if self.output_activation is not None:
            x = self.output_activation(x)

        return x

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass with automatic batching.

        Args:
            x: Input of shape [input_dim] or [batch, input_dim]

        Returns:
            Output of shape [output_dim] or [batch, output_dim]
        """
        if x.ndim == 1:
            return self._forward_single(x)
        else:
            # Batched input: vmap over batch dimension
            return jax.vmap(self._forward_single)(x)


# ==================== Initialization Utilities ====================


def orthogonal_init_mlp(
    mlp: MLP,
    hidden_gain: float = math.sqrt(2),
    output_gain: float = 1.0,
    *,
    key: jax.Array,
) -> MLP:
    """
    Apply orthogonal initialization to MLP (PPO standard).

    Args:
        mlp: MLP module to initialize
        hidden_gain: Gain for hidden layers (sqrt(2) for ReLU)
        output_gain: Gain for output layer (1.0 or 0.01)
        key: JAX random key

    Returns:
        New MLP with orthogonal weights
    """

    def init_linear(linear: eqx.nn.Linear, gain: float, key: jax.Array) -> eqx.nn.Linear:
        # Orthogonal initialization
        weight = jax.nn.initializers.orthogonal(scale=gain)(key, linear.weight.shape, linear.weight.dtype)
        bias = jnp.zeros_like(linear.bias)
        return eqx.tree_at(lambda l: (l.weight, l.bias), linear, (weight, bias))

    num_linears = len(mlp.linears)
    keys = jax.random.split(key, num_linears)

    new_linears = []
    for i, (linear, k) in enumerate(zip(mlp.linears, keys)):
        is_output = i == num_linears - 1
        gain = output_gain if is_output else hidden_gain
        new_linears.append(init_linear(linear, gain, k))

    return eqx.tree_at(lambda m: m.linears, mlp, new_linears)


# ==================== Running Normalization ====================


class RunningNormalization(eqx.Module):
    """
    Running mean/variance normalization.

    Note: In JAX, this is stateful. For pure functional style,
    return updated state explicitly.
    """

    running_mean: jax.Array
    running_var: jax.Array
    count: jax.Array
    epsilon: float = eqx.field(static=True)

    def __init__(self, num_inputs: int, epsilon: float = 1e-8):
        self.running_mean = jnp.zeros(num_inputs)
        self.running_var = jnp.ones(num_inputs)
        self.count = jnp.array(1.0)
        self.epsilon = epsilon

    def normalize(self, x: jax.Array) -> jax.Array:
        """Normalize without updating stats."""
        return (x - self.running_mean) / jnp.sqrt(self.running_var + self.epsilon)

    def update(self, x: jax.Array) -> "RunningNormalization":
        """Return new normalizer with updated statistics."""
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.running_mean
        total_count = self.count + batch_count

        new_mean = self.running_mean + (delta * batch_count / total_count)
        new_var = (
            self.running_var * self.count + batch_var * batch_count + delta**2 * self.count * batch_count / total_count
        ) / total_count

        return RunningNormalization.__new__(RunningNormalization).replace(
            running_mean=new_mean,
            running_var=new_var,
            count=total_count,
        )

    def replace(self, **kwargs) -> "RunningNormalization":
        """Functional update helper."""
        new = object.__new__(RunningNormalization)
        new.running_mean = kwargs.get("running_mean", self.running_mean)
        new.running_var = kwargs.get("running_var", self.running_var)
        new.count = kwargs.get("count", self.count)
        object.__setattr__(new, "epsilon", self.epsilon)
        return new
