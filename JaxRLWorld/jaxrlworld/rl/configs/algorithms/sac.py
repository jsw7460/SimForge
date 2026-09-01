from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

from ..base_config import BaseConfig

if TYPE_CHECKING:
    pass


@dataclass
class SACConfig(BaseConfig):
    algorithm_name: str = field(default="SAC")
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    n_steps: int = 1
    tau: float = 0.005
    ent_coef: Union[str, float] = "auto"
    target_entropy: Union[str, float] = "auto"
    obs_normalization: bool = False
    # Gradient-norm clipping for actor/critic/alpha optimizers. 10.0
    # matches the historical OffPolicyRunner fallback so existing SAC
    # benchmarks are unchanged; legged presets tighten this to 1.0.
    max_grad_norm: float = 10.0
    optimizer: str = "adam"
    optimizer_betas: tuple = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    learning_starts: int = 100
    # Half-width of the uniform warmup action distribution: during the
    # ``learning_starts`` warmup the runner samples actions from
    # ``U[-warmup_action_range, +warmup_action_range]`` instead of the
    # policy. 1.0 = the historical full [-1, 1] range. Lower it to make
    # warmup exploration gentler — with the joint-limit action mapping
    # full-range warmup drives every joint to its soft-limit extremes,
    # which can spam joint-limit / body-contact penalties before learning
    # starts.
    warmup_action_range: float = 1.0
    batch_size: int = 512
    buffer_size: int = 1_000_000
    # Where the replay buffer's storage lives. "host" keeps it in NumPy,
    # which is right when transitions are born on the host (a CPU
    # Gymnasium env). "device" keeps it on the accelerator, which is
    # right when they are born there — with a GPU simulator the host
    # buffer makes every transition cross twice, down to be stored and
    # back up to be sampled. That costs device memory: 5M transitions
    # across 8192 environments is roughly 4 GB the physics no longer has,
    # and vision observations make it far worse. Default "host" because
    # it is what every existing preset was measured on.
    replay_buffer_device: str = "host"
    policy_delay: int = 1
    num_gradient_steps: int = 1
    num_steps_per_env: int = 1
