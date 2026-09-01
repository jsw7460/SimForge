from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Dict, Literal, Sequence, Union

from .base_config import BaseConfig

if TYPE_CHECKING:
    from jaxrlworld.rl.vis.overlays.hud_items.items import HUDItem
    from jaxrlworld.rl.vis.viser.scene_config import ViserSceneConfig


@dataclass
class RewardConfig(BaseConfig):
    """Reward configuration. Terms are named class attributes of type RewardTermConfig.

    Subclass and add terms as class-level attributes::

        @dataclass
        class MyRewardsCfg(RewardConfig):
            track_lin_vel = RewardTermConfig(func=rf.track_lin_vel, weight=2.0)
            action_rate = RewardTermConfig(func=rf.action_rate, weight=-0.01)

    Reward modes:
        ``"sum"`` (default): traditional weighted sum of all terms.
        ``"exponential"``: fixed classification using per-term ``exp_shaping`` flag.
            ``total = (sum of non-exp terms) * exp((sum of exp terms) / sigma)``
        ``"exponential_auto"``: dynamic classification by sign of global sum.
            Each step, terms with negative global sum go inside exp().
    """

    reward_mode: str = "sum"
    shaping_sigma: float = 0.02
    # Optional (min, max) clip applied to the TOTAL per-step reward after
    # mode computation (per-term episode sums stay unclipped). E.g.
    # ``(0.0, 10000.0)`` floors the summed reward at zero so penalty-heavy
    # steps cannot go negative.
    total_clip: tuple[float, float] | None = None


@dataclass
class CommandConfig(BaseConfig):
    """Command configuration (shared).

    Holds a dict of named CommandTermCfg objects. Each term independently
    manages its own sampling ranges, resampling timer, and post-processing.
    """

    terms: dict[str, Any] = field(default_factory=dict)  # str -> CommandTermCfg


@dataclass
class GaitConfig(BaseConfig):
    """Gait configuration (shared across simulators).

    Two modes:
        ``"fixed"``
            Evenly-spaced phase offsets with constant period.
            No commands needed. Suitable for basic locomotion training.

        ``"command"``
            Frequency and stance duration read from CommandManager each step.
            Per-foot phase offsets produced by ``foot_offset_provider``.
            Use this for gait-conditioned training (e.g. Walk-These-Ways style).

    The ``foot_offset_provider`` is a callable
    ``(CommandManager) -> [num_envs, num_feet]`` that reads whatever commands
    it needs and returns per-foot phase offsets. Pre-built providers:
        - ``QuadrupedOffsets()``:  reads (phase, offset, bound) -> 4-foot offsets
        - ``DirectOffsets(names)``: reads N named commands -> N-foot offsets
    """

    foot_names: tuple[str, ...] | list[str] = field(default_factory=tuple)

    # "fixed" or "command"
    offset_mode: str = "fixed"

    # ── Fixed-mode settings ──
    gait_period: float = 0.8
    default_freq: float = 2.5
    default_duration: float = 0.5

    # ── Command-mode settings ──
    freq_command: str = "gait_freq"
    duration_command: str = "gait_duration"

    # Callable instance: (CommandManager) -> [num_envs, num_feet].
    # See QuadrupedOffsets and DirectOffsets in managers/common/gait.py.
    # Automatically serialized to string by recursive_to_dict().
    foot_offset_provider: Any = None

    # Von Mises smoothing for desired contact states
    contact_smoothing_sigma: float = 0.07


@dataclass
class TerminationsConfig(BaseConfig):
    """Termination configuration. Terms are named class attributes of type TerminationTermConfig.

    Subclass and add terms as class-level attributes::

        @dataclass
        class MyTerminationsCfg(TerminationsConfig):
            bad_orientation = TerminationTermConfig(func=tf.roll_pitch, params={...})
            time_out = TerminationTermConfig(func=tf.max_episode)
    """

    pass


@dataclass
class EventConfig(BaseConfig):
    """Event configuration. Terms are named class attributes of type EventTermConfig.

    Subclass and add terms as class-level attributes::

        @dataclass
        class MyEventsCfg(EventConfig):
            reset_dof = EventTermConfig(func=init_dof, mode="reset")
            push_robot = EventTermConfig(func=push, mode="interval", interval_range_s=(2, 20))
    """

    pass


@dataclass
class ObservationGroupConfig(BaseConfig):
    """An observation group. Terms are named class attributes of type ObservationTermConfig.

    Attributes:
        enable_corruption: When True (default), per-term ``noise`` configs are
            applied to this group. Set to False on the critic / privileged-
            observation group to feed raw signals while the actor still sees
            noise — the standard locomotion-from-state recipe. Mirrors
            ``ObservationGroupCfg.enable_corruption`` from IsaacLab.

    Example::

        @dataclass
        class ActorObsCfg(ObservationGroupConfig):
            base_ang_vel = ObservationTermConfig(func=base_ang_vel, scale=0.25)
            dof_pos = ObservationTermConfig(func=dof_pos, scale=1.0)

        @dataclass
        class CriticObsCfg(ObservationGroupConfig):
            enable_corruption: bool = False  # privileged: critic sees raw signals
            base_ang_vel = ObservationTermConfig(func=base_ang_vel, scale=0.25)
            ...
    """

    enable_corruption: bool = True

    concatenate_terms: bool = True
    """Whether to concatenate this group's terms into one tensor. Defaults to True.

    False keeps them separate and the group is served as a dict of named
    tensors. A group whose terms differ in shape MUST set this to False —
    an image and a joint vector have no axis along which to be joined.
    Mirrors ``ObservationGroupCfg.concatenate_terms`` in both mjlab and
    IsaacLab.
    """

    concatenate_dim: int = -1
    """Axis to concatenate along, ignoring the batch axis. Defaults to -1.

    The default joins vectors end to end. Image terms are stacked on the
    CHANNEL axis instead — for a ``(C, H, W)`` term that is 0, which is
    what mjlab's own YAM vision task passes to put a depth map and a
    target mask into one two-channel tensor. Mirrors
    ``ObservationGroupCfg.concatenate_dim``.
    """


def disable_corruption(observation_cfg) -> None:
    """Disable noise on every ObservationGroupConfig field of the given config.

    Convenience for eval / test runs that want raw observations on every
    group at once. Walks the dataclass fields and sets
    ``group.enable_corruption = False`` on each :class:`ObservationGroupConfig`
    attribute. Replaces the old ``observation.enable_noise = False`` call
    pattern.
    """
    for attr_name in vars(observation_cfg):
        val = getattr(observation_cfg, attr_name)
        if isinstance(val, ObservationGroupConfig):
            val.enable_corruption = False


# ════════════════════════════════════════════════════════════════════
# Neural network configuration — strict-typed
# ════════════════════════════════════════════════════════════════════
#
# Design (long-term clean, no legacy shims):
#   • StrEnum for closed value sets (Activation, DistributionType, StdType)
#       — IDE-rename refactorable, mypy strict, single source of truth.
#   • Union dataclass for mutually-exclusive init schemes
#       (OrthoInit | DefaultInit) — invariant strangler ("ortho gain
#       only meaningful with OrthoInit"), trivially extensible to
#       KaimingInit / XavierInit later.
#   • One dataclass per actor / critic class (MLPActorCfg,
#       SpaceTimeTransformerActorCfg, ...). The cfg type *is* the
#       actor-class identity — no string ``actor_class_name`` field
#       (registry is keyed by cfg type). isinstance-based dispatch.
#   • Backward compat with old `actor_kwargs`-style dict configs is
#       NOT preserved; every preset / benchmark gets migrated in one
#       pass. Checkpoints saved under the old schema cannot be loaded
#       (acceptable per design decision).


# ── Enums ───────────────────────────────────────────────────────────


class Activation(StrEnum):
    """Activation functions supported by MLP / decoder layers."""

    RELU = "relu"
    ELU = "elu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    SELU = "selu"
    GELU = "gelu"
    SILU = "silu"


class DistributionType(StrEnum):
    """Action-distribution families for stochastic policies."""

    GAUSSIAN = "gaussian"
    SQUASHED_GAUSSIAN = "squashed_gaussian"


class StdType(StrEnum):
    """How the policy parameterizes the action-distribution std."""

    # NN outputs a per-state std vector.
    STATE_DEPENDENT = "state_dependent"
    # Learnable per-dimension log-std vector (state-independent).
    STATE_INDEPENDENT = "state_independent"
    # Frozen constant std (not learned).
    FIXED = "fixed"
    # Learnable single scalar std (no log transform, shared across dims).
    SCALAR = "scalar"


# ── Init schemes (mutually-exclusive Union) ─────────────────────────


@dataclass
class OrthoInit(BaseConfig):
    """Orthogonal initialization.

    Hidden-layer gain is auto-derived from the activation (sqrt(2) for
    relu/elu, 1.0 for tanh/sigmoid/selu) unless ``hidden_gain`` pins it
    explicitly — SB3/openai-baselines use sqrt(2) for every hidden
    layer regardless of activation, so parity comparisons set
    ``hidden_gain=sqrt(2)`` on tanh networks. ``output_gain`` scales the
    head: actor heads typically use a small value (e.g. 0.01) so the
    policy starts near zero; critic heads ignore it and always use 1.0.
    """

    output_gain: float = 1.0
    hidden_gain: float | None = None


@dataclass
class DefaultInit(BaseConfig):
    """Framework default init (Equinox's Glorot / Lecun)."""

    pass


InitScheme = Union[OrthoInit, DefaultInit]


_INIT_CFG_CLASSES: Dict[str, type] = {
    "OrthoInit": OrthoInit,
    "DefaultInit": DefaultInit,
}


def _hydrate_init_scheme(val: Any) -> InitScheme:
    """Convert a dict (deserialized form) back to the right Init dataclass."""
    if isinstance(val, dict):
        cls_name = val.pop("_type", "OrthoInit")
        cls = _INIT_CFG_CLASSES[cls_name]
        return cls(**val)
    return val


# ── Actor configs (one dataclass per actor class) ───────────────────


@dataclass
class MLPActorCfg(BaseConfig):
    """Config for ``MLPActor`` (jaxrlworld/rl/modules/architectures/mlp/actor.py).

    The cfg type itself identifies the actor class — there is no
    separate ``actor_class_name`` string. Dispatch happens by
    ``isinstance(cfg.actor, MLPActorCfg)`` (or a cfg-type-keyed
    registry lookup).
    """

    hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    activation: Activation = Activation.ELU
    use_layer_norm: bool = False
    init: InitScheme = field(default_factory=OrthoInit)

    def __post_init__(self):
        if isinstance(self.activation, str):
            self.activation = Activation(self.activation)
        self.init = _hydrate_init_scheme(self.init)

    def recursive_to_dict(self) -> Dict:
        result = super().recursive_to_dict()
        result["_type"] = type(self).__name__
        if isinstance(result.get("init"), dict):
            result["init"]["_type"] = type(self.init).__name__
        return result

    @classmethod
    def from_dict(cls, config_dict: Dict):
        # Need to hydrate ``init`` (Union dataclass) manually because
        # BaseConfig.update_from_dict can't switch a Union member's
        # type. Then re-run __post_init__ so string → Enum coercion
        # picks up the value yaml.safe_load left as a plain string.
        from .base_config import update_from_dict

        d = dict(config_dict)
        init_dict = d.pop("init", None)
        obj = cls()
        update_from_dict(obj, d)
        if init_dict is not None:
            obj.init = _hydrate_init_scheme(dict(init_dict))
        obj.__post_init__()
        return obj


@dataclass
class SpaceTimeTransformerActorCfg(BaseConfig):
    """Config for ``SpaceTimeTransformerActor``."""

    tracked_body_names: Sequence[str] = field(default_factory=tuple)
    future_offsets: Sequence[int] = field(default_factory=tuple)
    actuated_joint_names: Sequence[str] | None = None
    ref_feature_dim: int = 9
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.0
    bottleneck_dim: int = 32
    tokenizer_hidden_dim: int | None = None
    decoder_hidden_dim: int | None = None
    decoder_activation: Activation = Activation.ELU
    use_kinematic_mask: bool = True
    pe_type: str = "learned"
    use_relational_bias: bool = False
    re_use_laplacian: bool = True
    re_use_spd: bool = True
    re_use_ppr: bool = True
    re_ppr_alpha: float = 0.15
    attention_mode: str = "factorized"

    def __post_init__(self):
        if isinstance(self.decoder_activation, str):
            self.decoder_activation = Activation(self.decoder_activation)

    def recursive_to_dict(self) -> Dict:
        result = super().recursive_to_dict()
        result["_type"] = type(self).__name__
        return result

    @classmethod
    def from_dict(cls, config_dict: Dict):
        # Re-run __post_init__ so decoder_activation gets coerced
        # str → Enum after yaml.safe_load.
        obj = super().from_dict(config_dict)
        obj.__post_init__()
        return obj


@dataclass
class CNNEncoderCfg(BaseConfig):
    """Config for ``CNNEncoder`` (jaxrlworld/rl/modules/architectures/cnn/encoder.py).

    Defaults are mjlab's vision recipe for the YAM wrist camera
    (``tasks/manipulation/config/yam/rl_cfg.py``): two strided
    convolutions, then a spatial softmax that reports WHERE each of the
    32 channels fired rather than how strongly, giving a 64-number
    latent regardless of image size.
    """

    output_channels: tuple[int, ...] = (16, 32)
    kernel_size: tuple[int, ...] | int = (5, 3)
    stride: tuple[int, ...] | int = (2, 2)
    dilation: tuple[int, ...] | int = 1
    # "zeros" keeps the output at ceil(input / stride); "none" does not pad.
    padding: str = "zeros"
    activation: Activation = Activation.ELU
    spatial_softmax: bool = True
    spatial_softmax_temperature: float = 1.0

    def __post_init__(self):
        if isinstance(self.activation, str):
            self.activation = Activation(self.activation)

    @classmethod
    def from_dict(cls, config_dict: Dict):
        # Re-run __post_init__: from_dict builds a default instance and
        # then writes the raw values over it, so the enum coercion above
        # never sees the string yaml.safe_load produced.
        obj = super().from_dict(config_dict)
        obj.__post_init__()
        return obj


@dataclass
class VisionActorCfg(BaseConfig):
    """Config for ``VisionActor``: image groups encoded, then an MLP.

    ``image_groups`` names the observation groups that arrive as
    ``(C, H, W)``; each gets its own encoder. Everything else the model
    reads is the ordinary state vector, and it alone is normalised.
    """

    trunk: MLPActorCfg = field(default_factory=lambda: MLPActorCfg(hidden_dims=[256, 256, 128]))
    cnn: CNNEncoderCfg = field(default_factory=CNNEncoderCfg)
    image_groups: tuple[str, ...] = ("camera",)

    def __post_init__(self):
        if isinstance(self.trunk, dict):
            self.trunk = MLPActorCfg.from_dict(self.trunk)
        if isinstance(self.cnn, dict):
            self.cnn = CNNEncoderCfg.from_dict(self.cnn)
        # The nested cfgs may have been written over field by field
        # rather than constructed, which skips their own coercion and
        # leaves an activation as the string yaml produced.
        self.trunk.__post_init__()
        self.cnn.__post_init__()
        self.image_groups = tuple(self.image_groups)

    @classmethod
    def from_dict(cls, config_dict: Dict):
        obj = super().from_dict(config_dict)
        obj.__post_init__()
        return obj

    def recursive_to_dict(self) -> Dict:
        result = super().recursive_to_dict()
        result["_type"] = type(self).__name__
        return result


ActorCfg = Union[MLPActorCfg, SpaceTimeTransformerActorCfg, VisionActorCfg]

_ACTOR_CFG_CLASSES: Dict[str, type] = {
    "MLPActorCfg": MLPActorCfg,
    "SpaceTimeTransformerActorCfg": SpaceTimeTransformerActorCfg,
    "VisionActorCfg": VisionActorCfg,
}


def _hydrate_actor_cfg(val: Any) -> ActorCfg:
    if isinstance(val, dict):
        cls_name = val.pop("_type", "MLPActorCfg")
        cls = _ACTOR_CFG_CLASSES[cls_name]
        return cls.from_dict(val)
    return val


# ── Critic configs (one dataclass per critic class) ─────────────────


@dataclass
class MLPCriticCfg(BaseConfig):
    """Config for ``MLPCritic``."""

    hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    activation: Activation = Activation.ELU
    use_layer_norm: bool = False
    init: InitScheme = field(default_factory=OrthoInit)

    def __post_init__(self):
        if isinstance(self.activation, str):
            self.activation = Activation(self.activation)
        self.init = _hydrate_init_scheme(self.init)

    def recursive_to_dict(self) -> Dict:
        result = super().recursive_to_dict()
        result["_type"] = type(self).__name__
        if isinstance(result.get("init"), dict):
            result["init"]["_type"] = type(self.init).__name__
        return result

    @classmethod
    def from_dict(cls, config_dict: Dict):
        # See MLPActorCfg.from_dict for rationale.
        from .base_config import update_from_dict

        d = dict(config_dict)
        init_dict = d.pop("init", None)
        obj = cls()
        update_from_dict(obj, d)
        if init_dict is not None:
            obj.init = _hydrate_init_scheme(dict(init_dict))
        obj.__post_init__()
        return obj


@dataclass
class SpaceTimeTransformerCriticCfg(BaseConfig):
    """Config for ``SpaceTimeTransformerCritic`` (no decoder fields)."""

    tracked_body_names: Sequence[str] = field(default_factory=tuple)
    future_offsets: Sequence[int] = field(default_factory=tuple)
    ref_feature_dim: int = 9
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.0
    tokenizer_hidden_dim: int | None = None
    use_kinematic_mask: bool = True
    pe_type: str = "learned"
    use_relational_bias: bool = False
    re_use_laplacian: bool = True
    re_use_spd: bool = True
    re_use_ppr: bool = True
    re_ppr_alpha: float = 0.15
    attention_mode: str = "factorized"

    def recursive_to_dict(self) -> Dict:
        result = super().recursive_to_dict()
        result["_type"] = type(self).__name__
        return result


@dataclass
class VisionCriticCfg(BaseConfig):
    """Config for ``VisionCritic``. See ``VisionActorCfg``."""

    trunk: MLPCriticCfg = field(default_factory=lambda: MLPCriticCfg(hidden_dims=[256, 256, 128]))
    cnn: CNNEncoderCfg = field(default_factory=CNNEncoderCfg)
    image_groups: tuple[str, ...] = ("camera",)

    def __post_init__(self):
        if isinstance(self.trunk, dict):
            self.trunk = MLPCriticCfg.from_dict(self.trunk)
        if isinstance(self.cnn, dict):
            self.cnn = CNNEncoderCfg.from_dict(self.cnn)
        # The nested cfgs may have been written over field by field
        # rather than constructed, which skips their own coercion and
        # leaves an activation as the string yaml produced.
        self.trunk.__post_init__()
        self.cnn.__post_init__()
        self.image_groups = tuple(self.image_groups)

    @classmethod
    def from_dict(cls, config_dict: Dict):
        obj = super().from_dict(config_dict)
        obj.__post_init__()
        return obj

    def recursive_to_dict(self) -> Dict:
        result = super().recursive_to_dict()
        result["_type"] = type(self).__name__
        return result


CriticCfg = Union[MLPCriticCfg, SpaceTimeTransformerCriticCfg, VisionCriticCfg]

_CRITIC_CFG_CLASSES: Dict[str, type] = {
    "MLPCriticCfg": MLPCriticCfg,
    "SpaceTimeTransformerCriticCfg": SpaceTimeTransformerCriticCfg,
    "VisionCriticCfg": VisionCriticCfg,
}


def _hydrate_critic_cfg(val: Any) -> CriticCfg:
    if isinstance(val, dict):
        cls_name = val.pop("_type", "MLPCriticCfg")
        cls = _CRITIC_CFG_CLASSES[cls_name]
        return cls.from_dict(val)
    return val


# ── Policy configs ──────────────────────────────────────────────────


@dataclass
class PolicyConfig(BaseConfig):
    """Base policy network configuration — common to all algorithms."""

    actor: ActorCfg = field(default_factory=MLPActorCfg)
    critic: CriticCfg = field(default_factory=MLPCriticCfg)

    def __post_init__(self):
        self.actor = _hydrate_actor_cfg(self.actor)
        self.critic = _hydrate_critic_cfg(self.critic)

    def to(self, target_cls: type) -> "PolicyConfig":
        """Convert to another PolicyConfig subclass, copying common fields
        (actor + critic). Subclass-specific fields (init_noise_std,
        distribution_type, etc.) keep the target's defaults.
        """
        return target_cls(actor=self.actor, critic=self.critic)

    def recursive_to_dict(self) -> Dict:
        result = super().recursive_to_dict()
        result["_type"] = self.__class__.__name__
        return result

    @classmethod
    def from_dict(cls, config_dict: Dict):
        """Hydrate actor / critic Union members from dict form.

        The base ``BaseConfig.from_dict`` builds a default instance then
        does an in-place ``update_from_dict``. That works for scalar /
        nested-dict fields but cannot switch the *type* of a Union
        field (e.g. default ``actor`` is :class:`MLPActorCfg` but the
        serialized dict describes a :class:`SpaceTimeTransformerActorCfg`).
        We extract ``actor`` / ``critic`` ahead of the update and
        rebuild them via the ``_type`` discriminator.
        """
        from .base_config import update_from_dict

        d = dict(config_dict)
        actor_dict = d.pop("actor", None)
        critic_dict = d.pop("critic", None)
        obj = cls()
        update_from_dict(obj, d)
        if actor_dict is not None:
            obj.actor = _hydrate_actor_cfg(dict(actor_dict))
        if critic_dict is not None:
            obj.critic = _hydrate_critic_cfg(dict(critic_dict))
        # Re-run post_init for fresh enum / actor / critic coercion.
        obj.__post_init__()
        return obj


@dataclass
class PPOPolicyConfig(PolicyConfig):
    """PPO policy settings."""

    init_noise_std: float = 1.0
    distribution_type: DistributionType = DistributionType.GAUSSIAN
    std_type: StdType = StdType.STATE_INDEPENDENT

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.distribution_type, str):
            self.distribution_type = DistributionType(self.distribution_type)
        if isinstance(self.std_type, str):
            self.std_type = StdType(self.std_type)


@dataclass
class SACPolicyConfig(PolicyConfig):
    """SAC policy settings."""

    init_noise_std: float = 0.05
    distribution_type: DistributionType = DistributionType.SQUASHED_GAUSSIAN
    log_std_min: float = -20.0
    log_std_max: float = 2.0

    small_output_init: bool = False
    """Re-initialize the actor output heads for calibrated initial
    exploration: the mean head's final layer gets N(0, 1e-3) weights
    with zero bias (initial pre-tanh actions near zero), and the
    log-std net's final layer gets zero weights (initial std exactly
    ``init_noise_std`` for every state and dimension). Default False
    keeps the historical orthogonal/Kaiming output init."""

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.distribution_type, str):
            self.distribution_type = DistributionType(self.distribution_type)


@dataclass
class TD3PolicyConfig(PolicyConfig):
    """TD3 policy settings (deterministic — no extra fields)."""

    pass


@dataclass
class FastTD3PolicyConfig(PolicyConfig):
    """FastTD3 policy settings (deterministic — no extra fields)."""

    pass


# Registry for deserialization
_POLICY_CONFIG_CLASSES: Dict[str, type] = {
    "PolicyConfig": PolicyConfig,
    "PPOPolicyConfig": PPOPolicyConfig,
    "SACPolicyConfig": SACPolicyConfig,
    "TD3PolicyConfig": TD3PolicyConfig,
    "FastTD3PolicyConfig": FastTD3PolicyConfig,
}


@dataclass
class NNConfig(BaseConfig):
    """Neural network configuration (shared)."""

    policy: PolicyConfig = field(default_factory=PPOPolicyConfig)
    # ``state_estimator`` removed: there were no read sites for it in
    # the framework or any preset. Re-introduce as a typed dataclass
    # if/when actually consumed.

    def __post_init__(self):
        if isinstance(self.policy, dict):
            cls_name = self.policy.pop("_type", "PPOPolicyConfig")
            cls = _POLICY_CONFIG_CLASSES[cls_name]
            self.policy = cls.from_dict(self.policy)

    @classmethod
    def from_dict(cls, config_dict: Dict):
        """Hydrate ``policy`` (PolicyConfig subclass + Union actor/critic)
        from dict form. Required because ``BaseConfig.from_dict`` 's
        in-place update cannot switch the policy *subclass* nor the
        Union actor/critic *types* — both must be discriminated by
        ``_type`` ahead of the update.
        """
        from .base_config import update_from_dict

        d = dict(config_dict)
        policy_dict = d.pop("policy", None)
        obj = cls()
        update_from_dict(obj, d)
        if policy_dict is not None:
            pd = dict(policy_dict)
            cls_name = pd.pop("_type", "PPOPolicyConfig")
            policy_cls = _POLICY_CONFIG_CLASSES[cls_name]
            obj.policy = policy_cls.from_dict(pd)
        return obj


@dataclass
class LoggingConfig(BaseConfig):
    """Fine-grained wandb/tensorboard output toggles.

    Central hub for per-category logging verbosity. Disable noisy
    blocks without touching runner/logger plumbing. Adding a new knob
    here + a corresponding guard in the runner/logger is the intended
    extension path (see ``action_dist`` / ``action_histogram`` below
    for the template).

    All flags default to ``False`` — callers opt in to extra logging
    rather than opting out. This keeps wandb dashboards quiet by
    default and makes new runs self-documenting about what is being
    tracked.
    """

    # Per-dim action statistics — ``ActionDist/{mean,std,min,max}/dim_*``.
    # Noisy for high-dim action spaces (e.g. 23-DoF humanoid → 92 scalar
    # keys per iteration).
    action_dist: bool = False
    # ``wandb.Histogram`` per action dimension. Requires ``action_dist``
    # data collection to run (or independently collects raw actions);
    # separate from the scalar toggle because histograms are an order
    # of magnitude more expensive both to compute and to store.
    action_histogram: bool = False


@dataclass
class RunnerConfig(BaseConfig):
    """Runner configuration (shared)."""

    checkpoint: int = -1
    log_interval: int = 1
    max_iterations: int = 99999
    init_at_random_ep_len: bool = False
    resume: bool = False
    resume_path: str | None = None
    run_name: str = ""
    logger: str = "wandb"
    wandb_project: str = "SimForge"
    save_interval: int = 1000
    # Cadence of the rolling ``checkpoint_latest`` (full-parameter D2H +
    # disk write). 1 restores the old save-every-iteration behavior; the
    # crash-recovery cost of the default is at most this many iterations.
    latest_checkpoint_interval: int = 10
    output_dir: str = "auto"
    upload_checkpoint: bool = False
    delete_local_after_upload: bool = False

    # Fine-grained wandb logging toggles. See :class:`LoggingConfig`.
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # In-training evaluation
    eval_interval: int = 50  # 0 = disabled
    eval_num_envs: int = 32
    eval_num_episodes: int = 10
    eval_deterministic: bool = True
    eval_disable_noise: bool = True
    eval_disable_interval_events: bool = True
    # Optional held-out eval sets, keyed by label, e.g.
    # ``{"near": ("a.npz", ...), "far": (...)}``. For each non-empty entry
    # the runner builds a separate deterministic eval env with the motion
    # command's ``motion_files`` swapped to that set and logs the metrics
    # under ``Eval/heldout_<label>/``. Empty (default) = no held-out eval.
    # Only meaningful for motion-tracking presets (those with a "motion"
    # command term); ignored otherwise.
    eval_extra_motion_files: dict = field(default_factory=dict)


@dataclass
class VisualizationConfig(BaseConfig):
    """Visualization configuration (shared)."""

    _EXCLUDE_FROM_SERIALIZATION = ("extra_hud_items", "viser_scene")

    show_viewer: bool = False
    record_video: bool = False
    video_dir: str = ""
    video_fps: int | None = None
    record_env_ids: list[int] = field(default_factory=lambda: [0])
    grid_layout: bool = True

    # 3D Overlay for Genesis
    enable_command_arrow: bool = True
    command_arrow_radius: float = 0.02
    command_arrow_length_scale: float = 0.5
    max_arrow_length: float = 1.0

    # 2D HUD for Genesis
    enable_text_hud: bool = True
    hud_position: str = "top_left"
    show_base_height: bool = True
    show_command_vel: bool = True
    show_feet_height: bool = True
    show_episode_info: bool = True
    feet_names: tuple[str, ...] = ("FL", "FR", "RL", "RR")
    extra_hud_items: "list[HUDItem]" = field(default_factory=list)

    # Viewer type
    viewer_type: Literal["gl", "viser", "rerun", "usd", "file"] = "gl"
    viser_port: int = 8080
    viser_share: bool = True
    rerun_web_port: int = 9191

    # Look of the unified Viser scene (Genesis/Newton bridge path). ``None``
    # → ViserSceneConfig defaults (near-white ground, dark metallic robot).
    viser_scene: "ViserSceneConfig | None" = None

    # Unified Viser viewer (SimForge custom)
    viser_enable_reward_plots: bool = True
    viser_enable_debug_viz: bool = False
