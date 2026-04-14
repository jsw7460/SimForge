"""Protocol contract for sim-specific builder modules.

Each preset's base config (e.g. ``Go2FlatConfig``, ``G1FlatConfig``)
delegates simulator-specific construction to a module like
``_newton_builders.py``, ``_genesis_builders.py``, or
``_mujoco_builders.py``.  This file documents what functions and
constants those modules must export so that the base config can call
into them uniformly.

Python modules satisfy these Protocols **structurally** — a module
that defines the right top-level names satisfies the Protocol
automatically, with no inheritance required.  The Protocols are used
purely as a documentation and IDE-autocomplete aid; type checkers
(pyright / ty) will flag missing members.

Optional hooks (not declared on the Protocol; called via ``hasattr``
guard at the call site so they can be omitted when not needed):

* ``customize_reset_root_params(cfg, params: dict) -> None``
    Mutate the ``reset_root_state_uniform`` params dict in place for
    sim-specific overrides.  Example: Newton converts its native
    xyzw initial quat into wxyz, Genesis Go2 shifts the spawn
    position by ``(1.5, 1.5)`` for scene layout reasons.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Protocol

if TYPE_CHECKING:
    from rlworld.rl.configs.events import EventTermConfig


class SimBuilderProtocol(Protocol):
    """Base contract shared by every sim builder module.

    The ``@staticmethod`` decorator on each method tells type
    checkers that a plain module-level ``def build_env(cfg, timing)``
    satisfies the slot — no implicit ``self`` parameter required.
    """

    #: Simulator-specific top-level ``ConfigsForRun`` container class
    #: (e.g. ``NewtonConfigsForRun``).  ``base.py`` instantiates this
    #: at the end of ``build()``.
    CONFIGS_FOR_RUN_CLS: type

    #: Simulator-specific observation config container class.  Used
    #: by preset ``_build_observation_config`` methods that subclass
    #: it for the actor/critic groups.
    OBSERVATION_CFG_CLS: type

    @staticmethod
    def build_env(cfg: Any, timing: Dict[str, Any]) -> Any: ...

    @staticmethod
    def build_scene(cfg: Any, timing: Dict[str, Any]) -> Any: ...

    @staticmethod
    def build_visualization(cfg: Any) -> Any: ...

    @staticmethod
    def build_action(cfg: Any) -> Any: ...

    @staticmethod
    def build_reward(cfg: Any) -> Any: ...

    @staticmethod
    def build_dr_terms(cfg: Any) -> "Dict[str, EventTermConfig]": ...


class Go2SimBuilderProtocol(SimBuilderProtocol, Protocol):
    """Go2 preset sim builder contract.

    Adds ``get_foot_names`` — Go2's base config uses this when
    building the gait config.  Go2 shares observation construction
    across simulators via ``_build_observation_config`` in
    ``go2_flat/base.py``, so sim builders do **not** need to provide
    a ``build_observation`` function.
    """

    @staticmethod
    def get_foot_names(robot: Any) -> List[str]: ...


class G1SimBuilderProtocol(SimBuilderProtocol, Protocol):
    """G1 preset sim builder contract.

    Adds ``build_observation`` — G1's observation config varies per
    simulator (different critic state helpers, different foot
    observation sources), so observation construction lives in each
    sim builder rather than in ``g1_29dof/base.py``.
    """

    @staticmethod
    def build_observation(cfg: Any) -> Any: ...
