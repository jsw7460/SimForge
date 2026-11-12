"""Simulator backends that live outside this package.

Genesis, Newton and MuJoCo are wired into the framework directly. A
backend developed in another package plugs into the same places through
this module instead of through edits here:

* the env class, named by ``cfgs.env.env_name = "package.module:ClassName"``
  (see :func:`resolve_env_class`);
* its managers, registered into :class:`ManagerRegistry` by
  :attr:`BackendSpec.register_managers`;
* its evaluation initializer (:attr:`BackendSpec.make_initializer`);
* its domain-randomization terms (:func:`register_dr_backend`);
* its preset builders
  (:func:`jaxrlworld.rl.configs.presets._sim_builder_registry.register_sim_builders`).

A package announces its backends with an entry point in the
``jaxrlworld.backends`` group whose target is a zero-argument callable
that performs the registrations::

    [project.entry-points."jaxrlworld.backends"]
    mysim = "mypkg.backend:register"

Entry points are loaded on the first lookup that misses, so a process
that never asks for an external backend never imports one.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from jaxrlworld.rl import envs as _envs

if TYPE_CHECKING:
    from jaxrlworld.rl.evals.sim_initializers import SimInitializer

ENTRY_POINT_GROUP = "jaxrlworld.backends"

# Simulators wired in directly; their sim_type values are reserved.
BUILTIN_SIM_TYPES: tuple[str, ...] = ("genesis", "newton", "mujoco")


@dataclass(frozen=True)
class BackendSpec:
    """What the framework needs to know about an external backend."""

    sim_type: str
    """Lowercase key: ``World.sim_type``, ``ConfigsForRun.sim_type`` and the
    :class:`ManagerRegistry` key."""

    eval_key: str
    """Key under which :func:`jaxrlworld.rl.evals.sim_initializers.get_initializer`
    returns this backend's initializer."""

    register_managers: Callable[[], None]
    """Registers the backend's scene/action/observation/contact/... managers
    with :class:`ManagerRegistry`. Called once per ``World`` setup, like the
    built-in registrars."""

    make_initializer: Callable[[], SimInitializer]
    """Returns a fresh :class:`SimInitializer` for offline evaluation."""

    preset_package: str | None = None
    """Package laid out like ``jaxrlworld.rl.configs.presets`` holding this
    backend's simulator-specific preset modules
    (``<package>.<robot>.<sim_type>.<task>``), so cross-simulator evaluation
    finds the counterpart of ``jaxrlworld.rl.configs.presets.go2.newton.<task>``
    and back. ``None`` when the backend has none."""


_BACKENDS: dict[str, BackendSpec] = {}
_DR_BACKENDS: dict[tuple[str, str], Callable[..., None]] = {}
_plugins_loaded = False


def register_backend(spec: BackendSpec) -> None:
    """Register an external backend. A sim_type is registered at most once."""
    if spec.sim_type in BUILTIN_SIM_TYPES:
        raise ValueError(f"sim_type {spec.sim_type!r} is a built-in backend and cannot be re-registered.")
    existing = _BACKENDS.get(spec.sim_type)
    if existing is not None and existing != spec:
        raise ValueError(f"sim_type {spec.sim_type!r} is already registered with a different spec: {existing}")
    _BACKENDS[spec.sim_type] = spec


def register_dr_backend(term: str, sim_type: str, fn: Callable[..., None]) -> None:
    """Register ``fn`` as the ``sim_type`` implementation of the unified DR
    term ``term`` (e.g. ``"randomize_friction"``).

    ``fn`` is called as ``fn(env, env_ids, sampler, **term_kwargs)`` with the
    term's own keyword arguments, after the term's backend-independent
    checks have run.
    """
    key = (term, sim_type)
    existing = _DR_BACKENDS.get(key)
    if existing is not None and existing is not fn:
        raise ValueError(f"DR term {term!r} already has a {sim_type!r} backend: {existing}")
    _DR_BACKENDS[key] = fn


def load_plugins() -> None:
    """Run every ``jaxrlworld.backends`` entry point once per process."""
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True
    for entry_point in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
        entry_point.load()()


def get_backend(sim_type: str) -> BackendSpec:
    """The external backend registered for ``sim_type``."""
    if sim_type not in _BACKENDS:
        load_plugins()
    if sim_type not in _BACKENDS:
        raise ValueError(
            f"Unknown sim_type={sim_type!r}. Built-in: {list(BUILTIN_SIM_TYPES)}; "
            f"registered through the {ENTRY_POINT_GROUP!r} entry points: {sorted(_BACKENDS)}."
        )
    return _BACKENDS[sim_type]


def find_backend_by_eval_key(eval_key: str) -> BackendSpec | None:
    """The external backend whose ``eval_key`` is ``eval_key``, if any."""
    load_plugins()
    for spec in _BACKENDS.values():
        if spec.eval_key == eval_key:
            return spec
    return None


def registered_backends() -> list[BackendSpec]:
    """Every external backend the entry points register."""
    load_plugins()
    return list(_BACKENDS.values())


def find_backend(sim_type: str) -> BackendSpec | None:
    """The external backend registered for ``sim_type``, if any."""
    load_plugins()
    return _BACKENDS.get(sim_type)


def get_dr_backend(term: str, sim_type: str) -> Callable[..., None]:
    """The registered ``sim_type`` implementation of DR term ``term``."""
    key = (term, sim_type)
    if key not in _DR_BACKENDS:
        load_plugins()
    if key not in _DR_BACKENDS:
        raise NotImplementedError(f"{term} has no backend for sim_type={sim_type!r}")
    return _DR_BACKENDS[key]


def resolve_env_class(env_name: str) -> type:
    """The env class named by ``cfgs.env.env_name``.

    A plain name (``"NewtonLocomotionEnv"``) is looked up on
    :mod:`jaxrlworld.rl.envs`, whose lazy ``__getattr__`` imports only that
    simulator. ``"package.module:ClassName"`` imports the class from
    another package.
    """
    if ":" in env_name:
        module_name, _, attr = env_name.partition(":")
        return getattr(importlib.import_module(module_name), attr)
    return getattr(_envs, env_name)
