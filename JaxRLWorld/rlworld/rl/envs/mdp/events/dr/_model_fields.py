"""Per-world model-field declarations for MuJoCo domain randomization.

MuJoCo Warp stores model fields as a single shared ``(1, ...)`` row: every
world reads the same ``body_mass``, the same ``dof_damping``, and so on.
Per-environment randomization therefore requires *expanding* the field to
``(num_envs, ...)`` first, via ``Simulation.expand_model_fields``. That call
reallocates GPU arrays and re-captures the CUDA graphs, so it runs exactly
once at env build time — long before any DR term executes. Something has to
tell it, up front, which fields the registered DR terms are going to write.

Two failure modes make getting that list right load-bearing:

* **Unexpanded target field.** Writing per-env values into a still-shared
  ``(1, ...)`` row makes every world take the last value written. There is no
  error and no warning: the logs show DR resampling every reset while the
  policy in fact trains against a single draw.
* **Unexpanded derived field.** ``recompute_constants(level)`` launches
  mujoco-warp kernels that iterate over every world and *write* derived
  constants. A derived field still at ``(1, ...)`` turns those writes into
  out-of-bounds stores — absorbed by memory-pool slack (silent corruption) at
  small ``num_envs``, and a hard ``CUDA_ERROR_ILLEGAL_ADDRESS`` at large ones.
  Small-scale smoke tests pass either way.

This module keeps the declaration next to the function that does the writing:

    @requires_model_fields("body_mass", recompute=RecomputeLevel.set_const)
    def randomize_body_mass(env, env_ids, ...):
        ...

:func:`collect_expand_fields` then reads the declarations off the registered
event terms, so adding a DR function cannot silently skip the expansion — and
:func:`collect_expand_fields` raises on any DR-package term that forgot the
decorator. Mirrors mjlab's ``requires_model_fields`` /
``EventManager.domain_randomization_fields``.

Simulator independence: nothing here imports mjlab or mujoco_warp, so
``unified.py`` stays importable in Genesis-/Newton-only processes. See
:class:`RecomputeLevel` for the one contract that couples us to mujoco-warp.
"""

from __future__ import annotations

import enum
from typing import Any, Callable

from rlworld.rl.configs.base_config import iter_terms
from rlworld.rl.configs.events.event_term_config import EventTermConfig

# Import path prefix of the DR term package. Any event term whose function is
# defined under here is expected to carry a ``model_fields`` declaration;
# :func:`collect_expand_fields` raises otherwise rather than silently skipping
# the expansion (which is the failure this module exists to prevent).
_DR_PACKAGE = "rlworld.rl.envs.mdp.events.dr"


class RecomputeLevel(enum.IntEnum):
    """How much derived model state goes stale after a DR write.

    Ordered: a higher level recomputes a superset of the lower ones, so
    batching several DR terms only needs ``max()`` over their levels.

    **Member names are a contract.** ``mjlab``'s ``Simulation.recompute_constants``
    dispatches with ``getattr(mujoco_warp, level.name)``, so these names must
    match mujoco-warp's ``set_const*`` entry points. Declaring our own enum
    (rather than importing mjlab's) is what keeps this module — and therefore
    ``unified.py`` — free of a module-level mjlab import.
    ``MujocoEnv._pre_manager_setup`` asserts the names still line up.
    """

    none = 0
    """Nothing to recompute (e.g. ``geom_friction``, ``dof_damping``)."""

    set_const_fixed = 1
    """Recompute after ``body_gravcomp`` changes."""

    set_const_0 = 2
    """Recompute after ``dof_armature`` / ``body_inertia`` / ``body_pos`` /
    ``body_quat`` / ``qpos0`` changes."""

    set_const = 3
    """Full recomputation. Required after ``body_mass`` / ``body_ipos``."""


# Fields WRITTEN by each mujoco-warp ``set_const*`` entry point, i.e. the
# fields that must already be per-world before a recompute at that level runs.
#
# Verified against the installed mujoco-warp by reading ``_src/io.py``:
# ``set_const`` calls ``set_const_0``, ``set_const_fixed`` and
# ``set_const_spring``; the write sets are as listed below (reads such as
# ``body_mass``, ``qpos0`` or ``qpos_spring`` are deliberately absent — only
# writes can go out of bounds).
#
# This table is intentionally a SUPERSET of mjlab's own
# ``event_manager._DERIVED_FIELDS``, which lists only five ``set_const_0``
# entries and omits the actuator / camera / light / equality writes below.
# Keep ours until upstream widens theirs; re-verify after every mujoco-warp
# bump (``rlworld/scripts/diag/dr_model_fields_diag.py`` checks it).
_SET_CONST_0_FIELDS: tuple[str, ...] = (
    "dof_invweight0",
    "body_invweight0",
    "tendon_length0",
    "tendon_invweight0",
    "actuator_acc0",
    "actuator_biasprm",
    "cam_pos0",
    "cam_poscom0",
    "cam_mat0",
    "light_pos0",
    "light_poscom0",
    "light_dir0",
    "eq_data",
)

DERIVED_FIELDS: dict[RecomputeLevel, tuple[str, ...]] = {
    RecomputeLevel.none: (),
    RecomputeLevel.set_const_fixed: ("body_subtreemass",),
    RecomputeLevel.set_const_0: _SET_CONST_0_FIELDS,
    RecomputeLevel.set_const: ("body_subtreemass",) + _SET_CONST_0_FIELDS + ("tendon_lengthspring",),
}


def requires_model_fields(
    *fields: str,
    recompute: RecomputeLevel = RecomputeLevel.none,
) -> Callable[[Callable], Callable]:
    """Declare the MuJoCo model fields a DR term writes.

    Attaches ``model_fields`` (the declared fields plus every derived field
    implied by *recompute*) and ``recompute`` to the decorated function.
    :func:`collect_expand_fields` reads the former; the MuJoCo backends read
    the latter so the expansion set and the recompute level cannot drift apart.

    Decorate DR entry points that touch no MuJoCo model field with a bare
    ``@requires_model_fields()`` — an empty declaration is a statement that the
    author considered the question, whereas a missing decorator is an error.

    Args:
        *fields: MuJoCo model field names written by this term.
        recompute: Level of derived-constant recomputation the write requires.

    Returns:
        A decorator that annotates the function in place.
    """
    derived = DERIVED_FIELDS[recompute]
    all_fields = tuple(fields) + tuple(f for f in derived if f not in fields)

    def decorator(func: Callable) -> Callable:
        func.model_fields = all_fields
        func.recompute = recompute
        return func

    return decorator


def collect_expand_fields(event_cfg: Any) -> tuple[str, ...]:
    """Model fields to hand to ``Simulation.expand_model_fields`` for *event_cfg*.

    Walks the registered event terms and unions:

    1. ``model_fields`` declared via :func:`requires_model_fields`.
    2. The ``field`` param of mjlab-native DR terms (``mode="startup"`` with an
       explicit ``field``), which name their target field in the config itself.

    Order is preserved and duplicates removed, so the result can be passed
    straight through.

    Raises:
        RuntimeError: If a term's function lives in the DR package but carries
            no ``model_fields`` declaration. That combination is always a
            missing decorator, and letting it through would reintroduce the
            silent single-value-shared-by-every-env failure.
    """
    fields: list[str] = []
    for term_name, term in iter_terms(event_cfg, EventTermConfig).items():
        func = term.resolved_func

        if term.mode == "startup" and "field" in term.params:
            fields.append(term.params["field"])

        declared = getattr(func, "model_fields", None)
        if declared is None:
            module = getattr(func, "__module__", "") or ""
            if module.startswith(_DR_PACKAGE):
                raise RuntimeError(
                    f"Event term {term_name!r} ({module}.{getattr(func, '__qualname__', func)}) "
                    f"is a domain-randomization term but declares no model fields. Decorate it "
                    f"with @requires_model_fields(...) from "
                    f"rlworld.rl.envs.mdp.events.dr._model_fields, listing the MuJoCo model "
                    f"fields its mujoco backend writes (use a bare @requires_model_fields() if "
                    f"it writes none). Without the declaration the field is never expanded "
                    f"per-world and every environment silently shares one randomized value."
                )
            continue
        fields.extend(declared)

    return tuple(dict.fromkeys(fields))
