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
* **Unexpanded derived field.** ``recompute_constants(groups)`` launches
  mujoco-warp kernels that iterate over every world and *write* derived
  constants. A derived field still at ``(1, ...)`` turns those writes into
  out-of-bounds stores — absorbed by memory-pool slack (silent corruption) at
  small ``num_envs``, and a hard ``CUDA_ERROR_ILLEGAL_ADDRESS`` at large ones.
  Small-scale smoke tests pass either way.

This module keeps the declaration next to the function that does the writing:

    @requires_model_fields("body_mass", recompute=RecomputeGroups.ALL)
    def randomize_body_mass(env, env_ids, ...):
        ...

:func:`collect_expand_fields` then reads the declarations off the registered
event terms, so adding a DR function cannot silently skip the expansion — and
:func:`collect_expand_fields` raises on any DR-package term that forgot the
decorator. Mirrors mjlab's ``requires_model_fields`` /
``EventManager.domain_randomization_fields``.

Simulator independence: nothing here imports mjlab or mujoco_warp, so
``unified.py`` stays importable in Genesis-/Newton-only processes. See
:class:`RecomputeGroups` for the one contract that couples us to mujoco-warp.
"""

from __future__ import annotations

import enum
from typing import Any, Callable

from jaxrlworld.rl.configs.base_config import iter_terms
from jaxrlworld.rl.configs.events.event_term_config import EventTermConfig

# Import path prefix of the DR term package. Any event term whose function is
# defined under here is expected to carry a ``model_fields`` declaration;
# :func:`collect_expand_fields` raises otherwise rather than silently skipping
# the expansion (which is the failure this module exists to prevent).
_DR_PACKAGE = "jaxrlworld.rl.envs.mdp.events.dr"


class RecomputeGroups(enum.IntFlag):
    """The groups of derived model constants a DR write leaves stale.

    Bit flags: several DR terms batched together OR their groups, and one
    ``Simulation.recompute_constants`` call runs the union.

    **Member names and values are a contract.** ``mjlab``'s
    ``Simulation.recompute_constants`` hands the flags to
    ``mujoco_warp.set_const(groups=...)``, which tests each bit against
    ``mujoco_warp.SetConstGroup``, so these members mirror that enum bit for
    bit. Declaring our own flags (rather than importing mujoco-warp's) is what
    keeps this module -- and therefore ``unified.py`` -- free of a module-level
    mujoco-warp import. ``MujocoEnv._pre_manager_setup`` asserts the two still
    line up.
    """

    NONE = 0
    """Nothing to recompute (e.g. ``geom_friction``, ``dof_damping``)."""

    FIXED = 1 << 0
    """Constants independent of the configuration: subtree masses, geom bounds.
    Needed after ``body_mass`` / ``geom_size`` changes."""

    KIN0 = 1 << 1
    """Kinematics in ``qpos0``: tendon rest lengths, camera / light reference
    poses, equality anchors. Needed after ``body_ipos`` / ``body_pos`` /
    ``body_quat`` / ``qpos0`` changes."""

    INERT0 = 1 << 2
    """Inertia matrix in ``qpos0``: the inverse weights and actuator
    accelerations. Needed after ``dof_armature`` / ``body_inertia`` /
    ``body_mass`` / ``body_ipos`` changes."""

    ALL = FIXED | KIN0 | INERT0


# Fields WRITTEN by each mujoco-warp ``set_const`` group, i.e. the fields that
# must already be per-world before a recompute of that group runs. Reads
# (``body_mass``, ``qpos0``) are deliberately absent: only writes can go out of
# bounds. The geom bounds FIXED recomputes (``geom_rbound``, ``geom_aabb``)
# are written modulo their own row count, so they are expanded only along
# with a per-world ``geom_size``, which no term of ours randomizes.
#
# Mirrors mjlab's ``event_manager._DERIVED_FIELDS``; re-verify after every
# mujoco-warp bump (``jaxrlworld/scripts/diag/dr/dr_model_fields_diag.py``
# checks it against the installed source).
DERIVED_FIELDS: dict[RecomputeGroups, tuple[str, ...]] = {
    RecomputeGroups.FIXED: ("body_subtreemass",),
    RecomputeGroups.KIN0: (
        "tendon_length0",
        "eq_data",
        "cam_pos0",
        "cam_poscom0",
        "cam_mat0",
        "light_pos0",
        "light_poscom0",
        "light_dir0",
    ),
    RecomputeGroups.INERT0: (
        "dof_invweight0",
        "body_invweight0",
        "tendon_invweight0",
        "actuator_acc0",
    ),
}


def derived_fields(groups: RecomputeGroups) -> tuple[str, ...]:
    """The fields a recompute of ``groups`` writes, in table order."""
    return tuple(field for group, fields in DERIVED_FIELDS.items() if group & groups for field in fields)


def requires_model_fields(
    *fields: str,
    recompute: RecomputeGroups = RecomputeGroups.NONE,
) -> Callable[[Callable], Callable]:
    """Declare the MuJoCo model fields a DR term writes.

    Attaches ``model_fields`` (the declared fields plus every derived field
    implied by *recompute*) and ``recompute`` to the decorated function.
    :func:`collect_expand_fields` reads the former; the MuJoCo backends read
    the latter so the expansion set and the recompute groups cannot drift apart.

    Decorate DR entry points that touch no MuJoCo model field with a bare
    ``@requires_model_fields()`` — an empty declaration is a statement that the
    author considered the question, whereas a missing decorator is an error.

    Args:
        *fields: MuJoCo model field names written by this term.
        recompute: Groups of derived constants the write leaves stale.

    Returns:
        A decorator that annotates the function in place.
    """
    derived = derived_fields(recompute)
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

        declared = getattr(func, "model_fields", None)  # dynamic-attr: optional @requires_model_fields marker
        if declared is None:
            module = getattr(func, "__module__", "") or ""  # dynamic-attr: arbitrary callable may lack it
            if module.startswith(_DR_PACKAGE):
                qualname = getattr(func, "__qualname__", func)  # dynamic-attr: arbitrary callable may lack it
                raise RuntimeError(
                    f"Event term {term_name!r} ({module}.{qualname}) "
                    f"is a domain-randomization term but declares no model fields. Decorate it "
                    f"with @requires_model_fields(...) from "
                    f"jaxrlworld.rl.envs.mdp.events.dr._model_fields, listing the MuJoCo model "
                    f"fields its mujoco backend writes (use a bare @requires_model_fields() if "
                    f"it writes none). Without the declaration the field is never expanded "
                    f"per-world and every environment silently shares one randomized value."
                )
            continue
        fields.extend(declared)

    return tuple(dict.fromkeys(fields))
