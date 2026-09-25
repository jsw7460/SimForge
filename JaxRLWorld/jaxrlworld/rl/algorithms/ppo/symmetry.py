"""Left/right symmetry (mirror) support for on-policy PPO.

A trained symmetric-morphology policy should be left/right equivariant:
``pi(mirror(o)) == mirror(pi(o))``. Two ways to get there, both from Mittal
et al. (ICRA 2024) and both available here:

- **mirror loss** (``use_mirror_loss``): an auxiliary
  ``MSE(pi(mirror(o)), mirror(pi(o)))`` on the actor observations.
- **data augmentation** (``use_data_augmentation``): every minibatch is
  doubled with its mirror image -- actor obs, critic obs and actions
  mirrored, the rollout scalars (old log-prob, value, advantage, return)
  repeated -- and the PPO losses run on all 2N samples. The paper's
  preferred option; see :func:`augment_minibatch` for the exact rule,
  which follows rsl_rl's ``Symmetry.augment_batch``.

The mirror operator on the (flat) observation and action vectors is a single
gather + sign flip: ``x_mirror = x[..., perm] * sign``. The ``(perm, sign)``
vectors are built AUTOMATICALLY from the observation layout — each obs term's
resolved-function name maps to a local mirror rule, and per-term slices come
from ``obs_manager._group_term_indices``. Joint terms reuse an L<->R joint
permutation derived from ``joint_names`` (roll/yaw DOFs flip sign, pitch keeps).

Add a new obs function to :data:`_FIXED_TERM_RULES` (or the joint/phase sets)
when introducing a new observation term, or the spec build will raise — that is
deliberate: a silently-unmirrored term would corrupt the symmetry loss.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from jaxrlworld.rl.configs.scene.entity_selector import ResolvedEntity


class MirrorSpec(NamedTuple):
    """Static mirror operators (int perm + float sign): actor obs, action, critic obs.

    The mirror loss needs only the actor group. Data augmentation also
    evaluates the critic on mirrored samples, so it needs the critic
    group too; ``critic_perm``/``critic_sign`` are ``None`` unless the spec
    was built with ``include_critic=True`` (a critic may carry privileged
    entries with no mirror image, e.g. raw DR draws, and building its
    operator raises on any such term).
    """

    actor_perm: jnp.ndarray
    actor_sign: jnp.ndarray
    action_perm: jnp.ndarray
    action_sign: jnp.ndarray
    critic_perm: jnp.ndarray | None = None
    critic_sign: jnp.ndarray | None = None


# ── per-term local mirror rules (perm within the term, sign per element) ──────
# Frames follow the standard left-right mirror: y-axis position flips, so
# roll(x)/yaw(z) rotational components and lateral(y) linear components flip.
_FIXED_TERM_RULES: dict[str, tuple[list[int], list[float]]] = {
    # angular velocity [wx, wy, wz]: roll & yaw flip, pitch keeps.
    "base_ang_vel": ([0, 1, 2], [-1.0, 1.0, -1.0]),
    "base_ang_vel_w": ([0, 1, 2], [-1.0, 1.0, -1.0]),
    # projected gravity / linear velocity [x, y, z]: only y flips.
    "projected_gravity": ([0, 1, 2], [1.0, -1.0, 1.0]),
    "base_lin_vel": ([0, 1, 2], [1.0, -1.0, 1.0]),
    # velocity command [lin_vel_x, lin_vel_y, ang_vel_z]: y & yaw flip.
    "velocity_command": ([0, 1, 2], [1.0, -1.0, -1.0]),
    # base height (z): scalar, invariant under a left-right mirror.
    "base_height": ([0], [1.0]),
    # per-foot [left, right] scalars: swap the two feet.
    "feet_contact": ([1, 0], [1.0, 1.0]),
    "feet_air_time": ([1, 0], [1.0, 1.0]),
    # per-foot world linear velocity [Lx,Ly,Lz, Rx,Ry,Rz]: swap feet + flip y.
    "feet_lin_vel_w": ([3, 4, 5, 0, 1, 2], [1.0, -1.0, 1.0, 1.0, -1.0, 1.0]),
    # Privileged per-foot terms (common.proprioception), feet ordered
    # [left, right] by the preset's ``body_names``: scalars swap feet.
    "foot_height": ([1, 0], [1.0, 1.0]),
    "foot_air_time": ([1, 0], [1.0, 1.0]),
    "foot_contact_indicator": ([1, 0], [1.0, 1.0]),
    # per-foot contact force [Lx,Ly,Lz, Rx,Ry,Rz] (log-scaled, sign kept by
    # the scaling): swap feet AND flip the lateral component. A force is a
    # vector; its y component reverses under the mirror like a velocity's.
    "foot_contact_forces": ([3, 4, 5, 0, 1, 2], [1.0, -1.0, 1.0, 1.0, -1.0, 1.0]),
}
# Terms that mirror with the L<->R joint permutation (pos/vel/torque/action-like).
_JOINT_TERMS = frozenset(
    {
        "dof_pos_nominal_difference",
        "dof_pos_nominal_difference_biased",
        "dof_vel",
        "raw_actions",
        "applied_torque",
        # Joint-subset terms: their selector's ``joint_ids`` restricts the
        # L<->R permutation to the selected joints (see _subset_joint_rule).
        "joint_pos_rel",
        "joint_vel_rel",
    }
)
# Per-foot gait phase encoding: assumed layout [left_block(2), right_block(2)].
_PHASE_TERMS = frozenset({"gait_phase_encoding"})
# Mutable so out-of-tree obs terms can register (via the functions below).
_JOINT_TERMS = set(_JOINT_TERMS)


def register_mirror_rule(func_name: str, perm: list[int], sign: list[float]) -> None:
    """Register the left/right mirror rule of an out-of-tree obs term.

    ``perm``/``sign`` are LOCAL to the term's slice. Must run before
    :func:`build_mirror_spec` (i.e. at import time of the module that
    defines the obs function). Re-registration must be identical.
    """
    if len(perm) != len(sign):
        raise ValueError(f"perm/sign length mismatch for {func_name!r}")
    existing = _FIXED_TERM_RULES.get(func_name)
    if existing is not None and existing != (perm, sign):
        raise ValueError(f"conflicting mirror rule re-registration for {func_name!r}")
    _FIXED_TERM_RULES[func_name] = (list(perm), list(sign))


def register_joint_mirror_term(func_name: str) -> None:
    """Mark an out-of-tree obs term as joint-shaped (L<->R joint perm)."""
    _JOINT_TERMS.add(func_name)


def _joint_perm_sign(joint_names: Sequence[str]) -> tuple[list[int], list[float]]:
    """L<->R joint permutation + sign. Roll/Yaw DOFs (incl. central head yaw)
    flip sign under a left-right mirror; pitch DOFs keep. Involutive."""
    names = list(joint_names)
    perm, sign = [], []
    for n in names:
        if "Left" in n:
            mate = n.replace("Left", "Right")
        elif "Right" in n:
            mate = n.replace("Right", "Left")
        else:
            mate = n  # central joint (head)
        perm.append(names.index(mate))
        flip = ("Roll" in n) or ("Yaw" in n) or ("yaw" in n)  # include AAHead_yaw
        sign.append(-1.0 if flip else 1.0)
    return perm, sign


def _subset_joint_rule(joint_ids: Sequence[int], jperm, jsign) -> tuple[list[int], list[float]]:
    """The L<->R joint rule restricted to the canonical joints ``joint_ids``.

    Local index ``k`` holds canonical joint ``joint_ids[k]``; its mirror
    partner ``jperm[joint_ids[k]]`` must itself be selected, otherwise the
    subset has no mirror image (a left-leg-only term, say) and the build
    fails rather than mirroring half of it. For the full ``arange`` subset
    this is exactly the global rule.
    """
    ids = [int(i) for i in joint_ids]
    where = {j: k for k, j in enumerate(ids)}
    perm, sign = [], []
    for k, j in enumerate(ids):
        mate = int(jperm[j])
        if mate not in where:
            raise ValueError(
                f"joint subset {ids} is not closed under the left/right mirror: joint {j}'s mirror "
                f"partner {mate} is not selected"
            )
        perm.append(where[mate])
        sign.append(float(jsign[j]))
    return perm, sign


def _term_local_rule(
    func_name: str, dim: int, jperm, jsign, joint_ids: Sequence[int] | None = None
) -> tuple[list[int], list[float]]:
    if func_name in _FIXED_TERM_RULES:
        p, s = _FIXED_TERM_RULES[func_name]
        if len(p) != dim:
            raise ValueError(f"mirror rule for '{func_name}' has dim {len(p)} != term dim {dim}")
        return list(p), list(s)
    if func_name in _JOINT_TERMS:
        if joint_ids is not None:
            if dim != len(joint_ids):
                raise ValueError(f"joint term '{func_name}' dim {dim} != {len(joint_ids)} selected joints")
            return _subset_joint_rule(joint_ids, jperm, jsign)
        if dim != len(jperm):
            raise ValueError(f"joint term '{func_name}' dim {dim} != num joints {len(jperm)}")
        return list(jperm), list(jsign)
    if func_name in _PHASE_TERMS:
        if dim != 4:
            raise ValueError(f"phase term '{func_name}' expected dim 4, got {dim}")
        # swap left/right 2-blocks; sign kept.
        return [2, 3, 0, 1], [1.0, 1.0, 1.0, 1.0]
    raise ValueError(
        f"no mirror rule for obs/action term '{func_name}' (dim {dim}); "
        f"add it to symmetry._FIXED_TERM_RULES / _JOINT_TERMS / _PHASE_TERMS"
    )


def _build_group(obs_manager, group: str, jperm, jsign) -> tuple[list[int], list[float]]:
    """Assemble the global (perm, sign) for one observation group from its term
    slices in obs_manager._group_term_indices."""
    idx = obs_manager._group_term_indices[group]  # {term: (start, end)}
    terms = obs_manager._group_terms[group]  # {term: ObservationTermConfig}
    dim = int(obs_manager.obs_dict[group].shape[-1])
    perm = list(range(dim))
    sign = [1.0] * dim
    covered = 0
    for tname, (s, e) in idx.items():
        fname = terms[tname].resolved_func.__name__
        # A joint term reads its joints through a resolved selector whose
        # ``joint_ids`` (canonical order) is the whole robot for the default
        # selector and a subset for an explicit ``joint_names`` pattern.
        selector = terms[tname].params.get("asset_cfg")
        joint_ids = (
            selector.joint_ids.tolist() if isinstance(selector, ResolvedEntity) and fname in _JOINT_TERMS else None
        )
        lp, ls = _term_local_rule(fname, e - s, jperm, jsign, joint_ids)
        for k in range(e - s):
            perm[s + k] = s + lp[k]  # local perm is within the term slice
            sign[s + k] = ls[k]
        covered += e - s
    if covered != dim:
        raise ValueError(f"group '{group}': term slices cover {covered}/{dim} dims (gap in layout)")
    return perm, sign


def build_mirror_spec(obs_manager, joint_names: Sequence[str], include_critic: bool = False) -> MirrorSpec:
    """Build the MirrorSpec from the env's observation layout + joint names.

    ``joint_names`` must be the ACTION/obs joint order (e.g.
    ``env.act_manager.actuated_joint_names``). Raises if any obs term lacks a
    mirror rule or if a group's slices leave a gap (fail loud, never silent).
    ``include_critic`` additionally builds the critic operator, which data
    augmentation needs; it raises the same way on a critic term without a
    rule."""
    # Ensure the (lazy) per-term slice map + obs_dict are populated.
    obs_manager.calculate_obs_dim()
    jperm, jsign = _joint_perm_sign(joint_names)
    ap, as_ = _build_group(obs_manager, "actor", jperm, jsign)
    critic_perm = critic_sign = None
    if include_critic:
        cp, cs = _build_group(obs_manager, "critic", jperm, jsign)
        critic_perm = jnp.asarray(cp, dtype=jnp.int32)
        critic_sign = jnp.asarray(cs, dtype=jnp.float32)
    return MirrorSpec(
        actor_perm=jnp.asarray(ap, dtype=jnp.int32),
        actor_sign=jnp.asarray(as_, dtype=jnp.float32),
        action_perm=jnp.asarray(jperm, dtype=jnp.int32),
        action_sign=jnp.asarray(jsign, dtype=jnp.float32),
        critic_perm=critic_perm,
        critic_sign=critic_sign,
    )


def joint_mirror_perm_sign(joint_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """The L<->R joint permutation and sign of :func:`_joint_perm_sign`, as arrays."""
    perm, sign = _joint_perm_sign(joint_names)
    return np.asarray(perm, dtype=np.int64), np.asarray(sign, dtype=np.float32)


def group_mirror_operator(obs_manager, group: str, joint_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """``(perm, sign)`` of one observation group's mirror, from its live term layout.

    The same construction ``build_mirror_spec`` uses for the actor and
    critic, for any flat group (the AMP discriminator's, say). Raises on a
    term without a rule or on a joint subset that is not mirror-closed.
    """
    obs_manager.calculate_obs_dim()
    jperm, jsign = _joint_perm_sign(joint_names)
    perm, sign = _build_group(obs_manager, group, jperm, jsign)
    return np.asarray(perm, dtype=np.int64), np.asarray(sign, dtype=np.float32)


def mirror(x: jnp.ndarray, perm: jnp.ndarray, sign: jnp.ndarray) -> jnp.ndarray:
    """Mirror a (..., dim) array: gather by perm then flip signs."""
    return x[..., perm] * sign


def mirror_qpos(qpos: jnp.ndarray, joint_perm: jnp.ndarray, joint_sign: jnp.ndarray) -> jnp.ndarray:
    """Left-right mirror a free-joint qpos = [base_pos(3), base_quat_wxyz(4), joints].

    Physical (not observation) mirror, for viser / state-space validation:
      - base position: flip y  ->  [x, -y, z]
      - base orientation (wxyz): reflect across the y=0 plane, R -> M R M with
        M=diag(1,-1,1), which for a unit quaternion is [w, -x, y, -z] (derived +
        verified). Improper reflection maps to a proper rotation here.
      - joints: L<->R permutation + roll/yaw sign flip.
    """
    base_pos = qpos[..., :3] * jnp.asarray([1.0, -1.0, 1.0])
    q = qpos[..., 3:7]
    base_quat = jnp.stack([q[..., 0], -q[..., 1], q[..., 2], -q[..., 3]], axis=-1)
    joints = qpos[..., 7:][..., joint_perm] * joint_sign
    return jnp.concatenate([base_pos, base_quat, joints], axis=-1)


def mirror_qvel(qvel: jnp.ndarray, joint_perm: jnp.ndarray, joint_sign: jnp.ndarray) -> jnp.ndarray:
    """Left-right mirror qvel = [base_lin(3), base_ang(3), joint_vel].

    base linear velocity flips y; base angular velocity flips roll(x)/yaw(z)
    (pitch keeps); joint velocities take the L<->R permutation + sign.
    """
    lin = qvel[..., :3] * jnp.asarray([1.0, -1.0, 1.0])
    ang = qvel[..., 3:6] * jnp.asarray([-1.0, 1.0, -1.0])
    jv = qvel[..., 6:][..., joint_perm] * joint_sign
    return jnp.concatenate([lin, ang, jv], axis=-1)


def augment_minibatch(
    spec: MirrorSpec,
    actor_obs: jnp.ndarray,
    critic_obs: jnp.ndarray,
    actions: jnp.ndarray,
    old_log_probs: jnp.ndarray,
    old_values: jnp.ndarray,
    advantages: jnp.ndarray,
    returns: jnp.ndarray,
) -> tuple[jnp.ndarray, ...]:
    """Double a minibatch with its mirror image (rsl_rl ``Symmetry.augment_batch``).

    Rows ``[:N]`` are the originals, rows ``[N:]`` their mirrors. Actor obs,
    critic obs and actions are mirrored; the rollout scalars are repeated,
    so a mirrored sample keeps the ORIGINAL sample's old log-prob, value,
    advantage and return. The PPO ratio of a mirrored row is therefore
    ``pi_new(K a | L o) / pi_old(a | o)``, which is what makes the update
    pull the policy toward equivariance (Mittal et al. 2024, Eq. 6). The
    caller keeps entropy and the adaptive-KL statistic on the originals
    only, as rsl_rl does. Call after advantage normalization, so the
    statistic is the one the N originals define.
    """
    if spec.critic_perm is None:
        raise ValueError("data augmentation needs a MirrorSpec built with include_critic=True")
    actor_obs = jnp.concatenate([actor_obs, mirror(actor_obs, spec.actor_perm, spec.actor_sign)], axis=0)
    critic_obs = jnp.concatenate([critic_obs, mirror(critic_obs, spec.critic_perm, spec.critic_sign)], axis=0)
    actions = jnp.concatenate([actions, mirror(actions, spec.action_perm, spec.action_sign)], axis=0)
    repeat = lambda x: jnp.concatenate([x, x], axis=0)  # noqa: E731 - one-line local helper
    return (
        actor_obs,
        critic_obs,
        actions,
        repeat(old_log_probs),
        repeat(old_values),
        repeat(advantages),
        repeat(returns),
    )


def symmetry_mirror_loss(model, actor_obs: jnp.ndarray, spec: MirrorSpec, key) -> jnp.ndarray:
    """MSE( pi(mirror(o)).mean , mirror(pi(o).mean) ).

    Enforces left/right equivariance of the policy mean. The mirrored-target
    branch is stop-gradient'd (rsl_rl convention) so the loss pulls the mirrored
    prediction toward the mirrored original, not both toward each other."""
    dist, _ = model.get_distribution(actor_obs, key=key)
    obs_m = mirror(actor_obs, spec.actor_perm, spec.actor_sign)
    dist_m, _ = model.get_distribution(obs_m, key=key)
    target = mirror(dist.mean, spec.action_perm, spec.action_sign)
    diff = dist_m.mean - jax.lax.stop_gradient(target)
    return jnp.mean(diff**2)
