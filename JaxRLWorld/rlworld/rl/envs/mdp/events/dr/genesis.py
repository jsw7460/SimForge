"""Genesis domain randomization terms.

All functions follow the standard event term signature::

    func(env: GenesisEnv, env_ids: torch.Tensor, **params) -> None

Genesis handles ``scale`` / ``add`` semantics internally via its
``set_friction_ratio``, ``set_mass_shift``, and ``set_COM_shift`` APIs,
so a ``DefaultCache`` is generally not needed here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rlworld.rl.utils import entity_utils as eu

from ._utils import sample

if TYPE_CHECKING:
    from rlworld.rl.envs import GenesisEnv


# ------------------------------------------------------------------ #
#  Friction                                                           #
# ------------------------------------------------------------------ #


def randomize_friction(
    env: GenesisEnv,
    env_ids: torch.Tensor,
    friction_range: tuple[float, float] = (0.6, 1.4),
    distribution: str = "uniform",
    entity_name: str = "robot",
    link_names: tuple[str, ...] | None = None,
    shared_random: bool = False,
) -> None:
    """Randomize friction ratio for robot links.

    Genesis ``set_friction_ratio`` multiplies the original friction by the
    given ratio, so *friction_range* is always a multiplier range.

    Args:
        env: Genesis environment instance.
        env_ids: Environment indices to randomize.
        friction_range: ``(min_ratio, max_ratio)`` multiplier on default friction.
        distribution: ``"uniform"`` | ``"log_uniform"`` | ``"gaussian"``.
        entity_name: Name of the robot entity in the scene.
        link_names: Optional link-name patterns (regex). When ``None``, every
            link is randomized (legacy behavior). When set, only matched
            links are touched — used to mirror mjlab's foot-only friction
            DR via ``cfg.robot.foot_names``.
        shared_random: When ``True``, draw a single ratio per env and
            broadcast it across all matched links. Mirrors mjlab's
            ``shared_random=True`` so the four feet receive the same
            friction ratio within an env.
    """
    if len(env_ids) == 0:
        return

    entity = env.scene_manager[entity_name]

    if link_names is None:
        links_idx = list(range(entity.n_links))
    else:
        links_idx, _ = eu.find_links(entity, list(link_names), global_ids=False)
    n_target = len(links_idx)

    if shared_random:
        ratios = (
            sample((len(env_ids), 1), *friction_range, env.device, distribution)
            .expand(len(env_ids), n_target)
            .contiguous()
        )
    else:
        ratios = sample((len(env_ids), n_target), *friction_range, env.device, distribution)

    entity.set_friction_ratio(
        friction_ratio=ratios,
        links_idx_local=links_idx,
        envs_idx=env_ids,
    )


# ------------------------------------------------------------------ #
#  Body mass                                                          #
# ------------------------------------------------------------------ #


def randomize_body_mass(
    env: GenesisEnv,
    env_ids: torch.Tensor,
    mass_ratio_range: tuple[float, float] = (0.85, 1.15),
    distribution: str = "uniform",
    link_names: str | list[str] | tuple[str, ...] = ("base",),
    entity_name: str = "robot",
) -> None:
    """Randomize mass for specified links via ``set_mass_shift``.

    Samples a mass *ratio* and converts to a shift:
    ``mass_shift = original_mass * (ratio - 1)``.

    Args:
        env: Genesis environment instance.
        env_ids: Environment indices to randomize.
        mass_ratio_range: ``(min_ratio, max_ratio)`` multiplier on default mass.
        distribution: ``"uniform"`` | ``"log_uniform"`` | ``"gaussian"``.
        link_names: Link name(s) or regex patterns to randomize.
        entity_name: Name of the robot entity in the scene.
    """
    if len(env_ids) == 0:
        return

    entity = env.scene_manager[entity_name]
    if isinstance(link_names, str):
        link_names = [link_names]
    else:
        link_names = list(link_names)
    links_idx, _ = eu.find_links(entity, link_names, global_ids=False)

    n_envs = len(env_ids)
    n_links = len(links_idx)
    ratios = sample((n_envs, n_links), *mass_ratio_range, env.device, distribution)

    # Convert ratios to shifts: shift = original * (ratio - 1)
    mass_shift = torch.zeros(n_envs, n_links, device=env.device)
    for i, idx in enumerate(links_idx):
        original_mass = entity.links[idx].get_mass()
        mass_shift[:, i] = original_mass * (ratios[:, i] - 1.0)

    entity.set_mass_shift(
        mass_shift=mass_shift,
        links_idx_local=links_idx,
        envs_idx=env_ids,
    )


# ------------------------------------------------------------------ #
#  Body COM offset                                                    #
# ------------------------------------------------------------------ #


def randomize_body_com_offset(
    env: GenesisEnv,
    env_ids: torch.Tensor,
    ranges: dict[int, tuple[float, float]],
    link_names: str | list[str] | tuple[str, ...] = ("torso_link",),
    entity_name: str = "robot",
) -> None:
    """Randomize body COM offset for specified links.

    Uses Genesis ``set_COM_shift`` which is additive on the original COM.

    Args:
        env: Genesis environment instance.
        env_ids: Environment indices to randomize.
        ranges: Per-axis ``{axis_index: (min, max)}`` offset in meters.
        link_names: Link name(s) or regex patterns.
        entity_name: Name of the robot entity in the scene.
    """
    if len(env_ids) == 0:
        return

    entity = env.scene_manager[entity_name]
    if isinstance(link_names, str):
        link_names = [link_names]
    else:
        link_names = list(link_names)
    links_idx, _ = eu.find_links(entity, link_names, global_ids=False)

    n_envs = len(env_ids)
    n_links = len(links_idx)

    com_shift = torch.zeros(n_envs, n_links, 3, device=env.device)
    for axis, (lo, hi) in ranges.items():
        com_shift[:, :, axis] = torch.empty(
            n_envs,
            n_links,
            device=env.device,
        ).uniform_(lo, hi)

    entity.set_COM_shift(
        com_shift=com_shift,
        links_idx_local=links_idx,
        envs_idx=env_ids,
    )


# ------------------------------------------------------------------ #
#  PD gains                                                           #
# ------------------------------------------------------------------ #


def randomize_pd_gains(
    env: GenesisEnv,
    env_ids: torch.Tensor,
    kp_range: tuple[float, float] | None = None,
    kd_range: tuple[float, float] | None = None,
    distribution: str = "uniform",
    entity_name: str = "robot",
) -> None:
    """Randomize PD gains (kp / kv) for all actuated DOFs.

    Samples a *ratio* and multiplies the current gains, since Genesis
    ``set_dofs_kp`` / ``set_dofs_kv`` set absolute values and there is no
    built-in ratio API.

    Args:
        env: Genesis environment instance.
        env_ids: Environment indices to randomize.
        kp_range: ``(min_ratio, max_ratio)`` for proportional gain. ``None`` to skip.
        kd_range: ``(min_ratio, max_ratio)`` for derivative gain. ``None`` to skip.
        distribution: ``"uniform"`` | ``"log_uniform"`` | ``"gaussian"``.
        entity_name: Name of the robot entity in the scene.
    """
    if len(env_ids) == 0:
        return

    entity = env.scene_manager[entity_name]
    n_dofs = entity.n_dofs

    if kp_range is not None:
        current_kp = entity.get_dofs_kp()  # (n_dofs,) or (n_envs, n_dofs)
        ratios = sample((len(env_ids), n_dofs), *kp_range, env.device, distribution)
        entity.set_dofs_kp(
            kp=(current_kp * ratios).cpu().numpy()
            if current_kp.dim() == 1
            else (current_kp[env_ids] * ratios).cpu().numpy(),
            envs_idx=env_ids,
        )

    if kd_range is not None:
        current_kv = entity.get_dofs_kv()
        ratios = sample((len(env_ids), n_dofs), *kd_range, env.device, distribution)
        entity.set_dofs_kv(
            kv=(current_kv * ratios).cpu().numpy()
            if current_kv.dim() == 1
            else (current_kv[env_ids] * ratios).cpu().numpy(),
            envs_idx=env_ids,
        )


# ------------------------------------------------------------------ #
#  Joint armature                                                     #
# ------------------------------------------------------------------ #


def randomize_joint_armature(
    env: GenesisEnv,
    env_ids: torch.Tensor,
    armature_range: tuple[float, float] = (0.9, 1.1),
    distribution: str = "uniform",
    entity_name: str = "robot",
) -> None:
    """Randomize joint armature (reflected rotor inertia).

    Samples a ratio and multiplies the current armature values.
    Requires ``batch_dofs_info=True`` in ``RigidOptions``.

    Args:
        env: Genesis environment instance.
        env_ids: Environment indices to randomize.
        armature_range: ``(min_ratio, max_ratio)`` multiplier on current armature.
        distribution: ``"uniform"`` | ``"log_uniform"`` | ``"gaussian"``.
        entity_name: Name of the robot entity in the scene.
    """
    if len(env_ids) == 0:
        return

    entity = env.scene_manager[entity_name]
    n_dofs = entity.n_dofs
    current = entity.get_dofs_armature()
    ratios = sample((len(env_ids), n_dofs), *armature_range, env.device, distribution)

    entity.set_dofs_armature(
        armature=(current * ratios).cpu().numpy() if current.dim() == 1 else (current[env_ids] * ratios).cpu().numpy(),
        envs_idx=env_ids,
    )


# ------------------------------------------------------------------ #
#  Joint friction loss                                                #
# ------------------------------------------------------------------ #


def randomize_joint_friction(
    env: GenesisEnv,
    env_ids: torch.Tensor,
    friction_range: tuple[float, float] = (0.0, 0.05),
    distribution: str = "uniform",
    entity_name: str = "robot",
) -> None:
    """Randomize joint friction loss (load-independent).

    Sets absolute friction values. Requires ``batch_dofs_info=True``
    in ``RigidOptions``.

    Args:
        env: Genesis environment instance.
        env_ids: Environment indices to randomize.
        friction_range: ``(min, max)`` absolute friction loss values.
        distribution: ``"uniform"`` | ``"log_uniform"`` | ``"gaussian"``.
        entity_name: Name of the robot entity in the scene.
    """
    if len(env_ids) == 0:
        return

    entity = env.scene_manager[entity_name]
    n_dofs = entity.n_dofs
    values = sample((len(env_ids), n_dofs), *friction_range, env.device, distribution)

    entity.set_dofs_frictionloss(
        frictionloss=values.cpu().numpy(),
        envs_idx=env_ids,
    )
