"""Genesis-specific event terms.

Functions that rely on Genesis-only APIs (e.g. ``rigid_solver``) and
have no cross-sim equivalent. General-purpose reset / push functions
live in ``common.py``; cross-sim domain-randomization terms live in
``dr/unified.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import genesis as gs
import torch

from jaxrlworld.rl.envs.mdp.events.keyed_sampler import TermSampler
from jaxrlworld.rl.utils import entity_utils as eu

if TYPE_CHECKING:
    from jaxrlworld.rl.envs import GenesisEnv


def apply_external_force_torque(
    env: GenesisEnv,
    env_ids: torch.Tensor,
    sampler: TermSampler,
    force_range: dict[str, tuple[float, float]],
    torque_range: dict[str, tuple[float, float]] | None = None,
    body_name: str = "base",
) -> None:
    """Apply random external force and torque to robot body.

    Args:
        env: The environment instance.
        env_ids: Environment indices to apply force to.
        force_range: Dict with 'x', 'y', 'z' keys, each mapping to (min, max) range.
        torque_range: Dict with 'x', 'y', 'z' keys, each mapping to (min, max) range.
            If None, no torque is applied.
        body_name: Name of the body to apply force to.
    """
    if len(env_ids) == 0:
        return

    robot = env.scene_manager["robot"]
    rigid_solver = env.scene.rigid_solver

    # Get global link index
    link_ids_global, _ = eu.find_links(robot, [body_name], global_ids=True)

    # Sample random forces: (n, 3), axis index as the coordinate (stream 0)
    n = len(env_ids)
    axes = (torch.arange(3, device=env.device)[None, :],)
    force_u = sampler.uniforms(env_ids, axes, stream=0)
    forces = torch.zeros(n, 3, device=env.device)
    for i, key in enumerate(["x", "y", "z"]):
        lo, hi = force_range.get(key, (0.0, 0.0))
        forces[:, i] = (hi - lo) * force_u[:, i] + lo

    rigid_solver.apply_links_external_force(
        force=forces,
        links_idx=link_ids_global,
        envs_idx=env_ids.tolist(),
        ref=gs.link_ref_frame.link_COM,
        local=False,
    )

    # Apply torque if specified
    if torque_range is not None:
        torque_u = sampler.uniforms(env_ids, axes, stream=1)
        torques = torch.zeros(n, 3, device=env.device)
        for i, key in enumerate(["x", "y", "z"]):
            lo, hi = torque_range.get(key, (0.0, 0.0))
            torques[:, i] = (hi - lo) * torque_u[:, i] + lo

        rigid_solver.apply_links_external_torque(
            torque=torques,
            links_idx=link_ids_global,
            envs_idx=env_ids.tolist(),
            ref=gs.link_ref_frame.link_COM,
            local=False,
        )
