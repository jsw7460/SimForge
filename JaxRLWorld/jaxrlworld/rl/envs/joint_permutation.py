"""Joint-order permutation between a simulator's joint order and a checkpoint's.

Simulators order the same robot's joints differently (Genesis follows the
URDF, Newton and MuJoCo the pattern order), so a policy trained on one
backend reads joint-indexed observation columns and writes action columns
in that backend's order. Cross-simulator evaluation maps between the two:
:class:`JointPermutation` reorders actions into the eval simulator's order
and observations back into the canonical one, over the column slices that
:func:`find_joint_obs_slices` identifies in each observation group.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple

import torch

from jaxrlworld.rl.envs.mdp.observations.joint_indexed import joint_indexed_flag

if TYPE_CHECKING:
    from jaxrlworld.rl.envs import World


class JointPermutation:
    """Maps between a sub-environment's joint order and the canonical order.

    The canonical order is the one the checkpoint was trained with
    (``canonical_joint_names`` in its train_state.yaml).

    Attributes:
        is_identity:  True when no reordering is needed (fast path).
        action_perm:  Index tensor to reorder canonical actions → sim order.
        obs_perms:    Per obs-group index tensor to reorder sim obs → canonical.
    """

    def __init__(
        self,
        canonical_names: List[str],
        sim_names: List[str],
        obs_group_joint_slices: Dict[str, List[Tuple[int, int]]],
        obs_group_dims: Dict[str, int],
        device: torch.device,
    ):
        n = len(canonical_names)
        assert len(sim_names) == n, f"Joint count mismatch: canonical={n}, sim={len(sim_names)}"

        # Strip prefix (Newton adds "g1_29dof/" etc.)
        def _bare(name: str) -> str:
            return name.rsplit("/", 1)[-1]

        canonical_bare = [_bare(n) for n in canonical_names]
        sim_bare = [_bare(n) for n in sim_names]

        # Validate same set of joints
        if set(canonical_bare) != set(sim_bare):
            only_canonical = set(canonical_bare) - set(sim_bare)
            only_sim = set(sim_bare) - set(canonical_bare)
            raise ValueError(f"Joint name mismatch!\n  Only in canonical: {only_canonical}\n  Only in sim: {only_sim}")

        # ── Build permutation indices ──
        # sim_to_canon[s] = c  means sim_names[s] == canonical_names[c]
        canonical_idx = {name: i for i, name in enumerate(canonical_bare)}
        sim_to_canon_list = [canonical_idx[b] for b in sim_bare]
        self._sim_to_canon = torch.tensor(sim_to_canon_list, device=device, dtype=torch.long)

        # canon_to_sim[c] = s  means canonical_names[c] == sim_names[s]
        sim_idx = {name: i for i, name in enumerate(sim_bare)}
        canon_to_sim_list = [sim_idx[b] for b in canonical_bare]
        self._canon_to_sim = torch.tensor(canon_to_sim_list, device=device, dtype=torch.long)

        # ── Identity check ──
        identity = torch.arange(n, device=device, dtype=torch.long)
        self.is_identity = bool(torch.equal(self._sim_to_canon, identity))

        # ── Action permutation (canonical → sim) ──
        # sim_actions[:, s] = canonical_actions[:, sim_to_canon[s]]
        # => sim_actions = canonical_actions[:, sim_to_canon]
        self.action_perm = self._sim_to_canon

        # ── Obs permutation per group (sim → canonical) ──
        self.obs_perms: Dict[str, torch.Tensor] = {}
        for group_name, joint_slices in obs_group_joint_slices.items():
            obs_dim = obs_group_dims[group_name]
            perm = torch.arange(obs_dim, device=device, dtype=torch.long)
            if not self.is_identity:
                for start, end in joint_slices:
                    # perm[start + c] = start + canon_to_sim[c]
                    for c in range(n):
                        perm[start + c] = start + self._canon_to_sim[c].item()
            self.obs_perms[group_name] = perm

    def permute_actions(self, canonical_actions: torch.Tensor) -> torch.Tensor:
        """Reorder actions from canonical joint order to this sim's order."""
        if self.is_identity:
            return canonical_actions
        return canonical_actions[:, self.action_perm]

    def permute_obs(self, sim_obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Reorder observations from this sim's joint order to canonical."""
        if self.is_identity:
            return sim_obs
        return {
            group: tensor[:, self.obs_perms[group]] if group in self.obs_perms else tensor
            for group, tensor in sim_obs.items()
        }


def find_joint_obs_slices(env: World, num_actions: int) -> Dict[str, List[Tuple[int, int]]]:
    """Find slices in each obs group's flat vector that are joint-indexed.

    A term is joint-indexed when its observation function is marked
    ``@joint_indexed`` (:mod:`jaxrlworld.rl.envs.mdp.observations.joint_indexed`);
    a stacked history widens the term to a multiple of ``num_actions`` and
    yields one slice per frame. An unmarked term whose width is a multiple of
    ``num_actions`` is refused rather than guessed about: the earlier
    name-list approach silently left ``dof_pos_biased``, ``prev_raw_actions``
    and the K1 terms in simulator order, scrambling the actor input on a
    cross-simulator evaluation with a non-identity joint permutation.
    """
    # Ensure term indices are built
    if not env.obs_manager._is_term_indices_built:
        env.obs_manager._build_term_indices()
        env.obs_manager._is_term_indices_built = True

    result: Dict[str, List[Tuple[int, int]]] = {}

    # ``_group_terms`` is ``{group_name: {term_name: cfg}}`` and
    # ``_group_term_indices`` is keyed by the same ``term_name``.
    for group_name, terms_dict in env.obs_manager._group_terms.items():
        slices = []
        term_indices = env.obs_manager._group_term_indices.get(group_name, {})

        for term_name, obs_term in terms_dict.items():
            if term_name not in term_indices:
                continue
            start, end = term_indices[term_name]
            term_dim = end - start
            flag = joint_indexed_flag(obs_term.func)
            if flag is None and term_dim % num_actions == 0:
                raise ValueError(
                    f"Observation term {group_name}.{term_name} ({obs_term.func!r}) is {term_dim} wide, a multiple "
                    f"of the {num_actions} actuated joints, but its function is not marked @joint_indexed or "
                    "@not_joint_indexed. Cross-simulator evaluation must know whether to permute its columns."
                )
            if not flag:
                continue
            if term_dim % num_actions != 0:
                raise ValueError(
                    f"Observation term {group_name}.{term_name} is marked @joint_indexed but is {term_dim} wide, "
                    f"not a multiple of the {num_actions} actuated joints."
                )
            frames = term_dim // num_actions
            for h in range(frames):
                slices.append((start + h * num_actions, start + (h + 1) * num_actions))

        result[group_name] = slices

    return result
