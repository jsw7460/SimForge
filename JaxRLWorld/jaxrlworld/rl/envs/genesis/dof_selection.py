"""How a canonical-ordered set of an entity's DOFs is addressed in Genesis.

Genesis takes the DOFs a setter or getter touches in two forms, and
they cost very differently. A ``range`` becomes a slice of the solver's
buffer: the zero-copy path, no host synchronisation. An index tensor
takes advanced indexing, which Genesis bounds-checks and de-duplicates
on the host: a synchronisation on every call, several in the position
setter. The actuated joints of a robot are one contiguous block of its
DOFs in every asset this repo loads, so the canonical selection is a
range once its columns are put in simulator order.

:class:`GenesisDofSelection` carries both forms. Managers hand Genesis
``dofs`` (and ``qs`` for configuration writes) and permute the columns
of what they write or read with :meth:`to_sim_order` /
:meth:`to_canonical_order`, which are the identity (no copy) whenever
the canonical order already is the simulator order.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class GenesisDofSelection:
    """A set of an entity's DOFs, in canonical order, with the index forms Genesis is handed."""

    sim_ids: Tensor
    """Local (entity) DOF indices in canonical order. The index-tensor form."""

    dofs: range | Tensor
    """What Genesis DOF setters and getters are handed: a ``range`` when the DOFs form one contiguous
    block (the zero-copy path), else ``sim_ids``."""

    qs: range | Tensor
    """The local configuration (``qpos``) indices of the same DOFs, in the order of ``dofs``. Only hinge and
    slide joints are selected, so each DOF has exactly one ``q``."""

    to_sim: Tensor | None
    """Column permutation taking canonical order to the order of ``dofs``, or None when they coincide."""

    to_canonical: Tensor | None
    """Inverse of ``to_sim``, or None when the orders coincide."""

    def to_sim_order(self, values: Tensor) -> Tensor:
        """Columns of ``values`` (canonical order) reordered to match ``dofs``."""
        return values if self.to_sim is None else values[..., self.to_sim]

    def to_canonical_order(self, values: Tensor) -> Tensor:
        """Columns of ``values`` (order of ``dofs``) reordered to canonical order."""
        return values if self.to_canonical is None else values[..., self.to_canonical]


def build_dof_selection(entity, sim_ids: Tensor | list[int], device: torch.device) -> GenesisDofSelection:
    """Resolve the index forms of ``sim_ids`` (local DOF indices, canonical order) on a built entity.

    Raises:
        ValueError: If a selected DOF belongs to a joint whose configuration is not one scalar per DOF
            (a free or spherical joint), since its position cannot be written as a ``qpos`` entry.
    """
    ids = torch.as_tensor(sim_ids, dtype=torch.long, device=device)
    dof_to_q: dict[int, int] = {}
    for joint in entity.joints:
        if joint.n_qs != joint.n_dofs:
            continue
        for dof_local, q_local in zip(joint.dofs_idx_local, joint.qs_idx_local, strict=True):
            dof_to_q[dof_local] = q_local
    ids_list = ids.tolist()
    unmapped = [dof for dof in ids_list if dof not in dof_to_q]
    if unmapped:
        raise ValueError(
            f"DOFs {unmapped} of entity {entity.name!r} belong to joints whose position is not one qpos per DOF; "
            "only hinge and slide joints can be selected"
        )

    order = sorted(range(len(ids_list)), key=ids_list.__getitem__)
    sorted_ids = [ids_list[i] for i in order]
    sorted_qs = [dof_to_q[dof] for dof in sorted_ids]
    is_contiguous = sorted_ids == list(range(sorted_ids[0], sorted_ids[0] + len(sorted_ids))) if ids_list else True
    is_q_contiguous = sorted_qs == list(range(sorted_qs[0], sorted_qs[0] + len(sorted_qs))) if ids_list else True
    if is_contiguous and is_q_contiguous:
        dofs: range | Tensor = range(sorted_ids[0], sorted_ids[0] + len(sorted_ids)) if ids_list else range(0)
        qs: range | Tensor = range(sorted_qs[0], sorted_qs[0] + len(sorted_qs)) if ids_list else range(0)
        if order == list(range(len(order))):
            to_sim = to_canonical = None
        else:
            to_sim = torch.tensor(order, dtype=torch.long, device=device)
            to_canonical = torch.argsort(to_sim)
    else:
        dofs = ids
        qs = torch.tensor([dof_to_q[dof] for dof in ids_list], dtype=torch.long, device=device)
        to_sim = to_canonical = None
    return GenesisDofSelection(sim_ids=ids, dofs=dofs, qs=qs, to_sim=to_sim, to_canonical=to_canonical)
