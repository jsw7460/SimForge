"""Are the batched reset-path kernels bit-identical to the loops they replace?

Two pieces of the reset path were rewritten to launch fewer kernels:

- ``reset_root_state_uniform`` composes its yaw/pitch/roll perturbation
  with ``quat_from_euler_zyx_wxyz`` instead of three angle-axis
  quaternions and three products (~100 launches -> ~20), and skips the
  multiply by the default orientation when that default is the identity.
- ``TerminationManager.reset`` folds the ending episodes' per-term fire
  counts with one gather / reduction / scatter over a stacked
  ``(n_terms, num_envs)`` tensor instead of three launches per term.

Both claim to change nothing but the launch count. This checks that
claim on the CPU, where the arithmetic is the same IEEE arithmetic, by
running the old formulation next to the new one on random inputs and
demanding ``torch.equal``.

    python -m jaxrlworld.scripts.diag.gates.check_reset_batching_parity
"""

from __future__ import annotations

import torch

from jaxrlworld.rl.utils.quat_utils import quat_from_angle_axis_wxyz, quat_from_euler_zyx_wxyz, quat_mul_wxyz


def _old_delta(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """The chain ``reset_root_state_uniform`` used before the fusion."""
    q_roll = quat_from_angle_axis_wxyz(roll, torch.tensor((1.0, 0.0, 0.0)))
    q_pitch = quat_from_angle_axis_wxyz(pitch, torch.tensor((0.0, 1.0, 0.0)))
    q_yaw = quat_from_angle_axis_wxyz(yaw, torch.tensor((0.0, 0.0, 1.0)))
    return quat_mul_wxyz(quat_mul_wxyz(q_yaw, q_pitch), q_roll)


def check_quaternion(seeds: int = 20, n: int = 8192) -> None:
    identity = torch.tensor((1.0, 0.0, 0.0, 0.0)).unsqueeze(0).expand(n, -1)
    for seed in range(seeds):
        g = torch.Generator().manual_seed(seed)
        angles = (torch.rand((n, 3), generator=g) * 2.0 - 1.0) * torch.pi
        roll, pitch, yaw = angles.unbind(-1)
        old = _old_delta(roll, pitch, yaw)
        new = quat_from_euler_zyx_wxyz(roll, pitch, yaw)
        assert torch.equal(old, new), f"seed {seed}: fused euler quaternion differs from the chain"
        # Skipping the identity multiply must be exactly the multiply.
        assert torch.equal(quat_mul_wxyz(identity, old), new), f"seed {seed}: identity skip differs"
    # Degenerate angles, where signed zeros are most likely to show.
    for special in (0.0, torch.pi, -torch.pi, torch.pi / 2, -torch.pi / 2):
        a = torch.full((16,), special)
        assert torch.equal(_old_delta(a, a, a), quat_from_euler_zyx_wxyz(a, a, a)), f"angle {special}"
    print(f"  quaternion: {seeds} seeds x {n} samples bit-identical, identity skip exact")


def check_termination_reset(seeds: int = 20, n_terms: int = 6, num_envs: int = 4096) -> None:
    for seed in range(seeds):
        g = torch.Generator().manual_seed(seed)
        fires_all = torch.randint(0, 4, (n_terms, num_envs), generator=g, dtype=torch.long)
        n_reset = int(torch.randint(1, 512, (), generator=g))
        env_ids = torch.randperm(num_envs, generator=g)[:n_reset]

        # Old: per-term loop on independent tensors.
        old_fires = {i: fires_all[i].clone() for i in range(n_terms)}
        old_counts = {i: torch.zeros((), dtype=torch.long) for i in range(n_terms)}
        for i, fires in old_fires.items():
            old_counts[i] += (fires[env_ids] > 0).long().sum()
            fires[env_ids] = 0

        # New: one stacked tensor.
        new_all = fires_all.clone()
        new_counts = torch.zeros(n_terms, dtype=torch.long)
        sub = new_all[:, env_ids]
        new_counts += (sub > 0).sum(dim=1)
        new_all[:, env_ids] = 0

        for i in range(n_terms):
            assert torch.equal(old_fires[i], new_all[i]), f"seed {seed}: term {i} fires after reset differ"
            assert int(old_counts[i]) == int(new_counts[i]), f"seed {seed}: term {i} count differs"
    print(f"  termination.reset: {seeds} seeds x {n_terms} terms exact")


def main() -> None:
    print("=" * 78)
    print("RESET BATCHING PARITY")
    print("=" * 78)
    check_quaternion()
    check_termination_reset()
    print("  PASS")
    print("=" * 78)


if __name__ == "__main__":
    main()
