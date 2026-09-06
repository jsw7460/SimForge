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


def _old_contact_frame(prev, cur_air, cur_con, last_air, last_con, is_contact, dt):
    """The rebinding formulation ``_apply_contact_frame`` had."""
    is_landing = ~prev & is_contact
    is_liftoff = prev & ~is_contact
    last_air = torch.where(is_landing, cur_air + dt, last_air)
    last_con = torch.where(is_liftoff, cur_con + dt, last_con)
    cur_con = torch.where(is_contact, cur_con + dt, torch.zeros_like(cur_con))
    cur_air = torch.where(~is_contact, cur_air + dt, torch.zeros_like(cur_air))
    return is_contact, cur_air, cur_con, last_air, last_con


def check_contact_frame(seeds: int = 10, num_envs: int = 2048, n: int = 4, substeps: int = 64) -> None:
    from types import SimpleNamespace

    from jaxrlworld.rl.envs.managers.common.contact import BaseContactManager

    dt = 0.005
    fields = ("_prev_is_contact", "current_air_time", "current_contact_time", "last_air_time", "last_contact_time")
    for seed in range(seeds):
        g = torch.Generator().manual_seed(seed)
        shape = (num_envs, n)
        state = [torch.zeros(shape, dtype=torch.bool), *(torch.rand(shape, generator=g) * 0.1 for _ in range(4))]
        group = SimpleNamespace(**{name: value.clone() for name, value in zip(fields, state)})
        handed_in = [getattr(group, name) for name in fields]
        for _ in range(substeps):
            is_contact = torch.rand(shape, generator=g) < 0.5
            state = _old_contact_frame(*state, is_contact, dt)
            BaseContactManager._apply_contact_frame(group, is_contact, dt)
        for name, old in zip(fields, state):
            assert torch.equal(old, getattr(group, name)), f"seed {seed}: {name} differs"
        # The buffers must still be the objects handed in — that is the point.
        for name, before in zip(fields, handed_in):
            assert getattr(group, name) is before, f"{name} was rebound"
    print(f"  contact frame: {seeds} seeds x {substeps} substeps bit-identical, buffers never rebound")


def check_peak_height_update(seeds: int = 10, num_envs: int = 2048, n: int = 2) -> None:
    for seed in range(seeds):
        g = torch.Generator().manual_seed(seed)
        peak = torch.rand((num_envs, n), generator=g)
        foot = torch.rand((num_envs, n), generator=g)
        in_air = torch.rand((num_envs, n), generator=g) < 0.5
        first = torch.rand((num_envs, n), generator=g) < 0.2
        old = torch.where(in_air, torch.maximum(peak, foot), peak)
        old = torch.where(first, torch.zeros_like(old), old)
        new = peak.clone()
        torch.where(in_air, torch.maximum(new, foot), new, out=new)
        torch.where(first, torch.zeros_like(new), new, out=new)
        assert torch.equal(old, new), f"seed {seed}: peak height update differs"
    print(f"  peak heights: {seeds} seeds bit-identical in place")


def main() -> None:
    print("=" * 78)
    print("RESET BATCHING PARITY")
    print("=" * 78)
    check_quaternion()
    check_termination_reset()
    check_contact_frame()
    check_peak_height_update()
    print("  PASS")
    print("=" * 78)


if __name__ == "__main__":
    main()
