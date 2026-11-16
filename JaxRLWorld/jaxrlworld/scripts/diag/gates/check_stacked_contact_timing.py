"""Is the stacked contact-timing update bit-identical to the per-group one?

``BaseContactManager.advance`` updates every contact group's air / contact
timers in one pass over a stacked buffer whose columns are the groups'
own buffers. This runs it next to ``_apply_contact_frame`` applied to each
group on independent tensors, through resets, eager and through
``torch.compile`` (``compile_contact_kernels``), on the CPU, and demands
``torch.equal``.

    python -m jaxrlworld.scripts.diag.gates.check_stacked_contact_timing
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from jaxrlworld.rl.envs.managers.common.contact import BaseContactManager


def check_stacked_timing(
    seeds: int = 10, num_envs: int = 2048, substeps: int = 64, widths=(2, 1, 4), compile_timing: bool = False
) -> None:
    """``BaseContactManager.advance`` runs the frame arithmetic ONCE over
    every group's buffers stacked side by side, each group's own buffers
    being column views of that stack. Per column that is the elementwise
    arithmetic ``_apply_contact_frame`` runs on one group; this checks it,
    through resets, against that per-group update on independent tensors.

    With ``compile_timing`` the manager runs that arithmetic through
    ``torch.compile`` (what mjlab / Newton do under
    ``compile_contact_kernels``); boolean logic, ``where`` and ``+ dt``
    must come out bit-identical to the eager loop."""

    class _Manager(BaseContactManager):
        frames: dict[str, torch.Tensor] = {}

        def _compute_group_contact_force(self, group):
            return None

        def _compute_group_is_contact(self, group):
            return self.frames[group.name]

    dt = 0.005
    fields = BaseContactManager._TIMING_FIELDS
    for seed in range(seeds):
        g = torch.Generator().manual_seed(seed)
        m = _Manager(
            SimpleNamespace(num_envs=num_envs, control_dt=0.02, device="cpu", _env_step_counter=0),
            compile_timing=compile_timing,
        )
        for i, w in enumerate(widths):
            m._register_group(f"g{i}", [f"b{i}{j}" for j in range(w)], ("found",))
        refs = {}
        for name, grp in m._groups.items():
            snapshot = {f: getattr(grp, f).clone() for f in fields}  # dynamic-attr: ContactGroup timing field names
            refs[name] = SimpleNamespace(**snapshot)
        for k in range(substeps):
            m.frames = {
                name: torch.rand((num_envs, grp.num_tracked), generator=g) < 0.5 for name, grp in m._groups.items()
            }
            if k % 9 == 0:
                n_reset = int(torch.randint(1, 256, (), generator=g))
                env_ids = torch.randperm(num_envs, generator=g)[:n_reset]
                m.reset(env_ids)
                for ref in refs.values():
                    for f in fields:
                        vars(ref)[f][env_ids] = 0
            m.advance(dt)
            for name, ref in refs.items():
                BaseContactManager._apply_contact_frame(ref, m.frames[name], dt)
        for name, ref in refs.items():
            grp = m._groups[name]
            for f in fields:
                live = getattr(grp, f)  # dynamic-attr: ContactGroup timing field names
                assert torch.equal(vars(ref)[f], live), f"seed {seed}: {name}.{f} differs"
            # The accessors must hand out the live column views, not copies.
            assert m.current_air_time(name).data_ptr() == grp.current_air_time.data_ptr()
    how = "compiled" if compile_timing else "eager"
    print(f"  stacked contact timing ({how}): {seeds} seeds x {substeps} substeps x {len(widths)} groups bit-identical")


def main() -> None:
    print("=" * 78)
    print("STACKED CONTACT TIMING PARITY")
    print("=" * 78)
    check_stacked_timing()
    check_stacked_timing(compile_timing=True)
    print("  PASS")
    print("=" * 78)


if __name__ == "__main__":
    main()
