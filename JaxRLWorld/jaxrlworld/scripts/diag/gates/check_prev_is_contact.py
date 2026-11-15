"""Is ``prev_is_contact`` the contact state the previous control step ended on?

The timing edges keep a substep-level previous state, overwritten on every
substep; ``prev_is_contact`` used to return that buffer, so a reward
reading it after the physics saw this step's contact again and
``contact | prev_contact`` collapsed to ``contact``. This drives the real
:class:`BaseContactManager` the way ``World.step`` does (``begin_step``,
then one ``advance`` per substep) over a scripted per-substep contact
pattern, with the timing arithmetic eager and compiled, and requires:

1. ``prev_is_contact`` = the last substep's contact of the PREVIOUS
   control step, every step, with declared order != native order;
2. it differs from ``is_contact`` on the steps where contact changed
   (the bug read them equal);
3. ``False`` on the step after a reset, for the reset envs only;
4. the air/contact timing buffers are what they were before the change
   (an independent reference implementation of the substep edges).

    python -m jaxrlworld.scripts.diag.gates.check_prev_is_contact
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

from jaxrlworld.rl.envs.managers.common.contact import BaseContactManager, ContactGroup

GROUP = "feet"
NATIVE = ["L", "R", "M"]
PATTERNS = ("R", "M", "L")  # declared order differs from native
NUM_ENVS = 5
SUBSTEPS = 4
STEPS = 40
DT = 0.005


class _Stub(BaseContactManager):
    def __init__(self, env, compile_timing: bool) -> None:
        super().__init__(env, compile_timing=compile_timing)
        self._register_group(GROUP, list(NATIVE), ("found",), patterns=PATTERNS)
        self.state = torch.zeros((NUM_ENVS, len(NATIVE)), dtype=torch.bool)

    def _compute_group_contact_force(self, group: ContactGroup) -> torch.Tensor | None:
        return None

    def _compute_group_is_contact(self, group: ContactGroup) -> torch.Tensor:
        return self.state


def _pattern(steps: int) -> torch.Tensor:
    """(steps, substeps, envs, native bodies) bool, with mid-step flips."""
    g = torch.Generator().manual_seed(0)
    return torch.rand((steps, SUBSTEPS, NUM_ENVS, len(NATIVE)), generator=g) < 0.5


def _run(compile_timing: bool) -> list[str]:
    env = SimpleNamespace(num_envs=NUM_ENVS, device="cpu", control_dt=DT * SUBSTEPS)
    m = _Stub(env, compile_timing)
    contacts = _pattern(STEPS)
    declared = [NATIVE.index(n) for n in PATTERNS]
    reset_at = {13: torch.tensor([1, 3]), 27: torch.tensor([0])}

    # Reference substep timing, written out independently of the manager.
    ref = {k: torch.zeros((NUM_ENVS, len(NATIVE))) for k in ("cat", "cct", "lat", "lct")}
    ref_prev = torch.zeros((NUM_ENVS, len(NATIVE)), dtype=torch.bool)

    problems: list[str] = []
    expected_prev = torch.zeros((NUM_ENVS, len(NATIVE)), dtype=torch.bool)  # native order
    differ_steps = 0
    for k in range(STEPS):
        m.begin_step()
        for s in range(SUBSTEPS):
            m.state = contacts[k, s]
            m.advance(DT)
            c = contacts[k, s]
            land, lift = ~ref_prev & c, ref_prev & ~c
            ref["lat"] = torch.where(land, ref["cat"] + DT, ref["lat"])
            ref["lct"] = torch.where(lift, ref["cct"] + DT, ref["lct"])
            ref["cct"] = torch.where(c, ref["cct"] + DT, torch.zeros_like(ref["cct"]))
            ref["cat"] = torch.where(~c, ref["cat"] + DT, torch.zeros_like(ref["cat"]))
            ref_prev = c.clone()

        prev = m.prev_is_contact(GROUP)
        now = m.is_contact(GROUP)
        want = expected_prev[:, declared]
        if not torch.equal(prev, want):
            problems.append(f"step {k}: prev_is_contact {prev.int().tolist()} != previous step's {want.int().tolist()}")
        if not torch.equal(now, contacts[k, -1][:, declared]):
            problems.append(f"step {k}: is_contact is not this step's last substep")
        if not torch.equal(prev, now):
            differ_steps += 1
        for key, field in (
            ("cat", "current_air_time"),
            ("cct", "current_contact_time"),
            ("lat", "last_air_time"),
            ("lct", "last_contact_time"),
        ):
            got = getattr(m, field)(GROUP)
            if not torch.allclose(got, ref[key][:, declared], atol=1e-7):
                problems.append(f"step {k}: {field} drifted from the reference substep timing")
        expected_prev = contacts[k, -1].clone()

        if k in reset_at:
            ids = reset_at[k]
            m.reset(ids)
            for key in ref:
                ref[key][ids] = 0.0
            ref_prev[ids] = False
            expected_prev[ids] = False
            # The reset envs restart from no contact; the others keep theirs.
            m.begin_step()
            after = m.prev_is_contact(GROUP)
            if after[ids].any():
                problems.append(f"step {k}: reset envs {ids.tolist()} keep a previous contact")
            others = [i for i in range(NUM_ENVS) if i not in ids.tolist()]
            if not torch.equal(after[others], contacts[k, -1][others][:, declared]):
                problems.append(f"step {k}: a reset changed the previous contact of envs {others}")
            # ``begin_step`` again at the next step start is idempotent here.

    if differ_steps == 0:
        problems.append("prev_is_contact equalled is_contact on every step (the scenario does not test anything)")
    label = "compiled" if compile_timing else "eager"
    print(f"  {label:<9} {STEPS} steps x {SUBSTEPS} substeps: prev != now on {differ_steps} steps")
    return problems


def main() -> int:
    problems = _run(False) + _run(True)
    for p in problems[:20]:
        print(f"  [FAIL] {p}")
    if problems:
        print(f"=== RESULT: {len(problems)} FAILED ===")
        return 1
    print("=== RESULT: ALL OK ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
