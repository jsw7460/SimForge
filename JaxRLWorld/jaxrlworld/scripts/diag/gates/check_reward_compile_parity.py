"""Does the compiled reward chain match the eager terms on a live env?

With ``RewardConfig.compile_terms`` the reward manager runs every term,
the weighting and the mode combination as one ``torch.compile``d
program over a per-step snapshot of the engine reads (reward_view.py).
This drives a live env and, on every step, computes the eager reference
from the same state first — saving and restoring the stateful terms'
buffers around it, so both paths see identical state and the state
advances once — then lets the compiled chain run, and compares every
term to 1e-5 relative of its scale (fused multiply-adds round
differently, so bit-identity is not expected). After the compile has
warmed, any recompile fails the gate.

Usage (GPU box, from the SimForge root):
    jaxpy -m jaxrlworld.scripts.diag.gates.check_reward_compile_parity --sim mujoco
    jaxpy -m jaxrlworld.scripts.diag.gates.check_reward_compile_parity --sim newton
    jaxpy -m jaxrlworld.scripts.diag.gates.check_reward_compile_parity --sim genesis
"""

from __future__ import annotations

import argparse
import importlib
import types

import torch
import torch._dynamo

from jaxrlworld.rl.runners import BaseRunner

_DEFAULT_PRESET = "jaxrlworld.rl.configs.presets.k1_joystick.g1_recipe:K1G1RecipeConfig"


def _build_env(preset: str, sim: str, num_envs: int):
    mod_path, cls_name = preset.split(":", 1)
    cfg_cls = getattr(importlib.import_module(mod_path), cls_name)
    cfgs = cfg_cls(sim_type=sim, num_envs=num_envs).build()
    runner = BaseRunner.create_with_env(cfgs, use_wandb=False)
    return runner.env


class TermState:
    """Every tensor a stateful term keeps, and where it lives, so a trial
    evaluation can be undone.

    A term advances its state either in place (``buf.copy_(...)``,
    ``buf[ids] = 0``) or by rebinding the attribute to a fresh tensor
    (``self.air_time = self.air_time * ~contact``). Restoring by copying
    into the tensor seen at save time handles only the first: after a
    rebind that tensor is no longer what the term reads. So each slot is
    (container, key, tensor, copy), and :meth:`restore` puts the original
    tensor object back under its key before refilling it.

    A backend wrapper class holds the shared tracker as an attribute
    (``self._impl``), so the walk descends into plain attribute objects
    and dicts; it stops at the env, which every instance also keeps a
    reference to. ``check_all_presets`` uses this too, for its shadow
    mode.
    """

    def __init__(self, mgr):
        self._slots: list[tuple[object, object, torch.Tensor, torch.Tensor]] = []
        seen: set[int] = set()

        def walk(obj, depth: int) -> None:
            if id(obj) in seen or depth > 3 or obj is mgr.env:
                return
            seen.add(id(obj))
            items = obj.items() if isinstance(obj, dict) else vars(obj).items()
            for key, value in items:
                if isinstance(value, torch.Tensor):
                    self._slots.append((obj, key, value, value.clone()))
                elif isinstance(value, dict) or (
                    hasattr(value, "__dict__") and not isinstance(value, types.FunctionType | types.MethodType | type)
                ):
                    # Not ``callable``: the trackers define ``__call__``.
                    walk(value, depth + 1)

        for inst in mgr._instances.values():
            walk(inst, 0)
        walk(mgr._fd_prev_foot_pos, 0)

    def restore(self) -> None:
        for container, key, tensor, copy in self._slots:
            current = container[key] if isinstance(container, dict) else getattr(container, key)
            if current is not tensor:
                if isinstance(container, dict):
                    container[key] = tensor
                else:
                    setattr(container, key, tensor)
            tensor.copy_(copy)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=_DEFAULT_PRESET)
    ap.add_argument("--sim", default="mujoco", choices=("genesis", "newton", "mujoco"))
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    # 1e-4, not 1e-5: a term built on world-frame positions
    # (capture_point_support) loses ~4e-5 relative to float32
    # cancellation before any fusion is involved.
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    env = _build_env(args.preset, args.sim, args.num_envs)
    mgr = env.reward_manager
    if not mgr._compile_terms:
        raise RuntimeError("compile_terms is off for this preset; nothing to compare.")
    names = list(mgr.reward_terms)
    worst: dict[str, float] = {name: 0.0 for name in names}
    worst["total"] = 0.0
    failures = 0
    compared = 0
    original = mgr.set_rewards

    def checked_set_rewards(reward_buffer, reward_buffer_per_type):
        nonlocal failures, compared
        if mgr._view is None:
            # The recording (eager) first call: nothing to compare yet.
            original(reward_buffer=reward_buffer, reward_buffer_per_type=reward_buffer_per_type)
            return
        state = TermState(mgr)
        ref = mgr._compute_stacked(mgr.env)
        ref_total = reward_buffer + ref.sum(dim=0) if mgr.config.reward_mode == "sum" else None
        state.restore()
        original(reward_buffer=reward_buffer, reward_buffer_per_type=reward_buffer_per_type)
        compared += 1
        for i, name in enumerate(names):
            scale = ref[i].abs().max().clamp_min(1e-6)
            rel = ((reward_buffer_per_type[name] - ref[i]).abs().max() / scale).item()
            worst[name] = max(worst[name], rel)
            if rel > args.tol:
                failures += 1
                print(f"[step {compared}] {name}: rel diff {rel:.3e}")
        if ref_total is not None:
            scale = ref_total.abs().max().clamp_min(1e-6)
            rel = ((reward_buffer - ref_total).abs().max() / scale).item()
            worst["total"] = max(worst["total"], rel)

    mgr.set_rewards = checked_set_rewards

    action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    for step in range(args.steps):
        env.step(0.1 * torch.randn_like(action))
        if step == 2:
            # Step 0 records, step 1 compiles; from here on the program
            # must be stable (the weights are constants in this preset).
            torch._dynamo.config.error_on_recompile = True

    print(f"compiled vs eager reward chain ({args.sim}, {compared} steps compared, tol {args.tol:g} relative)")
    for name in [*names, "total"]:
        print(f"  {name:<34} max rel diff {worst[name]:.2e}")
    if failures:
        print(f"\nFAIL — {failures} term/step mismatches")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
