"""Validate the K1 mirror spec BEFORE wiring symmetry into PPO.

The mirror (perm, sign) vectors are the ONLY error-prone part of symmetry
augmentation: a wrong index or sign silently corrupts the mirror loss. This
diag checks them structurally so we never train on a bad spec:

  1. build_mirror_spec covers EVERY obs term (raises if a rule is missing or a
     group's term slices leave a gap in the layout).
  2. Involutive: perm[perm] == identity  AND  sign * sign[perm] == 1 (a double
     mirror cancels) for actor obs, critic obs, and action.
  3. Numeric: mirror(mirror(o)) == o on a real observation sample.

It also prints the L<->R joint permutation so the pairing can be eyeballed.

Run (server; jaxpy for JAX):
    jaxpy -m jaxrlworld.scripts.diag.k1.k1_mirror_spec_diag
"""

from __future__ import annotations

import argparse


def main() -> int:
    import jax.numpy as jnp
    import numpy as np

    from jaxrlworld.rl.algorithms.ppo.symmetry import build_mirror_spec, mirror
    from jaxrlworld.rl.configs.presets.k1_joystick.g1_recipe import K1G1RecipeConfig
    from jaxrlworld.rl.evals.sim_initializers import get_initializer

    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="mujoco", choices=("mujoco", "newton", "genesis"))
    ap.add_argument("--num-envs", type=int, default=4)
    args = ap.parse_args()

    cfgs = K1G1RecipeConfig(sim_type=args.sim, num_envs=args.num_envs).build()
    env = get_initializer({"mujoco": "MujocoEnv", "newton": "Newton", "genesis": "Genesis"}[args.sim]).init_environment(
        cfgs
    )
    jn = list(env.act_manager.actuated_joint_names)

    print("=" * 70)
    print(f"K1 mirror spec validation  ({args.sim}, {len(jn)} joints)")
    print("=" * 70)

    # This raises if any term lacks a rule or a layout gap exists.
    spec = build_mirror_spec(env.obs_manager, jn)
    print(
        f"\nbuild_mirror_spec OK — actor_dim={spec.actor_perm.shape[0]} "
        f"critic_dim={spec.critic_perm.shape[0]} action_dim={spec.action_perm.shape[0]}"
    )

    print("\nL<->R joint permutation + sign:")
    ap_, as_ = np.array(spec.action_perm), np.array(spec.action_sign)
    for i, n in enumerate(jn):
        print(f"  {i:2d} {n:<22} -> {ap_[i]:2d} {jn[ap_[i]]:<22} sign={as_[i]:+.0f}")

    def check(perm, sign, name) -> bool:
        p, s = np.array(perm), np.array(sign)
        inv = bool((p[p] == np.arange(len(p))).all())
        sgn = bool(np.allclose(s * s[p], 1.0))
        print(f"  {name:<8} perm_involutive={inv}  sign_cancels={sgn}")
        return inv and sgn

    print("\n--- structural checks ---")
    ok = True
    ok &= check(spec.actor_perm, spec.actor_sign, "actor")
    ok &= check(spec.critic_perm, spec.critic_sign, "critic")
    ok &= check(spec.action_perm, spec.action_sign, "action")

    print("\n--- numeric: mirror(mirror(o)) == o ---")
    obs = env.obs_manager.get_observation()  # populated now (build called calculate_obs_dim)
    ao = jnp.asarray(obs["actor"].detach().cpu().numpy())
    co = jnp.asarray(obs["critic"].detach().cpu().numpy())
    a2 = mirror(mirror(ao, spec.actor_perm, spec.actor_sign), spec.actor_perm, spec.actor_sign)
    c2 = mirror(mirror(co, spec.critic_perm, spec.critic_sign), spec.critic_perm, spec.critic_sign)
    na = bool(jnp.allclose(ao, a2, atol=1e-6))
    nc = bool(jnp.allclose(co, c2, atol=1e-6))
    print(f"  actor  double-mirror==identity: {na}")
    print(f"  critic double-mirror==identity: {nc}")
    ok &= na and nc

    print("\n" + "=" * 70)
    print("OVERALL:", "PASS — spec is valid, safe to wire into PPO" if ok else "FAIL — fix rules above")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
