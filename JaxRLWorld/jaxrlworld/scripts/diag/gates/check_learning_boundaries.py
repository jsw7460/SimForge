"""Gate: the learning stack's boundary configurations are handled or refused.

Four settings the configs allowed and the code got wrong, each at a
boundary the default presets never reach:

1. a packed PPO step's observation of width 1 came back as ``(N,)`` from
   ``layout_column`` (the squeeze meant for reward and flags), and the
   policy's matmul failed on it. Vector fields now keep their axis at
   every width; the scalar fields are named.
2. TD-MPC2's VICReg covariance divides by ``batch_size - 1`` and is
   summed into the loss whatever its coefficient, so ``batch_size=1``
   was NaN at ``vicreg_coef=0``. The algorithm refuses a batch under 2.
3. TD-MPC2 ``num_bins=1`` had a two-hot encode / decode branch but no
   regression branch in ``soft_ce``: the log-softmax of one logit is
   zero, so reward and value losses were identically zero. The two-hot
   config refuses fewer than two bins.
4. AMP with ``joint_velocity_from_positions`` differences consecutive
   frames, which a one-frame window does not have; the constructor
   accepted ``num_amp_obs_steps=1`` and failed later on a shape. It
   refuses the combination.

Run::

    python -m jaxrlworld.scripts.diag.gates.check_learning_boundaries
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxrlworld.rl.algorithms.amp_ppo.amp import AdversarialMotionPrior, velocity_from_positions
from jaxrlworld.rl.algorithms.amp_ppo.expert_motion import ExpertMotionSet
from jaxrlworld.rl.algorithms.tdmpc2.math import make_two_hot_config, soft_ce, two_hot, two_hot_inv
from jaxrlworld.rl.algorithms.tdmpc2.tdmpc2 import TDMPC2
from jaxrlworld.rl.configs.algorithms.amp_ppo import AmpConfig
from jaxrlworld.rl.modules.policies.tdmpc2_world_model import TDMPC2WorldModel
from jaxrlworld.rl.storages.rollout_storage import SCALAR_FIELDS, layout_column

N = 4


def raises(fn, exc_type, needle: str) -> tuple[bool, str]:
    try:
        fn()
    except exc_type as e:
        return needle in str(e), str(e)[:100]
    return False, "no exception"


def main() -> int:
    failures: list[str] = []

    def chk(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    print("=" * 78)
    print("LEARNING BOUNDARIES")
    print("=" * 78)

    print("\n-- 1. packed step: width-1 observation keeps its axis --")
    layout = (("actor", 0, 1), ("critic", 1, 3), ("reward", 4, 1), ("terminated", 5, 1), ("truncated", 6, 1))
    packed = jnp.arange(N * 7, dtype=jnp.float32).reshape(N, 7)
    chk(
        "scalar fields are reward and the flags",
        SCALAR_FIELDS == {"reward", "terminated", "truncated", "bootstrap_mask", "trunc_no_reset"},
        str(sorted(SCALAR_FIELDS)),
    )
    actor = layout_column(packed, layout, "actor")
    chk("actor of width 1 is (N, 1)", actor.shape == (N, 1), str(actor.shape))
    chk("... with the right column", bool(jnp.array_equal(actor[:, 0], packed[:, 0])))
    chk("critic of width 3 is (N, 3)", layout_column(packed, layout, "critic").shape == (N, 3))
    chk("reward is (N,)", layout_column(packed, layout, "reward").shape == (N,))
    chk("terminated is (N,)", layout_column(packed, layout, "terminated").shape == (N,))
    chk("a field the step lacks is None", layout_column(packed, layout, "final_critic") is None)
    w = jnp.ones((N, 1))
    chk("the policy matmul the audit crashed on now works", (actor @ w.T).shape == (N, N))
    ok, msg = raises(lambda: layout_column(packed, (("reward", 0, 2),), "reward"), ValueError, "scalar")
    chk("a scalar field of width 2 raises", ok, msg)

    print("\n-- 2. TD-MPC2 batch size --")
    model = TDMPC2WorldModel(
        obs_dim=4, action_dim=2, latent_dim=16, mlp_dim=16, num_q=2, num_bins=5, key=jax.random.PRNGKey(0)
    )
    ok, msg = raises(lambda: TDMPC2(model, num_envs=2, batch_size=1, num_bins=5), ValueError, "batch_size >= 2")
    chk("batch_size=1 is refused", ok, msg)
    alg = TDMPC2(model, num_envs=2, batch_size=2, num_bins=5)
    chk("batch_size=2 constructs", alg.batch_size == 2)

    print("\n-- 3. two-hot bins --")
    for bins in (0, 1):
        ok, msg = raises(lambda b=bins: make_two_hot_config(b, -10.0, 10.0), ValueError, "num_bins >= 2")
        chk(f"num_bins={bins} is refused", ok, msg)
    ok, msg = raises(lambda: make_two_hot_config(5, 1.0, 1.0), ValueError, "vmax > vmin")
    chk("vmax == vmin is refused", ok, msg)
    cfg = make_two_hot_config(2, -1.0, 1.0)
    x = jnp.array([[0.0], [0.5]])
    enc = two_hot(x, cfg)
    chk("two bins encode as a distribution", enc.shape == (2, 2) and bool(jnp.allclose(enc.sum(-1), 1.0)))
    dec = two_hot_inv(jnp.log(enc + 1e-9), cfg)
    chk("... and decode back", bool(jnp.allclose(dec, x, atol=1e-4)), str(np.asarray(dec).ravel()))
    ce = soft_ce(jnp.array([[7.0, 0.0]]), jnp.array([[0.5]]), cfg)
    chk("soft_ce is non-zero for a wrong prediction", float(ce[0, 0]) > 0.1, f"{float(ce[0, 0]):.3f}")

    print("\n-- 4. AMP window length with position-derived velocity --")
    D = 4
    expert = ExpertMotionSet(
        features=np.random.default_rng(0).standard_normal((20, D)).astype(np.float32),
        clip_start=np.array([0, 10, 20]),
        clip_names=("a", "b"),
        fps=50.0,
        source_index=np.array([0, 0]),
    )
    fd = ((0, 2), (2, 4))

    def build(steps: int):
        cfg = AmpConfig(
            num_amp_obs_steps=steps,
            joint_velocity_from_positions=True,
            replay_buffer_size=16,
            replay_insert_size=1,
            discriminator_hidden_dims=(8,),
        )
        return AdversarialMotionPrior(cfg, expert, D, 0.02, 1e-3, jax.random.PRNGKey(0), fd_blocks=fd)

    ok, msg = raises(lambda: build(1), ValueError, "num_amp_obs_steps >= 2")
    chk("K=1 with position-derived velocity is refused", ok, msg)
    amp = build(2)
    chk("K=2 constructs and its expert windows have the window width", amp.expert_windows.shape == (20, 2 * D))
    ok, msg = raises(lambda: velocity_from_positions(jnp.ones((2, 1, D)), fd, 0.02), Exception, "")
    chk("(the helper itself still cannot take a one-frame window)", ok, msg)

    print("=" * 78)
    print("=== RESULT: ALL OK ===" if not failures else f"=== RESULT: {len(failures)} FAILED ===")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
