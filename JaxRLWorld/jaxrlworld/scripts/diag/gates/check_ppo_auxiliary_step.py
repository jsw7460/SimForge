"""Does an actor's auxiliary loss take exactly one actor-only step per update?

An actor that declares ``has_auxiliary_loss`` gets one extra optimizer
step per PPO update on ``compute_auxiliary_loss`` over the last minibatch,
after the policy update, and then its ``post_update_step`` hook. This
checks, with a small actor whose loss lives in an encoder layer, that

- an externally registered actor cfg builds that actor and hydrates back
  from its saved form (``register_actor``);
- the step moves the encoder and nothing else: critic, std head and the
  actor's own decoder keep their values and their optimizer moments bit
  for bit;
- repeated steps drive the loss down;
- a full ``PPO.update`` reports the loss, runs the hook, and does both on
  the per-epoch-GAE path too;
- an actor without an auxiliary loss reports 0 and runs unchanged.

    python -m jaxrlworld.scripts.diag.gates.check_ppo_auxiliary_step
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Dict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jaxrlworld.rl.algorithms.ppo import PPO
from jaxrlworld.rl.algorithms.ppo.update import apply_auxiliary_step
from jaxrlworld.rl.configs.base_config import BaseConfig
from jaxrlworld.rl.configs.common_config_classes import PPOPolicyConfig
from jaxrlworld.rl.modules.architectures.actor_registry import build_actor, register_actor
from jaxrlworld.rl.modules.architectures.base import BaseActor
from jaxrlworld.rl.modules.normalization import EmpiricalNormalization
from jaxrlworld.rl.modules.policies.ppo_ac import PPOActorCritic

OBS_A, OBS_C, ACT, NUM_ENVS, NUM_STEPS, FEAT = 6, 8, 3, 16, 4, 8


@dataclass
class AuxTestActorCfg(BaseConfig):
    feature_dim: int = FEAT

    def recursive_to_dict(self) -> Dict:
        result = super().recursive_to_dict()
        result["_type"] = type(self).__name__
        return result


class AuxTestActor(BaseActor):
    """Encoder -> decoder; the auxiliary loss pulls the encoder output to zero."""

    encoder: eqx.nn.Linear
    decoder: eqx.nn.Linear
    schedule: jax.Array = eqx.field(metadata={"frozen": True})
    num_obs: int = eqx.field(static=True)
    num_actions: int = eqx.field(static=True)

    has_auxiliary_loss: ClassVar[bool] = True

    def __init__(self, num_obs: int, num_actions: int, feature_dim: int, *, key: jax.Array):
        k1, k2 = jax.random.split(key)
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.encoder = eqx.nn.Linear(num_obs, feature_dim, key=k1)
        self.decoder = eqx.nn.Linear(feature_dim, num_actions, key=k2)
        self.schedule = jnp.array(1.0)

    def __call__(self, obs: jax.Array, *, key: jax.Array | None = None) -> tuple[jax.Array, dict]:
        return self.decoder(jax.nn.tanh(self.encoder(obs))), {}

    def compute_auxiliary_loss(self, obs: jax.Array) -> jax.Array:
        return jnp.mean(jnp.square(self.encoder(obs)))

    def post_update_step(self) -> AuxTestActor:
        return eqx.tree_at(lambda a: a.schedule, self, self.schedule * 0.5)


def _build(cfg: AuxTestActorCfg, *, num_obs, num_actions, key, kinematic_tree, actuated_joint_names):
    return AuxTestActor(num_obs, num_actions, cfg.feature_dim, key=key)


register_actor(AuxTestActorCfg, AuxTestActor, _build)


def make_ppo(actor_cfg, recompute_gae: bool = False) -> PPO:
    pc = PPOPolicyConfig(actor=actor_cfg)
    model = PPOActorCritic(
        num_actor_obs=OBS_A,
        num_critic_obs=OBS_C,
        num_actions=ACT,
        actor_cfg=pc.actor,
        critic_cfg=pc.critic,
        init_noise_std=1.0,
        std_type=pc.std_type,
        distribution_type=pc.distribution_type,
        key=jax.random.PRNGKey(0),
        obs_normalization=True,
        obs_shapes={"actor": (OBS_A,), "critic": (OBS_C,)},
    )
    alg = PPO(
        actor_critic=model,
        num_learning_epochs=2,
        num_mini_batches=1 if recompute_gae else 2,
        recompute_gae_per_epoch=recompute_gae,
        key=jax.random.PRNGKey(1),
    )
    alg.init_storage(
        dict(
            num_envs=NUM_ENVS,
            num_transitions_per_env=NUM_STEPS,
            actor_obs_shape=[OBS_A],
            critic_obs_shape=[OBS_C],
            actions_shape=[ACT],
        )
    )
    return alg


def fill_rollout(alg: PPO, seed: int = 3) -> None:
    key = jax.random.PRNGKey(seed)
    for _ in range(NUM_STEPS):
        key, ka, kc, kr = jax.random.split(key, 4)
        a = jax.random.normal(ka, (NUM_ENVS, OBS_A))
        c = jax.random.normal(kc, (NUM_ENVS, OBS_C))
        alg.act(PPO.ActInput(a, c))
        rewards = jax.random.normal(kr, (NUM_ENVS,))
        zeros = jnp.zeros((NUM_ENVS,), dtype=bool)
        alg.process_env_step(rewards, zeros, zeros, {}, next_actor_obs=a, next_critic_obs=c)
    alg.compute_returns(c)


def _params(model):
    params, _ = eqx.partition(model, eqx.is_inexact_array, is_leaf=lambda x: isinstance(x, EmpiricalNormalization))
    return params


def _leaf_paths(tree) -> dict[str, np.ndarray]:
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {jax.tree_util.keystr(path): np.asarray(leaf) for path, leaf in flat}


def _changed(before: dict[str, np.ndarray], after: dict[str, np.ndarray], needle: str) -> list[str]:
    return [k for k in before if needle in k and not np.array_equal(before[k], after[k])]


def _unchanged(before: dict[str, np.ndarray], after: dict[str, np.ndarray], needle: str) -> list[str]:
    return [k for k in before if needle in k and np.array_equal(before[k], after[k])]


def check_registration(chk) -> None:
    cfg = AuxTestActorCfg(feature_dim=5)
    actor = build_actor(cfg, num_obs=OBS_A, num_actions=ACT, key=jax.random.PRNGKey(0))
    chk("registered cfg builds its actor", isinstance(actor, AuxTestActor) and actor.encoder.out_features == 5)
    saved = PPOPolicyConfig(actor=cfg).recursive_to_dict()
    loaded = PPOPolicyConfig.from_dict(saved)
    chk(
        "registered cfg hydrates from its saved form",
        isinstance(loaded.actor, AuxTestActorCfg) and loaded.actor.feature_dim == 5,
        f"got {type(loaded.actor).__name__}",
    )
    try:
        register_actor(AuxTestActorCfg, AuxTestActor, _build)
        same_ok = True
    except ValueError:
        same_ok = False
    chk("re-registering identical bindings is a no-op", same_ok)
    try:
        register_actor(AuxTestActorCfg, AuxTestActor, lambda *a, **k: None)
        rebound = True
    except ValueError:
        rebound = False
    chk("rebinding a registered cfg is refused", not rebound)


def check_direct_step(chk) -> None:
    alg = make_ppo(AuxTestActorCfg())
    obs = jax.random.normal(jax.random.PRNGKey(7), (NUM_ENVS, OBS_A))
    params, static = eqx.partition(
        alg.train_state.model, eqx.is_inexact_array, is_leaf=lambda x: isinstance(x, EmpiricalNormalization)
    )
    opt_state = alg.train_state.opt_state
    p0, s0 = _leaf_paths(params), _leaf_paths(opt_state)
    new_params, new_opt, loss0 = apply_auxiliary_step(params, static, opt_state, alg.optimizer, alg._param_labels, obs)
    p1, s1 = _leaf_paths(new_params), _leaf_paths(new_opt)

    chk("encoder weights move", len(_changed(p0, p1, ".actor.encoder")) == 2)
    chk("decoder untouched", not _changed(p0, p1, ".actor.decoder"), str(_changed(p0, p1, ".actor.decoder")))
    chk("critic untouched", not _changed(p0, p1, ".critic"), str(_changed(p0, p1, ".critic")))
    chk("std head untouched", not _changed(p0, p1, "std_module"), str(_changed(p0, p1, "std_module")))
    chk("encoder moments move", len(_changed(s0, s1, ".actor.encoder")) >= 2)
    for needle in (".actor.decoder", ".critic", "std_module"):
        chk(f"moments of {needle} untouched", not _changed(s0, s1, needle), str(_changed(s0, s1, needle))[:200])
    counts = _changed(s0, s1, "count")
    chk(
        "only the actor group's step counters advance",
        bool(counts) and all("'actor'" in c for c in counts),
        str(counts),
    )

    loss = loss0
    for _ in range(30):
        params, opt_state, loss = apply_auxiliary_step(
            new_params, static, new_opt, alg.optimizer, alg._param_labels, obs
        )
        new_params, new_opt = params, opt_state
    chk("30 repeated steps drive the loss down", float(loss) < float(loss0), f"{float(loss0):.4f} -> {float(loss):.4f}")


def check_full_update(chk, recompute_gae: bool) -> None:
    label = "recompute-GAE path" if recompute_gae else "scan path"
    alg = make_ppo(AuxTestActorCfg(), recompute_gae=recompute_gae)
    fill_rollout(alg)
    before = _leaf_paths(_params(alg.train_state.model))
    metrics = alg.update()
    after = _leaf_paths(_params(alg.train_state.model))
    chk(f"{label}: auxiliary loss reported", metrics.actor.auxiliary_loss > 0.0, f"{metrics.actor.auxiliary_loss:.4f}")
    chk(f"{label}: post_update_step hook ran once", float(alg.train_state.model.actor.schedule) == 0.5)
    chk(f"{label}: policy update still moves the decoder", len(_changed(before, after, ".actor.decoder")) == 2)
    chk(f"{label}: wandb dict carries actor/auxiliary_loss", "actor/auxiliary_loss" in metrics.to_wandb_dict())

    plain = make_ppo(PPOPolicyConfig().actor, recompute_gae=recompute_gae)
    fill_rollout(plain)
    pm = plain.update()
    chk(f"{label}: MLP actor reports 0 and updates", pm.actor.auxiliary_loss == 0.0 and pm.actor.policy_loss != 0.0)


def main() -> int:
    failures: list[str] = []

    def chk(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    print("=" * 78)
    print("PPO ACTOR AUXILIARY STEP")
    print("=" * 78)
    check_registration(chk)
    check_direct_step(chk)
    for recompute_gae in (False, True):
        check_full_update(chk, recompute_gae)
    print("=" * 78)
    print("=== RESULT: ALL OK ===" if not failures else f"=== RESULT: {len(failures)} FAILED ===")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
