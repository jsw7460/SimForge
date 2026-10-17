"""Gate: five small API contracts the managers and DR terms used to break.

Each is a one-line defect with a silent or late failure; each is tried here
on the real class with tensor stand-ins for the engine. No simulator.

1. contact ``order=``: the reindex cache is per manager, so a second manager
   that tracks the same group name in another order gets its own
   permutation (it used to reuse the first manager's, class-wide);
2. ``std_type="fixed"``: the std leaf is detached and its optimizer group
   frozen, so it stays exactly where it was set, with weight decay on too
   (it used to drift, 1.0 -> 0.999 after one update); the unfrozen
   optimizer is shown to move it, so the check is not vacuous;
3. an externally set command survives its resampling timer expiring (the
   resample used to run before the locked values were captured);
4. the mujoco DR backends forward ``distribution`` to mjlab for body mass,
   armature, joint friction and joint damping (they dropped it, so
   ``log_uniform`` was uniform on mujoco only), and ``shared_random`` /
   ``axes`` are refused on the backends that never implemented them
   instead of being ignored;
5. Newton joint-damping DR with ``operation="scale"`` reads a baseline the
   snapshot now holds (``joint_damping``; it raised AttributeError).

Run::

    python -m jaxrlworld.scripts.diag.gates.check_term_api_contracts
"""

from __future__ import annotations

import inspect
import sys
import types
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch
import warp as wp

from jaxrlworld.rl.algorithms.base import create_optimizer_with_labels
from jaxrlworld.rl.algorithms.ppo import PPO
from jaxrlworld.rl.configs.common_config_classes import PPOPolicyConfig
from jaxrlworld.rl.configs.scene.entity_selector import SceneEntitySelector
from jaxrlworld.rl.envs.managers.common.command import CommandManager, CommandManagerConfig
from jaxrlworld.rl.envs.managers.common.command_term import VelocityCommandTermCfg
from jaxrlworld.rl.envs.managers.common.contact import BaseContactManager
from jaxrlworld.rl.envs.mdp.events.dr import unified
from jaxrlworld.rl.envs.utils.newton import dr_baselines
from jaxrlworld.rl.modules.policies.ppo_ac import PPOActorCritic

OBS_A, OBS_C, ACT, NUM_ENVS, NUM_STEPS = 6, 8, 3, 8, 4
LAYOUT = (
    ("actor", 0, OBS_A),
    ("critic", OBS_A, OBS_C),
    ("reward", OBS_A + OBS_C, 1),
    ("terminated", OBS_A + OBS_C + 1, 1),
    ("truncated", OBS_A + OBS_C + 2, 1),
)


def chk_factory(failures: list[str]):
    def chk(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    return chk


# ── 1. contact reindex cache ─────────────────────────────────────────────


class _Contacts(BaseContactManager):
    def _compute_group_contact_force(self, group):
        return torch.zeros(1, group.num_tracked, 3)


def check_contact_reindex(chk) -> None:
    print("=== 1. contact order= permutation is per manager ===")
    env = SimpleNamespace(num_envs=1, device="cpu", control_dt=0.02)
    a, b = _Contacts(env), _Contacts(env)
    a._register_group("feet", ["left", "right"], ("force",))
    b._register_group("feet", ["right", "left"], ("force",))
    # Values are stored in each manager's own tracked order: a = [L, R], b = [R, L].
    out_a = a._apply_order(torch.tensor([[10.0, 20.0]]), "feet", ["left", "right"]).tolist()
    out_b = b._apply_order(torch.tensor([[20.0, 10.0]]), "feet", ["left", "right"]).tolist()
    chk("first manager, requested [left, right]", out_a == [[10.0, 20.0]], str(out_a))
    chk("second manager (tracks [right, left]), requested [left, right]", out_b == [[10.0, 20.0]], str(out_b))
    chk("caches are distinct objects", a._reindex_cache is not b._reindex_cache)


# ── 2. fixed std ─────────────────────────────────────────────────────────


def make_ppo(std_type: str, **kw) -> PPO:
    pc = PPOPolicyConfig()
    model = PPOActorCritic(
        num_actor_obs=OBS_A,
        num_critic_obs=OBS_C,
        num_actions=ACT,
        actor_cfg=pc.actor,
        critic_cfg=pc.critic,
        init_noise_std=1.0,
        std_type=std_type,
        distribution_type=pc.distribution_type,
        key=jax.random.PRNGKey(0),
        obs_normalization=False,
        obs_shapes={"actor": (OBS_A,), "critic": (OBS_C,)},
    )
    alg = PPO(actor_critic=model, num_learning_epochs=2, num_mini_batches=2, key=jax.random.PRNGKey(1), **kw)
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


def train(alg: PPO, iterations: int) -> None:
    a, c = jnp.ones((NUM_ENVS, OBS_A)), jnp.ones((NUM_ENVS, OBS_C))
    for it in range(iterations):
        for step in range(NUM_STEPS):
            alg.act(PPO.ActInput(a, c))
            k1, k2, k3 = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(7), it * NUM_STEPS + step), 3)
            packed = jnp.concatenate(
                [
                    jax.random.normal(k1, (NUM_ENVS, OBS_A)),
                    jax.random.normal(k2, (NUM_ENVS, OBS_C)),
                    jax.random.normal(k3, (NUM_ENVS, 1)),
                    jnp.zeros((NUM_ENVS, 2)),
                ],
                axis=1,
            )
            a, c = alg.process_env_step_packed(packed, LAYOUT)
        alg.compute_returns(c)
        alg.update()


def std_of(alg: PPO) -> np.ndarray:
    return np.asarray(alg.train_state.model.std_module.std)


def unfreeze(alg: PPO) -> None:
    """The optimizer as it was built before: the std group trained like the rest."""
    alg._frozen_labels = frozenset()
    alg.optimizer, _ = create_optimizer_with_labels(
        model=alg.train_state.model,
        label_fn=lambda path: "std" if "std_module" in ".".join(str(p) for p in path) else "actor",
        lr_config={"actor": alg.actor_lr, "std": alg.actor_lr},
        max_grad_norm=alg.max_grad_norm,
        optimizer_class=alg.optimizer_class,
    )
    params, _ = eqx.partition(alg.train_state.model, eqx.is_inexact_array)
    alg.train_state = alg.train_state._replace(opt_state=alg.optimizer.init(params))


def check_fixed_std(chk) -> None:
    print("\n=== 2. std_type='fixed' never moves ===")
    alg = make_ppo("fixed", optimizer="adamw", weight_decay=0.01)
    before = std_of(alg).copy()
    state = alg.train_state.opt_state[1].inner_states["std"]
    chk(
        "std group is frozen in the optimizer",
        type(state.inner_state).__name__ == "EmptyState",
        type(state.inner_state).__name__,
    )

    model = alg.train_state.model
    obs = jnp.ones((NUM_ENVS, OBS_A))
    actions = jnp.zeros((NUM_ENVS, ACT))

    def logp(m):
        dist, _ = m.get_distribution(obs, key=jax.random.PRNGKey(3))
        return dist.log_prob(actions).sum()

    grads = eqx.filter_grad(logp)(model)
    g_std = np.asarray(grads.std_module.std)
    chk("no gradient reaches the fixed std", bool(np.all(g_std == 0.0)), f"max |g| {np.abs(g_std).max():.3g}")

    train(alg, 3)
    after = std_of(alg)
    chk("fixed std bit-identical after 3 updates with weight decay", bool(np.array_equal(before, after)), f"{after}")

    control = make_ppo("fixed", optimizer="adamw", weight_decay=0.01)
    unfreeze(control)
    train(control, 3)
    moved = not np.array_equal(before, std_of(control))
    chk("the unfrozen optimizer does move it (control)", moved, f"{std_of(control)}")

    learnable = make_ppo("state_independent")
    s0 = np.asarray(learnable.train_state.model.std_module.log_std).copy()
    train(learnable, 1)
    chk(
        "learnable std still learns",
        not np.array_equal(s0, np.asarray(learnable.train_state.model.std_module.log_std)),
    )


# ── 3. external command lock ─────────────────────────────────────────────


def check_command_lock(chk) -> None:
    print("\n=== 3. locked command survives its timer ===")
    env = SimpleNamespace(num_envs=3, device="cpu", control_dt=0.02)
    cm = CommandManager(
        env,
        CommandManagerConfig(
            terms={
                "velocity": VelocityCommandTermCfg(
                    lin_vel_x_range=(1.0, 1.0),
                    lin_vel_y_range=(0.0, 0.0),
                    ang_vel_range=(0.0, 0.0),
                    resampling_time_range=(0.1, 0.1),
                )
            }
        ),
    )
    term = cm.get_term("velocity")
    cm.set_commands(torch.tensor([0]), velocity=torch.tensor([[9.0, 8.0, 7.0]]))
    term.time_left.zero_()  # every env's timer expires on the next compute
    cm.compute(0.02)
    cmd = cm.get_command("velocity")
    chk("locked env keeps its command across the resample", cmd[0].tolist() == [9.0, 8.0, 7.0], str(cmd[0].tolist()))
    chk("unlocked envs were resampled", cmd[1].tolist() == [1.0, 0.0, 0.0] and cmd[2].tolist() == [1.0, 0.0, 0.0])
    cm.release_commands(torch.tensor([0]), "velocity")
    term.time_left.zero_()
    cm.compute(0.02)
    chk("released env is resampled again", cm.get_command("velocity")[0].tolist() == [1.0, 0.0, 0.0])
    cm.set_commands(torch.tensor([1]), velocity=torch.tensor([[5.0, 4.0, 3.0]]))
    cm.reset(torch.tensor([1]))
    chk("episode reset clears the lock (documented)", not bool(term._externally_controlled[1].any()))


# ── 4. DR argument forwarding ────────────────────────────────────────────


def check_dr_forwarding(chk) -> None:
    print("\n=== 4. DR: distribution forwarded on mujoco, shared_random/axes refused elsewhere ===")
    recorded: dict[str, dict] = {}

    def recorder(name):
        def fn(**kwargs):
            recorded[name] = kwargs

        return fn

    fake_dr = types.ModuleType("mjlab.envs.mdp.dr")
    for name in (
        "body_mass",
        "joint_armature",
        "joint_friction",
        "joint_damping",
        "body_com_offset",
        "geom_friction",
        "pd_gains",
    ):
        setattr(fake_dr, name, recorder(name))
    saved = {k: sys.modules.get(k) for k in ("mjlab", "mjlab.envs", "mjlab.envs.mdp", "mjlab.envs.mdp.dr")}
    sys.modules["mjlab"] = types.ModuleType("mjlab")
    sys.modules["mjlab.envs"] = types.ModuleType("mjlab.envs")
    sys.modules["mjlab.envs.mdp"] = types.ModuleType("mjlab.envs.mdp")
    sys.modules["mjlab.envs.mdp.dr"] = fake_dr
    patched = {
        "_MujocoEnvAdapter": unified._MujocoEnvAdapter,
        "_selector_to_mjlab_cfg": unified._selector_to_mjlab_cfg,
        "_mujoco_recompute": unified._mujoco_recompute,
    }
    unified._MujocoEnvAdapter = lambda env: SimpleNamespace(scene=None, sim=None)
    unified._selector_to_mjlab_cfg = lambda cfg, scene: cfg
    unified._mujoco_recompute = lambda env, sim, level: None
    try:
        env = SimpleNamespace(sim_type="mujoco")
        ids = torch.tensor([0, 1])
        sel = SceneEntitySelector(name="robot")
        unified.randomize_body_mass(env, ids, sel, (0.9, 1.1), "scale", "log_uniform", False)
        unified.randomize_joint_armature(env, ids, sel, (0.9, 1.1), "scale", "log_uniform", False)
        unified.randomize_joint_friction(env, ids, sel, (0.0, 0.05), "abs", "log_uniform", True)
        unified.randomize_joint_damping(env, ids, sel, (0.0, 1.0), "abs", "gaussian", False)
        for name, dist in (
            ("body_mass", "log_uniform"),
            ("joint_armature", "log_uniform"),
            ("joint_friction", "log_uniform"),
            ("joint_damping", "gaussian"),
        ):
            got = recorded.get(name, {}).get("distribution")
            chk(f"mujoco {name} forwards distribution", got == dist, f"got {got!r}")
        chk("mujoco joint_friction forwards shared_random", recorded["joint_friction"].get("shared_random") is True)

        for sim in ("genesis", "newton"):
            env = SimpleNamespace(sim_type=sim)
            for fn, args in (
                (unified.randomize_body_mass, (env, ids, sel, (0.9, 1.1), "scale", "uniform", True)),
                (unified.randomize_joint_armature, (env, ids, sel, (0.9, 1.1), "scale", "uniform", True)),
                (unified.randomize_joint_friction, (env, ids, sel, (0.0, 0.05), "abs", "uniform", True)),
                (unified.randomize_joint_damping, (env, ids, sel, (0.0, 1.0), "abs", "uniform", True)),
            ):
                try:
                    fn(*args)
                    chk(f"{sim} {fn.__name__} shared_random=True refused", False, "ran")
                except NotImplementedError as e:
                    chk(f"{sim} {fn.__name__} shared_random=True refused", "mujoco backend only" in str(e))
            try:
                unified.randomize_body_com_offset(env, ids, {0: (-0.01, 0.01)}, sel, "add", [0], False)
                chk(f"{sim} randomize_body_com_offset axes refused", False, "ran")
            except NotImplementedError as e:
                chk(f"{sim} randomize_body_com_offset axes refused", "mujoco backend only" in str(e))
    finally:
        for k, v in patched.items():
            setattr(unified, k, v)
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


# ── 5. Newton damping baseline ───────────────────────────────────────────


def check_newton_damping_baseline(chk) -> None:
    print("\n=== 5. Newton joint-damping DR, operation='scale' ===")
    fields = {f.name for f in dr_baselines.NewtonDRBaselines.__dataclass_fields__.values()}
    chk("NewtonDRBaselines has joint_damping", "joint_damping" in fields)
    chk(
        "snapshot reads joint_damping off the articulation view",
        'get_attribute("joint_damping", model)' in inspect.getsource(dr_baselines.snapshot_newton_dr_baselines),
    )

    num_envs, dofs = 3, 4
    baseline = torch.arange(1.0, 1.0 + num_envs * dofs).reshape(num_envs, dofs)
    live = wp.array(np.asarray(baseline).reshape(num_envs, 1, dofs), dtype=wp.float32, device="cpu")
    written: dict = {}
    view = SimpleNamespace(
        get_attribute=lambda name, model: live,
        set_attribute=lambda name, model, value: written.setdefault(name, value.clone()),
    )
    env = SimpleNamespace(
        device="cpu",
        sim_type="newton",
        scene_manager=SimpleNamespace(robot_view=view, model=None, solver=None),
        act_manager=SimpleNamespace(indexing=SimpleNamespace(newton_qd_indices=torch.arange(dofs))),
        _dr_baselines=SimpleNamespace(joint_damping=baseline.clone()),
        _dr_pending_notify_flags=0,
    )
    saved_newton = sys.modules.get("newton")
    sys.modules["newton"] = SimpleNamespace(ModelFlags=SimpleNamespace(JOINT_DOF_PROPERTIES=2))
    saved_sel = unified._selected_joint_ids
    unified._selected_joint_ids = lambda env, cfg: None
    try:
        ids = torch.tensor([1])
        unified._newton_joint_damping_backend(
            env, ids, SceneEntitySelector(name="robot"), (2.0, 2.0), "scale", "uniform"
        )
    finally:
        unified._selected_joint_ids = saved_sel
        if saved_newton is None:
            sys.modules.pop("newton", None)
        else:
            sys.modules["newton"] = saved_newton
    out = written["joint_damping"].reshape(num_envs, dofs)
    expected = baseline.clone()
    expected[1] *= 2.0
    chk("scaled env is baseline x 2, others untouched", bool(torch.equal(out, expected)), f"{out[1].tolist()}")
    chk("notify flag folded into the pending batch", env._dr_pending_notify_flags == 2)


def main() -> int:
    failures: list[str] = []
    chk = chk_factory(failures)
    check_contact_reindex(chk)
    check_fixed_std(chk)
    check_command_lock(chk)
    check_dr_forwarding(chk)
    check_newton_damping_baseline(chk)
    print(f"\n=== RESULT: {'ALL OK' if not failures else f'{len(failures)} FAILED: {failures}'} ===")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
