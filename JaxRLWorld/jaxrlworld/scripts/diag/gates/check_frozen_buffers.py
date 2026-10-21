"""Gate: constant array buffers stay constant under the optimizer.

``SpatialSoftmax.pos_h / pos_w`` (the pixel coordinate grid) and
``GraphRelationalEmbedding.features`` (Laplacian / shortest-path / PageRank
of the kinematic tree) are constants that happen to be float arrays, so
``eqx.is_inexact_array`` puts them in the parameter partition and the
optimizer used to train them: one PPO update moved the grid by 1e-3 and
the policy's "where did the channel fire" coordinates drifted with it;
under adamw the weight decay shrank them with no gradient at all. They
are now declared ``eqx.field(metadata={"frozen": True})``, which
``create_optimizer_with_labels`` maps to a ``set_to_zero`` transform,
and read through ``stop_gradient``.

1. a vision PPO (adamw, weight decay on): every grid leaf is labelled
   frozen, an optimizer step with non-zero gradients leaves the grid bit
   for bit and moves the convolution weights;
2. the encoder's gradient with respect to the grid is exactly zero;
3. the graph embedding, built from a real kinematic tree: same two
   checks through the shared helper;
4. the fixed-std case the helper already handled still holds.

Run::

    python -m jaxrlworld.scripts.diag.gates.check_frozen_buffers
"""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import jaxrlworld
from jaxrlworld.rl.algorithms.base import FROZEN_LABEL, create_optimizer_with_labels
from jaxrlworld.rl.algorithms.ppo import PPO
from jaxrlworld.rl.configs.common_config_classes import (
    CNNEncoderCfg,
    PPOPolicyConfig,
    StdType,
    VisionActorCfg,
    VisionCriticCfg,
)
from jaxrlworld.rl.configs.robots.kinematic_tree import KinematicTree
from jaxrlworld.rl.modules.architectures.space_time_transformer.embeddings import GraphRelationalEmbedding
from jaxrlworld.rl.modules.policies.ppo_ac import PPOActorCritic

OBS_A, OBS_C, ACT = 6, 8, 3
IMAGE = (1, 16, 16)
URDF = Path(jaxrlworld.__file__).parent / "assets" / "g1_description" / "g1_23dof.urdf"


def leaf_paths(tree) -> dict[str, jax.Array]:
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {jax.tree_util.keystr(path): leaf for path, leaf in flat}


def labels_by_path(labels) -> dict[str, str]:
    flat, _ = jax.tree_util.tree_flatten_with_path(labels)
    return {jax.tree_util.keystr(path): label for path, label in flat}


def make_vision_ppo(std_type: StdType) -> PPO:
    cnn = CNNEncoderCfg(output_channels=(4, 8), kernel_size=(3, 3), stride=(2, 2))
    pc = PPOPolicyConfig(std_type=std_type)
    model = PPOActorCritic(
        num_actor_obs=OBS_A,
        num_critic_obs=OBS_C,
        num_actions=ACT,
        actor_cfg=VisionActorCfg(cnn=cnn),
        critic_cfg=VisionCriticCfg(cnn=cnn),
        init_noise_std=1.0,
        std_type=std_type,
        distribution_type=pc.distribution_type,
        key=jax.random.PRNGKey(0),
        obs_normalization=False,
        obs_shapes={"actor": (OBS_A,), "critic": (OBS_C,), "camera": IMAGE},
    )
    return PPO(actor_critic=model, optimizer="adamw", weight_decay=0.1, key=jax.random.PRNGKey(1))


def one_step(optimizer, params):
    """One optimizer step with a gradient of ones on every leaf."""
    opt_state = optimizer.init(params)
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    updates, _ = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates)


def main() -> int:
    failures: list[str] = []

    def chk(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    print("=" * 78)
    print("FROZEN BUFFERS")
    print("=" * 78)

    # ── 1. vision PPO: labels and one optimizer step ─────────────────
    print("\n-- 1. vision PPO under adamw --")
    alg = make_vision_ppo(StdType.STATE_INDEPENDENT)
    params, _ = eqx.partition(alg.train_state.model, eqx.is_inexact_array)
    labels = labels_by_path(alg._param_labels)
    grid_paths = [p for p in labels if p.endswith(".pos_h") or p.endswith(".pos_w")]
    chk("vision model has grid leaves in actor and critic", len(grid_paths) == 4, f"{len(grid_paths)} found")
    chk(
        "every grid leaf is labelled frozen",
        all(labels[p] == FROZEN_LABEL for p in grid_paths),
        ", ".join(f"{p}={labels[p]}" for p in grid_paths),
    )
    chk(
        "no other leaf is labelled frozen",
        all(label != FROZEN_LABEL for p, label in labels.items() if p not in grid_paths),
    )
    before = leaf_paths(params)
    after = leaf_paths(one_step(alg.optimizer, params))
    grid_same = all(np.array_equal(np.asarray(before[p]), np.asarray(after[p])) for p in grid_paths)
    chk("one step leaves the grid bit-identical", grid_same)
    conv_paths = [p for p in before if ".convs" in p and p.endswith(".weight")]
    conv_moved = all(not np.array_equal(np.asarray(before[p]), np.asarray(after[p])) for p in conv_paths)
    chk("the same step moves the convolution weights", conv_moved, f"{len(conv_paths)} conv weights")

    # ── 2. the encoder's gradient wrt the grid is exactly zero ───────
    print("\n-- 2. stop_gradient on the grid --")
    encoder = alg.train_state.model.actor.encoders["camera"]
    image = jax.random.normal(jax.random.PRNGKey(2), IMAGE)
    grads = eqx.filter_grad(lambda enc, x: jnp.sum(enc(x) ** 2))(encoder, image)
    chk(
        "d(encoder)/d(pos_h), d(pos_w) are exactly zero",
        bool(jnp.all(grads.spatial_softmax.pos_h == 0)) and bool(jnp.all(grads.spatial_softmax.pos_w == 0)),
    )
    conv_grad = grads.convs[0].weight
    chk("the convolution still receives a gradient", bool(jnp.any(conv_grad != 0)))

    # ── 3. graph relational embedding through the shared helper ──────
    print("\n-- 3. graph relational embedding --")
    tree = KinematicTree(urdf_path=str(URDF))
    emb = GraphRelationalEmbedding(tree, num_heads=2, key=jax.random.PRNGKey(3))
    optimizer, emb_labels = create_optimizer_with_labels(
        model=emb,
        label_fn=lambda path: "actor",
        lr_config={"actor": 1e-2},
        optimizer_class=optax.adamw,
    )
    labels = labels_by_path(emb_labels)
    chk(
        "features labelled frozen, projection labelled actor",
        labels[".features"] == FROZEN_LABEL and labels[".projection.weight"] == "actor",
        str(labels),
    )
    params, _ = eqx.partition(emb, eqx.is_inexact_array)
    before = leaf_paths(params)
    after = leaf_paths(one_step(optimizer, params))
    chk(
        "one step leaves the features bit-identical",
        np.array_equal(np.asarray(before[".features"]), np.asarray(after[".features"])),
    )
    chk(
        "the same step moves the projection",
        not np.array_equal(np.asarray(before[".projection.weight"]), np.asarray(after[".projection.weight"])),
    )
    grads = eqx.filter_grad(lambda m: jnp.sum(m() ** 2))(emb)
    chk("d(bias)/d(features) is exactly zero", bool(jnp.all(grads.features == 0)))
    chk("the projection still receives a gradient", bool(jnp.any(grads.projection.weight != 0)))

    # ── 4. the fixed std is still frozen ─────────────────────────────
    print("\n-- 4. fixed std (regression) --")
    alg = make_vision_ppo(StdType.FIXED)
    params, _ = eqx.partition(alg.train_state.model, eqx.is_inexact_array)
    labels = labels_by_path(alg._param_labels)
    std_paths = [p for p in labels if ".std_module" in p]
    before = leaf_paths(params)
    after = leaf_paths(one_step(alg.optimizer, params))
    chk(
        "fixed std labelled std and unchanged by a step",
        bool(std_paths)
        and all(labels[p] == "std" for p in std_paths)
        and all(np.array_equal(np.asarray(before[p]), np.asarray(after[p])) for p in std_paths),
        ", ".join(std_paths),
    )

    print("=" * 78)
    print("=== RESULT: ALL OK ===" if not failures else f"=== RESULT: {len(failures)} FAILED ===")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
