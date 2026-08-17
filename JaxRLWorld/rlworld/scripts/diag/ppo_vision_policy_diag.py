"""Does the vision policy actually see, and actually learn from, the image?

The camera diag proves the RENDER is right. This proves the rest of the
path: that the image reaches the network, that it changes the action,
that the gradient comes back to the convolution weights, and that adding
it changed nothing for a policy that has no camera.

Each of those fails silently on its own. An encoder wired to a constant
still produces actions. A latent concatenated in the wrong order still
trains, slowly and wrongly. A CNN whose gradient never arrives leaves
its weights at their initialisation and the policy still improves, on
the state vector alone, looking exactly like a working vision policy
that has learnt to ignore the camera.

Run::

    jaxpy -m rlworld.scripts.diag.ppo_vision_policy_diag
    jaxpy -m rlworld.scripts.diag.ppo_vision_policy_diag --num-envs 64 --resolution 32
"""

from __future__ import annotations

import argparse

import equinox as eqx
import jax
import jax.numpy as jnp

from rlworld.rl.configs.presets.yam_lift.base import YamLiftConfig
from rlworld.rl.configs.presets.yam_lift.vision import CAMERA_GROUP, YamLiftVisionConfig
from rlworld.rl.modules.architectures.cnn.encoder import CNNEncoder, SpatialSoftmax, compute_output_dim, compute_padding
from rlworld.rl.runners import BaseRunner


def _conv_weights(model) -> list[jax.Array]:
    """Every convolution weight in the actor's encoders, in a fixed order."""
    encoders = model.actor.encoders
    return [conv.weight for group in sorted(encoders) for conv in encoders[group].convs]


def _trunk_weight(model) -> jax.Array:
    """The actor trunk's first layer — the anchor for "did anything train".

    Read the TRAINED model through the algorithm, not the runner: the
    runner keeps the model it built at startup and never rebinds it,
    while the updates land in the algorithm's train state.
    """
    return model.actor.trunk.net.linears[0].weight


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--resolution", type=int, default=32)
    args = ap.parse_args()

    print("=" * 78)
    print(f"PPO VISION POLICY  [yam_lift_vision / mjlab  num_envs={args.num_envs}  {args.resolution}px]")
    print("=" * 78)

    results: dict[str, bool] = {}

    # ── 1. the encoder's arithmetic ──────────────────────────────────
    # rsl_rl computes padding so each strided layer halves the image.
    # Getting this wrong does not raise: it produces a differently-sized
    # feature map and a latent of the wrong width, which the MLP happily
    # accepts.
    print("\n-- 1. the convolution stack --")
    res = args.resolution
    encoder = CNNEncoder(
        input_hw=(res, res),
        input_channels=1,
        output_channels=(16, 32),
        kernel_size=(5, 3),
        stride=(2, 2),
        key=jax.random.PRNGKey(0),
    )
    hw = (res, res)
    for kernel, stride in ((5, 2), (3, 2)):
        pad = compute_padding(hw, kernel, stride, 1)
        hw = compute_output_dim(hw, kernel, stride, 1, pad)
        print(f"  k={kernel} s={stride}: pad {pad} -> {hw}")
    print(f"  latent width {encoder.output_dim} (expected 2 per output channel = 64)")
    results["the_latent_is_two_numbers_per_channel"] = encoder.output_dim == 64
    results["strided_layers_halve_the_image"] = hw == (res // 4, res // 4)

    latent = encoder(jnp.zeros((1, res, res)))
    results["the_encoder_returns_that_width"] = latent.shape == (64,)

    # ── 2. what the spatial softmax reports ──────────────────────────
    # It should return WHERE the channel fired. Drive one pixel hard and
    # the answer must be that pixel's coordinate, on a grid running
    # -1..1 over each axis.
    print("\n-- 2. spatial softmax --")
    softmax = SpatialSoftmax(height=8, width=8, temperature=0.01)
    feature = jnp.zeros((1, 8, 8)).at[0, 6, 1].set(100.0)
    coords = softmax(feature)
    expected_h = jnp.linspace(-1.0, 1.0, 8)[6]
    expected_w = jnp.linspace(-1.0, 1.0, 8)[1]
    print(f"  a spike at row 6, col 1 -> {coords} (expected [{expected_h:.4f}, {expected_w:.4f}])")
    results["the_softmax_reports_where_the_channel_fired"] = bool(
        jnp.allclose(coords, jnp.array([expected_h, expected_w]), atol=1e-3)
    )

    uniform = softmax(jnp.zeros((1, 8, 8)))
    print(f"  a flat map -> {uniform} (expected the centre, [0, 0])")
    results["a_flat_map_reports_the_centre"] = bool(jnp.allclose(uniform, jnp.zeros(2), atol=1e-6))

    # ── 3. the policy is built around the image ──────────────────────
    print("\n-- 3. the built policy --")
    cfg = YamLiftVisionConfig(
        sim_type="mujoco",
        num_envs=args.num_envs,
        camera_width=args.resolution,
        camera_height=args.resolution,
    )
    cfgs = cfg.build()
    cfgs.runner.max_iterations = 1
    cfgs.algorithm.num_steps_per_env = 8
    runner = BaseRunner.create_with_env(cfgs, use_wandb=False)
    model = runner.actor_critic

    shapes = runner.obs_shapes
    print(f"  group shapes: {dict(shapes)}")
    print(f"  actor reads {runner.actor_image_groups} + the '{model.actor_vector_group}' vector")
    results["the_actor_reads_the_camera_group"] = runner.actor_image_groups == (CAMERA_GROUP,)
    results["the_critic_reads_the_camera_group"] = runner.critic_image_groups == (CAMERA_GROUP,)

    vector_dim = shapes["actor"][0]
    trunk_in = model.actor.trunk.num_obs
    print(
        f"  trunk input {trunk_in} = state vector {vector_dim} + latent {model.actor.encoders[CAMERA_GROUP].output_dim}"
    )
    results["the_trunk_is_widened_by_the_latent"] = (
        trunk_in == vector_dim + model.actor.encoders[CAMERA_GROUP].output_dim
    )

    # The privileged terms must be gone from the actor and kept by the
    # critic — otherwise the camera is decoration on top of the answer.
    critic_dim = shapes["critic"][0]
    print(f"  actor vector {vector_dim}, critic vector {critic_dim}")
    results["the_actor_lost_the_cube_coordinates"] = vector_dim < critic_dim

    # ── 4. the image changes the action ──────────────────────────────
    print("\n-- 4. does the image reach the action --")
    obs = runner.env.get_observation()
    packed = runner._pack_obs(obs, "actor")
    key = jax.random.PRNGKey(0)
    action_real, _ = model.act(packed, key=key, deterministic=True)
    blank = {**packed, CAMERA_GROUP: jnp.zeros_like(packed[CAMERA_GROUP])}
    action_blank, _ = model.act(blank, key=key, deterministic=True)
    delta = float(jnp.abs(action_real - action_blank).max())
    print(f"  blanking the depth image moves the action by {delta:.6f}")
    results["the_image_changes_the_action"] = delta > 1e-6

    value_real, _ = model.evaluate_value(runner._pack_obs(obs, "critic"))
    critic_blank = runner._pack_obs(obs, "critic")
    critic_blank = {**critic_blank, CAMERA_GROUP: jnp.zeros_like(critic_blank[CAMERA_GROUP])}
    value_blank, _ = model.evaluate_value(critic_blank)
    value_delta = float(jnp.abs(value_real - value_blank).max())
    print(f"  and the value by {value_delta:.6f}")
    results["the_image_changes_the_value"] = value_delta > 1e-6

    # ── 5. one training iteration ────────────────────────────────────
    print("\n-- 5. one PPO iteration --")
    before = [w.copy() for w in _conv_weights(runner.alg.actor_critic)]
    trunk_before = _trunk_weight(runner.alg.actor_critic).copy()
    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
    trained = runner.alg.actor_critic
    after = _conv_weights(trained)

    trunk_moved = float(jnp.abs(_trunk_weight(trained) - trunk_before).max())
    moved = [float(jnp.abs(a - b).max()) for a, b in zip(after, before, strict=True)]
    print(f"  trunk first layer moved by {trunk_moved:.3e}")
    print(f"  conv weights moved by {[f'{m:.3e}' for m in moved]}")
    results["the_update_trains_the_trunk"] = trunk_moved > 0.0
    results["the_gradient_reaches_every_conv_layer"] = all(m > 0.0 for m in moved)
    results["the_weights_stay_finite"] = all(bool(jnp.isfinite(a).all()) for a in after)

    # The normalizer must be the width of the state vector, not the
    # image: running statistics over pixels would drift with whatever
    # the camera is pointing at.
    normalizer = trained.actor_obs_normalizer
    if normalizer is not None:
        print(f"  actor normalizer covers {normalizer.mean.shape} (state vector is {(vector_dim,)})")
        results["normalization_covers_the_vector_only"] = tuple(normalizer.mean.shape)[-1:] == (vector_dim,)

    # ── 6. the stored rollout keeps the image ────────────────────────
    print("\n-- 6. what the rollout stored --")
    storage = runner.alg.storage
    stored = storage.actor_obs
    print(f"  storage actor_obs keys: {sorted(stored)}")
    results["the_rollout_stores_the_image"] = set(stored) == {"actor", CAMERA_GROUP}
    image_buffer = stored[CAMERA_GROUP]
    expected_shape = (storage.num_steps, args.num_envs, 1, args.resolution, args.resolution)
    print(f"  image buffer {tuple(image_buffer.shape)} (expected {expected_shape})")
    results["the_image_buffer_has_the_image_shape"] = tuple(image_buffer.shape) == expected_shape
    results["the_stored_image_is_normalised_depth"] = bool(
        jnp.isfinite(image_buffer).all() and image_buffer.min() >= 0.0 and image_buffer.max() <= 1.0
    )

    # ── 7. the vector-only policy is untouched ───────────────────────
    # The whole point of a separate group is that a preset with no
    # camera keeps taking one array, through the same storage and the
    # same jitted forwards.
    print("\n-- 7. a policy with no camera --")
    plain_cfgs = YamLiftConfig(sim_type="mujoco", num_envs=args.num_envs).build()
    plain_cfgs.runner.max_iterations = 1
    plain_cfgs.algorithm.num_steps_per_env = 8
    plain_runner = BaseRunner.create_with_env(plain_cfgs, use_wandb=False)
    print(f"  image groups: {plain_runner.actor_image_groups} (expected ())")
    results["a_vector_policy_declares_no_image_groups"] = plain_runner.actor_image_groups == ()
    results["a_vector_policy_has_no_group_dict"] = plain_runner.alg.actor_critic.actor_vector_group is None

    plain_obs = plain_runner._pack_obs(plain_runner.env.get_observation(), "actor")
    results["a_vector_policy_still_takes_one_array"] = isinstance(plain_obs, jax.Array)

    plain_runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
    plain_storage = plain_runner.alg.storage
    print(f"  storage actor_obs {tuple(plain_storage.actor_obs.shape)}")
    results["a_vector_rollout_is_one_buffer"] = isinstance(plain_storage.actor_obs, jax.Array)
    plain_leaves = jax.tree.leaves(eqx.filter(plain_runner.alg.actor_critic, eqx.is_array))
    results["the_vector_policy_stays_finite"] = all(bool(jnp.isfinite(leaf).all()) for leaf in plain_leaves)

    print("\n" + "=" * 78)
    ok = True
    for name, passed in results.items():
        print(f"  {name:<48}: {'PASS' if passed else 'FAIL'}")
        ok = ok and passed
    print(f"  {'OVERALL':<48}: {'PASS' if ok else 'FAIL'}")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
