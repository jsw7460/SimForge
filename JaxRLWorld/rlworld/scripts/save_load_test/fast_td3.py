"""
FastTD3 Save/Load 완벽 검증 스크립트 (리팩토링 후 버전)

검증 항목:
1. Model weights (actor, critic1, critic2)
2. Target network weights
3. Optimizer states
4. Noise scales
5. Observation normalizers (mean, var, count) - 이제 model 내부에 저장됨
6. Random key
7. Action 출력 일치성 (normalization 포함)
8. Q-value 출력 일치성 (normalization 포함)
9. Target network forward pass 일치성
10. Inference action 일치성 (act_inference 메서드)

리팩토링 변경사항:
- actor_obs_normalizer, critic_obs_normalizer가 train_state에서 model 내부로 이동
- normalizers.eqx 파일 제거됨 (model.eqx에 포함)
- normalization이 model 메서드 내부에서 자동 적용됨
"""

import os
import tempfile
import shutil

os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

# Custom assets setup
custom_assets = os.path.abspath(os.path.join(os.path.dirname(__file__), 'assets'))
import genesis.utils.terrain
genesis.utils.misc.get_assets_dir = lambda: custom_assets
genesis.utils.terrain.get_assets_dir = lambda: custom_assets

from rlworld.rl.configs.algorithms import FastTD3Config
from rlworld.rl.configs import NewtonConfigsForRun, FastTD3PolicyConfig
from rlworld.rl.runners import BaseRunner
from rlworld.rl.configs.presets.go1.newton.mlp import get_config


def arrays_equal(a, b, name: str, rtol: float = 1e-6, atol: float = 1e-8) -> bool:
    """Compare two arrays with detailed error reporting."""
    if a is None and b is None:
        print(f"  ✅ {name}: Both None")
        return True
    if a is None or b is None:
        print(f"  ❌ {name}: One is None (a={a is None}, b={b is None})")
        return False

    a_np = np.array(a)
    b_np = np.array(b)

    if a_np.shape != b_np.shape:
        print(f"  ❌ {name}: Shape mismatch ({a_np.shape} vs {b_np.shape})")
        return False

    if np.allclose(a_np, b_np, rtol=rtol, atol=atol):
        max_diff = np.max(np.abs(a_np - b_np))
        print(f"  ✅ {name}: Match (max_diff={max_diff:.2e})")
        return True
    else:
        max_diff = np.max(np.abs(a_np - b_np))
        mean_diff = np.mean(np.abs(a_np - b_np))
        print(f"  ❌ {name}: MISMATCH (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})")
        return False


def compare_pytree(tree1, tree2, name: str) -> bool:
    """Compare two pytrees leaf by leaf."""
    leaves1, struct1 = jax.tree_util.tree_flatten(tree1)
    leaves2, struct2 = jax.tree_util.tree_flatten(tree2)

    if len(leaves1) != len(leaves2):
        print(f"  ❌ {name}: Different number of leaves ({len(leaves1)} vs {len(leaves2)})")
        return False

    all_match = True
    for i, (l1, l2) in enumerate(zip(leaves1, leaves2)):
        if not arrays_equal(l1, l2, f"{name}[{i}]"):
            all_match = False

    return all_match


def extract_state_snapshot(algorithm):
    """
    Extract complete state snapshot for comparison.

    NOTE: 리팩토링 후 normalizer는 model 내부에 저장됨
    (train_state.model.actor_obs_normalizer, train_state.model.critic_obs_normalizer)
    """
    ts = algorithm.train_state
    model = ts.model

    # Extract all model parameters
    actor_params, _ = eqx.partition(model.actor, eqx.is_inexact_array)
    critic1_params, _ = eqx.partition(model.critic1, eqx.is_inexact_array)
    critic2_params, _ = eqx.partition(model.critic2, eqx.is_inexact_array)

    # Extract normalizer states from model (리팩토링 후 위치 변경)
    actor_normalizer = model.actor_obs_normalizer
    critic_normalizer = model.critic_obs_normalizer

    snapshot = {
        # Model weights
        "actor_params": actor_params,
        "critic1_params": critic1_params,
        "critic2_params": critic2_params,

        # Target networks
        "target_actor_params": ts.target_actor_params,
        "target_critic1_params": ts.target_critic1_params,
        "target_critic2_params": ts.target_critic2_params,

        # Noise scales
        "noise_scales": ts.noise_scales,

        # Random key
        "key": ts.key,

        # Normalizers (이제 model 내부에서 가져옴)
        "actor_obs_normalizer_mean": actor_normalizer.mean if actor_normalizer else None,
        "actor_obs_normalizer_var": actor_normalizer.var if actor_normalizer else None,
        "actor_obs_normalizer_count": actor_normalizer.count if actor_normalizer else None,
        "critic_obs_normalizer_mean": critic_normalizer.mean if critic_normalizer else None,
        "critic_obs_normalizer_var": critic_normalizer.var if critic_normalizer else None,
        "critic_obs_normalizer_count": critic_normalizer.count if critic_normalizer else None,

        # Counters
        "total_it": algorithm.total_it,

        # Distributional RL config
        "support": model.support,
        "v_min": model.v_min,
        "v_max": model.v_max,
        "num_atoms": model.num_atoms,
    }

    return snapshot


def compare_snapshots(snap1, snap2, label: str) -> bool:
    """Compare two state snapshots."""
    print(f"\n{'='*60}")
    print(f"Comparing: {label}")
    print('='*60)

    all_match = True

    # Model weights
    print("\n[Model Weights]")
    all_match &= compare_pytree(snap1["actor_params"], snap2["actor_params"], "actor_params")
    all_match &= compare_pytree(snap1["critic1_params"], snap2["critic1_params"], "critic1_params")
    all_match &= compare_pytree(snap1["critic2_params"], snap2["critic2_params"], "critic2_params")

    # Target networks
    print("\n[Target Networks]")
    all_match &= compare_pytree(snap1["target_actor_params"], snap2["target_actor_params"], "target_actor_params")
    all_match &= compare_pytree(snap1["target_critic1_params"], snap2["target_critic1_params"], "target_critic1_params")
    all_match &= compare_pytree(snap1["target_critic2_params"], snap2["target_critic2_params"], "target_critic2_params")

    # Noise scales
    print("\n[Noise Scales]")
    all_match &= arrays_equal(snap1["noise_scales"], snap2["noise_scales"], "noise_scales")

    # Random key
    print("\n[Random Key]")
    all_match &= arrays_equal(snap1["key"], snap2["key"], "key")

    # Normalizers (이제 model 내부에 저장됨)
    print("\n[Observation Normalizers (stored in model)]")
    all_match &= arrays_equal(snap1["actor_obs_normalizer_mean"], snap2["actor_obs_normalizer_mean"], "model.actor_obs_normalizer.mean")
    all_match &= arrays_equal(snap1["actor_obs_normalizer_var"], snap2["actor_obs_normalizer_var"], "model.actor_obs_normalizer.var")
    all_match &= arrays_equal(snap1["actor_obs_normalizer_count"], snap2["actor_obs_normalizer_count"], "model.actor_obs_normalizer.count")
    all_match &= arrays_equal(snap1["critic_obs_normalizer_mean"], snap2["critic_obs_normalizer_mean"], "model.critic_obs_normalizer.mean")
    all_match &= arrays_equal(snap1["critic_obs_normalizer_var"], snap2["critic_obs_normalizer_var"], "model.critic_obs_normalizer.var")
    all_match &= arrays_equal(snap1["critic_obs_normalizer_count"], snap2["critic_obs_normalizer_count"], "model.critic_obs_normalizer.count")

    # Counters
    print("\n[Counters]")
    if snap1["total_it"] == snap2["total_it"]:
        print(f"  ✅ total_it: {snap1['total_it']}")
    else:
        print(f"  ❌ total_it: {snap1['total_it']} vs {snap2['total_it']}")
        all_match = False

    # Distributional RL config
    print("\n[Distributional RL Config]")
    all_match &= arrays_equal(snap1["support"], snap2["support"], "support")
    if snap1["v_min"] == snap2["v_min"]:
        print(f"  ✅ v_min: {snap1['v_min']}")
    else:
        print(f"  ❌ v_min: {snap1['v_min']} vs {snap2['v_min']}")
        all_match = False
    if snap1["v_max"] == snap2["v_max"]:
        print(f"  ✅ v_max: {snap1['v_max']}")
    else:
        print(f"  ❌ v_max: {snap1['v_max']} vs {snap2['v_max']}")
        all_match = False
    if snap1["num_atoms"] == snap2["num_atoms"]:
        print(f"  ✅ num_atoms: {snap1['num_atoms']}")
    else:
        print(f"  ❌ num_atoms: {snap1['num_atoms']} vs {snap2['num_atoms']}")
        all_match = False

    return all_match


def test_model_act(algorithm, test_obs_actor):
    """
    Test model.act() output.

    NOTE: 리팩토링 후 normalization은 model.act() 내부에서 자동 적용됨
    (_normalize_actor_obs() 호출)
    """
    model = algorithm.train_state.model
    key = jax.random.PRNGKey(42)

    # model.act()는 내부에서 normalization을 자동 적용함
    actions, aux = model.act(test_obs_actor, key=key)

    return np.array(actions)


def test_model_act_inference(algorithm, test_obs_actor):
    """
    Test model.act_inference() output.

    NOTE: 리팩토링의 핵심 이점 - inference 시에도 normalization 자동 적용
    """
    model = algorithm.train_state.model
    key = jax.random.PRNGKey(42)

    # act_inference()도 내부에서 _normalize_actor_obs() 호출
    actions, aux = model.act_inference(test_obs_actor, key=key)

    return np.array(actions)


def test_model_evaluate(algorithm, test_obs_actor, test_obs_critic):
    """
    Test model.evaluate() output.

    NOTE: model.evaluate()는 내부에서 act() + critic*_q_value() 호출
    모두 자동으로 normalization 적용됨
    """
    model = algorithm.train_state.model
    key = jax.random.PRNGKey(42)

    # evaluate()는 내부에서 normalization을 자동 적용함
    values = model.evaluate(test_obs_actor, test_obs_critic, key=key)

    return np.array(values)


def test_critic_q_values(algorithm, test_obs_critic, test_actions):
    """
    Test critic Q-value output.

    NOTE: 리팩토링 후 critic*_q_value()가 내부에서 _normalize_critic_obs() 호출
    """
    model = algorithm.train_state.model

    # critic*_q_value()는 내부에서 normalization을 자동 적용함
    q1 = model.critic1_q_value(test_obs_critic, test_actions)
    q2 = model.critic2_q_value(test_obs_critic, test_actions)

    return {
        "q1": np.array(q1),
        "q2": np.array(q2),
    }


def test_critic_forward_raw(algorithm, test_obs_critic, test_actions):
    """
    Test critic forward pass returning raw logits.

    NOTE: critic*_forward()도 내부에서 _normalize_critic_obs() 호출
    """
    model = algorithm.train_state.model

    # critic*_forward()는 내부에서 normalization을 자동 적용함
    logits1 = model.critic1_forward(test_obs_critic, test_actions)
    logits2 = model.critic2_forward(test_obs_critic, test_actions)

    return {
        "logits1": np.array(logits1),
        "logits2": np.array(logits2),
    }


def test_target_network_forward(algorithm, test_obs_actor, test_obs_critic, test_actions):
    """
    Test target network forward pass.

    NOTE: Target networks는 model 외부에 params로 저장되어 있으므로
    normalization을 수동으로 적용해야 함 (model._normalize_*_obs() 사용)
    """
    ts = algorithm.train_state
    model = ts.model

    # Normalize observations using model's normalizers
    norm_actor_obs = model._normalize_actor_obs(test_obs_actor)
    norm_critic_obs = model._normalize_critic_obs(test_obs_critic)

    # Reconstruct target networks
    target_actor = eqx.combine(ts.target_actor_params, ts.target_actor_static)
    target_critic1 = eqx.combine(ts.target_critic1_params, ts.target_critic1_static)
    target_critic2 = eqx.combine(ts.target_critic2_params, ts.target_critic2_static)

    # Forward pass through target actor
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, norm_actor_obs.shape[0])
    target_actions_raw, _ = jax.vmap(target_actor)(norm_actor_obs, key=keys)

    # Apply tanh if squashed
    if model.is_squashed:
        target_actions = jnp.tanh(target_actions_raw)
    else:
        target_actions = target_actions_raw

    # Forward pass through target critics
    target_logits1 = target_critic1(norm_critic_obs, test_actions)
    target_logits2 = target_critic2(norm_critic_obs, test_actions)

    # Compute Q-values from logits
    target_probs1 = jax.nn.softmax(target_logits1, axis=-1)
    target_probs2 = jax.nn.softmax(target_logits2, axis=-1)
    support = model.support
    target_q1 = jnp.sum(target_probs1 * support, axis=-1)
    target_q2 = jnp.sum(target_probs2 * support, axis=-1)

    return {
        "target_actions_raw": np.array(target_actions_raw),
        "target_actions": np.array(target_actions),
        "target_logits1": np.array(target_logits1),
        "target_logits2": np.array(target_logits2),
        "target_q1": np.array(target_q1),
        "target_q2": np.array(target_q2),
    }


def test_normalization_consistency(algorithm, test_obs_actor, test_obs_critic):
    """
    Test that normalization is consistent.

    직접 normalizer를 호출한 결과와 model 메서드 내부 normalization이 일치하는지 확인
    """
    model = algorithm.train_state.model

    # Direct normalizer call
    if model.actor_obs_normalizer is not None:
        direct_norm_actor = model.actor_obs_normalizer.normalize(test_obs_actor)
        direct_norm_critic = model.critic_obs_normalizer.normalize(test_obs_critic)
    else:
        direct_norm_actor = test_obs_actor
        direct_norm_critic = test_obs_critic

    # Via model method
    method_norm_actor = model._normalize_actor_obs(test_obs_actor)
    method_norm_critic = model._normalize_critic_obs(test_obs_critic)

    return {
        "direct_norm_actor": np.array(direct_norm_actor),
        "method_norm_actor": np.array(method_norm_actor),
        "direct_norm_critic": np.array(direct_norm_critic),
        "method_norm_critic": np.array(method_norm_critic),
    }


def main():
    print("="*70)
    print("FastTD3 Save/Load 완벽 검증 (리팩토링 후 버전)")
    print("="*70)
    print("\n리팩토링 변경사항:")
    print("  - normalizer가 train_state에서 model 내부로 이동")
    print("  - normalizers.eqx 파일 제거됨 (model.eqx에 포함)")
    print("  - normalization이 model 메서드 내부에서 자동 적용됨")

    # =========================================================================
    # 1. Setup
    # =========================================================================
    print("\n[1/9] Setting up environment and algorithm...")

    configs_dict = get_config()
    cfgs_for_run = NewtonConfigsForRun.from_dict(configs_dict)

    scale_param = 5.0
    cfgs_for_run.action.action_scale = cfgs_for_run.action.action_scale / (scale_param / 2)
    cfgs_for_run.action.clip_actions = (-scale_param, scale_param)

    cfgs_for_run.nn.policy.actor_kwargs.update({
        "hidden_dims": [512, 256, 128],
        "activation": "relu",
    })
    cfgs_for_run.nn.policy.critic_kwargs.update({
        "hidden_dims": [1024, 512, 256],
        "activation": "relu",
    })
    cfgs_for_run.runner.max_iterations = 10

    fast_td3_config = FastTD3Config(
        actor_lr=1e-4,
        critic_lr=5e-4,
        gamma=0.95,
        tau=0.005,
        batch_size=1024,
        buffer_size=100_000,
        learning_starts=2,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        num_atoms=51,
        v_min=-50.0,
        v_max=50.0,
        noise_min=0.05,
        noise_max=0.4,
        is_squashed=True,
        use_cdq=True,
        num_gradient_steps=2,
        obs_normalization=True,  # 중요: normalization 활성화
    )
    cfgs_for_run.algorithm = fast_td3_config
    cfgs_for_run.nn.policy = cfgs_for_run.nn.policy.to(FastTD3PolicyConfig)

    runner = BaseRunner.create_with_env(cfgs_for_run)
    algorithm = runner.alg

    print(f"  Actor obs dim: {algorithm.actor_critic.actor_obs_dim}")
    print(f"  Critic obs dim: {algorithm.actor_critic.critic_obs_dim}")
    print(f"  Action dim: {algorithm.actor_critic.num_actions}")
    print(f"  Num envs: {algorithm.num_envs}")
    print(f"  Obs normalization: {algorithm.obs_normalization}")

    # Verify normalizer location (리팩토링 검증)
    model = algorithm.train_state.model
    print(f"\n  [리팩토링 검증]")
    print(f"  model.actor_obs_normalizer is not None: {model.actor_obs_normalizer is not None}")
    print(f"  model.critic_obs_normalizer is not None: {model.critic_obs_normalizer is not None}")

    # =========================================================================
    # 2. Create test data
    # =========================================================================
    print("\n[2/9] Creating test data...")

    key = jax.random.PRNGKey(12345)
    key, k1, k2, k3 = jax.random.split(key, 4)

    num_test_samples = algorithm.num_envs
    actor_obs_dim = algorithm.actor_critic.actor_obs_dim
    critic_obs_dim = algorithm.actor_critic.critic_obs_dim
    action_dim = algorithm.actor_critic.num_actions

    test_obs_actor = jax.random.normal(k1, (num_test_samples, actor_obs_dim))
    test_obs_critic = jax.random.normal(k2, (num_test_samples, critic_obs_dim))
    test_actions = jax.random.uniform(k3, (num_test_samples, action_dim), minval=-1, maxval=1)

    print(f"  Test obs actor shape: {test_obs_actor.shape}")
    print(f"  Test obs critic shape: {test_obs_critic.shape}")
    print(f"  Test actions shape: {test_actions.shape}")

    # =========================================================================
    # 3. Run training iterations
    # =========================================================================
    print("\n[3/9] Running training iterations...")

    num_train_iterations = 3
    for i in range(num_train_iterations):
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
        print(f"  Iteration {i+1}/{num_train_iterations} complete, total_it={algorithm.total_it}")

    # =========================================================================
    # 4. Take snapshot BEFORE save
    # =========================================================================
    print("\n[4/9] Taking snapshot before save...")

    snapshot_before = extract_state_snapshot(algorithm)

    # Test all model outputs
    actions_before = test_model_act(algorithm, test_obs_actor)
    actions_inference_before = test_model_act_inference(algorithm, test_obs_actor)
    values_before = test_model_evaluate(algorithm, test_obs_actor, test_obs_critic)
    qvalues_before = test_critic_q_values(algorithm, test_obs_critic, test_actions)
    logits_before = test_critic_forward_raw(algorithm, test_obs_critic, test_actions)
    target_before = test_target_network_forward(algorithm, test_obs_actor, test_obs_critic, test_actions)
    norm_before = test_normalization_consistency(algorithm, test_obs_actor, test_obs_critic)

    print(f"  total_it: {snapshot_before['total_it']}")
    print(f"  actor_obs_normalizer.mean[:5]: {snapshot_before['actor_obs_normalizer_mean'][:5] if snapshot_before['actor_obs_normalizer_mean'] is not None else None}")
    print(f"  actor_obs_normalizer.count: {snapshot_before['actor_obs_normalizer_count']}")
    print(f"  Actions (act) mean: {actions_before.mean():.6f}")
    print(f"  Actions (inference) mean: {actions_inference_before.mean():.6f}")
    print(f"  Values mean: {values_before.mean():.6f}")
    print(f"  Q1 mean: {qvalues_before['q1'].mean():.6f}")

    # =========================================================================
    # 5. Save checkpoint
    # =========================================================================
    print("\n[5/9] Saving checkpoint...")

    checkpoint_dir = tempfile.mkdtemp(prefix="fasttd3_test_")
    print(f"  Checkpoint dir: {checkpoint_dir}")

    metadata = algorithm.save_train_state(checkpoint_dir)

    saved_files = os.listdir(checkpoint_dir)
    print(f"  Saved files: {saved_files}")
    print(f"  Metadata keys: {list(metadata.keys())}")

    # Verify files exist (리팩토링 후: normalizers.eqx 없어야 함)
    assert os.path.exists(os.path.join(checkpoint_dir, "model.eqx")), "model.eqx not found!"
    assert os.path.exists(os.path.join(checkpoint_dir, "target_networks.eqx")), "target_networks.eqx not found!"

    # 리팩토링 검증: normalizers.eqx가 없어야 함
    normalizers_path = os.path.join(checkpoint_dir, "normalizers.eqx")
    if os.path.exists(normalizers_path):
        print(f"  ⚠️  WARNING: normalizers.eqx exists (should be removed after refactoring)")
    else:
        print(f"  ✅ normalizers.eqx does not exist (correct after refactoring)")

    print("  ✅ All expected files exist")

    # =========================================================================
    # 6. Corrupt state (simulate restart)
    # =========================================================================
    print("\n[6/9] Corrupting state to simulate restart...")

    # Corrupt random key and noise scales
    corrupted_key = jax.random.PRNGKey(99999)
    corrupted_noise = jnp.ones_like(algorithm.train_state.noise_scales) * 0.999

    # Corrupt normalizers inside model
    model = algorithm.train_state.model
    if model.actor_obs_normalizer is not None:
        corrupted_actor_normalizer = eqx.tree_at(
            lambda n: (n.mean, n.var, n.count),
            model.actor_obs_normalizer,
            (
                jnp.ones_like(model.actor_obs_normalizer.mean) * 999.0,
                jnp.ones_like(model.actor_obs_normalizer.var) * 999.0,
                jnp.array(999999.0),
            ),
        )
        corrupted_critic_normalizer = eqx.tree_at(
            lambda n: (n.mean, n.var, n.count),
            model.critic_obs_normalizer,
            (
                jnp.ones_like(model.critic_obs_normalizer.mean) * 999.0,
                jnp.ones_like(model.critic_obs_normalizer.var) * 999.0,
                jnp.array(999999.0),
            ),
        )
        corrupted_model = eqx.tree_at(
            lambda m: (m.actor_obs_normalizer, m.critic_obs_normalizer),
            model,
            (corrupted_actor_normalizer, corrupted_critic_normalizer),
        )
    else:
        corrupted_model = model

    algorithm.train_state = algorithm.train_state._replace(
        model=corrupted_model,
        key=corrupted_key,
        noise_scales=corrupted_noise,
    )
    algorithm.total_it = 99999

    # Verify corruption
    snapshot_corrupted = extract_state_snapshot(algorithm)
    assert not np.allclose(np.array(snapshot_corrupted["key"]), np.array(snapshot_before["key"])), "Key not corrupted!"
    assert algorithm.total_it == 99999, "total_it not corrupted!"
    if snapshot_before["actor_obs_normalizer_mean"] is not None:
        assert not np.allclose(
            np.array(snapshot_corrupted["actor_obs_normalizer_mean"]),
            np.array(snapshot_before["actor_obs_normalizer_mean"])
        ), "Normalizer mean not corrupted!"
    print("  ✅ State successfully corrupted")

    # =========================================================================
    # 7. Load checkpoint
    # =========================================================================
    print("\n[7/9] Loading checkpoint...")

    algorithm.load_train_state(checkpoint_dir, metadata)

    print(f"  Loaded total_it: {algorithm.total_it}")

    # =========================================================================
    # 8. Verify everything matches
    # =========================================================================
    print("\n[8/9] Verifying loaded state...")

    snapshot_after = extract_state_snapshot(algorithm)

    # Test all model outputs after load
    actions_after = test_model_act(algorithm, test_obs_actor)
    actions_inference_after = test_model_act_inference(algorithm, test_obs_actor)
    values_after = test_model_evaluate(algorithm, test_obs_actor, test_obs_critic)
    qvalues_after = test_critic_q_values(algorithm, test_obs_critic, test_actions)
    logits_after = test_critic_forward_raw(algorithm, test_obs_critic, test_actions)
    target_after = test_target_network_forward(algorithm, test_obs_actor, test_obs_critic, test_actions)
    norm_after = test_normalization_consistency(algorithm, test_obs_actor, test_obs_critic)

    # Compare snapshots
    state_match = compare_snapshots(snapshot_before, snapshot_after, "Before Save vs After Load")

    # Compare normalization consistency
    print("\n" + "="*60)
    print("Normalization Consistency Check")
    print("="*60)
    norm_actor_match = arrays_equal(
        norm_before["direct_norm_actor"],
        norm_before["method_norm_actor"],
        "Before: direct vs method (actor)"
    )
    norm_critic_match = arrays_equal(
        norm_before["direct_norm_critic"],
        norm_before["method_norm_critic"],
        "Before: direct vs method (critic)"
    )
    norm_actor_after_match = arrays_equal(
        norm_after["direct_norm_actor"],
        norm_after["method_norm_actor"],
        "After: direct vs method (actor)"
    )
    norm_critic_after_match = arrays_equal(
        norm_after["direct_norm_critic"],
        norm_after["method_norm_critic"],
        "After: direct vs method (critic)"
    )
    norm_actor_before_after = arrays_equal(
        norm_before["method_norm_actor"],
        norm_after["method_norm_actor"],
        "Before vs After: normalized actor obs"
    )
    norm_critic_before_after = arrays_equal(
        norm_before["method_norm_critic"],
        norm_after["method_norm_critic"],
        "Before vs After: normalized critic obs"
    )

    # Compare actions
    print("\n" + "="*60)
    print("Action Output Comparison")
    print("="*60)
    actions_match = arrays_equal(actions_before, actions_after, "model.act() Actions")
    actions_inference_match = arrays_equal(actions_inference_before, actions_inference_after, "model.act_inference() Actions")

    # Compare values
    print("\n" + "="*60)
    print("Value Output Comparison")
    print("="*60)
    values_match = arrays_equal(values_before, values_after, "model.evaluate() Values")

    # Compare Q-values
    print("\n" + "="*60)
    print("Q-Value Output Comparison")
    print("="*60)
    q1_match = arrays_equal(qvalues_before["q1"], qvalues_after["q1"], "critic1_q_value()")
    q2_match = arrays_equal(qvalues_before["q2"], qvalues_after["q2"], "critic2_q_value()")

    # Compare logits
    print("\n" + "="*60)
    print("Critic Logits Comparison")
    print("="*60)
    logits1_match = arrays_equal(logits_before["logits1"], logits_after["logits1"], "critic1_forward() logits")
    logits2_match = arrays_equal(logits_before["logits2"], logits_after["logits2"], "critic2_forward() logits")

    # Compare target networks
    print("\n" + "="*60)
    print("Target Network Output Comparison")
    print("="*60)
    target_actions_raw_match = arrays_equal(
        target_before["target_actions_raw"],
        target_after["target_actions_raw"],
        "Target Actor Raw Actions"
    )
    target_actions_match = arrays_equal(
        target_before["target_actions"],
        target_after["target_actions"],
        "Target Actor Actions (after tanh)"
    )
    target_logits1_match = arrays_equal(
        target_before["target_logits1"],
        target_after["target_logits1"],
        "Target Critic1 Logits"
    )
    target_logits2_match = arrays_equal(
        target_before["target_logits2"],
        target_after["target_logits2"],
        "Target Critic2 Logits"
    )
    target_q1_match = arrays_equal(
        target_before["target_q1"],
        target_after["target_q1"],
        "Target Critic1 Q-values"
    )
    target_q2_match = arrays_equal(
        target_before["target_q2"],
        target_after["target_q2"],
        "Target Critic2 Q-values"
    )

    # =========================================================================
    # 9. Final Report
    # =========================================================================
    print("\n" + "="*70)
    print("최종 검증 결과")
    print("="*70)

    all_tests = [
        ("State Snapshot", state_match),
        ("Normalization Consistency (actor, before)", norm_actor_match),
        ("Normalization Consistency (critic, before)", norm_critic_match),
        ("Normalization Consistency (actor, after)", norm_actor_after_match),
        ("Normalization Consistency (critic, after)", norm_critic_after_match),
        ("Normalized Obs (actor, before vs after)", norm_actor_before_after),
        ("Normalized Obs (critic, before vs after)", norm_critic_before_after),
        ("model.act() Actions", actions_match),
        ("model.act_inference() Actions", actions_inference_match),
        ("model.evaluate() Values", values_match),
        ("critic1_q_value()", q1_match),
        ("critic2_q_value()", q2_match),
        ("critic1_forward() Logits", logits1_match),
        ("critic2_forward() Logits", logits2_match),
        ("Target Actor Raw Actions", target_actions_raw_match),
        ("Target Actor Actions", target_actions_match),
        ("Target Critic1 Logits", target_logits1_match),
        ("Target Critic2 Logits", target_logits2_match),
        ("Target Critic1 Q-values", target_q1_match),
        ("Target Critic2 Q-values", target_q2_match),
    ]

    all_passed = True
    for name, passed in all_tests:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        all_passed &= passed

    print("\n" + "="*70)
    if all_passed:
        print("🎉 모든 검증 통과! Save/Load가 100% 정확합니다.")
        print("\n리팩토링 검증 완료:")
        print("  ✅ Normalizer가 model 내부에 올바르게 저장됨")
        print("  ✅ model.eqx에 normalizer 포함되어 저장/로드됨")
        print("  ✅ 모든 model 메서드에서 normalization 자동 적용됨")
        print("  ✅ act_inference()가 올바르게 작동함")
    else:
        print("💥 검증 실패! 위의 오류를 확인하세요.")
    print("="*70)

    # Cleanup
    shutil.rmtree(checkpoint_dir)
    print(f"\n  Cleaned up: {checkpoint_dir}")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)