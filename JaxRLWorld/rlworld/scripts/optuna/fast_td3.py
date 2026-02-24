import os
from typing import Optional

import optuna

os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'

custom_assets = os.path.abspath(os.path.join(os.path.dirname(__file__), 'assets'))
import genesis.utils.terrain

genesis.utils.misc.get_assets_dir = lambda: custom_assets
genesis.utils.terrain.get_assets_dir = lambda: custom_assets

from rlworld.rl.configs.algorithms import FastTD3Config
from rlworld.rl.configs import NewtonConfigsForRun
from rlworld.rl.runners import BaseRunner
from rlworld.rl.configs.presets.go1.newton.abdnet import get_config


def run_trial(trial: optuna.Trial, num_iterations: int = 5000) -> float:
    """Run single training trial with sampled hyperparameters."""

    # Sample hyperparameters
    batch_size = trial.suggest_categorical("batch_size", [8192, 16384, 32768, 65536])
    target_policy_noise = trial.suggest_float("target_policy_noise", 0.1, 0.4)
    target_noise_clip = trial.suggest_float("target_noise_clip", 0.3, 0.8)
    num_atoms = trial.suggest_categorical("num_atoms", [51, 101, 201])
    v_min = trial.suggest_float("v_min", -20.0, 0.0)
    v_max = trial.suggest_float("v_max", 50.0, 150.0)
    noise_min = trial.suggest_float("noise_min", 0.01, 0.1)
    noise_max = trial.suggest_float("noise_max", 0.2, 0.5)
    utd_ratio = trial.suggest_int("utd_ratio", 1, 16)

    # Constraint: noise_min < noise_max
    if noise_min >= noise_max:
        raise optuna.TrialPruned()

    # Get base config
    configs_dict = get_config()
    cfgs_for_run = NewtonConfigsForRun.from_dict(configs_dict)

    scale_param = 1.0
    cfgs_for_run.action.action_scale = cfgs_for_run.action.action_scale / scale_param
    cfgs_for_run.action.clip_actions = (-scale_param, scale_param)
    cfgs_for_run.runner.max_iterations = num_iterations
    cfgs_for_run.runner.run_name = f"FastTD3_optuna_trial{trial.number}"

    # Apply sampled hyperparameters
    fast_td3_config = FastTD3Config(
        actor_lr=1e-4,
        critic_lr=5e-4,
        gamma=0.99,
        tau=0.005,
        batch_size=batch_size,
        buffer_size=2_000_000,
        learning_starts=100,
        policy_delay=2,
        target_policy_noise=target_policy_noise,
        target_noise_clip=target_noise_clip,
        num_atoms=num_atoms,
        v_min=v_min,
        v_max=v_max,
        noise_min=noise_min,
        noise_max=noise_max,
        is_squashed=True,
        use_cdq=True,
        utd_ratio=utd_ratio,
    )
    cfgs_for_run.algorithm = fast_td3_config

    # Train
    runner = BaseRunner.create_with_env(cfgs_for_run, use_wandb=False)
    runner.learn(
        num_learning_iterations=num_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len
    )

    # Get final metric
    return_buffer = runner.reward_statistics.get_returns_buffer()
    if len(return_buffer) > 0:
        mean_return = sum(return_buffer[-100:]) / len(return_buffer[-100:])
    else:
        mean_return = float('-inf')

    runner.close()
    return mean_return


def main():
    study = optuna.create_study(
        study_name="fasttd3-hparam-tuning",
        storage="sqlite:///fasttd3_tuning.db",
        direction="maximize",
        load_if_exists=True,
    )

    n_trials = 50

    for i in range(n_trials):
        trial = study.ask()
        print(f"\n{'=' * 60}")
        print(f"Trial {trial.number} / {n_trials}")
        print(f"{'=' * 60}")

        try:
            value = run_trial(trial, num_iterations=2500)
            study.tell(trial, value)
            print(f"Trial {trial.number} finished: {value:.2f}")
        except optuna.TrialPruned:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            print(f"Trial {trial.number} pruned")
        except Exception as e:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            print(f"Trial {trial.number} failed: {e}")

    # Print results
    print(f"\n{'=' * 60}")
    print("Optimization finished")
    print(f"{'=' * 60}")
    print(f"Best value: {study.best_value:.2f}")
    print(f"Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()