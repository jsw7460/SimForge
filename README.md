# SimForge

A modular reinforcement learning framework for legged-robot locomotion that runs the **same** policy / observation / reward stack across **Newton**, **Genesis**, and **mjlab (MuJoCo)** physics backends. Train on one simulator, evaluate on another, or train on several at once and let the policy see the union of their dynamics.

## Overview

SimForge separates "what the task is" from "which simulator runs it". A task is defined once as a tree of dataclass configs (env, scene, observation, reward, action, command, event, algorithm, runner) and a small per-simulator builder module supplies the few sim-native pieces (entity construction, contact sensors, terrain). Everything downstream — observation functions, reward functions, event terms, runners, neural networks — is sim-agnostic and reads / writes the simulator only through narrow protocol interfaces.

The result is that the **bit-identical reward function** defines `feet_clearance` for all three backends, the **same `dof_pos`/`base_ang_vel`/`projected_gravity`** observation feeds the policy regardless of which sim the rollout came from, and a checkpoint trained under Newton can be loaded into a Genesis or mjlab environment without any conversion step.

## Architecture

```
                  ┌──────────────────────────────────────────┐
                  │  Preset (Go2FlatConfig / G1FlatConfig)   │
                  │   ─ env, scene, obs, reward, ...         │
                  │   ─ build()  →  ConfigsForRun            │
                  └──────────────────────┬───────────────────┘
                                         │
              ┌──────────────────────────┴──────────────────────────┐
              │                                                     │
              ▼                                                     ▼
  ┌──────────────────────┐                              ┌──────────────────────┐
  │  _newton_builders.py │                              │  _genesis_builders   │
  │  _genesis_builders   │  ── per-sim pieces only ──▶  │  _mujoco_builders    │
  │  _mujoco_builders    │                              │  (entities, sensors, │
  └──────────────────────┘                              │   gait, terrain...)  │
                                                        └──────────────────────┘
                                         │
                                         ▼
            ┌────────────────────────────────────────────────────┐
            │            World (NewtonEnv / GenesisEnv /         │
            │                    MujocoEnv)                      │
            │  ┌──────────────────────────────────────────────┐  │
            │  │  Managers (Scene, Action, Observation,       │  │
            │  │   Reward, Command, Event, Contact, ...)      │  │
            │  └──────────────────────────────────────────────┘  │
            │  ┌──────────────────────────────────────────────┐  │
            │  │  RobotData protocol  +  RobotStateWriter     │  │
            │  │  (reads / writes robot state through one     │  │
            │  │   stable interface, regardless of backend)   │  │
            │  └──────────────────────────────────────────────┘  │
            └────────────────────────┬───────────────────────────┘
                                     │
                                     ▼
            ┌────────────────────────────────────────────────────┐
            │   mdp/{observations,rewards,events,...}/common/    │
            │   ─ sim-agnostic functions, called identically     │
            │     by every backend                               │
            └────────────────────────────────────────────────────┘
                                     │
                                     ▼
            ┌────────────────────────────────────────────────────┐
            │   Runner (PPO / SAC / TD3 / FastTD3 / TDMPC2 /     │
            │           SimMPC) ─ no simulator-specific code     │
            └────────────────────────────────────────────────────┘
```

Key design decisions:

- **Single preset, three backends.** `Go2FlatConfig(sim_type="newton").build()` and `Go2FlatConfig(sim_type="genesis").build()` produce two `ConfigsForRun` objects that share every cross-sim field (reward weights, command ranges, network sizes, etc.) and only diverge in the small set of bits that *must* be sim-native (entity URDF/MJCF, contact sensors, solver tolerances).
- **`RobotData` protocol.** A read-only `Protocol` (`rl/envs/robot_data.py`) declares a small set of properties — `joint_pos`, `joint_vel`, `root_link_pos_w`, `body_pos_w(names)`, `site_pos_w(names)`, `angular_momentum_w(...)`, etc. Each backend supplies a thin wrapper (`NewtonRobotData`, `GenesisRobotData`, `MujocoRobotData`) that satisfies the protocol structurally. Reward and observation code reads `env.get_robot_data().joint_pos` and never knows which simulator it is on.
- **`RobotStateWriter`.** Mutation API (`set_dof_positions`, `set_root_state`, `eval_fk`) lives in a separate writer class (currently Newton-only — Genesis and mjlab event terms write through their native APIs and gain a writer when reset/event terms are unified).
- **Manager-based environment.** The same `World` ABC is shared by all backends. Observations, rewards, actions, commands, events, contacts, and terminations are independent managers; the env just orchestrates them.
- **`MultiSimWorld`.** Stacks multiple sim envs into a single vectorized interface for the algorithm. Joint-order permutation (canonical ↔ per-sim) is built automatically from the action manager so the policy sees a stable joint layout even when each sim's URDF parsed joints in a different order.
- **Sim-agnostic reward/observation library.** `mdp/rewards/common/reward_terms.py` and `mdp/observations/common/` hold the canonical implementations. Per-sim modules (`mdp/rewards/newton/`, etc.) exist mainly as thin redirect wrappers that preserve historical names for older presets.

## Installation

### Prerequisites
- Python 3.11+
- CUDA-compatible GPU
- Linux (recommended; macOS works for development but the simulators target Linux)

### Setup

```bash
conda create -n rl python=3.11
conda activate rl

# PyTorch (pick the index URL for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# JAX with CUDA (optional — only some algorithms use it)
pip install --upgrade "jax[cuda13]"

# Simulators (install whichever you intend to use)
pip install genesis-world                          # Genesis
pip install mjlab                                  # mjlab (MuJoCo wrapper)
# Newton: see https://github.com/newton-physics/newton

# SimForge
cd SimForge
pip install -e .
```

## Supported Platforms

### Robots

| Robot | Type      | DOF | Backends                |
|-------|-----------|-----|-------------------------|
| Go2   | Quadruped | 12  | Newton · Genesis · mjlab |
| G1    | Humanoid  | 29  | Newton · Genesis · mjlab |

(Older `go1` / `g1_12dof` presets were removed during the unification cleanup.)

### Backends

| Backend | Substrate           | GPU | Multi-env       |
|---------|---------------------|-----|-----------------|
| Newton  | MuJoCo-Warp + warp  | yes | thousands       |
| Genesis | Taichi              | yes | thousands       |
| mjlab   | MuJoCo-Warp wrapper | yes | thousands       |

### Algorithms

| Algorithm | Family       | Runner                      |
|-----------|--------------|-----------------------------|
| PPO       | on-policy    | `OnPolicyRunner`            |
| SAC       | off-policy   | `OffPolicyRunner`           |
| TD3       | off-policy   | `OffPolicyRunner`           |
| FastTD3   | off-policy   | `OffPolicyRunner`           |
| TDMPC2    | model-based  | `ModelBasedRunner`          |
| SimMPC    | sim-based MPC| `SimMPCRunner`              |

### Network Architectures

MLP, ABA (Articulated Body Algorithm), ABA + GNN, Transformer, Body Transformer, CRBA, Rodrigues, DR3.

## Quick Start

### Training a single backend

```python
from rlworld.rl.configs.presets.go2_flat.mlp import get_config
from rlworld.rl.runners import BaseRunner

cfg = get_config(sim="newton").with_cli_overrides()
runner = BaseRunner.create_with_env(cfg)
runner.learn(num_learning_iterations=cfg.runner.max_iterations)
```

The `sim=` argument is the only thing you change to switch backends — `"newton"`, `"genesis"`, and `"mujoco"` all return a `ConfigsForRun` of the corresponding sim-specific subtype (`NewtonConfigsForRun`, `GenesisConfigsForRun`, `MujocoConfigsForRun`).

Pre-baked launchers live under `rlworld/scripts/`:

```bash
python rlworld/scripts/go2/newton/mlp.py
python rlworld/scripts/go2/genesis/mlp.py
python rlworld/scripts/go2/mujoco/ppo_mlp.py

python rlworld/scripts/g1_29dof/newton/mlp.py
python rlworld/scripts/g1_29dof/genesis/mlp.py
python rlworld/scripts/g1_29dof/mujoco/ppo_mlp.py
```

CLI overrides go through the standard dotted-key form:

```bash
python rlworld/scripts/go2/newton/mlp.py env.num_envs=2048 algorithm.actor_lr=3e-4
```

### Multi-simulator training

`MultiSimWorld` runs Newton + Genesis (and / or mjlab) side by side, with one policy seeing batches from both. The script handles per-sim env counts and joint permutation:

```bash
python rlworld/scripts/go2/multi_sim/ppo_mlp.py \
    --genesis_num_envs 2048 --newton_num_envs 2048
```

### Evaluation

Each backend has a thin entry script under `rlworld/scripts/evaluation/`. They all share the same `PolicyEvaluator` underneath; the only differences are the viser viewer setup and any sim-specific overrides:

```bash
python rlworld/scripts/evaluation/eval_newton.py
python rlworld/scripts/evaluation/eval_genesis.py
python rlworld/scripts/evaluation/eval_mujoco.py
```

Edit `policy_path` (or `wandb_run_path`) inside the script to point at the checkpoint you want to visualize. By default the script opens an interactive viser viewer; pass `--eval` to run a batch evaluation with statistics instead.

For cross-simulator evaluation (train on one backend, evaluate on another), use `eval_target`:

```python
from rlworld.rl.evals import PolicyEvaluator

evaluator = PolicyEvaluator(
    policy_path="outputs/models/.../checkpoint_latest/",
    eval_target="newton",   # or "genesis" / "mujoco"
)
evaluator.play()       # interactive viser viewer
# evaluator.evaluate() # batch eval with statistics
```

The evaluator automatically resolves the right preset class from the checkpoint metadata (`preset_class_name` + `preset_kwargs`) so you do not need to remember which config the policy was trained with.

## Project Structure

```
JaxRLWorld/rlworld/
├── assets/                              # Robot URDFs, MJCFs, meshes
├── rl/
│   ├── algorithms/                      # PPO, SAC, TD3, FastTD3, TDMPC2, SimMPC, ...
│   ├── configs/
│   │   ├── presets/
│   │   │   ├── go2_flat/                # Unified Go2 preset
│   │   │   │   ├── base.py              #   Go2FlatConfig dataclass
│   │   │   │   ├── mlp.py               #   get_config(sim=...) entry point
│   │   │   │   ├── _newton_builders.py  #   per-sim scene / event / reward
│   │   │   │   ├── _genesis_builders.py #     builders that the base class
│   │   │   │   ├── _mujoco_builders.py  #     dispatches to at build()
│   │   │   │   ├── newton/              #   Variants: gait_conditioned, ...
│   │   │   │   ├── genesis/             #     (one file per variant)
│   │   │   │   └── mujoco/
│   │   │   └── g1_29dof/                # Same layout for G1 29-DOF
│   │   └── robots/                      # Go2Config, G1Config, kinematic tree
│   ├── envs/
│   │   ├── world.py                     # World ABC + manager orchestration
│   │   ├── robot_data.py                # RobotData protocol (read API)
│   │   ├── multi_sim_world.py           # MultiSimWorld + joint permutation
│   │   ├── newton/
│   │   │   ├── newton_env.py            # NewtonEnv (subclass of World)
│   │   │   ├── robot_data.py            # NewtonRobotData (read impl)
│   │   │   └── robot_state_writer.py    # NewtonRobotStateWriter (write API)
│   │   ├── genesis/
│   │   │   ├── genesis_env.py
│   │   │   └── robot_data.py            # GenesisRobotData
│   │   ├── mujoco/
│   │   │   ├── mjlab_env.py
│   │   │   └── robot_data.py            # MujocoRobotData
│   │   ├── managers/
│   │   │   ├── common/                  # Cross-sim manager bases
│   │   │   │   ├── action.py            #   BaseActionManager
│   │   │   │   ├── command.py           #   CommandManager
│   │   │   │   ├── command_term.py      #   VelocityCommandTerm, ...
│   │   │   │   ├── contact.py           #   BaseContactManager (named groups)
│   │   │   │   ├── event.py             #   EventManager
│   │   │   │   ├── observation.py       #   ObservationManager
│   │   │   │   ├── reward.py            #   RewardManager
│   │   │   │   ├── termination.py       #   TerminationManager
│   │   │   │   ├── gait.py              #   GaitManager
│   │   │   │   ├── scene_helpers.py     #   build_kinematic_trees, ...
│   │   │   │   └── scene_protocol.py    #   SceneManagerProtocol
│   │   │   ├── newton/                  # Sim-specific subclasses
│   │   │   ├── genesis/                 # (mostly scene + contact + visualization)
│   │   │   └── mujoco/
│   │   └── mdp/
│   │       ├── observations/
│   │       │   ├── common/              # Sim-agnostic obs functions
│   │       │   ├── newton/ genesis/ mujoco/   # Thin redirects
│   │       ├── rewards/
│   │       │   ├── common/reward_terms.py     # Canonical reward library
│   │       │   ├── newton/mjlab_rewards.py    # Thin redirect wrappers
│   │       │   ├── genesis/mjlab_rewards.py
│   │       │   └── mujoco/reward_terms.py
│   │       ├── events/                  # Reset / startup / interval terms
│   │       ├── reset/                   # Per-sim reset helpers
│   │       ├── terminations/
│   │       └── commands/
│   ├── evals/
│   │   ├── evaluator.py                 # PolicyEvaluator
│   │   └── sim_initializers/            # Per-sim eval setup (strategy pattern)
│   ├── modules/
│   │   ├── architectures/               # MLP, ABA, GNN, Transformer, ...
│   │   ├── policies/                    # PPO/SAC/TD3/TDMPC2 actor-critics
│   │   └── dynamics/                    # World models
│   ├── runners/                         # On-policy, off-policy, model-based, ...
│   ├── storages/                        # Rollout / replay buffers
│   ├── utils/                           # Quaternions, string matching, ...
│   └── vis/viser/                       # Viser viewer + sim bridges + overlays
└── scripts/
    ├── go2/{newton,genesis,mujoco,multi_sim}/
    ├── g1_29dof/{newton,genesis,mujoco,multi_sim,playground}/
    ├── evaluation/                      # eval_newton, eval_genesis, eval_mujoco, eval_cross_sim
    ├── benchmark/
    ├── gymnasium/
    └── maniskill/
```

## Configuration

Configs are dataclasses, composable from Python or overridable from CLI:

```python
from rlworld.rl.configs.presets.go2_flat.mlp import get_config

cfg = get_config(sim="newton")
cfg.env.num_envs = 2048
cfg.algorithm.actor_lr = 3e-4
cfg.runner.max_iterations = 10000
```

CLI overrides use dotted keys:

```bash
python rlworld/scripts/go2/newton/mlp.py env.num_envs=2048 algorithm.actor_lr=3e-4
```

A preset's `build()` method stamps `preset_module`, `preset_class_name`, and `preset_kwargs` onto the resulting `ConfigsForRun`. At checkpoint reload time the evaluator uses these to re-instantiate the exact same preset, so non-serializable fields (mjlab `EntityCfg`, Newton solver options, etc.) are reconstructed from code rather than YAML.

## Visualization

All viewers use [Viser](https://viser.studio/) — a web-based 3D viewer that works over SSH without port forwarding (it can serve a public share URL).

- **PlayViewer** (`evaluator.play()`): real-time pacing with budget accumulator. Play/Pause, Speed (1/32× — 8×), Single Step, Reset. Command-velocity arrows, reward plots, and HUD overlays.
- **Passive viewer** (`evaluator.evaluate()` with `viewer_type="viser"`): runs during batch evaluation at full speed.

## Logging

Training metrics are logged to [Weights & Biases](https://wandb.ai/) when `use_wandb=True`:

```python
runner = OnPolicyRunner(env=env, cfgs=cfg, use_wandb=True)
```

Checkpoints are saved to `outputs/models/{date}/{time}/` together with the full config metadata so they can be reloaded for cross-sim evaluation.
