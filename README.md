# SimForge

A modular reinforcement learning framework for robot locomotion, built for training across multiple physics simulators simultaneously.

## Overview

SimForge provides a unified environment interface (`World`) that abstracts over Genesis, Newton, and MuJoCo simulators. A single policy can be trained on rollouts from multiple simulators at once ("simulator randomization"), then evaluated on any of them — including cross-simulator transfer.

Key design choices:
- **Manager-based environment**: Observations, rewards, actions, commands, events, contacts, and terminations are modular managers that plug into any simulator backend.
- **Common observation functions**: Simulator-agnostic obs terms via the `RobotData` protocol — the same `dof_pos`, `base_ang_vel`, `projected_gravity` functions work on Genesis, Newton, and MuJoCo.
- **Multi-sim training**: `MultiSimWorld` wraps multiple simulator environments, handles joint ordering permutations automatically, and presents a single vectorized interface to the algorithm.
- **Config presets**: Dataclass-based configs per robot/simulator/architecture, composable and overridable from CLI.

## Installation

### Prerequisites
- Python 3.11+
- CUDA-compatible GPU

### Setup

```bash
conda create -n rl python=3.11
conda activate rl

# PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# JAX with CUDA
pip install --upgrade "jax[cuda13]"

# Simulators (install the ones you need)
pip install genesis-world                          # Genesis
pip install mjlab                                  # MuJoCo (mjlab wrapper)
# Newton: follow https://github.com/newton-physics/newton

# SimForge
cd SimForge
pip install -e .
```

## Supported Platforms

### Robots

| Robot | Type | DOFs | Simulators |
|-------|------|------|------------|
| Go1 | Quadruped | 12 | Genesis, Newton, MuJoCo |
| Go2 | Quadruped | 12 | Genesis, Newton, MuJoCo |
| G1 | Humanoid | 12 / 29 | Genesis, Newton, MuJoCo |

### Simulators

| Simulator | Backend | GPU | Multi-env |
|-----------|---------|-----|-----------|
| Genesis | JAX-native | Yes | Yes (thousands) |
| Newton | MuJoCo-Warp | Yes | Yes (thousands) |
| MuJoCo | mjlab (MuJoCo-Warp) | Yes | Yes (thousands) |

### Algorithms

| Algorithm | Type | Runner |
|-----------|------|--------|
| PPO | On-policy | `OnPolicyRunner` |
| TD3 | Off-policy | `OffPolicyRunner` |
| SAC | Off-policy | `OffPolicyRunner` |
| FastTD3 | Off-policy | `OffPolicyRunner` |
| TDMPC2 | Model-based | `ModelBasedRunner` |
| SimMPC | Sim-based MPC | `SimMPCRunner` |

### Neural Network Architectures

MLP, ABA (Articulated Body Algorithm), ABA+GNN, Transformer, Body Transformer, CRBA, Rodrigues, DR3.

## Quick Start

### Training

```python
from rlworld.rl.configs.presets.go2_flat.genesis.mlp import get_config
from rlworld.rl.runners import BaseRunner

cfg = get_config()
runner = BaseRunner.create_with_env(cfg.with_cli_overrides())
runner.learn(num_learning_iterations=cfg.runner.max_iterations)
```

```bash
python -m rlworld.scripts.go2.genesis.mlp
python -m rlworld.scripts.g1_29dof.genesis.mlp
```

### Multi-Simulator Training

Train on Genesis + Newton simultaneously:

```python
from rlworld.rl.configs.presets.go2_flat.genesis.mlp import get_config as genesis_cfg
from rlworld.rl.configs.presets.go2_flat.newton.mlp import get_config as newton_cfg
from rlworld.rl.envs.multi_sim_world import MultiSimWorld
from rlworld.rl.runners import BaseRunner, OnPolicyRunner

g_env = BaseRunner._create_env_from_config(genesis_cfg())
n_env = BaseRunner._create_env_from_config(newton_cfg())

multi_env = MultiSimWorld([g_env, n_env])
runner = OnPolicyRunner(env=multi_env, cfgs=genesis_cfg())
runner.learn(num_learning_iterations=6000)
```

```bash
python -m rlworld.scripts.go2.multi_sim.ppo_mlp
```

### Evaluation

#### Interactive viewer (real-time, default)

```bash
python -m rlworld.scripts.evaluation.eval_cross_sim \
    --policy_path outputs/models/.../checkpoint_latest/ \
    --eval_sim newton
```

Opens a Viser web viewer with Play/Pause, Speed (1/32x — 8x), Step, and Reset controls. No SSH tunnel needed — uses a share URL.

#### Batch evaluation

```bash
python -m rlworld.scripts.evaluation.eval_cross_sim \
    --policy_path outputs/models/.../checkpoint_latest/ \
    --eval_sim genesis --eval --record_video
```

#### Cross-simulator evaluation

A checkpoint trained on Genesis can be evaluated on Newton or MuJoCo:

```python
from rlworld.rl.evals import PolicyEvaluator

evaluator = PolicyEvaluator(
    policy_path="outputs/models/.../checkpoint_latest/",
    eval_target="newton",
)
evaluator.play()       # Interactive viewer
# evaluator.evaluate() # Batch eval with statistics
```

## Project Structure

```
JaxRLWorld/rlworld/
├── assets/                          # Robot URDFs and meshes
├── rl/
│   ├── algorithms/                  # PPO, TD3, SAC, FastTD3, TDMPC2, SimMPC
│   ├── configs/
│   │   ├── components/              # Reusable observation/reward components
│   │   │   └── observations/        # Per-simulator LocomotionObservations
│   │   ├── presets/                 # Ready-to-use configs
│   │   │   ├── go1/                 #   genesis/ newton/ mujoco/
│   │   │   ├── go2_flat/            #   genesis/ newton/ mujoco/
│   │   │   ├── g1_12dof/            #   genesis/ newton/
│   │   │   └── g1_29dof/            #   genesis/ newton/ mujoco/
│   │   └── robots/                  # Go1, Go2, G1 robot configs
│   ├── envs/
│   │   ├── world.py                 # World ABC — base for all envs
│   │   ├── multi_sim_world.py       # MultiSimWorld + joint permutation
│   │   ├── genesis/                 # Genesis env implementation
│   │   ├── newton/                  # Newton env implementation
│   │   ├── mujoco/                  # MuJoCo (mjlab) env implementation
│   │   ├── managers/                # Modular managers per simulator
│   │   │   ├── common/              #   observation, action, reward, command, ...
│   │   │   ├── genesis/             #   scene, contact, visualization
│   │   │   ├── newton/              #   scene, contact, visualization
│   │   │   └── mujoco/              #   scene, contact, visualization
│   │   └── mdp/                     # MDP components
│   │       ├── observations/        #   common/ genesis/ newton/ mujoco/
│   │       ├── rewards/             #   common/ genesis/ newton/ mujoco/
│   │       ├── terminations/
│   │       └── commands/
│   ├── evals/                       # PolicyEvaluator + SimInitializers
│   ├── modules/                     # Neural networks
│   │   ├── architectures/           #   MLP, ABA, GNN, Transformer, ...
│   │   ├── policies/                #   PPO, SAC, TD3, TDMPC2 actor-critics
│   │   └── dynamics/                #   World models
│   ├── runners/                     # Training loops
│   │   ├── on_policy_runner.py      #   PPO
│   │   ├── off_policy_runner.py     #   TD3, SAC, FastTD3
│   │   └── model_based_runner.py    #   TDMPC2
│   ├── storages/                    # Rollout buffer, replay buffer
│   └── vis/                         # Visualization
│       └── viser/                   #   PlayViewer, ViserScene, bridges, overlays
└── scripts/                         # Training and evaluation scripts
    ├── go1/                         #   genesis/ newton/ mujoco/
    ├── go2/                         #   genesis/ newton/ mujoco/ multi_sim/
    ├── g1_29dof/                    #   genesis/ newton/ mujoco/ multi_sim/
    └── evaluation/                  #   eval_cross_sim, eval_genesis, eval_newton
```

## Configuration

Configs are Python dataclasses, composable and overridable:

```python
from rlworld.rl.configs.presets.go2_flat.genesis.mlp import get_config

cfg = get_config()
cfg.env.num_envs = 2048
cfg.algorithm.learning_rate = 3e-4
cfg.runner.max_iterations = 10000
```

CLI overrides:

```bash
python script.py env.num_envs=2048 algorithm.learning_rate=3e-4
```

### Config Presets

Each preset is organized as `presets/{robot}/{simulator}/{architecture}.py` and includes env, scene, observation, reward, action, command, event, nn, algorithm, and runner configs.

## Visualization

All viewers use [Viser](https://viser.studio/) — a web-based 3D viewer that works over SSH without port forwarding (share URL).

- **PlayViewer** (`evaluator.play()`): Real-time pacing with budget accumulator. Play/Pause, Speed (1/32x — 8x), Single Step, Reset. Command velocity arrows, reward plots.
- **Passive viewer** (`--eval` mode): Runs during batch evaluation at full speed. Enabled via `viewer_type: "viser"` in visualization config.

## Logging

Training metrics are logged to [Weights & Biases](https://wandb.ai/):

```python
runner = OnPolicyRunner(env=env, cfgs=cfg, use_wandb=True)
```

Checkpoints are saved to `outputs/models/{date}/{time}/` with full config metadata for reproducibility and cross-sim evaluation.
