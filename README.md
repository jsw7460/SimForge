# SimForge

A modular reinforcement learning framework for robot locomotion training with support for multiple physics simulators and robot platforms.

## Overview

SimForge (rlworld) provides a flexible and extensible framework for training RL policies using reinforcement learning. It supports multiple physics simulators and robot configurations with a clean, composable configuration system.

## Installation

### Prerequisites
- Python 3.11
- CUDA-compatible GPU

### Step-by-step Setup

1. **Create a conda environment**
```bash
   conda create -n rl python=3.11
   conda activate rl
```

2. **Install PyTorch**
```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

3. **Install Genesis**
```bash
   pip install genesis-world
```
   For more details, see [Genesis](https://github.com/Genesis-Embodied-AI/Genesis).

4. **Install Newton**

   Follow **Method 3: Manual Setup Using Pip in a Virtual Environment** from the [Newton documentation](https://github.com/newton-physics/newton).

5. **Install Mjlab**
```bash
   pip install mjlab
```
   For more details, see [Mjlab](https://github.com/mujocolab/mjlab).

6. **Install JAX with CUDA support**
```bash
   pip install --upgrade "jax[cuda13]"
```

7. **Install SimForge**
```bash
   cd SimForge
   pip install -e .
```

## Supported Robots

| Robot | Type | DOFs | Description |
|-------|------|------|-------------|
| **Go1** | Quadruped | 12 | Unitree Go1 quadruped robot |
| **Go2** | Quadruped | 12 | Unitree Go2 quadruped robot |
| **G1** | Humanoid | 12 | Unitree G1 humanoid robot (lower body) |

## Supported Simulators

| Simulator | Backend | Description |
|-----------|---------|-------------|
| **Genesis** | Custom | High-performance GPU-accelerated physics simulator |
| **Newton** | MuJoCo Warp | GPU-accelerated simulator with MuJoCo physics|

## Features

- **Modular Configuration System**: Dataclass-based configs for robots, observations, rewards, and environments
- **Multiple RL Algorithms**: PPO, SAC, TD3, TDMPC2
- **Various Neural Network Architectures**: MLP, GNN, Transformer, ABA, CRBA, Body Transformer
- **Domain Randomization**: Mass, friction, external force randomization
- **Curriculum Learning**: Progressive difficulty adjustment
- **WandB Integration**: Experiment tracking and logging

## Project Structure

```
rlworld/
├── assets/                     # Robot URDF models and terrain meshes
├── rl/
│   ├── algorithms/             # RL algorithms (PPO, SAC, TD3)
│   ├── configs/
│   │   ├── components/         # Reusable observation/reward components
│   │   ├── presets/            # Ready-to-use configurations
│   │   │   ├── go1/            # Go1 robot presets
│   │   │   ├── go2_flat/       # Go2 flat terrain presets
│   │   │   └── g1/             # G1 humanoid presets
│   │   └── robots/             # Robot-specific configurations
│   ├── envs/                   # Environment implementations
│   │   ├── mdp/                # MDP components (observations, rewards, terminations)
│   │   └── managers/           # Simulator-specific managers
│   ├── modules/                # Neural network architectures
│   │   └── architectures/      # MLP, GNN, Transformer, ABA, CRBA, etc.
│   └── runners/                # Training runners
└── scripts/                    # Training scripts
    ├── go1/                    # Go1 training examples
    ├── go2/                    # Go2 training examples
    └── g1/                     # G1 training examples
```

## Quick Start

### Training Go1 with MLP Policy (Genesis)

```python
from rlworld.rl.configs import GenesisConfigsForRun
from rlworld.rl.runners import BaseRunner
from rlworld.rl.configs.presets.go1.genesis.mlp import get_config

def main():
    # Get preset configuration
    configs_dict = get_config()

    # Create configs and runner
    cfgs_for_run = GenesisConfigsForRun.from_dict_with_overrides(configs_dict)
    runner = BaseRunner.create_with_env(cfgs_for_run)

    # Start training
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len
    )

if __name__ == "__main__":
    main()
```

### Training G1 Humanoid

```python
from rlworld.rl.configs import GenesisConfigsForRun
from rlworld.rl.runners import OnPolicyRunner
from rlworld.rl.configs.presets.g1.genesis.mlp import get_config

def main():
    configs_dict = get_config()
    cfgs_for_run = GenesisConfigsForRun.from_dict_with_overrides(configs_dict)

    # Customize network architecture
    cfgs_for_run.nn.policy["actor_kwargs"].update({
        "hidden_dims": [700, 512, 512]
    })

    runner = OnPolicyRunner.create_with_env(cfgs_for_run)
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len
    )

if __name__ == "__main__":
    main()
```

## Configuration System

### Robot Configuration

Each robot has a dedicated configuration class:

```python
@dataclass
class Go1Config(RobotConfig):
    name: str = "go1"
    urdf_path: str = "./rlworld/assets/go1_model_clean/urdf/go1_simplified_stl.urdf"
    base_init_height: float = 0.41
    default_joint_angles: Dict[str, float] = ...
    p_gains: Dict[str, float] = ...
    d_gains: Dict[str, float] = ...
```

### Environment Configuration

Configurations compose multiple components:

```python
@dataclass
class Go1FlatGenesisConfig:
    robot: Go1Config = field(default_factory=Go1Config)
    observations: LocomotionObservations = ...
    tracking_rewards: TrackingRewards = ...
    regularization_rewards: RegularizationRewards = ...
    num_envs: int = 4096
    episode_length_s: float = 20.0
    ...
```

### Reward Components

Rewards are modularly defined and combined:

- **TrackingRewards**: Linear/angular velocity tracking
- **RegularizationRewards**: Action smoothness, base height, default pose
- **ContactRewards**: Feet height, air time, impact forces
- **PostureRewards**: Body orientation, joint limits

## Running Training Scripts

```bash
# Go1 quadruped
python -m rlworld.scripts.go1.genesis.mlp

# Go2 quadruped
python -m rlworld.scripts.go2.genesis.mlp

# G1 humanoid
python -m rlworld.scripts.g1.mlp
```

## Available Presets

### Go1 (Quadruped)
- `rlworld.rl.configs.presets.go1.genesis.mlp` - MLP policy with Genesis
- `rlworld.rl.configs.presets.go1.newton.mlp` - MLP policy with Newton

### Go2 (Quadruped)
- `rlworld.rl.configs.presets.go2_flat.genesis.mlp` - Flat terrain MLP
- `rlworld.rl.configs.presets.go2_flat.newton.mlp` - Flat terrain with Newton

### G1 (Humanoid)
- `rlworld.rl.configs.presets.g1.genesis.mlp` - MLP policy for humanoid locomotion

## Key Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_envs` | Number of parallel environments | 4096 |
| `episode_length_s` | Episode duration in seconds | 20.0 |
| `decimation` | Action repeat count | 4 |
| `action_scale` | Action scaling factor | 0.4 |
| `max_iterations` | Training iterations | 6000 |
| `actor_hidden_dims` | Actor network architecture | [256, 256, 256] |

## Logging

Training metrics are logged to WandB by default. Configure in the runner settings:

```python
"logger": "wandb",
"wandb_project": "RLArchitecture",
```

## Requirements

- Python 3.10+
- PyTorch
- Genesis simulator
- Newton simulator
- WandB (optional, for logging)
