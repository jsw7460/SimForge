# SimForge

A JAX-based reinforcement learning framework for legged-robot locomotion,
with first-class support for training and evaluating **one policy across
three simulators** — [Genesis][genesis], [Newton][newton], and MuJoCo
(via [mjlab][mjlab]) — using a single sim-agnostic API. The framework
itself is `rlworld/` inside [`JaxRLWorld/`](JaxRLWorld); `SimForge/` is the
umbrella repo that pins specific simulator versions as git submodules so
external users can clone a single, reproducible stack.

<p align="center">
  <img src="docs/demo.gif" alt="A single PPO policy trained in Newton, evaluated in Genesis, Newton, and MuJoCo" width="900"/>
</p>

<p align="center">
  <em>One PPO policy trained on <code>go2/newton/gait_conditioned</code>, evaluated across all three simulators.</em>
</p>

## Highlights

- **Sim-to-sim out of the box.** A policy trained in one simulator can be
  evaluated in the other two without code changes. The same task config
  drives all three backends.
- **5 task presets × 3 simulators = 15 ready combinations** covering
  Unitree G1 (29-DOF humanoid), Unitree Go2 (quadruped), and the
  Booster T1 humanoid.
- **PPO is the default for all locomotion tasks** across the three
  simulators. **SAC, TD3, FastTD3, and TDMPC2** are validated on a
  small subset of a Gymnasium-based benchmark suite (see
  `JaxRLWorld/rlworld/scripts/benchmark/`), with FastTD3 additionally
  validated on [mujoco_playground][mjpg].
- **2 actor/critic architectures**: MLP and Space-Time Transformer.
- **Domain randomization, motion tracking, and viser-based 3-D
  visualization** are wired up across all simulators.

## Supported tasks

The table below lists (task, simulator) combinations that have been
trained and evaluated end-to-end with PPO.

|              | Robot        | Genesis | Newton | MuJoCo |
| ------------ | ------------ | :-----: | :----: | :----: |
| `g1_29dof`   | Unitree G1   | ✓       | ✓      | ✓      |
| `g1_tracking`| Unitree G1   | ✓       | ✓      | ✓      |
| `go2_flat`   | Unitree Go2  | ✓       | ✓      | ✓      |
| `t1_getup`   | Booster T1   | ✓       | ✓      | ✓      |
| `t1_tracking`| Booster T1   | ✓       | ✓      | ✓      |

## Installation

JaxRLWorld pins specific versions of [Genesis][genesis], [Newton][newton],
and [mjlab][mjlab] as git submodules under this `SimForge/` repo.

### 1. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/jsw7460/SimForge.git
cd SimForge
# or, if already cloned: git submodule update --init
```

### 2. Create a conda env

Python 3.10–3.12 is required.

```bash
conda create -n jrw python=3.11 -y
conda activate jrw
```

Any other env manager (`venv`, `uv`, `pyenv`) works too — just make
sure you are running inside a clean, isolated Python and that the
later steps install into that same env.

### 3. Install the simulators (editable, from submodules)

Each simulator has its own install notes — consult its README for CUDA
and system prerequisites. Typically:

```bash
pip install -e Mjlab/
pip install -e Newton/
pip install -e Genesis/
```

### 4. Install JaxRLWorld and JAX-CUDA

```bash
pip install -e "JaxRLWorld/[all]"
pip install -U "jax[cuda12]"   # match your system CUDA
```

> CUDA versions across JAX, Genesis, and Newton's [Warp][warp] backend
> must be mutually compatible — consult each simulator's docs.

## Quickstart

Train PPO on Go2 gait-conditioned locomotion in Newton:

```bash
python JaxRLWorld/rlworld/scripts/go2/newton/gait_conditioned.py
```

The same task in Genesis or MuJoCo:

```bash
python JaxRLWorld/rlworld/scripts/go2/genesis/gait_conditioned.py
python JaxRLWorld/rlworld/scripts/go2/mujoco/gait_conditioned.py
```

## Cross-sim evaluation

`eval_cross_sim.py` is the single entry point for evaluating any
checkpoint on any simulator. The robot, observation, algorithm, and
network configs are auto-detected from the checkpoint, so you only
specify which simulator to roll out on. Without `--eval`, the script
launches an interactive viser-based viewer; with `--eval`, it runs
batched statistics.

Training writes checkpoints to
`./outputs/models/<date>/<time>/checkpoint_latest/` by default. Pass
that directory to `--policy_path`:

```bash
python JaxRLWorld/rlworld/scripts/evaluation/eval_cross_sim.py \
    --policy_path outputs/models/<date>/<time>/checkpoint_latest/ \
    --eval_sim mujoco
```

To pull a checkpoint from W&B instead, set `--policy_path None` and
provide `--wandb_run_path`:

```bash
python JaxRLWorld/rlworld/scripts/evaluation/eval_cross_sim.py \
    --policy_path None \
    --wandb_run_path <entity>/<task>/<run-id> \
    --eval_sim mujoco \
    --eval
```

The W&B path is only resolvable if the training run uploaded its
checkpoint. Enable that either in your runner config or as a CLI
override:

```bash
python JaxRLWorld/rlworld/scripts/g1_29dof/genesis/mlp.py \
    runner.upload_checkpoint=True
```

`--eval_sim` accepts `genesis`, `newton`, or `mujoco`.

## Architecture

JaxRLWorld separates "what the task is" from "which simulator runs it".
A task is defined once as a tree of dataclass configs (env, scene,
observation, reward, action, command, event, algorithm, runner) and a
small per-simulator builder module supplies the few sim-native pieces
(entity construction, contact sensors, terrain). Everything downstream
— observation functions, reward functions, event terms, runners, neural
networks — is sim-agnostic and reads/writes the simulator only through
narrow protocol interfaces.

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
            │   Runner (PPO / SAC / TD3 / FastTD3 / TDMPC2) ─    │
            │   no simulator-specific code                       │
            └────────────────────────────────────────────────────┘
```

Key design decisions:

- **Single preset, three backends.** `Go2FlatConfig(sim_type="newton").build()`
  and `Go2FlatConfig(sim_type="genesis").build()` produce two
  `ConfigsForRun` objects that share every cross-sim field (reward
  weights, command ranges, network sizes, etc.) and only diverge in the
  small set of bits that *must* be sim-native (entity URDF/MJCF, contact
  sensors, solver tolerances).
- **`RobotData` protocol.** A read-only `Protocol`
  (`rl/envs/robot_data.py`) declares a small set of properties —
  `joint_pos`, `joint_vel`, `root_link_pos_w`, `body_pos_w(names)`,
  `site_pos_w(names)`, `angular_momentum_w(...)`, etc. Each backend
  supplies a thin wrapper (`NewtonRobotData`, `GenesisRobotData`,
  `MujocoRobotData`) that satisfies the protocol structurally. Reward
  and observation code reads `env.get_robot_data().joint_pos` and never
  knows which simulator it is on.
- **`RobotStateWriter`.** Mutation API (`set_dof_positions`,
  `set_root_state`, `eval_fk`) lives in a separate writer class.
- **Manager-based environment.** The same `World` ABC is shared by all
  backends. Observations, rewards, actions, commands, events, contacts,
  and terminations are independent managers; the env just orchestrates
  them.
- **`MultiSimWorld`.** Stacks multiple sim envs into a single vectorized
  interface for the algorithm. Joint-order permutation (canonical ↔
  per-sim) is built automatically from the action manager so the policy
  sees a stable joint layout even when each sim's URDF parses joints in
  a different order.
- **Sim-agnostic reward/observation library.**
  `mdp/rewards/common/reward_terms.py` and `mdp/observations/common/`
  hold the canonical implementations. Per-sim modules
  (`mdp/rewards/newton/`, etc.) exist mainly as thin redirect wrappers
  that preserve historical names for older presets.

## Repository layout

```
SimForge/                       # this repo
├── Genesis/                    # submodule — Genesis physics simulator
├── Newton/                     # submodule — Newton (NVIDIA Warp)
├── Mjlab/                      # submodule — MuJoCo (MJX) framework
└── JaxRLWorld/                 # the framework
    ├── pyproject.toml
    ├── LICENSE
    └── rlworld/
        ├── rl/
        │   ├── algorithms/     # PPO, SAC, TD3, FastTD3, TDMPC2
        │   ├── configs/        # Task presets + algorithm configs
        │   ├── envs/           # Sim-agnostic env + per-sim adapters
        │   ├── modules/        # Network architectures
        │   ├── runners/        # On-policy / Off-policy / Model-based
        │   ├── evals/          # Policy evaluation + cross-sim transfer
        │   └── vis/            # viser-based visualization
        └── scripts/            # Launch scripts per (task, simulator)
```

## Acknowledgements

JaxRLWorld would not exist without prior open-source work; we are
indebted to the authors for releasing high-quality, well-documented
code. In particular:

- **Environment / scene design** — the manager, scene, observation,
  command, event, and randomization abstractions follow conventions
  established by [IsaacLab][isaaclab] and [mjlab][mjlab]. We borrowed
  liberally from their designs while porting the runtime to JAX.
- **RL framework backbone and PPO** — adapted from
  [RSL_RL][rsl_rl] (ETH Robotic Systems Lab), which served as the
  reference implementation for our on-policy training loop and PPO
  update.
- **FastTD3** — JAX port adapted from the authors' original
  implementation: [FastTD3][fasttd3].
- **TDMPC2** — JAX port adapted from the authors' original
  implementation: [TD-MPC2][tdmpc2].

If you build on JaxRLWorld, please also cite the upstream projects
above.

## Citation

If you use JaxRLWorld in academic work, please cite it as:

```bibtex
@misc{shin2026jaxrlworld,
  author       = {Sangwoo Shin},
  title        = {{JaxRLWorld}: A JAX-based RL framework for
                  cross-simulator legged locomotion},
  year         = {2026},
  howpublished = {\url{https://github.com/jsw7460/SimForge}},
}
```

## License

JaxRLWorld is released under the MIT License. See
[JaxRLWorld/LICENSE](JaxRLWorld/LICENSE) for details. Note that the
upstream projects we build on carry their own licenses; please consult
each one when redistributing.

[genesis]: https://github.com/Genesis-Embodied-AI/Genesis
[newton]: https://github.com/newton-physics/newton
[mjlab]: https://github.com/mujocolab/mjlab
[isaaclab]: https://github.com/isaac-sim/IsaacLab
[rsl_rl]: https://github.com/leggedrobotics/rsl_rl
[fasttd3]: https://github.com/younggyoseo/FastTD3
[tdmpc2]: https://github.com/nicklashansen/tdmpc2
[mjpg]: https://github.com/google-deepmind/mujoco_playground
[warp]: https://github.com/NVIDIA/warp
