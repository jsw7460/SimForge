"""SB3 PPO on Swimmer-v5 — parity contract in ``_common.py``.

Run:  python rlworld/scripts/benchmark/sb3_compare/sb3_ppo_swimmer.py
"""

from rlworld.scripts.benchmark.sb3_compare._common import run_sb3

if __name__ == "__main__":
    run_sb3("ppo", "swimmer")
