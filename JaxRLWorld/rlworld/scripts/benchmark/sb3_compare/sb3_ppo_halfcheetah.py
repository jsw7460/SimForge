"""SB3 PPO on HalfCheetah-v5 — parity contract in ``_common.py``.

Run:  python rlworld/scripts/benchmark/sb3_compare/sb3_ppo_halfcheetah.py
"""

from rlworld.scripts.benchmark.sb3_compare._common import run_sb3

if __name__ == "__main__":
    run_sb3("ppo", "halfcheetah")
