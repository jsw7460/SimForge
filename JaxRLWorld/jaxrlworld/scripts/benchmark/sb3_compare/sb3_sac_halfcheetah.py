"""SB3 SAC on HalfCheetah-v5 — parity contract in ``_common.py``.

Run:  python jaxrlworld/scripts/benchmark/sb3_compare/sb3_sac_halfcheetah.py
"""

from jaxrlworld.scripts.benchmark.sb3_compare._common import run_sb3

if __name__ == "__main__":
    run_sb3("sac", "halfcheetah")
