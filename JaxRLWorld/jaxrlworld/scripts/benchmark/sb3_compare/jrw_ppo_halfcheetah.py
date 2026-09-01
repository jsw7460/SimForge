"""JaxRLWorld PPO on HalfCheetah-v5 — parity contract in ``_common.py``.

Run:  jaxpy jaxrlworld/scripts/benchmark/sb3_compare/jrw_ppo_halfcheetah.py
"""

from jaxrlworld.scripts.benchmark.sb3_compare._common import run_jrw

if __name__ == "__main__":
    run_jrw("ppo", "halfcheetah")
