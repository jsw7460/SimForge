#!/usr/bin/env bash
# Control experiment for the PPO halfcheetah gap: the SAME learner and
# hyperparameters as jrw_ppo_halfcheetah.py, but driven directly on a
# SyncVectorEnv instead of through BaseRunner + GymnasiumEnv. Seeds
# 0/1/2 in parallel; wandb runs are named JRW-norunner_ppo_halfcheetah_s<seed>.
#
# Compare its Train/mean_return against JRW_ppo_halfcheetah_s<seed>:
# equal => the runner/env plumbing is not responsible for the gap.
#
# Run from anywhere:  bash JaxRLWorld/jaxrlworld/scripts/benchmark/sb3_compare/ppo_halfcheetah_norunner.bash
# CPU pinning:        BENCH_CPUS=0-7,12-31 bash .../ppo_halfcheetah_norunner.bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

TASKSET=()
if [ -n "${BENCH_CPUS:-}" ]; then
    TASKSET=(taskset -c "$BENCH_CPUS")
fi

PIDS=()
for SEED in 0 1 2; do
    BENCH_SEED=$SEED XLA_PYTHON_CLIENT_PREALLOCATE=false \
        ${TASKSET[@]+"${TASKSET[@]}"} python "$SCRIPT_DIR/jrw_ppo_halfcheetah_norunner.py" \
        > "$LOG_DIR/jrw_norunner_ppo_halfcheetah_s$SEED.log" 2>&1 &
    PIDS+=($!)
    sleep 3
done

echo "launched ${#PIDS[@]} processes: ${PIDS[@]}"
echo "logs: $LOG_DIR/jrw_norunner_ppo_halfcheetah_s{0,1,2}.log"

FAIL=0
for PID in "${PIDS[@]}"; do
    wait "$PID" || FAIL=$((FAIL + 1))
done
echo "--- FINAL ---"
grep -H FINAL "$LOG_DIR"/jrw_norunner_ppo_halfcheetah_s*.log
if [ "$FAIL" -ne 0 ]; then
    echo "$FAIL process(es) exited non-zero — check the logs above."
    exit 1
fi
echo "all three runs finished."
