#!/usr/bin/env bash
# SAC on swimmer: SB3 vs JaxRLWorld, seeds 0/1/2, all six processes
# in parallel. Logs land in sb3_compare/logs/; results in the wandb
# project SB3_vs_JRW (runs suffixed _s<seed>).
#
# Run from anywhere:  bash JaxRLWorld/rlworld/scripts/benchmark/sb3_compare/sac_swimmer.bash
# CPU pinning:        BENCH_CPUS=0-7,12-31 bash .../sac_swimmer.bash
#   (applies `taskset -c $BENCH_CPUS` to every process; unset = no pinning)
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
    BENCH_SEED=$SEED \
        ${TASKSET[@]+"${TASKSET[@]}"} python "$SCRIPT_DIR/sb3_sac_swimmer.py" \
        > "$LOG_DIR/sb3_sac_swimmer_s$SEED.log" 2>&1 &
    PIDS+=($!)

    # XLA_PYTHON_CLIENT_PREALLOCATE=false is what the jaxpy alias sets;
    # required here so three JAX processes can share the GPU.
    BENCH_SEED=$SEED XLA_PYTHON_CLIENT_PREALLOCATE=false \
        ${TASKSET[@]+"${TASKSET[@]}"} python "$SCRIPT_DIR/jrw_sac_swimmer.py" \
        > "$LOG_DIR/jrw_sac_swimmer_s$SEED.log" 2>&1 &
    PIDS+=($!)

    # JRW stamps its model/log directory with a per-second timestamp;
    # simultaneous starts collide on the same path. Stagger the seeds.
    sleep 3
done

echo "launched ${#PIDS[@]} processes: ${PIDS[@]}"
echo "logs: $LOG_DIR/{sb3,jrw}_sac_swimmer_s{0,1,2}.log"

FAIL=0
for PID in "${PIDS[@]}"; do
    wait "$PID" || FAIL=$((FAIL + 1))
done
if [ "$FAIL" -ne 0 ]; then
    echo "$FAIL process(es) exited non-zero — check the logs above."
    exit 1
fi
echo "all six runs finished."
