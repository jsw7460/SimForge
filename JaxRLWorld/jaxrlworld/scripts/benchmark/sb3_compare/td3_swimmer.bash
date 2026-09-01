#!/usr/bin/env bash
# TD3 on swimmer: SB3 vs JaxRLWorld, seeds 0/1/2, all six processes
# in parallel. Logs land in sb3_compare/logs/; results in the wandb
# project SB3_vs_JRW (runs suffixed _s<seed>).
#
# Run from anywhere:  bash JaxRLWorld/jaxrlworld/scripts/benchmark/sb3_compare/td3_swimmer.bash
# CPU pinning:        BENCH_CPUS=0-7,12-31 bash .../td3_swimmer.bash
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
    ( START=$(date +%s); BENCH_SEED=$SEED \
        ${TASKSET[@]+"${TASKSET[@]}"} python "$SCRIPT_DIR/sb3_td3_swimmer.py" \
        > "$LOG_DIR/sb3_td3_swimmer_s$SEED.log" 2>&1; \
        echo $(($(date +%s) - START)) > "$LOG_DIR/sb3_td3_swimmer_s$SEED.sec" ) &
    PIDS+=($!)

    # XLA_PYTHON_CLIENT_PREALLOCATE=false is what the jaxpy alias sets;
    # required here so three JAX processes can share the GPU.
    ( START=$(date +%s); BENCH_SEED=$SEED XLA_PYTHON_CLIENT_PREALLOCATE=false \
        ${TASKSET[@]+"${TASKSET[@]}"} python "$SCRIPT_DIR/jrw_td3_swimmer.py" \
        > "$LOG_DIR/jrw_td3_swimmer_s$SEED.log" 2>&1; \
        echo $(($(date +%s) - START)) > "$LOG_DIR/jrw_td3_swimmer_s$SEED.sec" ) &
    PIDS+=($!)

    # JRW stamps its model/log directory with a per-second timestamp;
    # simultaneous starts collide on the same path. Stagger the seeds.
    sleep 3
done

echo "launched ${#PIDS[@]} processes: ${PIDS[@]}"
echo "logs: $LOG_DIR/{sb3,jrw}_td3_swimmer_s{0,1,2}.log"

FAIL=0
for PID in "${PIDS[@]}"; do
    wait "$PID" || FAIL=$((FAIL + 1))
done
if [ "$FAIL" -ne 0 ]; then
    echo "$FAIL process(es) exited non-zero — check the logs above."
    exit 1
fi
echo "--- wall clock (s) and final return ---"
for SEED in 0 1 2; do
    for FW in sb3 jrw; do
        SECS=$(cat "$LOG_DIR/${FW}_td3_swimmer_s$SEED.sec" 2>/dev/null || echo "?")
        if [ "$FW" = jrw ]; then
            RET=$(grep -a "Mean Return" "$LOG_DIR/jrw_td3_swimmer_s$SEED.log" | tail -1 | awk '{print $NF}')
        else
            RET=$(grep -a "ep_rew_mean" "$LOG_DIR/sb3_td3_swimmer_s$SEED.log" | tail -1 | awk '{print $(NF-1)}')
        fi
        printf '%-4s s%s  %6ss  %s\n' "$FW" "$SEED" "$SECS" "${RET:-n/a}"
    done
done
echo "all six runs finished."
