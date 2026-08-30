#!/usr/bin/env bash
# Every (algorithm, task) cell of the SB3 comparison, one after another.
#
# A cell is six processes — three seeds of each framework — run together
# so both sides see the same contention. Cells run one at a time, since
# six is already what the machine was sized for.
#
# Each finished cell appends to logs/sweep_results.tsv and prints a
# table, so a sweep that dies overnight still leaves everything it got
# through. Re-running skips cells already in the file; delete their rows
# (or the file) to redo them.
#
# Run from anywhere:
#   bash JaxRLWorld/rlworld/scripts/benchmark/sb3_compare/sweep.bash
#   BENCH_CPUS=0-7,12-31 bash .../sweep.bash
#   ALGOS="sac td3" TASKS="hopper walker2d" bash .../sweep.bash
#
# Budget: PPO cells are ~2.0M env steps, SAC/TD3 cells 100k steps with a
# gradient step each. On one RTX 4090 with six processes sharing it, a
# cell lands somewhere between 15 and 60 minutes depending on the task's
# observation size, so the full 3 x 6 sweep is an overnight job.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
RESULTS="$LOG_DIR/sweep_results.tsv"
mkdir -p "$LOG_DIR"

ALGOS="${ALGOS:-ppo sac td3}"
TASKS="${TASKS:-halfcheetah swimmer hopper walker2d ant humanoid}"
SEEDS="${SEEDS:-0 1 2}"

TASKSET=()
if [ -n "${BENCH_CPUS:-}" ]; then
    TASKSET=(taskset -c "$BENCH_CPUS")
fi

if [ ! -f "$RESULTS" ]; then
    printf 'algo\ttask\tframework\tseed\tseconds\treturn\n' > "$RESULTS"
fi

cell_done() {
    # $1 algo, $2 task — a cell counts as done once every row is present.
    local want=$(( $(echo "$SEEDS" | wc -w) * 2 ))
    local have
    have=$(awk -F'\t' -v a="$1" -v t="$2" '$1==a && $2==t' "$RESULTS" | wc -l)
    [ "$have" -ge "$want" ]
}

final_return() {
    # $1 framework, $2 log file
    if [ "$1" = jrw ]; then
        grep -a "Mean Return" "$2" 2>/dev/null | tail -1 | awk '{print $NF}'
    else
        grep -a "ep_rew_mean" "$2" 2>/dev/null | tail -1 | awk '{print $(NF-1)}'
    fi
}

TOTAL=0
for ALGO in $ALGOS; do for TASK in $TASKS; do TOTAL=$((TOTAL + 1)); done; done
INDEX=0
STARTED=$(date +%s)

for ALGO in $ALGOS; do
    for TASK in $TASKS; do
        INDEX=$((INDEX + 1))
        TAG="$ALGO/$TASK"

        if cell_done "$ALGO" "$TASK"; then
            echo "[$INDEX/$TOTAL] $TAG — already in $(basename "$RESULTS"), skipping"
            continue
        fi

        echo "[$INDEX/$TOTAL] $TAG — starting $(date '+%F %T')"
        PIDS=()
        for SEED in $SEEDS; do
            ( START=$(date +%s); BENCH_SEED=$SEED \
                ${TASKSET[@]+"${TASKSET[@]}"} python "$SCRIPT_DIR/sb3_run.py" --algo "$ALGO" --task "$TASK" \
                > "$LOG_DIR/sb3_${ALGO}_${TASK}_s$SEED.log" 2>&1; \
                echo $(($(date +%s) - START)) > "$LOG_DIR/sb3_${ALGO}_${TASK}_s$SEED.sec" ) &
            PIDS+=($!)

            # XLA_PYTHON_CLIENT_PREALLOCATE=false is what the jaxpy alias
            # sets; required so three JAX processes can share the GPU.
            ( START=$(date +%s); BENCH_SEED=$SEED XLA_PYTHON_CLIENT_PREALLOCATE=false \
                ${TASKSET[@]+"${TASKSET[@]}"} python "$SCRIPT_DIR/jrw_run.py" --algo "$ALGO" --task "$TASK" \
                > "$LOG_DIR/jrw_${ALGO}_${TASK}_s$SEED.log" 2>&1; \
                echo $(($(date +%s) - START)) > "$LOG_DIR/jrw_${ALGO}_${TASK}_s$SEED.sec" ) &
            PIDS+=($!)

            # JRW stamps its model/log directory with a per-second
            # timestamp; simultaneous starts collide on the same path.
            sleep 3
        done

        FAIL=0
        for PID in "${PIDS[@]}"; do
            wait "$PID" || FAIL=$((FAIL + 1))
        done

        echo "  framework seed   seconds   return"
        for SEED in $SEEDS; do
            for FW in sb3 jrw; do
                SEC=$(cat "$LOG_DIR/${FW}_${ALGO}_${TASK}_s$SEED.sec" 2>/dev/null || echo "")
                RET=$(final_return "$FW" "$LOG_DIR/${FW}_${ALGO}_${TASK}_s$SEED.log")
                printf '  %-9s %-4s %8s   %s\n' "$FW" "s$SEED" "${SEC:-?}" "${RET:-n/a}"
                printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "$ALGO" "$TASK" "$FW" "$SEED" "${SEC:-}" "${RET:-}" >> "$RESULTS"
            done
        done
        if [ "$FAIL" -ne 0 ]; then
            echo "  $FAIL of 6 processes exited non-zero — see $LOG_DIR/*_${ALGO}_${TASK}_s*.log"
        fi
        echo "  elapsed so far: $((($(date +%s) - STARTED) / 60)) min"
    done
done

echo
echo "sweep finished in $((($(date +%s) - STARTED) / 60)) min — results in $RESULTS"
awk -F'\t' 'NR>1 && $6!="" {n[$1"\t"$2"\t"$3]++; r[$1"\t"$2"\t"$3]+=$6; s[$1"\t"$2"\t"$3]+=$5}
     END {printf "\n%-6s %-13s %-4s %8s %10s\n", "algo", "task", "fw", "mean_s", "mean_ret";
          for (k in n) {split(k, p, "\t");
          printf "%-6s %-13s %-4s %8.0f %10.1f\n", p[1], p[2], p[3], s[k]/n[k], r[k]/n[k]}}' "$RESULTS" | sort
