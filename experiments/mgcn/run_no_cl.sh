#!/usr/bin/env bash

set -uo pipefail

ROOT="/root/autodl-tmp/MMRec"
EXP="$ROOT/experiments/mgcn"
RUN_DIR="$EXP/no_cl_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$RUN_DIR"

LOG="$RUN_DIR/full.log"

export OMP_NUM_THREADS=1

cd "$ROOT" || exit 1

git rev-parse HEAD > "$RUN_DIR/git_commit.txt"
git status --short > "$RUN_DIR/git_status.txt"

cp src/configs/model/MGCN.yaml "$RUN_DIR/MGCN.yaml"
cp "$EXP/run_no_cl.py" "$RUN_DIR/run_no_cl.py"

python -m pip freeze > "$RUN_DIR/environment.txt"
nvidia-smi > "$RUN_DIR/gpu.txt"

echo "Experiment directory: $RUN_DIR"

cd "$ROOT/src" || exit 1

python "$EXP/run_no_cl.py" > "$LOG" 2>&1
EXIT_CODE=$?

echo "Python exit code: $EXIT_CODE"

if [ "$EXIT_CODE" -ne 0 ]; then
    echo "ERROR: Training failed. Server will remain running."
    exit "$EXIT_CODE"
fi

if ! grep -q "All Over" "$LOG"; then
    echo "ERROR: Final experiment completion marker missing."
    exit 1
fi

if ! grep -q "BEST" "$LOG"; then
    echo "ERROR: Final BEST result missing."
    exit 1
fi

tail -n 40 "$LOG" > "$RUN_DIR/summary.txt"

echo "SUCCESS: MGCN w/o CL completed."
echo "Results saved in: $RUN_DIR"

sync
sleep 10

/usr/bin/shutdown -h now
