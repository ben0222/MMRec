#!/bin/bash

set -o pipefail

# ==========================================
# VBPR modality ablation experiment
# Usage:
#   ./run_vbpr_ablation.sh image_only
#   ./run_vbpr_ablation.sh text_only
#   ./run_vbpr_ablation.sh image_text
# ==========================================

MODE="$1"

ROOT="/root/autodl-tmp/MMRec"
SRC="$ROOT/src"
CONFIG="$SRC/configs/model/VBPR.yaml"
EXP_ROOT="$ROOT/experiments/vbpr"

if [ -z "$MODE" ]; then
    echo "Usage:"
    echo "  $0 image_only"
    echo "  $0 text_only"
    echo "  $0 image_text"
    exit 1
fi

case "$MODE" in
    image_only)
        USE_VISUAL=True
        USE_TEXTUAL=False
        ;;
    text_only)
        USE_VISUAL=False
        USE_TEXTUAL=True
        ;;
    image_text)
        USE_VISUAL=True
        USE_TEXTUAL=True
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Allowed: image_only | text_only | image_text"
        exit 1
        ;;
esac

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXP_NAME="VBPR_sports_${MODE}_${TIMESTAMP}"
EXP_DIR="$EXP_ROOT/$EXP_NAME"

mkdir -p "$EXP_DIR"

FULL_LOG="$EXP_DIR/full.log"
SUMMARY="$EXP_DIR/summary.txt"

echo "=========================================="
echo "Experiment: $EXP_NAME"
echo "Visual:     $USE_VISUAL"
echo "Textual:    $USE_TEXTUAL"
echo "Directory:  $EXP_DIR"
echo "=========================================="

# ------------------------------------------
# 1. Backup original VBPR config
# ------------------------------------------

cp "$CONFIG" "$EXP_DIR/VBPR.yaml.before"

# ------------------------------------------
# 2. Set modality
# ------------------------------------------

sed -i "s/^use_visual:.*/use_visual: $USE_VISUAL/" "$CONFIG"
sed -i "s/^use_textual:.*/use_textual: $USE_TEXTUAL/" "$CONFIG"

cp "$CONFIG" "$EXP_DIR/VBPR.yaml"

echo
echo "Current modality configuration:"
grep "use_visual\|use_textual" "$CONFIG"

# ------------------------------------------
# 3. Record reproducibility information
# ------------------------------------------

cd "$ROOT" || exit 1

git rev-parse HEAD > "$EXP_DIR/git_commit.txt"
git status --short > "$EXP_DIR/git_status.txt"

nvidia-smi > "$EXP_DIR/gpu.txt" 2>&1

{
    echo "Python:"
    python --version

    echo
    echo "PyTorch:"
    python - <<'PY'
import torch
print("torch =", torch.__version__)
print("cuda =", torch.version.cuda)
print("cuda available =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu =", torch.cuda.get_device_name(0))
PY
} > "$EXP_DIR/environment.txt" 2>&1

# ------------------------------------------
# 4. Verify Config before training
# ------------------------------------------

cd "$SRC" || exit 1

python - <<PY > "$EXP_DIR/config_check.txt"
from utils.configurator import Config

config = Config(model='VBPR', dataset='sports')

print("model       :", config['model'])
print("dataset     :", config['dataset'])
print("use_visual  :", config['use_visual'])
print("use_textual :", config['use_textual'])
print("vision file :", config['vision_feature_file'])
print("text file   :", config['text_feature_file'])
print("seed        :", config['seed'])
print("reg_weight  :", config['reg_weight'])
PY

cat "$EXP_DIR/config_check.txt"

# ------------------------------------------
# 5. Start experiment
# ------------------------------------------

START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
START_SECONDS=$(date +%s)

echo
echo "Experiment started at: $START_TIME"
echo

python main.py -m VBPR -d sports 2>&1 | tee "$FULL_LOG"

TRAIN_STATUS=${PIPESTATUS[0]}

END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
END_SECONDS=$(date +%s)

DURATION=$((END_SECONDS - START_SECONDS))

# ------------------------------------------
# 6. Restore original configuration
# ------------------------------------------

cp "$EXP_DIR/VBPR.yaml.before" "$CONFIG"

# ------------------------------------------
# 7. If training failed, DO NOT shutdown
# ------------------------------------------

if [ "$TRAIN_STATUS" -ne 0 ]; then
    {
        echo "=========================================="
        echo "EXPERIMENT FAILED"
        echo "=========================================="
        echo "Experiment : $EXP_NAME"
        echo "Mode       : $MODE"
        echo "Start      : $START_TIME"
        echo "End        : $END_TIME"
        echo "Exit code  : $TRAIN_STATUS"
        echo
        echo "Server was NOT shut down."
        echo "Check:"
        echo "$FULL_LOG"
    } | tee "$SUMMARY"

    exit "$TRAIN_STATUS"
fi

# ------------------------------------------
# 8. Extract final BEST block
# ------------------------------------------

BEST_BLOCK=$(tail -n 100 "$FULL_LOG" | \
    grep -A 3 "█████████████ BEST" | tail -n 4)

if [ -z "$BEST_BLOCK" ]; then
    echo "ERROR: Training finished but final BEST block was not found."
    echo "Server will NOT shut down."
    exit 2
fi

# ------------------------------------------
# 9. Create experiment summary
# ------------------------------------------

{
    echo "=========================================="
    echo "VBPR Ablation Experiment"
    echo "=========================================="
    echo
    echo "Experiment : $EXP_NAME"
    echo "Model      : VBPR"
    echo "Dataset    : sports"
    echo "Mode       : $MODE"
    echo
    echo "Visual     : $USE_VISUAL"
    echo "Textual    : $USE_TEXTUAL"
    echo
    echo "Start      : $START_TIME"
    echo "End        : $END_TIME"
    echo "Duration   : $DURATION seconds"
    echo
    echo "Git commit : $(cat "$EXP_DIR/git_commit.txt")"
    echo
    echo "=========================================="
    echo "FINAL BEST RESULT"
    echo "=========================================="
    echo
    echo "$BEST_BLOCK"
    echo
    echo "=========================================="
    echo "Files"
    echo "=========================================="
    echo
    echo "Full log   : $FULL_LOG"
    echo "Config     : $EXP_DIR/VBPR.yaml"
    echo "Environment: $EXP_DIR/environment.txt"
    echo "GPU        : $EXP_DIR/gpu.txt"
} > "$SUMMARY"

cat "$SUMMARY"

# ------------------------------------------
# 10. Final safety checks
# ------------------------------------------

if ! grep -q "^Parameters:" "$SUMMARY" && \
   ! grep -q "Parameters:" "$SUMMARY"; then
    echo "ERROR: Parameters missing from summary."
    echo "Server will NOT shut down."
    exit 3
fi

if ! grep -q "^Valid:" "$SUMMARY"; then
    echo "ERROR: Valid result missing from summary."
    echo "Server will NOT shut down."
    exit 4
fi

if ! grep -q "^Test:" "$SUMMARY"; then
    echo "ERROR: Test result missing from summary."
    echo "Server will NOT shut down."
    exit 5
fi

sync

echo
echo "=========================================="
echo "Experiment completed successfully."
echo "Results saved to:"
echo "$EXP_DIR"
echo
echo "AutoDL server will now shut down."
echo "=========================================="

sleep 10

/usr/bin/shutdown
