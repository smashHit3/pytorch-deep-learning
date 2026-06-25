#!/usr/bin/env bash
# Batch training script for all CV models on the dogs_vs_cats dataset.

set -u -o pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
readonly SCRIPT_DIR
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
readonly REPO_ROOT

RESULTS_DIR="$SCRIPT_DIR/results"
readonly RESULTS_DIR

PYTHON_BIN=${PYTHON_BIN:-python}
DATASET=${DATASET:-dogs_vs_cats}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-32}

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO_ROOT" || exit 1

echo "=========================================="
echo "CV Batch Training: All Models on Dogs vs Cats"
echo "=========================================="
echo "Python: $PYTHON_BIN"
echo "Dataset: $DATASET"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"

mapfile -t model_entries < <(
    "$PYTHON_BIN" - <<'PY'
from cv_sources.classification.train import MODEL_FILE_MAP

for model_name in sorted(MODEL_FILE_MAP):
    print(f"{model_name}:{MODEL_FILE_MAP[model_name]}")
PY
)

if [ ${#model_entries[@]} -eq 0 ]; then
    echo "No CV models were discovered from classification.train.MODEL_FILE_MAP."
    exit 1
fi

check_model_exists() {
    local weight_file=$1
    [ -f "$RESULTS_DIR/$weight_file" ]
}

mkdir -p "$RESULTS_DIR"

success_count=0
skip_count=0
failure_count=0

for entry in "${model_entries[@]}"; do
    model_name=${entry%%:*}
    weight_file=${entry#*:}

    echo ""
    echo "=========================================="
    echo "Processing: $model_name"
    echo "=========================================="

    if check_model_exists "$weight_file"; then
        echo "⏭️ Skipping $model_name (found existing weights: $weight_file)"
        skip_count=$((skip_count + 1))
        continue
    fi

    echo "🚀 Starting training for $model_name..."

    if "$PYTHON_BIN" -m cv_sources.classification.train \
        --model "$model_name" \
        --dataset "$DATASET" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE"; then
        echo "✅ $model_name training completed successfully."
        success_count=$((success_count + 1))
    else
        echo "❌ $model_name training failed."
        failure_count=$((failure_count + 1))
    fi
done

echo ""
echo "=========================================="
echo "CV training completed"
echo "=========================================="
echo "Succeeded: $success_count"
echo "Skipped: $skip_count"
echo "Failed: $failure_count"
echo "Weights directory: $RESULTS_DIR"
ls -la "$RESULTS_DIR"/*.pth 2>/dev/null || echo "No weight files found"

if [ "$failure_count" -ne 0 ]; then
    exit 1
fi
