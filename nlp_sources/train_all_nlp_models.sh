#!/bin/bash
# Batch training script for all NLP models
# This script is specifically for NLP tasks and is located in nlp_sources/

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR" || exit 1

RESULTS_DIR="results"

echo "=========================================="
echo "NLP Batch Training: All Models"
echo "=========================================="

declare -A model_weight_map=(
    ["lstm"]="lstm.pth"
    ["gru"]="gru.pth"
    ["transformer"]="transformer.pth"
)

dataset="imdb"
epochs=10
batch_size=32

check_model_exists() {
    local model_name=$1
    local weight_file=${model_weight_map[$model_name]}
    
    if [ -f "$RESULTS_DIR/$weight_file" ]; then
        echo "ℹ️ Model weight file '$weight_file' already exists"
        return 0
    else
        return 1
    fi
}

mkdir -p "$RESULTS_DIR"

for model in "${!model_weight_map[@]}"; do
    echo ""
    echo "=========================================="
    echo "Processing: $model"
    echo "=========================================="
    
    if check_model_exists "$model"; then
        echo "⏭️ Skipping training for $model (weights already exist)"
        continue
    fi
    
    echo "🚀 Starting training for $model..."
    
    python train.py \
        --model "$model" \
        --dataset "$dataset" \
        --epochs "$epochs" \
        --batch-size "$batch_size"
    
    if [ $? -eq 0 ]; then
        echo "✅ $model training completed successfully!"
    else
        echo "❌ $model training failed!"
    fi
done

echo ""
echo "=========================================="
echo "NLP training completed!"
echo "=========================================="
echo "Weights saved to: $RESULTS_DIR/"
ls -la "$RESULTS_DIR"/*.pth 2>/dev/null || echo "No weight files found"