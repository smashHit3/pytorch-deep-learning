#!/bin/bash
# Batch training script for all CV models on dogs_vs_cats dataset
# This script is specifically for CV tasks and is located in cv_sources/
# NLP scripts will be placed separately in nlp_sources/

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR" || exit 1

RESULTS_DIR="results"

echo "=========================================="
echo "CV Batch Training: All Models on Dogs vs Cats"
echo "=========================================="

declare -A model_weight_map=(
    ["alexnet"]="alexnet.pth"
    ["vgg11"]="vgg11.pth"
    ["vgg13"]="vgg13.pth"
    ["vgg16"]="vgg16.pth"
    ["vgg19"]="vgg19.pth"
    ["googlenet"]="googlenet.pth"
    ["resnet18"]="resnet18.pth"
    ["resnet34"]="resnet34.pth"
    ["resnet50"]="resnet50.pth"
    ["densenet121"]="densenet121.pth"
    ["densenet169"]="densenet169.pth"
    ["densenet201"]="densenet201.pth"
    ["mobilenet_x1_0"]="mobilenet_1_0.pth"
    ["mobilenet_x0_5"]="mobilenet_0_5.pth"
    ["mobilenet_x0_75"]="mobilenet_0_75.pth"
)

dataset="dogs_vs_cats"
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
    
    python classification/train.py \
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
echo "CV training completed!"
echo "=========================================="
echo "Weights saved to: $RESULTS_DIR/"
ls -la "$RESULTS_DIR"/*.pth 2>/dev/null || echo "No weight files found"