#!/bin/bash
# Batch training script for all models on dogs_vs_cats dataset
# Each model will be trained for 10 epochs

echo "=========================================="
echo "Batch Training: All Models on Dogs vs Cats"
echo "=========================================="

# Define all models
models=(
    "alexnet"
    "vgg11"
    "vgg13"
    "vgg16"
    "vgg19"
    "googlenet"
    "resnet18"
    "resnet34"
    "resnet50"
    "densenet121"
    "densenet169"
    "densenet201"
    "mobilenet"
    "mobilenet_0_5"
    "mobilenet_0_75"
)

# Training parameters
dataset="dogs_vs_cats"
epochs=10
batch_size=32

# Train each model
for model in "${models[@]}"; do
    echo ""
    echo "=========================================="
    echo "Training: $model"
    echo "=========================================="
    
    python cv_sources/classification/train.py \
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
echo "All training completed!"
echo "=========================================="
echo "Weights saved to: cv_sources/results/"
ls -la cv_sources/results/*.pth 2>/dev/null || echo "No weight files found"