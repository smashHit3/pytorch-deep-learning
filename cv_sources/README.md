# CV Sources - PyTorch Deep Learning Computer Vision Library

A unified computer vision framework for training and inference with classic CNN architectures.

## Directory Structure

```
cv_sources/
├── classification/       # Training & Inference pipeline
│   ├── train.py          # Unified training script
│   └── inference.py      # Inference script with CLI support
├── data_processor/       # Dataset loaders
│   ├── dogs_vs_cats.py   # Kaggle Dogs vs Cats dataset
│   ├── fashion_mnist.py  # Fashion MNIST dataset
│   └── download.sh       # Dataset download script
├── models/               # Neural network implementations
│   ├── alexnet.py        # AlexNet
│   ├── vgg.py            # VGG-11/13/16/19
│   ├── resnet.py         # ResNet-18/34/50
│   ├── googlenet.py      # GoogLeNet (Inception)
│   └── densenet.py       # DenseNet-121/169/201
└── results/              # Trained model weights (auto-generated)
```

## Supported Models

| Model | Variants | Key Feature |
|-------|----------|-------------|
| **AlexNet** | alexnet | Classic 5-conv + 3-fc architecture |
| **VGG** | vgg11, vgg13, vgg16, vgg19 | Deep 3x3 convolution stacks |
| **ResNet** | resnet18, resnet34, resnet50 | Skip connections (residual learning) |
| **GoogLeNet** | googlenet | Inception modules (multi-scale features) |
| **DenseNet** | densenet121, densenet169, densenet201 | Dense connectivity |

## Supported Datasets

| Dataset | Classes | Description |
|---------|---------|-------------|
| **Dogs vs Cats** | 2 | Kaggle binary classification (cat/dog) |
| **Fashion MNIST** | 10 | Zalando clothing items |

## Training

```bash
# Train AlexNet on Dogs vs Cats
python cv_sources/classification/train.py --model alexnet --dataset dogs_vs_cats

# Train ResNet-18 on Fashion MNIST
python cv_sources/classification/train.py --model resnet18 --dataset fashion_mnist

# Train VGG-16 with custom parameters
python cv_sources/classification/train.py --model vgg16 --epochs 20 --batch-size 64 --lr 0.0001

# Available arguments
--model         # Model type (alexnet, vgg11, resnet18, etc.)
--dataset       # Dataset (dogs_vs_cats, fashion_mnist)
--epochs        # Training epochs (default: 10)
--batch-size    # Batch size (default: 32)
--lr            # Learning rate (auto-set per model)
--optimizer     # Optimizer (adam, sgd)
--save-path     # Custom weight save path
```

Weights are auto-saved to `cv_sources/results/<model>.pth`.

## Inference (CLI)

```bash
# Classify an image
python cv_sources/classification/inference.py \
    --image path/to/image.jpg \
    --model alexnet \
    --dataset dogs_vs_cats

# Custom preprocessing
python cv_sources/classification/inference.py \
    --image image.jpg \
    --model resnet18 \
    --num-classes 2 \
    --top-k 3

# FP16 acceleration (CUDA)
python cv_sources/classification/inference.py \
    --image image.jpg \
    --model vgg16 \
    --fp16
```

## Web Application

A FastAPI web interface for interactive image classification.

```bash
# Start the server
python app.py
```

Open `http://localhost:8000` in browser:
- Select model from dropdown
- Upload image
- View prediction results with confidence scores

## Quick Start

```bash
# 1. Train a model
python cv_sources/classification/train.py --model resnet18 --dataset dogs_vs_cats

# 2. Run inference
python cv_sources/classification/inference.py \
    --image test_image.jpg \
    --model resnet18 \
    --dataset dogs_vs_cats

# 3. Or use web app
python app.py
```

## Model Hyperparameters (Auto-configured)

| Model | LR | Optimizer | Weight Decay |
|-------|-----|-----------|--------------|
| AlexNet | 0.01 | SGD | 5e-4 |
| VGG | 0.0001 | Adam | 5e-4 |
| ResNet | 0.001 | SGD | 1e-4 |
| GoogLeNet | 0.001 | Adam | 5e-4 |
| DenseNet | 0.001 | SGD | 1e-4 |