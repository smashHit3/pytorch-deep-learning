# PyTorch Deep Learning

A local project for learning and experimenting with PyTorch image classification models.

## Structure

```
pytorch-deep-learning/
├── cv_sources/                 # Computer Vision training framework
│   ├── classification/         # Unified classification training
│   │   └── train.py            # Main training script
│   ├── models/                 # Model implementations
│   │   ├── alexnet.py          # AlexNet implementation
│   │   ├── vgg.py              # VGG implementations
│   │   ├── googlenet.py        # GoogLeNet implementation
│   │   ├── resnet.py           # ResNet implementations
│   │   ├── densenet.py         # DenseNet implementations
│   │   └── mobilenet.py        # MobileNetV1 implementations
│   ├── data_processor.py       # Data loading utilities
│   ├── results/                # Trained model weights
│   └── train_all_cv_models.sh  # Batch train all CV models
├── nlp_sources/                # Future NLP experiments
├── CNN_d2l/                    # Tutorial examples from Deep Learning with PyTorch
├── CV_paper/                   # Legacy CV experiments
└── train_all_models.sh         # Root batch training script
```

## Supported Models

| Model | Variants |
|-------|----------|
| AlexNet | alexnet |
| VGG | vgg11, vgg13, vgg16, vgg19 |
| GoogLeNet | googlenet |
| ResNet | resnet18, resnet34, resnet50 |
| DenseNet | densenet121, densenet169, densenet201 |
| MobileNetV1 | mobilenet_1_0, mobilenet_0_75, mobilenet_0_5 |

## Quick Start

### Train a single model

```bash
cd cv_sources/classification
python train.py --model mobilenet_1_0 --dataset dogs_vs_cats --epochs 10
```

### Train all CV models (with skip if weights exist)

```bash
cd cv_sources
./train_all_cv_models.sh
```

### Available options

```bash
python train.py --help
```

## Training Options

- `--model`: Select model (see table above)
- `--dataset`: `dogs_vs_cats` or `fashion_mnist`
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--lr`: Learning rate
- `--optimizer`: `adam` or `sgd`
- `--save-path`: Custom path to save weights

## Dataset Requirements

- **Dogs vs Cats**: Place `cat.*.jpg` and `dog.*.jpg` files in `dataset/dogs_vs_cats/train/`
- **Fashion MNIST**: Automatically downloaded if not present

## Notes

- Model weights are saved to `cv_sources/results/` by default
- The batch training script skips models with existing weights
- MobileNet variants use width multipliers: 1.0 (full size), 0.75, 0.5