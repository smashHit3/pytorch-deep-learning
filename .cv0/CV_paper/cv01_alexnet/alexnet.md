# AlexNet

This folder contains a PyTorch implementation of AlexNet for binary cat-vs-dog classification, along with dataset loading, training, and inference utilities.

## Files

- `sources/alexnet.py` - AlexNet model definition.
- `sources/train.py` - Training entrypoint for the cats-vs-dogs dataset.
- `sources/inference.py` - Single-image inference script using a saved model checkpoint.
- `sources/tools/data_processor.py` - Dataset loader, raw split utility, and transforms.
- `results/alexnet.pth` - Default output path for the trained model weights.

## Model architecture

The implementation follows the original AlexNet design with the following structure:

- Convolutional feature extractor:
  - `Conv2d(3, 64, kernel_size=11, stride=4, padding=2)`
  - `ReLU`
  - `MaxPool2d(kernel_size=3, stride=2)`
  - `Conv2d(64, 192, kernel_size=5, padding=2)`
  - `ReLU`
  - `MaxPool2d(kernel_size=3, stride=2)`
  - `Conv2d(192, 384, kernel_size=3, padding=1)`
  - `ReLU`
  - `Conv2d(384, 256, kernel_size=3, padding=1)`
  - `ReLU`
  - `Conv2d(256, 256, kernel_size=3, padding=1)`
  - `ReLU`
  - `MaxPool2d(kernel_size=3, stride=2)`
- Adaptive average pooling to `6x6`
- Classifier:
  - `Dropout(p=0.5)`
  - `Linear(256 * 6 * 6, 4096)`
  - `ReLU`
  - `Dropout(p=0.5)`
  - `Linear(4096, 4096)`
  - `ReLU`
  - `Linear(4096, num_classes)`

The model is defined in `AlexNet` and supports a configurable number of classes. For the cats-vs-dogs experiment, `num_classes=2`.

## Dataset and preprocessing

The dataset loader supports two modes:

- Pre-split dataset under `dataset/dogs_vs_cats/split/train`, `split/val`, and optional `split/test`.
- Raw dataset under `dataset/dogs_vs_cats/train` with filenames like `cat.123.jpg` and `dog.456.jpg`.

When pre-split data exists, `get_data_loaders` uses `torchvision.datasets.ImageFolder`.
If only raw files are available, `CatDogDataset` performs a deterministic split using `train_ratio` and `val_ratio`.

Default transforms:

- Train: resize to `256x256`, random crop `224x224`, random horizontal flip, convert to tensor, normalize with ImageNet mean/std.
- Validation: resize to `256x256`, center crop `224x224`, convert to tensor, normalize.

## Training

Run the training script from `CV_paper/cv01_alexnet/sources`:

```bash
cd /workspace/pytorch-deep-learning/CV_paper/cv01_alexnet/sources
python train.py --data-root ../dataset
```

Default training settings:

- `--batch-size 32`
- `--epochs 10`
- `--lr 0.01`
- `--momentum 0.9`
- `--weight-decay 5e-4`
- `--num-workers 4`
- `--save-path ../results/alexnet.pth`

Training uses `CrossEntropyLoss`, SGD optimizer, and a `StepLR` scheduler with `step_size=5` and `gamma=0.1`.

## Inference

Run inference on a single image:

```bash
cd /workspace/pytorch-deep-learning/CV_paper/cv01_alexnet/sources
python inference.py --image /path/to/image.jpg --model-path ../results/alexnet.pth --device cpu
```

The inference script:

- loads the saved model state dict
- preprocesses the image with resize, center crop, and normalization
- runs a forward pass
- prints the top-k predictions using the labels `cat` and `dog`

## Dataset split utility

The dataset utility in `sources/tools/data_processor.py` also provides a split tool:

```bash
python sources/tools/data_processor.py --action split --data-root ../dataset --train-ratio 0.8 --val-ratio 0.2 --verbose
```

This will create `dataset/dogs_vs_cats/split/train`, `split/val`, and optionally `split/test` directories if raw files are present.
