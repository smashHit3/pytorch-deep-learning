# PyTorch Deep Learning

Local PyTorch experiments for image classification, text classification, and a small web UI for running both.

## Project layout

```text
pytorch-deep-learning/
├── cv_sources/
│   ├── classification/          # CV training and inference entrypoints
│   ├── data_processor/          # Dogs vs Cats and FashionMNIST loaders
│   ├── models/                  # AlexNet, VGG, GoogLeNet, ResNet, DenseNet, MobileNetV1
│   ├── results/                 # CV model weights
│   └── train_all_cv_models.sh   # Batch CV training script
├── nlp_sources/
│   ├── data_processor/          # IMDB and AG News loaders
│   ├── models/                  # LSTM, GRU, Transformer classifiers
│   ├── results/                 # NLP model weights and vocab/config files
│   ├── inference.py             # NLP inference entrypoint
│   ├── train.py                 # NLP training entrypoint
│   └── train_all_nlp_models.sh  # Batch NLP training script
├── web/
│   ├── app.py                   # FastAPI app for CV and NLP demos
│   ├── static/                  # CSS and JavaScript
│   └── templates/               # HTML templates
└── README.md
```

## Supported models

### Computer vision

| Family | Variants |
| --- | --- |
| AlexNet | `alexnet` |
| VGG | `vgg11`, `vgg13`, `vgg16`, `vgg19` |
| GoogLeNet | `googlenet` |
| ResNet | `resnet18`, `resnet34`, `resnet50` |
| DenseNet | `densenet121`, `densenet169`, `densenet201` |
| MobileNetV1 | `mobilenet_x1_0`, `mobilenet_x0_75`, `mobilenet_x0_5` |

### NLP

| Family | Variants |
| --- | --- |
| Recurrent | `lstm`, `gru` |
| Transformer | `transformer` |

## Quick start

### Run the web app

```bash
python web/app.py
```

Then open `http://localhost:8000`.

### Train one CV model

```bash
python cv_sources/classification/train.py --model mobilenet_x1_0 --dataset dogs_vs_cats --epochs 10
```

### Train one NLP model

```bash
python nlp_sources/train.py --model lstm --dataset imdb --epochs 10
```

### Batch training

```bash
./cv_sources/train_all_cv_models.sh
```

```bash
./nlp_sources/train_all_nlp_models.sh
```

## Dataset notes

- **Dogs vs Cats**: extract raw images to `dataset/dogs_vs_cats/train/` with filenames like `cat.123.jpg` and `dog.123.jpg`.
- **FashionMNIST**: downloaded automatically under `dataset/`.
- **IMDB / AG News**: handled by the NLP data loaders.

## Useful commands

### CV training help

```bash
python cv_sources/classification/train.py --help
```

### CV inference from the CLI

```bash
python cv_sources/classification/inference.py \
  --image path/to/image.jpg \
  --model resnet18 \
  --dataset dogs_vs_cats
```

### NLP inference from the CLI

```bash
python nlp_sources/inference.py \
  --model lstm \
  --model-path nlp_sources/results/lstm.pth \
  --text "This movie was surprisingly good." \
  --output-json
```

## Notes

- CV weights are stored in `cv_sources/results/`.
- NLP weights are stored in `nlp_sources/results/`.
- MobileNet model names use the public `mobilenet_x...` form, while the existing checkpoint filenames remain `mobilenet_1_0.pth`, `mobilenet_0_75.pth`, and `mobilenet_0_5.pth`.
- The web app only shows models whose weight files already exist on disk.
