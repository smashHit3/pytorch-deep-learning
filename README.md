# PyTorch Deep Learning

A local project for learning and experimenting with PyTorch image classification models.

## Structure

- `CNN_d2l/` - tutorial examples from Deep Learning with PyTorch (linear regression, softmax, LeNet, AlexNet, VGG, ResNet, and more)
- `CV_paper/` - custom computer vision experiments and paper code
  - `A_AlexNet/` - legacy AlexNet training and inference scripts
  - `cv01_alexnet/` - current AlexNet implementation and dataset handling
  - `dataset/` - dataset storage, including `dogs_vs_cats`

## Current focus

- Train AlexNet on the cats-vs-dogs dataset
- Data processing and splitting are handled in `CV_paper/cv01_alexnet/src/tools/data_processor.py`
- Training entrypoint: `CV_paper/cv01_alexnet/src/train.py`
- Model outputs saved to `CV_paper/cv01_alexnet/results/alexnet.pth`

## Quick start

```bash
cd /workspace/pytorch-deep-learning/CV_paper/cv01_alexnet/src
python train.py --data-root ../dataset
```

## Notes

- `CV_paper/dataset/dogs_vs_cats/train` is expected to contain raw `cat.*.jpg` and `dog.*.jpg` files.
- If a split folder exists at `CV_paper/dataset/dogs_vs_cats/split`, it will be used instead.
- Default dataset split is 80% train and 20% validation.
