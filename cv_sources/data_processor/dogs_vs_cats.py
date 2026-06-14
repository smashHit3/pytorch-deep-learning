"""
Dogs vs Cats dataset
@File: dogs_vs_cats.py
@Description: Data loading for Dogs vs Cats classification with dataset preparation

Note: This dataset must be downloaded from Kaggle:
https://www.kaggle.com/c/dogs-vs-cats/data

Extract the train.zip to: dataset/dogs_vs_cats/train/
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import random
import shutil

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from .base import _resolve_dataset_dir

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
DATASET_NAME_DOGS_VS_CATS = "dogs_vs_cats"
NUM_CLASSES = 2


def default_transforms(resize: int = 224):
    """Get default transforms for training and validation"""
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(resize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return train_tf, val_tf


def group_files_by_label(source_dir: Path):
    """Group image files by their label (cat/dog)"""
    files_by_label = {"cat": [], "dog": []}
    for path in sorted(source_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VALID_EXT:
            continue
        name = path.name.lower()
        if name.startswith("cat."):
            files_by_label["cat"].append(path)
        elif name.startswith("dog."):
            files_by_label["dog"].append(path)
    return files_by_label


def split_files(file_paths, train_ratio, val_ratio, seed):
    """Split file paths into train/val/test sets"""
    random.seed(seed)
    file_paths = list(file_paths)
    random.shuffle(file_paths)
    total = len(file_paths)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    train_files = file_paths[:train_end]
    val_files = file_paths[train_end:val_end]
    test_files = file_paths[val_end:]
    return train_files, val_files, test_files


def copy_or_move_files(file_paths, target_dir: Path, move=False, verbose=False):
    """Copy or move files to target directory"""
    if len(file_paths) == 0:
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    action = "moved" if move else "copied"
    for source_path in file_paths:
        destination = target_dir / source_path.name
        if move:
            source_path.rename(destination)
        else:
            shutil.copy2(source_path, destination)
    if verbose:
        print(f"{action} {len(file_paths)} files to {target_dir}")


def split_raw_dataset(
    source_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    seed: int = 42,
    move: bool = False,
    verbose: bool = False,
):
    """Split raw dataset into train/val/test directories"""
    source_path = Path(source_dir)
    if source_path.is_dir() and source_path.name == "dogs_vs_cats" and (source_path / "train").exists():
        source_path = source_path / "train"

    if not source_path.exists() or not source_path.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_path}")

    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio > 1.0:
        raise ValueError("train_ratio and val_ratio must satisfy 0 < train_ratio < 1 and train_ratio + val_ratio <= 1")

    if output_dir is None:
        output_dir = source_path.parent / "split"
    else:
        output_dir = Path(output_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)

    files_by_label = group_files_by_label(source_path)
    if not files_by_label["cat"] and not files_by_label["dog"]:
        raise ValueError(
            f"No cat/dog images found in {source_path}. "
            "Ensure files are named like cat.123.jpg or dog.123.jpg."
        )

    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"

    for label, paths in files_by_label.items():
        train_files, val_files, test_files = split_files(paths, train_ratio, val_ratio, seed)
        copy_or_move_files(train_files, train_dir / label, move=move, verbose=verbose)
        copy_or_move_files(val_files, val_dir / label, move=move, verbose=verbose)
        copy_or_move_files(test_files, test_dir / label, move=move, verbose=verbose)
        print(f"{label}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test (total {len(paths)})")
        if verbose:
            print(f"  sample train: {[p.name for p in train_files[:3]]}")
            print(f"  sample val:   {[p.name for p in val_files[:3]]}")
            print(f"  sample test:  {[p.name for p in test_files[:3]]}")

    if test_dir.exists():
        print(f"Split completed. Train: {train_dir}, Val: {val_dir}, Test: {test_dir}")
    else:
        print(f"Split completed. Train: {train_dir}, Val: {val_dir}. No test set created.")
    return train_dir, val_dir, test_dir


def load_data_dogs_vs_cats(
    data_root: Optional[Union[str, Path]] = None,
    batch_size: int = 32,
    resize: int = 224,
    num_workers: int = 4,
    pin_memory: bool = True,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    seed: int = 42,
    train_transform: Optional[transforms.Compose] = None,
    val_transform: Optional[transforms.Compose] = None,
    test_transform: Optional[transforms.Compose] = None,
    verbose: bool = True,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Load Dogs vs Cats dataset
    
    Args:
        data_root: Root directory for datasets
        batch_size: Batch size for dataloader
        resize: Image resize size
        num_workers: Number of worker threads
        pin_memory: Enable pin memory
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        seed: Random seed for splitting
        train_transform: Custom train transforms
        val_transform: Custom validation transforms
        test_transform: Custom test transforms
        verbose: Print loading info
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    dataset_dir = _resolve_dataset_dir(DATASET_NAME_DOGS_VS_CATS, data_root)
    split_dir = dataset_dir / "split"
    default_train_tf, default_val_tf = default_transforms(resize)

    train_tf = train_transform or default_train_tf
    val_tf = val_transform or default_val_tf
    test_tf = test_transform or val_tf

    if verbose:
        print(f"📦 Loading Dogs vs Cats dataset from: {dataset_dir}")

    # Check for prepared split
    if (split_dir / "train").exists() and (split_dir / "val").exists():
        if verbose:
            print(f"✅ Using prepared split dataset in: {split_dir}")
        train_ds = datasets.ImageFolder(str(split_dir / "train"), transform=train_tf)
        val_ds = datasets.ImageFolder(str(split_dir / "val"), transform=val_tf)

        test_loader = None
        if (split_dir / "test").exists():
            test_ds = datasets.ImageFolder(str(split_dir / "test"), transform=test_tf)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        if verbose:
            print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
            if test_loader is not None:
                print(f"Test samples: {len(test_ds)}")
        return train_loader, val_loader, test_loader

    # Check for raw train directory
    raw_train = dataset_dir / "train"
    if raw_train.exists():
        if verbose:
            print(f"🔄 No prepared split found; creating one from raw files in {raw_train}")

        split_raw_dataset(
            source_dir=raw_train,
            output_dir=split_dir,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            move=False,
            verbose=verbose,
        )

        train_ds = datasets.ImageFolder(str(split_dir / "train"), transform=train_tf)
        val_ds = datasets.ImageFolder(str(split_dir / "val"), transform=val_tf)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = None
        test_dir = split_dir / "test"
        if test_dir.exists():
            test_ds = datasets.ImageFolder(str(test_dir), transform=test_tf)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        if verbose:
            if test_loader is not None:
                print(f"✅ Loaded: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")
            else:
                print(f"✅ Loaded: {len(train_ds)} train, {len(val_ds)} val")
        return train_loader, val_loader, test_loader

    # Dataset not found
    raise FileNotFoundError(
        f"❌ No usable dataset found under {dataset_dir}\n"
        "Please download the Dogs vs Cats dataset from Kaggle:\n"
        "https://www.kaggle.com/c/dogs-vs-cats/data\n"
        "Then extract train.zip to: dataset/dogs_vs_cats/train/"
    )