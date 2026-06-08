"""A lightweight PyTorch Dataset for the Kaggle cats-vs-dogs filenames.

This dataset expects files named like `cat.123.jpg` or `dog.456.jpg` inside
`<root>/train` and supports deterministic train/val/test splits.

It also provides dataset loader helpers and a raw split utility for
`CV_paper/dataset/dogs_vs_cats`.
"""
from pathlib import Path
from typing import List, Optional, Tuple, Union
import argparse
import random
import shutil

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torchvision.transforms as T


VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_dataset_dir(data_root: Optional[Union[str, Path]] = None) -> Path:
    if data_root is None:
        data_root = _project_root() / "dataset"
    else:
        data_root = Path(data_root)
    return data_root if data_root.name == "dogs_vs_cats" else data_root / "dogs_vs_cats"


def default_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return train_tf, val_tf


def group_files_by_label(source_dir: Path):
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
    if len(file_paths) == 0:
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    for source_path in file_paths:
        destination = target_dir / source_path.name
        if move:
            source_path.rename(destination)
            action = "moved"
        else:
            shutil.copy2(source_path, destination)
            action = "copied"
        if verbose:
            print(f"{action}: {source_path.name} -> {destination}")


def split_raw_dataset(
    source_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    seed: int = 42,
    move: bool = False,
    verbose: bool = False,
):
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
    dataset_dir = _resolve_dataset_dir(data_root)
    split_dir = dataset_dir / "split"
    default_train_tf, default_val_tf = default_transforms()

    train_tf = train_transform or default_train_tf
    val_tf = val_transform or default_val_tf
    test_tf = test_transform or val_tf

    if verbose:
        print(f"Loading dataset from: {dataset_dir}")

    if (split_dir / "train").exists() and (split_dir / "val").exists():
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

    raw_train = dataset_dir / "train"
    if raw_train.exists():
        if verbose:
            print(f"Using CatDogDataset on raw files in {raw_train}")

        train_ds = DogsVsCatsDataset(
            data_dir=str(raw_train),
            mode="train",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            random_seed=seed,
            transform=train_tf,
        )
        val_ds = DogsVsCatsDataset(
            data_dir=str(raw_train),
            mode="val",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            random_seed=seed,
            transform=val_tf,
        )
        test_ds = DogsVsCatsDataset(
            data_dir=str(raw_train),
            mode="test",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            random_seed=seed,
            transform=test_tf,
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        if verbose:
            print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Test samples: {len(test_ds)}")
        return train_loader, val_loader, test_loader

    raise FileNotFoundError(f"No usable dataset found under {dataset_dir}")


class DogsVsCatsDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        mode: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.2,
        random_seed: int = 42,
        transform: Optional[T.Compose] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"data_dir not found: {data_dir}")

        self.mode = mode.lower()
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.random_seed = int(random_seed)
        self.transform = transform or T.Compose([
            T.Resize((256, 256)),
            T.CenterCrop(224),
            T.ToTensor(),
        ])

        if self.train_ratio <= 0 or self.val_ratio < 0 or self.train_ratio + self.val_ratio > 1.0:
            raise ValueError("train_ratio and val_ratio must satisfy 0 < train_ratio < 1 and train_ratio + val_ratio <= 1")

        files = [p for p in sorted(self.data_dir.iterdir()) if p.is_file() and p.suffix.lower() in VALID_EXT]
        labeled: List[Tuple[Path, int]] = []
        for p in files:
            name = p.name.lower()
            if name.startswith("cat."):
                labeled.append((p, 0))
            elif name.startswith("dog." ):
                labeled.append((p, 1))

        if not labeled:
            raise ValueError(f"No cat/dog files found in {data_dir}")

        random.seed(self.random_seed)
        random.shuffle(labeled)

        num_total = len(labeled)
        num_train = int(num_total * self.train_ratio)
        num_val = int(num_total * self.val_ratio)
        train_part = labeled[:num_train]
        val_part = labeled[num_train:num_train + num_val]
        test_part = labeled[num_train + num_val:]

        if self.mode == "train":
            selected = train_part
        elif self.mode in {"val", "valid", "validation"}:
            selected = val_part
        elif self.mode == "test":
            selected = test_part
        else:
            raise ValueError("mode must be 'train', 'val', or 'test'")

        self.paths = [p for p, _ in selected]
        self.labels = [lbl for _, lbl in selected]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        label = self.labels[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split or validate the cats-vs-dogs dataset.")
    parser.add_argument(
        "--action",
        choices=["split", "sanity"],
        default="split",
        help="Split raw files or print dataset loader stats.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root path for dogs_vs_cats or dataset directory.",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=_project_root() / "dataset" / "dogs_vs_cats" / "train",
        help="Source directory containing raw files for splitting.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output split directory.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits.")
    parser.add_argument("--move", action="store_true", help="Move raw files instead of copying.")
    parser.add_argument("--verbose", action="store_true", help="Verbose output.")
    args = parser.parse_args()

    if args.action == "split":
        source_dir = args.source_dir or args.data_root
        if source_dir is None:
            raise ValueError("--source-dir or --data-root must be provided for split action")
        split_raw_dataset(
            source_dir=source_dir,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            move=args.move,
            verbose=args.verbose,
        )
    else:
        loaders = load_data_dogs_vs_cats(data_root=args.data_root, verbose=args.verbose)
        train_loader, val_loader, test_loader = loaders
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        if test_loader is not None:
            print(f"Test batches: {len(test_loader)}")
