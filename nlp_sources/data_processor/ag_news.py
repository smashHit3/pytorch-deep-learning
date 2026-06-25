"""
AG News classification dataset
@File: ag_news.py
@Description: Data loading and downloading for AG News dataset
"""

import csv
import shutil
from pathlib import Path
import urllib.request
import zipfile

from nlp_sources.data_processor.base import build_vocab_and_loaders

DATASET_NAME_AG_NEWS = "ag_news"
DATASET_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv.zip"


def download_ag_news(data_dir: Path) -> None:
    """
    Download and extract AG News dataset if not already present
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already extracted
    train_file = data_dir / 'train.csv'
    test_file = data_dir / 'test.csv'
    if train_file.exists() and test_file.exists():
        print(f"✅ AG News dataset already exists in {data_dir}")
        return
    
    # Download the dataset
    print(f"📥 Downloading AG News dataset from {DATASET_URL}...")
    zip_path = data_dir / 'ag_news_csv.zip'
    
    try:
        urllib.request.urlretrieve(DATASET_URL, str(zip_path))
        print("✅ Download completed")
    except Exception as e:
        raise RuntimeError(f"Failed to download AG News dataset: {e}")
    
    # Extract the dataset
    print(f"📦 Extracting AG News dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    # Move files from subdirectory if needed
    sub_dir = data_dir / 'ag_news_csv'
    if sub_dir.exists():
        for file in sub_dir.iterdir():
            target = data_dir / file.name
            if target.exists():
                target.unlink()
            shutil.move(str(file), str(target))
        sub_dir.rmdir()
    
    # Clean up zip file
    zip_path.unlink()
    print(f"✅ AG News dataset extracted to {data_dir}")


def load_ag_news_data(file_path: Path, max_samples: int | None = None) -> tuple[list[str], list[int]]:
    """
    Load AG News classification dataset
    """
    texts: list[str] = []
    labels: list[int] = []

    with file_path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if max_samples is not None and i >= max_samples:
                break
            if len(row) < 3:
                continue
            label = int(row[0]) - 1
            # Keep both title and description for a stronger text signal.
            text = f"{row[1]} {row[2]}".strip()
            labels.append(label)
            texts.append(text)

    return texts, labels


def load_data_ag_news(data_root=None, max_samples=None, batch_size=32, max_seq_len=512):
    """
    Load and preprocess AG News data with auto-download
    """
    if data_root is None:
        data_root = Path(__file__).resolve().parents[1] / 'dataset'
    elif not isinstance(data_root, Path):
        data_root = Path(data_root)
    
    ag_news_dir = data_root / 'ag_news'
    
    # Download if needed
    download_ag_news(ag_news_dir)
    
    train_file = ag_news_dir / 'train.csv'
    test_file = ag_news_dir / 'test.csv'
    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(f"AG News files not found in {ag_news_dir}")

    # Load canonical split shipped with the dataset.
    train_texts, train_labels = load_ag_news_data(train_file, max_samples)
    test_texts, test_labels = load_ag_news_data(test_file, max_samples)
    num_classes = 4
    
    # Build vocab and create loaders
    train_loader, test_loader, vocab = build_vocab_and_loaders(
        train_texts, train_labels, test_texts, test_labels,
        batch_size=batch_size, max_seq_len=max_seq_len
    )
    
    return train_loader, test_loader, vocab, num_classes