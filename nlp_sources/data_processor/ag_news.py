"""
AG News classification dataset
@File: ag_news.py
@Description: Data loading and downloading for AG News dataset
"""

from pathlib import Path
import zipfile
import urllib.request
from base import build_vocab_and_loaders

DATASET_NAME_AG_NEWS = "ag_news"
DATASET_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv.zip"


def download_ag_news(data_dir: Path):
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
        import shutil
        for file in sub_dir.iterdir():
            shutil.move(str(file), str(data_dir))
        sub_dir.rmdir()
    
    # Clean up zip file
    zip_path.unlink()
    print(f"✅ AG News dataset extracted to {data_dir}")


def load_ag_news_data(data_dir: Path, max_samples=None):
    """
    Load AG News classification dataset
    """
    texts = []
    labels = []
    
    for filename in ['train.csv', 'test.csv']:
        file_path = data_dir / filename
        if not file_path.exists():
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                parts = line.strip().split(',', 2)
                if len(parts) >= 3:
                    labels.append(int(parts[0]) - 1)
                    texts.append(parts[2])
                    
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
    
    # Load data
    all_texts, all_labels = load_ag_news_data(ag_news_dir, max_samples)
    split_idx = int(0.8 * len(all_texts))
    train_texts, train_labels = all_texts[:split_idx], all_labels[:split_idx]
    test_texts, test_labels = all_texts[split_idx:], all_labels[split_idx:]
    num_classes = 4
    
    # Build vocab and create loaders
    train_loader, test_loader, vocab = build_vocab_and_loaders(
        train_texts, train_labels, test_texts, test_labels,
        batch_size=batch_size, max_seq_len=max_seq_len
    )
    
    return train_loader, test_loader, vocab, num_classes