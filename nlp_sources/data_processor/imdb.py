"""
IMDB sentiment analysis dataset
@File: imdb.py
@Description: Data loading and downloading for IMDB dataset
"""

from pathlib import Path
import tarfile
import urllib.request
from nlp_sources.data_processor.base import build_vocab_and_loaders

DATASET_NAME_IMDB = "imdb"
DATASET_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


def download_imdb(data_dir: Path):
    """
    Download and extract IMDB dataset if not already present
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # The tar.gz extracts to an 'aclImdb' subdirectory
    acl_dir = data_dir / 'aclImdb'
    train_dir = acl_dir / 'train'
    test_dir = acl_dir / 'test'
    
    # Check if already extracted
    if train_dir.exists() and test_dir.exists():
        print(f"✅ IMDB dataset already exists in {acl_dir}")
        return
    
    # Download the dataset
    print(f"📥 Downloading IMDB dataset from {DATASET_URL}...")
    tar_path = data_dir / 'aclImdb_v1.tar.gz'
    
    try:
        urllib.request.urlretrieve(DATASET_URL, str(tar_path))
        print("✅ Download completed")
    except Exception as e:
        raise RuntimeError(f"Failed to download IMDB dataset: {e}")
    
    # Extract the dataset
    print(f"📦 Extracting IMDB dataset...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(data_dir)
    
    # Clean up tar file
    tar_path.unlink()
    print(f"✅ IMDB dataset extracted to {acl_dir}")


def load_imdb_data(data_dir: Path, max_samples=None):
    """
    Load IMDB sentiment analysis dataset
    """
    texts = []
    labels = []
    
    for label, folder in enumerate(['neg', 'pos']):
        folder_path = data_dir / folder
        if not folder_path.exists():
            continue
            
        files = list(folder_path.iterdir())
        if max_samples:
            files = files[:max_samples]
            
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(label)
                
    return texts, labels


def load_data_imdb(data_root=None, max_samples=None, batch_size=32, max_seq_len=512):
    """
    Load and preprocess IMDB data with auto-download
    """
    if data_root is None:
        data_root = Path(__file__).resolve().parents[1] / 'dataset'
    elif not isinstance(data_root, Path):
        data_root = Path(data_root)
    
    imdb_dir = data_root / 'imdb'
    
    # Download if needed
    download_imdb(imdb_dir)
    
    # The actual data is inside the aclImdb subdirectory
    acl_dir = imdb_dir / 'aclImdb'
    
    # Load data
    train_texts, train_labels = load_imdb_data(acl_dir / 'train', max_samples)
    test_texts, test_labels = load_imdb_data(acl_dir / 'test', max_samples)
    num_classes = 2
    
    # Build vocab and create loaders
    train_loader, test_loader, vocab = build_vocab_and_loaders(
        train_texts, train_labels, test_texts, test_labels,
        batch_size=batch_size, max_seq_len=max_seq_len
    )
    
    return train_loader, test_loader, vocab, num_classes