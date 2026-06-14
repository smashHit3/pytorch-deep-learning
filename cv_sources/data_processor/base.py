"""
Base data processing utilities for CV
@File: base.py
@Description: Shared utilities for computer vision data processing
"""

from pathlib import Path
from typing import Optional, Union


def _project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).resolve().parents[1]


def _resolve_dataset_dir(dataset_name: str, data_root: Optional[Union[str, Path]] = None) -> Path:
    """Resolve the dataset directory path"""
    if data_root is None:
        data_root = _project_root() / "dataset"
    else:
        data_root = Path(data_root)
    return data_root if data_root.name == dataset_name else data_root / dataset_name