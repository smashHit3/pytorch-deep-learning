#!/bin/bash
# Download script for Dogs vs Cats dataset from Kaggle
# Requires Kaggle CLI to be installed and configured: https://www.kaggle.com/docs/api

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR" || exit 1

DATASET_DIR="../dataset/dogs_vs_cats"

echo "=========================================="
echo "Downloading Dogs vs Cats dataset"
echo "=========================================="
echo ""

# Check if kaggle CLI is available
if ! command -v kaggle &> /dev/null; then
    echo "❌ Error: Kaggle CLI not found!"
    echo "Please install and configure Kaggle CLI:"
    echo "  pip install kaggle"
    echo "  https://www.kaggle.com/docs/api"
    exit 1
fi

# Remove existing dataset
echo "🗑️  Removing existing dataset..."
rm -rf "$DATASET_DIR"
mkdir -p "$DATASET_DIR"
cd "$DATASET_DIR" || exit 1

# Download from Kaggle
echo "📥 Downloading from Kaggle..."
if kaggle competitions download -c dogs-vs-cats-redux-kernels-edition; then
    echo "✅ Download completed"
else
    echo "❌ Failed to download dataset"
    exit 1
fi

# Extract files
echo "📦 Extracting files..."
unzip -qq dogs-vs-cats-redux-kernels-edition.zip
unzip -qq train.zip
unzip -qq test.zip

# Cleanup
echo "🧹 Cleaning up..."
rm -f dogs-vs-cats-redux-kernels-edition.zip train.zip test.zip

echo ""
echo "✅ Dataset is ready!"
echo "Location: $DATASET_DIR"
echo "Train images: $(ls train/ | wc -l)"
echo "Test images: $(ls test/ | wc -l)"