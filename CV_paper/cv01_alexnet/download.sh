#!/bin/bash
rm -rf ../dataset/dogs_vs_cats
mkdir -p ../dataset/dogs_vs_cats
cd ../dataset/dogs_vs_cats

echo "Downloading dogs vs cats dataset from Kaggle..."
kaggle competitions download -c dogs-vs-cats-redux-kernels-edition

unzip -qq dogs-vs-cats-redux-kernels-edition.zip
unzip -qq train.zip
unzip -qq test.zip

rm -f dogs-vs-cats-redux-kernels-edition.zip train.zip test.zip
echo "Dataset is ready"