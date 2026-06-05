#!/bin/bash
mkdir -p ../dataset/dogs_vs_cats
cd ../dataset/dogs_vs_cats
echo "Downloading dogs vs cats dataset from Kaggle..."
kaggle competitions download -c dogs-vs-cats-redux-kernels-edition
echo "Done"
unzip dogs-vs-cats-redux-kernels-edition.zip 
unzip train.zip
unzip test.zip
rm dogs-vs-cats-redux-kernels-edition.zip
rm train.zip
rm test.zip
echo "Dataset is ready"