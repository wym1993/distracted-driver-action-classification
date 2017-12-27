# distracted-driver-action-classification

## Introduction
This repository is the code for Kaggle [State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection). We further extend the classification task to continuous frames.

## Methodology
### Traditional CNN models
* AlexNet
* VGG16 (with and without pretrained weight)
* ResNet

### Modified K Nearest Neighbors
Calculate the average output of 10 most similar images in test dataset to stabilize the prediction.

### Skin detection
Extract useful features for input training images.

## Ranking
Ranking top 10% out of over 1, 400 teams
