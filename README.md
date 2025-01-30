# Super-Resolution on GTA V Screenshots

## Overview

This repository contains multiple deep learning models trained to achieve 2× super-resolution on a dataset of screenshots from GTA V. These architectures have been explored: SRResNet, SwinIR, U-Net, and a custom-designed model.

## Repository Structure

```
├── train_SRResNet.ipynb               # Training script for SRResNet
├── SwinIR.ipynb                        # Training and evaluation of SwinIR
├── Unet.ipynb                           # Training script for U-Net which accepts a lower resolution image upscaled to the target resolution with a simple method like bilinear
├── Unet_different_sizes.ipynb           # U-Net with the size of the input frame being smaller than the output, where resizing is done in the model
├── custom_architecture.ipynb            # A custom-built super-resolution model
├── Unet_tuning.ipynb                     # Hyperparameter and architecture tuning for U-Net
├── train_SRResNet_tuning.ipynb           # Hyperparameter and architecture tuning for SRResNet
```

## Models Implemented

### 1. **SRResNet**

- A deep residual network for image super-resolution.
- Uses content loss with VGG
- Training and hyperparameter tuning available in `train_SRResNet.ipynb` and `train_SRResNet_tuning.ipynb`.

### 2. **SwinIR**

- A transformer-based model for image restoration.
- Includes training and **proper performance evaluation**.
- Implemented in `SwinIR.ipynb`.

### 3. **U-Net**

- Originally developed for segmentation, repurposed for super-resolution.
- Implemented in `Unet.ipynb` and `Unet_different_sizes.ipynb`.
- Hyperparameter tuning performed in `Unet_tuning.ipynb`.

### 4. **Custom Architecture**

- A uniquely designed super-resolution model.
- Implemented in `custom_architecture.ipynb`.

## Performance Evaluation

Currently, **SwinIR.ipynb** contains the most complete performance testing.

## How to Use

1. Open the desired notebook in Jupyter Notebook or Google Colab.
2. Put the data you want to train your models on into 'data/GTAV/small'
3. Execute the cells to train and evaluate the models.
4. Use the cells below to generate example outputs


