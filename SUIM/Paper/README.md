# SUIM-Net: Semantic Segmentation of Underwater Imagery

This repository contains implementations and experiments for underwater image semantic segmentation using the SUIM-Net architecture. The project is based on the paper [Semantic Segmentation of Underwater Imagery: Dataset and Benchmark](https://arxiv.org/pdf/2004.01241.pdf) (IROS 2020) and includes both PyTorch and Keras implementations with multiple backbone architectures.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

## Overview

SUIM-Net is a fully-convolutional encoder-decoder network designed for semantic segmentation of natural underwater images. This project explores different backbone architectures and data augmentation strategies to improve segmentation performance.

### Segmentation Categories

The model segments underwater images into the following classes:
- **BW**: Background/waterbody
- **HD**: Human divers
- **PF**: Aquatic plants and sea-grass
- **WR**: Wrecks/ruins
- **RO**: Robots/instruments
- **RI**: Reefs/invertebrates
- **FV**: Fish and vertebrates
- **SR**: Sea-floor/rocks

## Dataset

**SUIM Dataset** (Segmentation of Underwater IMagery):
- 1,525 annotated images for training/validation
- 110 samples for testing
- Multiple object categories for semantic segmentation
- Dataset available at: [SUIM Dataset](http://irvlab.cs.umn.edu/resources/suim-dataset)

## Models

This project implements SUIM-Net with three different backbone architectures:

### 1. **SUIM-Net (RSB)** - ResNet-style Backbone
- Residual Skip Blocks (RSB) for feature extraction
- Lightweight and fast inference
- Input resolution: 320×256×3
- Good balance between speed and accuracy

### 2. **SUIM-Net (VGG)** - VGG16 Backbone
- Pre-trained VGG16 encoder
- Better generalization performance
- Input resolution: 320×256×3
- Higher accuracy with more parameters

### 3. **SUIM-Net (SWIN)** - Swin Transformer Backbone
- Transformer-based architecture
- State-of-the-art feature extraction
- Input resolution: 256×256×3
- Best performance for complex scenes

### Data Augmentation
Each architecture includes variants trained with data augmentation (denoted by `_aug` suffix):
- Rotation, flipping, scaling
- Color jittering
- Improved generalization to diverse underwater conditions

## Project Structure

```
Assignment_2/
├── README.md                          # This file
├── visualizemasks.py                  # Visualization script for comparing model outputs
│
├── SUIM/                              # Main SUIM implementation directory
│   │
│   ├── Paper/                         # Keras/TensorFlow implementation (from paper)
│   │   ├── README.md                  # Original SUIM paper README
│   │   ├── train_suimnet.py          # Training script
│   │   ├── test_suimnet.py           # Testing script
│   │   ├── models/                    # Keras model definitions
│   │   │   └── suim_net.py
│   │   ├── utils/                     # Data utilities
│   │   │   └── data_utils.py
│   │   └── ckpt_keras/                # Keras checkpoints and outputs
│   │       ├── suimnet_rsb5.hdf5
│   │       ├── suimnet_vgg5.hdf5
│   │       └── output_keras_*/        # Output predictions
│   │
│   ├── Pytorch/                       # PyTorch implementation (main implementation)
│   │   ├── train_suimnet_colab.ipynb # Training notebook for Google Colab
│   │   ├── models/                    # PyTorch model definitions
│   │   │   ├── suim_net.py           # RSB and VGG variants
│   │   │   ├── suim_net_swin.py      # Swin Transformer variant
│   │   │   └── suim_net_swin_PPM.py  # Swin with Pyramid Pooling Module
│   │   ├── utils/                     # Data utilities
│   │   │   └── data_utils.py
│   │   ├── test/                      # Testing scripts
│   │   │   └── test_suimnet.py
│   │   └── ckpt_*/                    # Model checkpoints for each variant
│   │       ├── ckpt_rsb/              # RSB model checkpoints
│   │       ├── ckpt_rsb_aug/          # RSB with augmentation
│   │       ├── ckpt_vgg/              # VGG model checkpoints
│   │       ├── ckpt_vgg_aug/          # VGG with augmentation
│   │       ├── ckpt_swin/             # Swin Transformer checkpoints
│   │       └── ckpt_swin_aug/         # Swin with augmentation
│   │
│   ├── TEST/                          # Test dataset
│   │   ├── images/                    # Test images
│   │   └── masks/                     # Ground truth masks by category
│   │       ├── FV/, HD/, PF/, RI/, RO/, SR/, WR/
│   │       └── Saliency/
│   │
│   ├── train_val/                     # Training and validation dataset
│   │   ├── images/
│   │   └── masks/
│   │
│   └── utils/                         # Utility scripts
│       ├── create_rgb_from_masks.py   # RGB mask generation
│       ├── get_f1_iou.py             # Performance evaluation metrics
│       └── measure_utils.py           # Measurement utilities
```

#### PyTorch (Recommended)
```bash
# Open the Jupyter notebook for training
jupyter notebook SUIM/Pytorch/train_suimnet_colab.ipynb
```

The notebook allows you to:
- Select backbone architecture (RSB, VGG, or SWIN)
- Configure data augmentation
- Adjust hyperparameters
- Monitor training progress

### Testing

```bash
cd SUIM/Pytorch/test
python test_suimnet.py
```

Configure the test script by modifying:
- `test_dir`: Path to test images
- `ckpt_dir`: Path to model checkpoint
- `base_`: Model backbone ('RSB', 'VGG', or 'SWIN')
- `samples_dir`: Output directory for predictions

### Visualization

Compare predictions from multiple models:

```bash
python visualizemasks.py
```

This script generates a grid visualization comparing:
- Original images
- Ground truth masks
- Predictions from different model variants

### Evaluation

Compute F1 scores and IoU metrics:
- **F-score**: Region similarity measure
- **mIOU**: Mean Intersection over Union for contour accuracy

```bash
cd SUIM/utils
python get_f1_iou.py
```
Output predictions for each model are stored in their respective `Output/` directories, organized by segmentation category.

## References

### Original Paper
```bibtex
@inproceedings{islam2020suim,
  title={{Semantic Segmentation of Underwater Imagery: Dataset and Benchmark}},
  author={Islam, Md Jahidul and Edge, Chelsey and Xiao, Yuyang and Luo, Peigen and Mehtaz, 
          Muntaqim and Morse, Christopher and Enan, Sadman Sakib and Sattar, Junaed},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2020},
  organization={IEEE/RSJ}
}
```

### Links
- **Paper**: [ArXiv](https://arxiv.org/pdf/2004.01241.pdf)
- **Virtual Talk**: [YouTube](https://youtu.be/LxWrhVeIkdg)
- **Dataset**: [SUIM Dataset](http://irvlab.cs.umn.edu/resources/suim-dataset)
- **Original Repository**: [SUIM-dev](https://github.com/xahidbuffon/SUIM)

