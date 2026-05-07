# 🌌 FIIF-Net: Focus Information Interaction Fusion Network

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[![DOI](https://zenodo.org/badge/1230791790.svg)](https://doi.org/10.5281/zenodo.20065381)

> Official PyTorch implementation of the paper: **"Multi-focus Image Fusion with Alignment and Interactive Weighting for Cosmic Dust Microscopic Imaging"**.

---

## 📌 About & Citation

This code is directly related to the manuscript currently submitted to ***The Visual Computer***. 

If you find our code, dataset, or framework useful for your research, we kindly request that you cite our paper:

```bibtex
@article{xian2026fiifnet,
  title={Multi-focus Image Fusion with Alignment and Interactive Weighting for Cosmic Dust Microscopic Imaging},
  author={Xian, Yongli and Gong, Zhijie and Zhao, Guangxin and Wang, Congzheng and Zhao, Chengxuan},
  journal={Submitted to The Visual Computer},
  year={2026}
}
```

---

## 🌟 Network Architecture

FIIF-Net is designed to address the challenges of cosmic dust microscopic imaging through three specialized sub-networks:

*   🎯 **Alignment-Net**: An optical flow-based alignment network using the RAFT paradigm, enhanced with **RFDP** (Residual Feature Downsampling Pyramid) and **DLO** (Dynamic Lookup Operator).
*   🔍 **Focus-Net**: A single image focus estimation network featuring an encoder-decoder architecture, **MHSA** (Multi-Head Self-Attention), and auxiliary focus map predictions.
*   ✨ **Fusion-Net**: A dual-path interactive fusion network equipped with the **STA** (Super Token Attention) module and ZSSR for high-quality super-resolution reconstruction.

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/gxiaopang/FIIF-Net.git
cd FIIF-Net

# Install dependencies
pip install -r requirements.txt
```

## 📂 Dataset Preparation

> **Note on Data Availability:** Due to confidentiality constraints, the full dataset is not publicly available at this time. We provide a limited subset for project demonstration.

Please organize the dataset using the following directory layout:

```text
dataset/
├── train/
│   ├── images/         # Input defocus images (scene_xxx/defocus_x.tif)
│   ├── focusmap/       # Ground truth focus maps (scene_xxx/focusmap_x.png)
│   └── labels/         # All-in-focus reference images (scene_xxx/all_in_focus_1.tif)
└── test/
    └── (Same structure as train)
```

## 🚀 Training

We support both step-by-step training and a unified full-pipeline training mode.

### Option A: Full Pipeline (Recommended)

Train both stages automatically with a single command:

```bash
python train.py --stage both --train_dir ./dataset/train --val_dir ./dataset/test
```

### Option B: Step-by-Step Training

**Stage 1: Individual Pre-training**

```bash
# 1. Pre-train Alignment-Net
python train_alignment.py --train_dir ./dataset/train --epochs 100 --batch_size 18 --lr 0.0001

# 2. Pre-train Focus-Net
python train_focus.py --train_dir ./dataset/train --epochs 50 --batch_size 2 --lr 0.0001
```

**Stage 2: Joint Training**

```bash
# Train Fusion-Net using pre-trained weights from Stage 1
python train_fusion.py \
    --train_dir ./dataset/train \
    --val_dir ./dataset/test \
    --alignment_pth ./model_pth/alignment_net_best.pth \
    --focus_pth ./model_pth/focus_net_best.pth \
    --batch_size 64 \
    --epochs 50
```

## 🖼️ Inference

To test the model on your own images using pre-trained weights:

```bash
python inference.py \
    --test_dir ./dataset/test \
    --output_dir ./results \
    --checkpoint ./model_pth/fiif_net_best.pth
```

## 📖 Project Structure

<details>
<summary>Click to expand the detailed project structure</summary>

```text
FIIF-Net/
├── model/
│   ├── __init__.py           
│   ├── common.py             # Common modules (WFD, ConvBlock, BasicBlock, etc.)
│   ├── alignment_net.py      # Alignment-Net (RFDP & DLO)
│   ├── focus_net.py          # Focus-Net (WFD & auxiliary branches)
│   ├── fusion_net.py         # Fusion-Net (STA & dual-path interactive weighting)
│   ├── sta.py                # Super Token Attention module
│   ├── zssr.py               # Zero-Shot Super-Resolution
│   └── fiif_net.py           # Overall FIIF-Net pipeline
├── dataset.py                # Dataset classes for all training stages
├── loss.py                   # Loss functions (FocusNetLoss, FusionNetLoss, MEFSSIM, VGG)
├── train.py                  # Unified training script
├── train_focus.py            # Stage 1: Focus-Net training script
├── train_alignment.py        # Stage 1: Alignment-Net training script
├── train_fusion.py           # Stage 2: Joint training script
├── inference.py              # Inference pipeline
├── requirements.txt          # Dependencies
└── model_pth/                # Directory for saving model weights
```

