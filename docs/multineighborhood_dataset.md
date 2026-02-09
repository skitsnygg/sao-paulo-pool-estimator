# Multi-Neighborhood Pseudo-Label Dataset Creation

This document describes the process of creating a multi-neighborhood pseudo-label dataset for pool detection.

## Overview

We have generated pseudo-labels for multiple neighborhoods in São Paulo using a segmentation model. This document outlines how to combine these pseudo-labels into a single dataset that can be used for training.

## Dataset Structure

The pseudo-labels are organized in the following structure:
```
runs/
└── segment/
    ├── brooklin_v3_pseudolabels_conf015/
    ├── moema_v3_pseudolabels_conf015/
    ├── pinheiros_v3_pseudolabels_conf015/
    ├── sao-paulo-pools-seg-v3/
    ├── sao-paulo-pools-seg-v4/
    ├── v3_preds_jardins/
    └── vila_olimpia_v3_pseudolabels_conf015/
```

Each neighborhood directory contains:
- Images directly in the directory (e.g., `r0000_c0000.jpg`)
- A `labels` subdirectory with corresponding label files (e.g., `r0000_c0000.txt`)

## Building the Dataset

We created a script to build a unified dataset from all neighborhoods:

```bash
python tools/build_multineighborhood_pseudolabel_dataset.py \
    --neighborhoods-dir runs/segment \
    --out-dir data/pseudolabels_multineighborhood \
    --val-split 0.2
```

This creates a dataset with:
- 616 total images (from all neighborhoods)
- 80% training split (492 images)
- 20% validation split (124 images)
- YOLO segmentation format with class ID 0 (pool)

## Dataset Format

The dataset uses the standard YOLO segmentation format:
- First value: class ID (0 for pool)
- Following values: x, y coordinate pairs (normalized to [0,1])

Example label file content:
```
0 0.451172 0.0214844 0.451172 0.0253906 0.450195 0.0263672 ...
```

## Usage

The resulting dataset can be used directly with YOLOv8 segmentation models:

```python
from ultralytics import YOLO

# Train the model
model = YOLO('yolov8s-seg.pt')  # Start with a pre-trained model
model.train(data='data/pseudolabels_multineighborhood/dataset.yaml', epochs=100)
```

## Validation

The dataset has been validated to ensure:
- Proper YOLO format
- Correct class mapping
- Proper train/validation split
- Consistent directory structure