# Pseudo-Label Dataset Tools

This directory contains tools for building pseudo-label datasets from model predictions.

## build_multineighborhood_pseudolabel_dataset.py

This script builds a YOLO segmentation dataset from pseudo-labels generated across multiple neighborhoods.

### Usage

```bash
python build_multineighborhood_pseudolabel_dataset.py \
    --neighborhoods-dir runs/segment \
    --out-dir data/pseudolabels_multineighborhood \
    --val-split 0.2
```

### Arguments

- `--neighborhoods-dir`: Directory containing neighborhood pseudo-labels (each neighborhood should be a subdirectory)
- `--out-dir`: Output directory for the dataset
- `--symlink`: Create symbolic links instead of copying files (optional)
- `--val-split`: Validation split ratio (default: 0.2)

### Dataset Structure

The script creates a standard YOLO dataset structure:

```
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── dataset.yaml
```

### Neighborhood Directory Structure

Each neighborhood directory should contain:
- Images directly in the directory (e.g., `r0000_c0000.jpg`)
- A `labels` subdirectory with corresponding label files (e.g., `r0000_c0000.txt`)

The script will:
1. Collect all images and labels from all neighborhoods
2. Shuffle them randomly (with seed=0 for reproducibility)
3. Split into train/validation sets (default 80/20 split)
4. Create symbolic links or copy files based on the `--symlink` flag
5. Generate a `dataset.yaml` file with the proper configuration

### Notes

- Not all images will have corresponding labels (negative samples are skipped)
- The script handles multiple neighborhoods automatically
- The dataset is configured for a single class (pool)