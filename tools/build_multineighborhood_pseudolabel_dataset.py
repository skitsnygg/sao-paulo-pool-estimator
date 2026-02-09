#!/usr/bin/env python3
"""
Script to build pseudo-label YOLO segmentation datasets from multiple neighborhoods.
"""
import argparse
import os
import shutil
import random
import glob
from pathlib import Path
from typing import List, Dict


def main():
    parser = argparse.ArgumentParser(description="Build pseudo-label YOLO dataset from multiple neighborhoods")
    parser.add_argument("--neighborhoods-dir", required=True, help="Directory containing neighborhood pseudo-labels")
    parser.add_argument("--out-dir", required=True, help="Output directory for the dataset")
    parser.add_argument("--symlink", action="store_true", help="Create symbolic links instead of copying files")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio (default: 0.2)")

    args = parser.parse_args()

    neighborhoods_dir = Path(args.neighborhoods_dir)
    out_dir = Path(args.out_dir)
    val_split = args.val_split

    # Create output directories
    images_train_dir = out_dir / "images" / "train"
    images_val_dir = out_dir / "images" / "val"
    labels_train_dir = out_dir / "labels" / "train"
    labels_val_dir = out_dir / "labels" / "val"

    for d in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Get all neighborhood directories - make sure we're not including the labels subdirectory
    neighborhood_dirs = []
    for d in neighborhoods_dir.iterdir():
        if d.is_dir() and d.name != "labels":
            neighborhood_dirs.append(d)

    print(f"Found {len(neighborhood_dirs)} neighborhoods: {[d.name for d in neighborhood_dirs]}")

    # Collect all image files from all neighborhoods
    all_image_files = []
    all_label_files = []

    for neighborhood_dir in neighborhood_dirs:
        print(f"Processing neighborhood: {neighborhood_dir.name}")

        # Get all image files directly in neighborhood directory
        image_files = list(neighborhood_dir.glob("*.png")) + list(neighborhood_dir.glob("*.jpg"))
        image_files = sorted(image_files)

        if not image_files:
            print(f"Warning: No image files found in {neighborhood_dir}")
            continue

        print(f"  Found {len(image_files)} images in {neighborhood_dir.name}")

        # Get labels directory for this neighborhood
        labels_dir = neighborhood_dir / "labels"
        if not labels_dir.exists():
            print(f"Warning: Labels directory not found in {neighborhood_dir}")
            continue

        # Get all label files
        label_files = list(labels_dir.glob("*.txt"))
        label_files = sorted(label_files)

        print(f"  Found {len(label_files)} labels in {neighborhood_dir.name}")

        # Create mapping from image name to label file
        label_map = {label_file.stem: label_file for label_file in label_files}

        # Add to our collections
        for image_file in image_files:
            # Check if there's a corresponding label file
            label_file = label_map.get(image_file.stem)
            if label_file:
                all_image_files.append(image_file)
                all_label_files.append(label_file)
            # We don't warn here since it's expected that not all images have labels

    if not all_image_files:
        raise ValueError("No image files found in any neighborhood")

    print(f"Total images: {len(all_image_files)}")
    print(f"Total labels: {len(all_label_files)}")

    # Shuffle and split images (with seed=0 for reproducibility)
    random.seed(0)
    combined = list(zip(all_image_files, all_label_files))
    random.shuffle(combined)
    all_image_files, all_label_files = zip(*combined)

    split_idx = int((1 - val_split) * len(all_image_files))
    train_images = all_image_files[:split_idx]
    val_images = all_image_files[split_idx:]
    train_labels = all_label_files[:split_idx]
    val_labels = all_label_files[split_idx:]

    print(f"Train images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")

    # Process images and labels
    def process_images_and_labels(image_list, label_list, image_out_dir, label_out_dir):
        for image_file, label_file in zip(image_list, label_list):
            # Copy or symlink image
            if args.symlink:
                os.symlink(image_file, image_out_dir / image_file.name)
            else:
                shutil.copy2(image_file, image_out_dir / image_file.name)

            # Copy or symlink label
            if args.symlink:
                os.symlink(label_file, label_out_dir / label_file.name)
            else:
                shutil.copy2(label_file, label_out_dir / label_file.name)

    # Process training images
    process_images_and_labels(train_images, train_labels, images_train_dir, labels_train_dir)

    # Process validation images
    process_images_and_labels(val_images, val_labels, images_val_dir, labels_val_dir)

    # Create dataset.yaml
    dataset_yaml = out_dir / "dataset.yaml"
    dataset_yaml_content = f"""
path: {out_dir}
train: images/train
val: images/val

# Classes
nc: 1
names: [pool]
"""

    with open(dataset_yaml, "w") as f:
        f.write(dataset_yaml_content.strip())

    print(f"Dataset created in {out_dir}")
    print(f"Dataset YAML: {dataset_yaml}")


if __name__ == "__main__":
    main()