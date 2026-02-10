#!/usr/bin/env python3
"""
Script to build pseudo-label YOLO segmentation datasets from model predictions.
"""
import argparse
import os
import shutil
import random
from pathlib import Path
from typing import List


def main():
    parser = argparse.ArgumentParser(description="Build pseudo-label YOLO dataset")
    parser.add_argument("--images-dir", required=True, help="Directory with images")
    parser.add_argument("--pred-labels-dir", required=True, help="Directory with prediction labels (.txt files)")
    parser.add_argument("--out-dir", required=True, help="Output directory for the dataset")
    parser.add_argument("--symlink", action="store_true", help="Create symbolic links instead of copying files")

    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    pred_labels_dir = Path(args.pred_labels_dir)
    out_dir = Path(args.out_dir)

    # Create output directories
    images_train_dir = out_dir / "images" / "train"
    images_val_dir = out_dir / "images" / "val"
    labels_train_dir = out_dir / "labels" / "train"
    labels_val_dir = out_dir / "labels" / "val"

    for d in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    image_files = sorted(image_files)

    if not image_files:
        raise ValueError(f"No image files found in {images_dir}")

    # Shuffle and split images (80/20 split with seed=0)
    random.seed(0)
    random.shuffle(image_files)

    split_idx = int(0.8 * len(image_files))
    train_images = image_files[:split_idx]
    val_images = image_files[split_idx:]

    print(f"Total images: {len(image_files)}")
    print(f"Train images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")

    # Process images and labels
    def process_images_and_labels(image_list, image_out_dir, label_out_dir):
        for image_file in image_list:
            # Copy or symlink image
            if args.symlink:
                os.symlink(image_file, image_out_dir / image_file.name)
            else:
                shutil.copy2(image_file, image_out_dir / image_file.name)

            # Check if there's a corresponding label file
            label_file = pred_labels_dir / f"{image_file.stem}.txt"
            if label_file.exists():
                # Copy or symlink label
                if args.symlink:
                    os.symlink(label_file, label_out_dir / label_file.name)
                else:
                    shutil.copy2(label_file, label_out_dir / label_file.name)
            # If no label file exists, we skip it (negative sample)

    # Process training images
    process_images_and_labels(train_images, images_train_dir, labels_train_dir)

    # Process validation images
    process_images_and_labels(val_images, images_val_dir, labels_val_dir)

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