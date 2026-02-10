#!/usr/bin/env python3
"""
Script to merge multiple YOLO segmentation datasets.
"""
import argparse
import os
import shutil
from pathlib import Path
from typing import List


def main():
    parser = argparse.ArgumentParser(description="Merge multiple YOLO datasets")
    parser.add_argument("--input-dirs", required=True, nargs="+", help="Input dataset directories")
    parser.add_argument("--out-dir", required=True, help="Output directory for the merged dataset")
    parser.add_argument("--symlink", action="store_true", help="Create symbolic links instead of copying files")

    args = parser.parse_args()

    input_dirs = [Path(d) for d in args.input_dirs]
    out_dir = Path(args.out_dir)

    # Create output directories
    images_train_dir = out_dir / "images" / "train"
    images_val_dir = out_dir / "images" / "val"
    labels_train_dir = out_dir / "labels" / "train"
    labels_val_dir = out_dir / "labels" / "val"

    for d in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Process each input directory
    for input_dir in input_dirs:
        print(f"Processing input directory: {input_dir}")

        # Copy images and labels from train
        train_images = input_dir / "images" / "train"
        train_labels = input_dir / "labels" / "train"

        if train_images.exists() and train_labels.exists():
            for img_file in train_images.glob("*"):
                if args.symlink:
                    os.symlink(img_file, images_train_dir / img_file.name)
                else:
                    shutil.copy2(img_file, images_train_dir / img_file.name)

            for label_file in train_labels.glob("*"):
                if args.symlink:
                    os.symlink(label_file, labels_train_dir / label_file.name)
                else:
                    shutil.copy2(label_file, labels_train_dir / label_file.name)

        # Copy images and labels from val
        val_images = input_dir / "images" / "val"
        val_labels = input_dir / "labels" / "val"

        if val_images.exists() and val_labels.exists():
            for img_file in val_images.glob("*"):
                if args.symlink:
                    os.symlink(img_file, images_val_dir / img_file.name)
                else:
                    shutil.copy2(img_file, images_val_dir / img_file.name)

            for label_file in val_labels.glob("*"):
                if args.symlink:
                    os.symlink(label_file, labels_val_dir / label_file.name)
                else:
                    shutil.copy2(label_file, labels_val_dir / label_file.name)

    # Create dataset.yaml
    dataset_yaml = out_dir / "dataset.yaml"
    dataset_yaml_content = """
path: {out_dir}
train: images/train
val: images/val

# Classes
nc: 1
names: [pool]
""".format(out_dir=out_dir)

    with open(dataset_yaml, "w") as f:
        f.write(dataset_yaml_content.strip())

    print(f"Merged dataset created in {out_dir}")
    print(f"Dataset YAML: {dataset_yaml}")


if __name__ == "__main__":
    main()