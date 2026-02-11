#!/usr/bin/env python3
"""
Script to run segmentation inference on orthophoto tiles and save results.

This script:
1. Loads a trained YOLO segmentation model
2. Accepts an input directory of orthophoto tiles
3. Runs segmentation inference on all images
4. Saves annotated images, YOLO txt outputs, GeoJSON polygons, and CSV outputs
5. Writes pool counts and a run manifest for reproducibility
6. Uses batch processing where possible with deterministic settings
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from shapely.geometry import Polygon, mapping

from src.utils.config import load_config

if TYPE_CHECKING:
    from ultralytics import YOLO

try:
    import cv2
except Exception:
    cv2 = None

GeoTransform = Tuple[float, float, float, float, float, float]


def set_deterministic(seed: int) -> Dict[str, object]:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)

    deterministic_info: Dict[str, object] = {"seed": seed}
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        deterministic_info["torch_deterministic"] = True
    except Exception as exc:
        deterministic_info["torch_deterministic"] = False
        deterministic_info["torch_error"] = str(exc)

    return deterministic_info


def get_geotiff_georeference(image_path: Path) -> Tuple[Optional[GeoTransform], Optional[str]]:
    if image_path.suffix.lower() not in {".tif", ".tiff"}:
        return None, None

    try:
        import rasterio
    except Exception:
        rasterio = None

    if rasterio is not None:
        try:
            with rasterio.open(image_path) as src:
                if src.crs and src.transform:
                    return src.transform.to_gdal(), src.crs.to_string()
        except Exception:
            pass

    try:
        from osgeo import gdal
    except Exception:
        return None, None

    try:
        dataset = gdal.Open(str(image_path), gdal.GA_ReadOnly)
        if dataset is None:
            return None, None
        geotransform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()
        if geotransform and projection:
            return geotransform, projection
    except Exception:
        return None, None

    return None, None


def pixel_coords_to_geo(coords: np.ndarray, transform: GeoTransform) -> List[List[float]]:
    origin_x, pixel_w, rot_x, origin_y, rot_y, pixel_h = transform
    geo_coords = []
    for x, y in coords:
        geo_x = origin_x + x * pixel_w + y * rot_x
        geo_y = origin_y + x * rot_y + y * pixel_h
        geo_coords.append([geo_x, geo_y])
    return geo_coords


def write_geojson(features: List[Dict[str, object]], output_path: Path, crs: Optional[str]) -> None:
    collection: Dict[str, object] = {"type": "FeatureCollection", "features": features}
    if crs:
        collection["crs"] = {"type": "name", "properties": {"name": crs}}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(collection, f, indent=2)


def write_run_manifest(output_dir: Path, manifest: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)


def safe_relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def save_segmentation_result(
    model: YOLO,
    image_path: Path,
    source_root: Path,
    output_dir: Path,
    conf: float,
    iou: float,
    max_det: int,
    save_images: bool = True,
    save_labels: bool = True,
    save_geojson: bool = True,
    min_area_px: int = 0,
    max_area_px: int = 0,
) -> Tuple[List[Tuple[str, float, float]], int]:
    """
    Run inference on a single image and save results.

    Args:
        model: Loaded YOLO model
        image_path: Path to input image
        source_root: Root directory for input images
        output_dir: Directory to save results
        conf: Confidence threshold for inference
        iou: IoU threshold for inference
        max_det: Maximum detections per image
        save_images: Whether to save annotated images
        save_labels: Whether to save YOLO format labels
        save_geojson: Whether to save GeoJSON polygons per image

    Returns:
        Tuple of detections list and pool count
    """
    # Run inference
    results = model.predict(
        source=str(image_path),
        conf=conf,
        iou=iou,
        max_det=max_det,
        stream=False,
    )

    result = results[0]  # Get first (and only) result

    # Process detections
    detections = []
    image_name = image_path.stem
    relative_path = safe_relative_path(image_path, source_root)

    if result.boxes is not None and len(result.boxes) > 0:
        image_size = (result.orig_shape[1], result.orig_shape[0])  # width, height

        # Save annotated image if requested
        if save_images:
            if cv2 is None:
                raise RuntimeError("cv2 is required to save annotated images; install opencv-python or disable --save-images.")
            annotated_image = result.plot()
            annotated_path = output_dir / "images" / f"{image_name}_annotated.jpg"
            annotated_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(annotated_path), annotated_image)

        label_file = None
        if save_labels:
            label_path = output_dir / "labels" / f"{image_name}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_file = open(label_path, "w")

        try:
            for box in result.boxes:
                # Get box coordinates (xyxy format)
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy

                # Calculate confidence and area
                confidence = float(box.conf[0].cpu().numpy())
                area = float((x2 - x1) * (y2 - y1))

                # Apply area filtering if specified
                if (min_area_px > 0 and area < min_area_px) or (max_area_px > 0 and area > max_area_px):
                    continue

                if label_file is not None:
                    # Convert to YOLO format (cx, cy, w, h)
                    cx = (x1 + x2) / 2 / image_size[0]
                    cy = (y1 + y2) / 2 / image_size[1]
                    w = (x2 - x1) / image_size[0]
                    h = (y2 - y1) / image_size[1]

                    # Write in YOLO format (class_id, cx, cy, w, h)
                    label_file.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

                detections.append((image_name, confidence, area))
        finally:
            if label_file is not None:
                label_file.close()

    # Build GeoJSON features from segmentation masks
    features: List[Dict[str, object]] = []
    geotransform, crs = get_geotiff_georeference(image_path)
    coord_system = "geographic" if geotransform and crs else "pixel"

    if result.masks is not None and result.masks.xy:
        mask_polygons = result.masks.xy
        boxes = list(result.boxes) if result.boxes is not None else []

        for idx, mask_coords in enumerate(mask_polygons):
            if mask_coords is None or len(mask_coords) < 3:
                continue
            coords_array = np.asarray(mask_coords)
            if coords_array.shape[0] < 3:
                continue

            if geotransform and crs:
                coords = pixel_coords_to_geo(coords_array, geotransform)
            else:
                coords = coords_array.tolist()

            polygon = Polygon(coords)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if polygon.is_empty:
                continue

            confidence = None
            area = None
            if idx < len(boxes):
                box = boxes[idx]
                confidence = float(box.conf[0].cpu().numpy())
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                area = float((x2 - x1) * (y2 - y1))

            feature = {
                "type": "Feature",
                "geometry": mapping(polygon),
                "properties": {
                    "image_name": image_name,
                    "image_path": relative_path,
                    "pool_index": idx,
                    "confidence": confidence,
                    "bbox_area_px": area,
                    "mask_area": float(polygon.area),
                    "coord_system": coord_system,
                    "crs": crs if crs else None,
                },
            }
            features.append(feature)

    if save_geojson:
        geojson_path = output_dir / "geojson" / f"{image_name}.geojson"
        write_geojson(features, geojson_path, crs if coord_system == "geographic" else None)

    pool_count = len(result.boxes) if result.boxes is not None else len(features)
    return detections, pool_count


def main():
    parser = argparse.ArgumentParser(description="Run segmentation inference on orthophoto tiles")
    parser.add_argument("--weights", default="runs/segment/sao-paulo-pools-seg-v3/weights/best.pt",
                        help="Path to the trained model weights (default: runs/segment/sao-paulo-pools-seg-v3/weights/best.pt)")
    parser.add_argument("--source", required=True,
                        help="Input directory containing orthophoto tiles")
    parser.add_argument("--output", default="data/processed/predictions",
                        help="Output directory for results (default: data/processed/predictions)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for inference (default: 8)")
    parser.add_argument("--conf", type=float, default=None,
                        help="Confidence threshold for inference (default: configs/project.yaml inference.conf)")
    parser.add_argument("--iou", type=float, default=None,
                        help="IoU threshold for inference (default: configs/project.yaml inference.iou)")
    parser.add_argument("--max-det", type=int, default=None,
                        help="Maximum detections per image (default: configs/project.yaml inference.max_det)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for deterministic inference (default: configs/project.yaml dataset.random_seed)")
    parser.add_argument("--save-images", action="store_true",
                        help="Save annotated images")
    parser.add_argument("--save-labels", action="store_true",
                        help="Save YOLO format labels")
    parser.add_argument("--save-csv", action="store_true",
                        help="Save CSV with detections")
    parser.add_argument("--min-area-px", type=int, default=0,
                        help="Minimum area (in pixels) for detections to be saved (default: 0, no minimum)")
    parser.add_argument("--max-area-px", type=int, default=0,
                        help="Maximum area (in pixels) for detections to be saved (default: 0, no maximum)")

    args = parser.parse_args()

    # Load configuration
    config = load_config("configs/project.yaml")
    inference_cfg = config.get("inference", {})
    dataset_cfg = config.get("dataset", {})

    # Validate input
    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_path}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    conf = args.conf if args.conf is not None else inference_cfg.get("conf", 0.25)
    iou = args.iou if args.iou is not None else inference_cfg.get("iou", 0.5)
    max_det = args.max_det if args.max_det is not None else inference_cfg.get("max_det", 300)
    seed = args.seed if args.seed is not None else dataset_cfg.get("random_seed", 42)
    save_images = args.save_images
    if save_images and cv2 is None:
        print("cv2 not available; disabling annotated image output.")
        save_images = False

    deterministic_info = set_deterministic(seed)

    from ultralytics import YOLO

    # Load model
    print(f"Loading model from {args.weights}")
    try:
        model = YOLO(args.weights)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {args.weights}: {e}")

    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    image_files = []

    for file_path in source_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)

    if not image_files:
        raise FileNotFoundError(f"No image files found in {source_path}")

    image_files = sorted(image_files)
    print(f"Found {len(image_files)} image files")

    manifest = {
        "run_date": datetime.now(timezone.utc).isoformat(),
        "model_path": str(Path(args.weights).resolve()),
        "source": str(source_path.resolve()),
        "output": str(output_dir.resolve()),
        "image_count": len(image_files),
        "parameters": {
            "batch_size": args.batch_size,
            "conf": conf,
            "iou": iou,
            "max_det": max_det,
            "seed": seed,
            "save_images": save_images,
            "save_labels": args.save_labels,
            "save_csv": args.save_csv,
            "save_geojson": True,
        },
        "deterministic": deterministic_info,
        "command": " ".join(sys.argv),
    }
    write_run_manifest(output_dir, manifest)

    # Process images in batches
    all_detections = []
    pool_counts = []
    total_batches = (len(image_files) + args.batch_size - 1) // args.batch_size

    for i in range(0, len(image_files), args.batch_size):
        batch_files = image_files[i:i + args.batch_size]
        print(f"Processing batch {i//args.batch_size + 1} of {total_batches}")

        # For segmentation, we process one image at a time to maintain control
        for image_path in batch_files:
            detections = []
            pool_count = 0
            try:
                detections, pool_count = save_segmentation_result(
                    model,
                    image_path,
                    source_path,
                    output_dir,
                    conf,
                    iou,
                    max_det,
                    save_images,
                    args.save_labels,
                    True,
                    args.min_area_px,
                    args.max_area_px,
                )
                all_detections.extend(detections)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
            pool_counts.append((image_path.stem, safe_relative_path(image_path, source_path), pool_count))

    # Save CSV with all detections
    if args.save_csv and all_detections:
        csv_path = output_dir / "detections.csv"
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['image_name', 'confidence', 'bbox_area'])
            writer.writerows(all_detections)
        print(f"Saved detections to {csv_path}")

    # Save pool counts per image
    counts_path = output_dir / "pool_counts.csv"
    with open(counts_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_name", "image_path", "pool_count"])
        writer.writerows(pool_counts)
    print(f"Saved pool counts to {counts_path}")

    print(f"Completed processing {len(all_detections)} detections")


if __name__ == "__main__":
    main()
