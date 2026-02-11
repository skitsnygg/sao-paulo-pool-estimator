from __future__ import annotations

import argparse
import csv
import fnmatch
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from src.models.predict import collect_predictions, write_outputs
from src.utils.config import load_config, project_root


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
DEFAULT_EXCLUDE_PATTERNS = ["*/jardim_europa.tif", "*/wms.xml"]


def _infer_tile_id(path: Path) -> str:
    stem = path.stem
    parent = path.parent.name
    if parent.isdigit() and "_" in stem:
        return f"{parent}_{stem}"
    return stem


def _matches_exclude_patterns(path: Path, patterns: list[str]) -> bool:
    path_str = path.as_posix()
    return any(fnmatch.fnmatch(path_str, pattern) for pattern in patterns)


def _prefer_tiles_over_root(root_dir: Path, path: Path) -> bool:
    relative = path.relative_to(root_dir)
    if len(relative.parts) == 1:
        return (root_dir / "tiles").is_dir()
    if len(relative.parts) == 2:
        neighborhood_dir = root_dir / relative.parts[0]
        return (neighborhood_dir / "tiles").is_dir()
    return False


def _find_images(root_dir: Path, exclude_patterns: list[str]) -> list[Path]:
    images: list[Path] = []
    for path in root_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        if _matches_exclude_patterns(path, exclude_patterns):
            continue
        if _prefer_tiles_over_root(root_dir, path):
            continue
        images.append(path)
    return sorted(images)


def _group_by_neighborhood(root_dir: Path, images: list[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for image in images:
        relative = image.relative_to(root_dir)
        if len(relative.parts) == 1:
            neighborhood = root_dir.name
        else:
            neighborhood = relative.parts[0]
        grouped[neighborhood].append(image)
    return {name: sorted(paths) for name, paths in grouped.items()}


def _apply_area_filter(
    rows: list[dict],
    detections: list[dict],
    min_area_px: float | None,
    max_area_px: float | None,
) -> tuple[list[dict], list[dict]]:
    filtered_detections: list[dict] = []
    if min_area_px is None and max_area_px is None:
        filtered_detections = list(detections)
    else:
        for detection in detections:
            area = detection.get("bbox_area", 0.0)
            if min_area_px is not None and area < min_area_px:
                continue
            if max_area_px is not None and area > max_area_px:
                continue
            filtered_detections.append(detection)

    counts_by_tile = {row["tile_id"]: 0 for row in rows}
    for detection in filtered_detections:
        tile_id = detection["tile_id"]
        counts_by_tile[tile_id] = counts_by_tile.get(tile_id, 0) + 1

    filtered_rows: list[dict] = []
    for row in rows:
        updated = dict(row)
        updated["count"] = counts_by_tile.get(row["tile_id"], 0)
        filtered_rows.append(updated)

    return filtered_rows, filtered_detections


def _compute_starts(size: int, slice_size: int, stride: int) -> list[int]:
    if size <= slice_size:
        return [0]
    starts = list(range(0, size - slice_size + 1, stride))
    if starts[-1] != size - slice_size:
        starts.append(size - slice_size)
    return starts


def _iou(box_a: dict, box_b: dict) -> float:
    x1 = max(box_a["x1"], box_b["x1"])
    y1 = max(box_a["y1"], box_b["y1"])
    x2 = min(box_a["x2"], box_b["x2"])
    y2 = min(box_a["y2"], box_b["y2"])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area == 0.0:
        return 0.0

    area_a = (box_a["x2"] - box_a["x1"]) * (box_a["y2"] - box_a["y1"])
    area_b = (box_b["x2"] - box_b["x1"]) * (box_b["y2"] - box_b["y1"])
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def _nms(detections: list[dict], iou_threshold: float) -> list[dict]:
    if not detections:
        return []
    remaining = sorted(detections, key=lambda det: det["confidence"], reverse=True)
    kept: list[dict] = []
    while remaining:
        current = remaining.pop(0)
        kept.append(current)
        remaining = [det for det in remaining if _iou(current, det) <= iou_threshold]
    return kept


def _nms_by_tile(detections: list[dict], iou_threshold: float) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for det in detections:
        grouped[det["tile_id"]].append(det)

    merged: list[dict] = []
    for tile_detections in grouped.values():
        merged.extend(_nms(tile_detections, iou_threshold))
    return merged


def _collect_predictions_sliced(
    model,
    sources: list[Path],
    config: dict,
    slice_size: int,
    slice_overlap: float,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    detections: list[dict] = []
    iou_threshold = config["inference"].get("iou", 0.5)
    conf_threshold = config["inference"].get("conf", 0.25)
    max_det = config["inference"].get("max_det", 300)

    stride = max(1, int(slice_size * (1 - slice_overlap)))

    for path in sources:
        tile_id = _infer_tile_id(path)
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        width, height = image.size

        if width <= slice_size and height <= slice_size:
            slices = [image]
            offsets = [(0, 0)]
        else:
            x_starts = _compute_starts(width, slice_size, stride)
            y_starts = _compute_starts(height, slice_size, stride)
            slices = []
            offsets = []
            for y in y_starts:
                for x in x_starts:
                    crop = image.crop((x, y, x + slice_size, y + slice_size))
                    slices.append(crop)
                    offsets.append((x, y))

        results_iter = model.predict(
            source=slices,
            conf=conf_threshold,
            iou=iou_threshold,
            max_det=max_det,
            stream=True,
        )

        tile_detections: list[dict] = []
        for result, (x_offset, y_offset) in zip(results_iter, offsets):
            if result.boxes is None or len(result.boxes) == 0:
                continue
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                x1 += x_offset
                x2 += x_offset
                y1 += y_offset
                y2 += y_offset
                x1 = max(0.0, min(x1, width))
                x2 = max(0.0, min(x2, width))
                y1 = max(0.0, min(y1, height))
                y2 = max(0.0, min(y2, height))
                if x2 <= x1 or y2 <= y1:
                    continue
                area = (x2 - x1) * (y2 - y1)
                tile_detections.append(
                    {
                        "tile_id": tile_id,
                        "confidence": float(box.conf[0].cpu().numpy()),
                        "bbox_area": float(area),
                        "path": str(path),
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                    }
                )

        rows.append({"tile_id": tile_id, "count": 0, "path": str(path)})
        detections.extend(tile_detections)

    return rows, detections


def _save_annotated_images(
    model,
    sources: list[Path],
    output_dir: Path,
    config: dict,
    min_area_px: float | None,
    max_area_px: float | None,
) -> None:
    if not sources:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    results_iter = model.predict(
        source=[str(path) for path in sources],
        conf=config["inference"].get("conf", 0.25),
        iou=config["inference"].get("iou", 0.5),
        max_det=config["inference"].get("max_det", 300),
        stream=True,
    )

    for result in results_iter:
        if result.boxes is None or len(result.boxes) == 0:
            continue

        xyxy = result.boxes.xyxy.cpu().numpy()
        filtered_boxes: list[tuple[float, float, float, float]] = []
        for box in xyxy:
            x1, y1, x2, y2 = box.tolist()
            area = (x2 - x1) * (y2 - y1)
            if min_area_px is not None and area < min_area_px:
                continue
            if max_area_px is not None and area > max_area_px:
                continue
            filtered_boxes.append((x1, y1, x2, y2))

        if not filtered_boxes:
            continue

        image = None
        try:
            with Image.open(result.path) as opened:
                image = opened.convert("RGB")
        except Exception:
            if result.orig_img is not None:
                array = result.orig_img
                if array.ndim == 2:
                    image = Image.fromarray(array)
                else:
                    image = Image.fromarray(array[..., ::-1])

        if image is None:
            continue

        draw = ImageDraw.Draw(image)
        width, height = image.size
        for x1, y1, x2, y2 in filtered_boxes:
            x1 = max(0.0, min(x1, width))
            x2 = max(0.0, min(x2, width))
            y1 = max(0.0, min(y1, height))
            y2 = max(0.0, min(y2, height))
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        output_path = output_dir / Path(result.path).name
        image.save(output_path)


def _save_annotated_images_from_detections(
    detections: list[dict],
    output_dir: Path,
) -> None:
    if not detections:
        return

    grouped: dict[str, list[dict]] = defaultdict(list)
    for detection in detections:
        if not {"x1", "y1", "x2", "y2"}.issubset(detection):
            continue
        grouped[detection["path"]].append(detection)

    if not grouped:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for path_str, dets in grouped.items():
        path = Path(path_str)
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size
        for det in dets:
            x1 = max(0.0, min(det["x1"], width))
            x2 = max(0.0, min(det["x2"], width))
            y1 = max(0.0, min(det["y1"], height))
            y2 = max(0.0, min(det["y2"], height))
            if x2 <= x1 or y2 <= y1:
                continue
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        image.save(output_dir / path.name)


def _write_summary(
    output_dir: Path,
    neighborhood: str,
    rows: list[dict],
    detections: list[dict],
) -> None:
    summary_path = output_dir / "summary.csv"
    tiles = len(rows)
    detections_count = len(detections)
    pools_sum = sum(row["count"] for row in rows)
    if detections_count > 0:
        mean_confidence = sum(det["confidence"] for det in detections) / detections_count
        mean_bbox_area = sum(det["bbox_area"] for det in detections) / detections_count
    else:
        mean_confidence = 0.0
        mean_bbox_area = 0.0

    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "neighborhood",
                "tiles",
                "detections",
                "pools_sum",
                "mean_confidence",
                "mean_bbox_area",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "neighborhood": neighborhood,
                "tiles": tiles,
                "detections": detections_count,
                "pools_sum": pools_sum,
                "mean_confidence": f"{mean_confidence:.6f}",
                "mean_bbox_area": f"{mean_bbox_area:.6f}",
            }
        )


def run_batch_inference(
    input_dir: Path,
    config_path: str,
    weights: str | None,
    min_area_px: float | None,
    max_area_px: float | None,
    save_images: bool,
    slice_size: int | None,
    slice_overlap: float,
    exclude_patterns: list[str],
) -> None:
    config = load_config(config_path)
    root = project_root()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is required. Install requirements-train.txt") from exc

    if weights is None:
        weights = str(root / "checkpoints" / "best.pt")

    model = YOLO(weights)

    images = _find_images(input_dir, exclude_patterns)
    if not images:
        raise FileNotFoundError(f"No images found under {input_dir}")

    grouped = _group_by_neighborhood(input_dir, images)
    output_root = root / "data" / "processed" / "batch"

    for neighborhood in sorted(grouped.keys()):
        neighborhood_paths = grouped[neighborhood]
        sources = [str(path) for path in neighborhood_paths]
        if slice_size is None:
            rows, detections = collect_predictions(model, sources, {}, config)
        else:
            rows, detections = _collect_predictions_sliced(
                model,
                neighborhood_paths,
                config,
                slice_size,
                slice_overlap,
            )
            detections = _nms_by_tile(detections, 0.5)
            rows, detections = _apply_area_filter(rows, detections, min_area_px, max_area_px)
        if slice_size is None:
            rows, detections = _apply_area_filter(rows, detections, min_area_px, max_area_px)

        output_dir = output_root / neighborhood
        predictions_path = output_dir / "predictions.csv"
        write_outputs(
            root,
            rows,
            detections,
            output_dir=output_dir,
            predictions_path=predictions_path,
            demo_mode_used=False,
            demo_fallback_type="none",
        )
        _write_summary(output_dir, neighborhood, rows, detections)

        if save_images:
            annotated_dir = output_dir / "annotated"
            if slice_size is None:
                positive_sources = [Path(row["path"]) for row in rows if row["count"] > 0]
                _save_annotated_images(
                    model,
                    positive_sources,
                    annotated_dir,
                    config,
                    min_area_px,
                    max_area_px,
                )
            else:
                _save_annotated_images_from_detections(detections, annotated_dir)

        print(f"{neighborhood}: tiles={len(rows)} detections={len(detections)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch inference across neighborhood tiles.")
    parser.add_argument("--input-dir", required=True, help="Top-level directory of neighborhood tile folders.")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--min-area-px", type=float, default=None)
    parser.add_argument("--max-area-px", type=float, default=None)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--slice-size", type=int, default=None)
    parser.add_argument("--slice-overlap", type=float, default=0.2)
    parser.add_argument("--exclude-pattern", action="append", default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    if args.min_area_px is not None and args.max_area_px is not None and args.min_area_px > args.max_area_px:
        raise ValueError("min-area-px must be less than or equal to max-area-px")
    if args.slice_size is not None and args.slice_size <= 0:
        raise ValueError("slice-size must be a positive integer")
    if args.slice_overlap < 0 or args.slice_overlap >= 1:
        raise ValueError("slice-overlap must be in [0, 1)")

    if args.exclude_pattern:
        exclude_patterns = DEFAULT_EXCLUDE_PATTERNS + args.exclude_pattern
    else:
        exclude_patterns = list(DEFAULT_EXCLUDE_PATTERNS)

    run_batch_inference(
        input_dir=input_dir,
        config_path=args.config,
        weights=args.weights,
        min_area_px=args.min_area_px,
        max_area_px=args.max_area_px,
        save_images=args.save_images,
        slice_size=args.slice_size,
        slice_overlap=args.slice_overlap,
        exclude_patterns=exclude_patterns,
    )


if __name__ == "__main__":
    main()
