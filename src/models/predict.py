from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

from src.utils.config import load_config, project_root


def _load_tiles_csv(tiles_csv: Path) -> tuple[list[str], dict[str, str]]:
    df = pd.read_csv(tiles_csv)
    if "path" not in df.columns:
        raise ValueError("tiles_csv must include a path column.")
    tile_id_by_path = {}
    if "tile_id" in df.columns:
        tile_id_by_path = dict(zip(df["path"], df["tile_id"]))
    sources = df["path"].tolist()
    return sources, tile_id_by_path


def _infer_tile_id(path: Path) -> str:
    stem = path.stem
    parent = path.parent.name
    if parent.isdigit() and "_" in stem:
        return f"{parent}_{stem}"
    return stem


def _positive_sample_dir(root: Path) -> Path:
    return root / "data" / "samples" / "sp_pools_positive"


def _positive_sample_labels_dir(root: Path) -> Path:
    return root / "data" / "processed" / "yolo" / "labels" / "train"


def _list_sample_images(sample_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(p for p in sample_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)


def _collect_predictions(model, sources, tile_id_by_path: dict[str, str], config: dict) -> tuple[list[dict], list[dict]]:
    results_iter = model.predict(
        source=sources,
        conf=config["inference"].get("conf", 0.25),
        iou=config["inference"].get("iou", 0.5),
        max_det=config["inference"].get("max_det", 300),
        stream=True,
    )

    rows: list[dict] = []
    detections: list[dict] = []

    for result in results_iter:
        path = Path(result.path)
        tile_id = tile_id_by_path.get(str(path), _infer_tile_id(path))
        count = len(result.boxes) if result.boxes is not None else 0
        rows.append({"tile_id": tile_id, "count": count, "path": str(path)})

        if result.boxes is None:
            continue
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy
            detections.append(
                {
                    "tile_id": tile_id,
                    "confidence": float(box.conf[0].cpu().numpy()),
                    "bbox_area": float((x2 - x1) * (y2 - y1)),
                    "path": str(path),
                }
            )

    return rows, detections


def _collect_label_fallback(sample_dir: Path, labels_dir: Path) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    detections: list[dict] = []

    image_paths = _list_sample_images(sample_dir)
    if not image_paths:
        raise FileNotFoundError(f"No sample images found in {sample_dir}")

    for image_path in image_paths:
        tile_id = _infer_tile_id(image_path)
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            lines = [line.strip() for line in label_path.read_text().splitlines() if line.strip()]
        else:
            lines = []

        count = len(lines)
        rows.append({"tile_id": tile_id, "count": count, "path": str(image_path)})

        if count == 0:
            continue

        with Image.open(image_path) as img:
            width, height = img.size
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            _, cx, cy, w, h = map(float, parts[:5])
            bbox_area = float(w * width * h * height)
            detections.append(
                {
                    "tile_id": tile_id,
                    "confidence": 1.0,
                    "bbox_area": bbox_area,
                    "path": str(image_path),
                }
            )

    return rows, detections


def _write_outputs(
    root: Path,
    rows: list[dict],
    detections: list[dict],
    output_dir: Path | None = None,
    predictions_path: Path | None = None,
    demo_mode_used: bool = False,
    demo_fallback_type: str = "none",
) -> Path:
    processed_dir = root / "data" / "processed"
    output_dir = output_dir or (processed_dir / "predictions")
    predictions_path = predictions_path or (processed_dir / "predictions.csv")

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    predictions_df = pd.DataFrame(rows, columns=["tile_id", "count", "path"])
    if demo_mode_used:
        predictions_df["demo_mode_used"] = demo_mode_used
        predictions_df["demo_fallback_type"] = demo_fallback_type
    predictions_df.to_csv(predictions_path, index=False)

    pool_counts_path = output_dir / "pool_counts.csv"
    pool_rows = [
        {"tile_id": row["tile_id"], "pool_count": row["count"], "path": row["path"]}
        for row in rows
    ]
    pool_counts_df = pd.DataFrame(pool_rows, columns=["tile_id", "pool_count", "path"])
    if demo_mode_used:
        pool_counts_df["demo_mode_used"] = demo_mode_used
        pool_counts_df["demo_fallback_type"] = demo_fallback_type
    pool_counts_df.to_csv(pool_counts_path, index=False)

    detections_path = output_dir / "detections.csv"
    detections_df = pd.DataFrame(
        detections,
        columns=["tile_id", "confidence", "bbox_area", "path"],
    )
    if demo_mode_used:
        detections_df["demo_mode_used"] = demo_mode_used
        detections_df["demo_fallback_type"] = demo_fallback_type
    detections_df.to_csv(detections_path, index=False)

    return predictions_path


def run_inference(
    config_path: str,
    weights: str | None,
    source: str | None,
    tiles_csv: str | None = None,
    demo_mode: bool = False,
) -> Path:
    config = load_config(config_path)
    root = project_root()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is required. Install requirements-train.txt") from exc

    if weights is None:
        weights = str(root / "checkpoints" / "best.pt")

    model = YOLO(weights)

    def resolve_sources() -> tuple[str | list[str], dict[str, str]]:
        tile_id_by_path: dict[str, str] = {}
        if tiles_csv:
            tiles_csv_path = Path(tiles_csv)
            if not tiles_csv_path.is_absolute():
                tiles_csv_path = root / tiles_csv_path
            sources, tile_id_by_path = _load_tiles_csv(tiles_csv_path)
            if not sources:
                raise FileNotFoundError(f"No tiles listed in {tiles_csv_path}")
            return sources, tile_id_by_path

        source_path = Path(source) if source else root / "data" / "processed" / "yolo" / "images" / "test"
        if not source_path.exists():
            raise FileNotFoundError(f"Missing inference source: {source_path}")
        return str(source_path), tile_id_by_path

    explicit_source = bool(source) or bool(tiles_csv)
    sources, tile_id_by_path = resolve_sources()
    rows, detections = _collect_predictions(model, sources, tile_id_by_path, config)
    total_detections = sum(row["count"] for row in rows)

    fallback_triggered = False
    fallback_type = "none"
    fallback_rows: list[dict] = []
    fallback_detections: list[dict] = []

    if demo_mode and total_detections == 0:
        sample_dir = _positive_sample_dir(root)
        if not sample_dir.exists():
            raise FileNotFoundError(f"Demo mode sample directory missing: {sample_dir}")
        if not any(sample_dir.glob("*.jpg")):
            raise FileNotFoundError(f"Demo mode sample directory is empty: {sample_dir}")
        print(
            "Demo mode: no pools detected in the provided tiles. "
            "Quickstart tiles may contain no pools, so rerunning on bundled Sao Paulo samples."
        )
        fallback_rows, fallback_detections = _collect_predictions(model, str(sample_dir), {}, config)
        fallback_type = "samples"
        total_fallback = sum(row["count"] for row in fallback_rows)
        if total_fallback == 0:
            labels_dir = _positive_sample_labels_dir(root)
            if not labels_dir.exists():
                raise FileNotFoundError(f"Demo mode labels directory missing: {labels_dir}")
            print(
                "Demo mode: model returned zero detections on bundled samples. "
                "Using labeled sample boxes to populate demo outputs."
            )
            fallback_rows, fallback_detections = _collect_label_fallback(sample_dir, labels_dir)
            fallback_type = "labels"
        fallback_triggered = True

    demo_output_dir = root / "data" / "processed" / "predictions" / "demo"
    demo_predictions_path = demo_output_dir / "predictions.csv"

    if fallback_triggered:
        if explicit_source:
            source_output_dir = demo_output_dir / "source"
            source_predictions_path = source_output_dir / "predictions.csv"
            _write_outputs(
                root,
                rows,
                detections,
                output_dir=source_output_dir,
                predictions_path=source_predictions_path,
                demo_mode_used=True,
                demo_fallback_type="none",
            )
        output_path = _write_outputs(
            root,
            fallback_rows,
            fallback_detections,
            output_dir=demo_output_dir,
            predictions_path=demo_predictions_path,
            demo_mode_used=True,
            demo_fallback_type=fallback_type,
        )
        summary_rows = fallback_rows
        summary_detections = fallback_detections
        summary_fallback = fallback_type
    else:
        output_path = _write_outputs(
            root,
            rows,
            detections,
            demo_mode_used=demo_mode,
            demo_fallback_type="none",
        )
        summary_rows = rows
        summary_detections = detections
        summary_fallback = "none"

    print(
        f"tiles={len(summary_rows)} detections={len(summary_detections)} demo_fallback={summary_fallback}"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pool detector inference.")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--tiles-csv", default=None)
    parser.add_argument("--demo-mode", action="store_true", help="Rerun on bundled Sao Paulo samples if no pools found.")
    args = parser.parse_args()

    run_inference(args.config, args.weights, args.source, args.tiles_csv, args.demo_mode)


if __name__ == "__main__":
    main()
