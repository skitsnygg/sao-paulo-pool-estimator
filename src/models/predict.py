from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.config import load_config, project_root


def run_inference(config_path: str, weights: str | None, source: str | None) -> Path:
    config = load_config(config_path)
    root = project_root()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is required. Install requirements-train.txt") from exc

    if weights is None:
        weights = str(root / "checkpoints" / "best.pt")

    source_path = Path(source) if source else root / "data" / "processed" / "yolo" / "images" / "test"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing inference source: {source_path}")

    model = YOLO(weights)
    results_iter = model.predict(
        source=str(source_path),
        conf=config["inference"].get("conf", 0.25),
        iou=config["inference"].get("iou", 0.5),
        max_det=config["inference"].get("max_det", 300),
        stream=True,
    )

    rows = []
    for result in results_iter:
        path = Path(result.path)
        tile_id = path.stem
        count = len(result.boxes) if result.boxes is not None else 0
        rows.append({"tile_id": tile_id, "count": count, "path": str(path)})

    output_dir = root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predictions.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pool detector inference.")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--source", default=None)
    args = parser.parse_args()

    output_path = run_inference(args.config, args.weights, args.source)
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
