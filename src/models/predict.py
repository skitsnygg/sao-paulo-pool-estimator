from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

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


def run_inference(
    config_path: str,
    weights: str | None,
    source: str | None,
    tiles_csv: str | None = None,
) -> Path:
    config = load_config(config_path)
    root = project_root()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is required. Install requirements-train.txt") from exc

    if weights is None:
        weights = str(root / "checkpoints" / "best.pt")

    tile_id_by_path: dict[str, str] = {}
    if tiles_csv:
        tiles_csv_path = Path(tiles_csv)
        if not tiles_csv_path.is_absolute():
            tiles_csv_path = root / tiles_csv_path
        sources, tile_id_by_path = _load_tiles_csv(tiles_csv_path)
        if not sources:
            raise FileNotFoundError(f"No tiles listed in {tiles_csv_path}")
    else:
        source_path = Path(source) if source else root / "data" / "processed" / "yolo" / "images" / "test"
        if not source_path.exists():
            raise FileNotFoundError(f"Missing inference source: {source_path}")
        sources = str(source_path)

    model = YOLO(weights)
    results_iter = model.predict(
        source=sources,
        conf=config["inference"].get("conf", 0.25),
        iou=config["inference"].get("iou", 0.5),
        max_det=config["inference"].get("max_det", 300),
        stream=True,
    )

    rows = []
    for result in results_iter:
        path = Path(result.path)
        tile_id = tile_id_by_path.get(str(path), _infer_tile_id(path))
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
    parser.add_argument("--tiles-csv", default=None)
    args = parser.parse_args()

    output_path = run_inference(args.config, args.weights, args.source, args.tiles_csv)
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
