from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.utils.config import load_config, project_root


def _load_tiles_csv(tiles_csv: Path) -> tuple[list[Path], dict[str, str]]:
    df = pd.read_csv(tiles_csv)
    if "path" not in df.columns:
        raise ValueError("tiles_csv must include a path column.")
    tile_id_by_path = {}
    if "tile_id" in df.columns:
        tile_id_by_path = dict(zip(df["path"], df["tile_id"]))
    image_paths = [Path(path) for path in df["path"].tolist()]
    return image_paths, tile_id_by_path


def _infer_tile_id(path: Path) -> str:
    stem = path.stem
    parent = path.parent.name
    if parent.isdigit() and "_" in stem:
        return f"{parent}_{stem}"
    return stem


def _blue_ratio(image: Image.Image) -> float:
    arr = np.asarray(image.convert("RGB"))
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)

    # Simple heuristic: blue-dominant pixels
    mask = (b > r + 15) & (b > g + 10) & (b > 80)
    return float(mask.mean())


def run_baseline(
    config_path: str,
    source: str | None,
    threshold: float | None,
    tiles_csv: str | None = None,
) -> Path:
    config = load_config(config_path)
    root = project_root()
    tile_id_by_path: dict[str, str] = {}
    if tiles_csv:
        tiles_csv_path = Path(tiles_csv)
        if not tiles_csv_path.is_absolute():
            tiles_csv_path = root / tiles_csv_path
        image_paths, tile_id_by_path = _load_tiles_csv(tiles_csv_path)
        if not image_paths:
            raise FileNotFoundError(f"No tiles listed in {tiles_csv_path}")
    else:
        source_path = Path(source) if source else root / "data" / "processed" / "yolo" / "images" / "test"
        if not source_path.exists():
            raise FileNotFoundError(f"Missing baseline source: {source_path}")
        image_paths = sorted(source_path.rglob("*.jpg")) if source_path.is_dir() else [source_path]

    if threshold is None:
        threshold = config.get("baseline", {}).get("blue_ratio_threshold", 0.02)

    rows = []
    for image_path in image_paths:
        image = Image.open(image_path)
        ratio = _blue_ratio(image)
        count = 1 if ratio >= threshold else 0
        tile_id = tile_id_by_path.get(str(image_path), _infer_tile_id(image_path))
        rows.append({"tile_id": tile_id, "count": count, "blue_ratio": ratio, "path": str(image_path)})

    output_path = root / "data" / "processed" / "predictions_baseline.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a naive blue-pixel baseline.")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--source", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--tiles-csv", default=None)
    args = parser.parse_args()

    output_path = run_baseline(args.config, args.source, args.threshold, args.tiles_csv)
    print(f"Saved baseline predictions to {output_path}")


if __name__ == "__main__":
    main()
