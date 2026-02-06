from __future__ import annotations

import argparse
from pathlib import Path

import folium
import pandas as pd
import mercantile
from branca.colormap import linear

from src.utils.config import load_config, project_root


def _tile_center(tile_id: str) -> tuple[float, float]:
    z, x, y = (int(part) for part in tile_id.split("_"))
    bounds = mercantile.bounds(x, y, z)
    center_lon = (bounds.west + bounds.east) / 2.0
    center_lat = (bounds.south + bounds.north) / 2.0
    return center_lat, center_lon


def build_map(config_path: str, predictions_path: Path | None, output_path: Path | None = None) -> Path:
    config = load_config(config_path)
    root = project_root()
    if predictions_path is None:
        predictions_path = root / "data" / "processed" / "predictions.csv"
    predictions = pd.read_csv(predictions_path)

    bbox = config["aoi"]["bbox"]
    center_lat = (bbox[1] + bbox[3]) / 2.0
    center_lon = (bbox[0] + bbox[2]) / 2.0

    max_count = max(predictions["count"].max(), 1)
    colormap = linear.YlGnBu_09.scale(0, max_count)

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="cartodbpositron")
    for row in predictions.itertuples():
        lat, lon = _tile_center(row.tile_id)
        color = colormap(row.count)
        folium.CircleMarker(
            location=[lat, lon],
            radius=4 + min(row.count, 6),
            color=color,
            fill=True,
            fill_opacity=0.7,
            weight=0,
            popup=f"{row.tile_id}: {row.count}",
        ).add_to(fmap)

    colormap.caption = "Predicted pool count per tile"
    colormap.add_to(fmap)

    if output_path is None:
        output_path = root / "reports" / "figures" / "pool_density_map.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(output_path))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Folium map of pool density.")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--predictions", default="data/processed/predictions.csv")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output_path = build_map(args.config, Path(args.predictions), Path(args.output) if args.output else None)
    print(f"Saved map to {output_path}")


if __name__ == "__main__":
    main()
