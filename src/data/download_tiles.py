from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time

import pandas as pd
import requests
from tqdm import tqdm

from src.utils.config import load_config, project_root
from src.utils.geo import TileSpec, sample_tiles, tiles_from_bbox
from src.utils.io import ensure_dir


def _tile_path(output_dir: Path, tile: TileSpec) -> Path:
    return output_dir / str(tile.z) / f"{tile.x}_{tile.y}.jpg"


def _download_one(tile: TileSpec, output_dir: Path, url_template: str, headers: dict, timeout: int) -> tuple[TileSpec, Path, str]:
    url = url_template.format(z=tile.z, x=tile.x, y=tile.y)
    path = _tile_path(output_dir, tile)
    ensure_dir(path.parent)
    if path.exists():
        return tile, path, "cached"

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code != 200:
            return tile, path, f"http_{response.status_code}"
        path.write_bytes(response.content)
        return tile, path, "downloaded"
    except requests.RequestException:
        return tile, path, "error"


def _load_tiles_from_csv(tiles_csv: Path) -> list[TileSpec]:
    df = pd.read_csv(tiles_csv)
    tiles: list[TileSpec] = []
    if {"z", "x", "y"}.issubset(df.columns):
        for row in df.itertuples():
            tiles.append(TileSpec(int(row.z), int(row.x), int(row.y)))
    elif "tile_id" in df.columns:
        for tile_id in df["tile_id"].tolist():
            z, x, y = (int(part) for part in str(tile_id).split("_"))
            tiles.append(TileSpec(z, x, y))
    else:
        raise ValueError("tiles_csv must include z/x/y columns or tile_id.")

    seen = set()
    unique_tiles = []
    for tile in tiles:
        if tile.id in seen:
            continue
        seen.add(tile.id)
        unique_tiles.append(tile)
    return unique_tiles


def download_tiles(
    config_path: str,
    max_tiles: int | None = None,
    tiles_csv: str | None = None,
    output_csv: str | None = None,
) -> Path:
    config = load_config(config_path)
    bbox = config["aoi"]["bbox"]
    zoom = config["tiles"]["zoom"]
    max_tiles = max_tiles or config["tiles"].get("max_tiles")
    threads = config["tiles"].get("download_threads", 4)
    url_template = config["imagery"]["tile_url"]
    rate_limit = config["imagery"].get("rate_limit_s", 0.0)
    headers = {"User-Agent": config["imagery"].get("user_agent", "pool-estimator")}

    if tiles_csv:
        tiles = _load_tiles_from_csv(Path(tiles_csv))
    else:
        tiles = tiles_from_bbox(bbox, zoom)
        tiles = sample_tiles(tiles, max_tiles, seed=config["dataset"].get("random_seed", 42))

    output_dir = project_root() / "data" / "raw" / "tiles"
    ensure_dir(output_dir)
    if output_csv:
        metadata_path = Path(output_csv)
        if not metadata_path.is_absolute():
            metadata_path = project_root() / metadata_path
        output_dir = metadata_path.parent
    else:
        metadata_path = output_dir / "tiles.csv"
    ensure_dir(output_dir)

    rows = []
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(_download_one, tile, output_dir, url_template, headers, 30) for tile in tiles]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading tiles"):
            tile, path, status = future.result()
            rows.append({"tile_id": tile.id, "z": tile.z, "x": tile.x, "y": tile.y, "path": str(path), "status": status})
            if rate_limit > 0:
                time.sleep(rate_limit)

    df = pd.DataFrame(rows).sort_values(["z", "x", "y"]).reset_index(drop=True)
    df.to_csv(metadata_path, index=False)
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download imagery tiles for the AOI.")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--tiles-csv", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    metadata_path = download_tiles(args.config, args.max_tiles, args.tiles_csv, args.output)
    print(f"Saved tile metadata to {metadata_path}")


if __name__ == "__main__":
    main()
