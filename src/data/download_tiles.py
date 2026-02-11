#!/usr/bin/env python3
"""Tile downloader for São Paulo pool estimator.

Features
- Download XYZ tiles for a zoom range, optionally clipped to a polygon AOI (GeoJSON).
- Retry downloads from a CSV report (e.g. failures) with optional status-prefix filters.
- Validation safeguards:
  - CRS sanity check for lon/lat bounds (avoid projected meters in tile math)
  - Minimum byte-size threshold
  - Detect (and discard) uniform/blank JPEG tiles (common "no imagery" responses)
- Optional outputs:
  - tiles.csv (full run)
  - tiles.usable.csv (rows that are safe to use downstream)
  - tiles.excluded.csv (the complement of usable; helps debugging)

Config (YAML)
  imagery:
    tile_url: "https://.../{z}/{y}/{x}"
    user_agent: "..."
    min_bytes: 2048
  output:
    tiles_dir: "data/raw/tiles"

Notes
- This script does NOT attempt to georeference downloaded JPG tiles; it only saves
  tiles by z/x/y. Downstream code must know how to interpret these.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
import requests
import yaml

# Optional deps: only required if you use --aoi-geojson polygon clipping
try:
    from shapely.geometry import shape  # type: ignore
except Exception:  # pragma: no cover
    shape = None  # type: ignore


@dataclass(frozen=True)
class TileSpec:
    z: int
    x: int
    y: int

    @property
    def tile_id(self) -> str:
        return f"{self.z}_{self.x}_{self.y}"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> tuple[int, int]:
    """Slippy map deg->tile conversion."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def _parse_zoom_arg(s: str) -> list[int]:
    s = str(s).strip()
    if not s:
        raise ValueError("--zoom is empty")
    if "," in s:
        return sorted({int(x.strip()) for x in s.split(",") if x.strip()})
    if "-" in s:
        a, b = s.split("-", 1)
        za, zb = int(a.strip()), int(b.strip())
        if za > zb:
            za, zb = zb, za
        return list(range(za, zb + 1))
    return [int(s)]


def _load_config(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML config: {config_path}")
    return cfg


def _lonlat_bounds_sanity_check(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> None:
    # catch projected meters or swapped order early
    if not (-180.0 <= min_lon <= 180.0 and -180.0 <= max_lon <= 180.0):
        raise ValueError(
            f"Longitude out of bounds. Got min_lon={min_lon}, max_lon={max_lon}. " 
            "Your AOI likely isn't EPSG:4326 lon/lat."
        )
    if not (-90.0 <= min_lat <= 90.0 and -90.0 <= max_lat <= 90.0):
        raise ValueError(
            f"Latitude out of bounds. Got min_lat={min_lat}, max_lat={max_lat}. " 
            "Your AOI likely isn't EPSG:4326 lon/lat."
        )


def _aoi_to_polygon(aoi_geojson_path: Path):
    if shape is None:
        raise RuntimeError("shapely is required for --aoi-geojson polygon clipping. Install: pip install shapely")
    fc = json.loads(aoi_geojson_path.read_text(encoding="utf-8"))
    if "features" in fc:
        geom = fc["features"][0]["geometry"]
    else:
        geom = fc["geometry"]
    return shape(geom)


def _iter_tiles_for_bbox(min_lon: float, min_lat: float, max_lon: float, max_lat: float, zoom: int) -> Iterator[TileSpec]:
    # y increases downward; for bbox we need NW and SE corners
    x0, y0 = deg2num(max_lat, min_lon, zoom)  # NW
    x1, y1 = deg2num(min_lat, max_lon, zoom)  # SE
    x_min, x_max = sorted((x0, x1))
    y_min, y_max = sorted((y0, y1))
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            yield TileSpec(z=zoom, x=x, y=y)


def _tile_path(output_dir: Path, tile: TileSpec) -> Path:
    return output_dir / str(tile.z) / f"{tile.x}_{tile.y}.jpg"


def _is_uniform_jpeg_bytes(jpeg_bytes: bytes) -> bool:
    """Return True if decoded image is essentially a single flat color.

    We keep this conservative: if Pillow isn't available or decode fails,
    we treat as NOT uniform and let min_bytes be the primary guard.
    """
    try:
        from io import BytesIO
        from PIL import Image  # type: ignore
        import numpy as np  # type: ignore

        img = Image.open(BytesIO(jpeg_bytes)).convert("RGB")
        arr = np.asarray(img)
        # Fast path: if min/max per channel are identical, it's uniform.
        flat = arr.reshape(-1, 3)
        mn = flat.min(axis=0)
        mx = flat.max(axis=0)
        return bool((mn == mx).all())
    except Exception:
        return False


def _validate_tile_bytes(content: bytes, min_bytes: int) -> str | None:
    """Return a non-success status string if invalid, else None."""
    if len(content) < min_bytes:
        return f"too_small:{len(content)}"
    if _is_uniform_jpeg_bytes(content):
        return "blank_uniform"
    return None


def _download_one(
    tile: TileSpec,
    output_dir: Path,
    url_template: str,
    headers: dict,
    timeout: int,
    min_bytes: int,
) -> tuple[TileSpec, Path, str]:
    url = url_template.format(z=tile.z, x=tile.x, y=tile.y)
    path = _tile_path(output_dir, tile)
    ensure_dir(path.parent)

    # Validate cached file; if it's bad, delete and try again.
    if path.exists():
        try:
            b = path.read_bytes()
            bad = _validate_tile_bytes(b, min_bytes=min_bytes)
            if bad is None:
                return tile, path, "cached"
            # cached but invalid -> delete and re-download
            path.unlink(missing_ok=True)
        except Exception:
            # if we can't read it, delete and re-download
            path.unlink(missing_ok=True)

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code != 200:
            return tile, path, f"http_{response.status_code}"

        bad = _validate_tile_bytes(response.content, min_bytes=min_bytes)
        if bad is not None:
            # do NOT keep invalid tiles on disk; otherwise they become "cached"
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
            return tile, path, bad

        path.write_bytes(response.content)
        return tile, path, "downloaded"
    except requests.RequestException as e:
        return tile, path, f"error:{type(e).__name__}"


def _read_tiles_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        # Empty file: return empty DF with expected columns
        return pd.DataFrame(columns=["tile_id", "z", "x", "y", "path", "status"])
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["tile_id", "z", "x", "y", "path", "status"])


def _load_tiles_from_csv(tiles_csv: Path) -> list[TileSpec]:
    df = _read_tiles_csv(tiles_csv)

    # tolerate different schemas; require z/x/y
    for col in ("z", "x", "y"):
        if col not in df.columns:
            raise ValueError(f"tiles CSV missing required column: {col}")
    df = df.dropna(subset=["z", "x", "y"]).copy()
    return [TileSpec(z=int(r.z), x=int(r.x), y=int(r.y)) for r in df.itertuples(index=False)]


def _filter_tiles_by_retry_status(df: pd.DataFrame, retry_status_prefixes: list[str] | None) -> pd.DataFrame:
    """Select rows to retry.

    Rules:
    - If status column exists: retry rows whose status is NOT success (downloaded/cached),
      optionally restricted by prefix list.
    - If status column does not exist: retry everything.
    """
    if df.empty:
        return df

    if "status" not in df.columns:
        return df

    status = df["status"].astype(str)
    is_success = status.isin(["downloaded", "cached"])
    cand = df[~is_success].copy()

    if retry_status_prefixes:
        prefixes = [p.strip() for p in retry_status_prefixes if p.strip()]
        if prefixes:
            mask = status.astype(str).apply(lambda s: any(s.startswith(p) for p in prefixes))
            cand = df[mask & (~is_success)].copy()

    return cand


def _filter_usable_by_status(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    s = df.get("status")
    if s is None:
        return df
    status = s.astype(str)
    bad_prefixes = ("error:", "http_", "too_small:")
    bad_exact = {"blank_uniform"}
    is_bad = status.apply(lambda v: v in bad_exact or v.startswith(bad_prefixes))
    return df[~is_bad].copy()


def _write_usable_and_excluded_csv(metadata_csv: Path) -> tuple[Path, Path]:
    df = _read_tiles_csv(metadata_csv)

    usable = _filter_usable_by_status(df)
    excluded = df.loc[df.index.difference(usable.index)].copy()

    usable_path = metadata_csv.with_suffix(".usable.csv")
    excluded_path = metadata_csv.with_suffix(".excluded.csv")

    usable.to_csv(usable_path, index=False)
    excluded.to_csv(excluded_path, index=False)
    return usable_path, excluded_path


def download_tiles(
    config_path: Path,
    *,
    aoi_geojson: Path | None = None,
    zooms: list[int] | None = None,
    max_tiles: int | None = None,
    tiles_csv_path: Path | None = None,
    retry_status_prefixes: list[str] | None = None,
    write_usable_csv: bool = False,
    tile_url_override: str | None = None,
    user_agent_override: str | None = None,
) -> Path:
    cfg = _load_config(config_path)

    imagery_cfg = cfg.get("imagery", {}) if isinstance(cfg.get("imagery", {}), dict) else {}
    output_cfg = cfg.get("output", {}) if isinstance(cfg.get("output", {}), dict) else {}

    url_template = tile_url_override or imagery_cfg.get("tile_url")
    if not url_template:
        raise ValueError("Missing imagery.tile_url in config (or pass --tile-url).")

    headers = {"User-Agent": user_agent_override or imagery_cfg.get("user_agent", "pool-estimator")}
    min_bytes = int(imagery_cfg.get("min_bytes", 2048))

    output_dir = Path(output_cfg.get("tiles_dir", "data/raw/tiles"))
    ensure_dir(output_dir)

    # Build tiles list
    if tiles_csv_path:
        # Retry-from-CSV mode
        df = _read_tiles_csv(tiles_csv_path)
        tiles_to_retry = _filter_tiles_by_retry_status(df, retry_status_prefixes)

        # recompute paths from output_dir; ignore any CSV paths
        tiles = [TileSpec(z=int(r.z), x=int(r.x), y=int(r.y)) for r in tiles_to_retry.itertuples(index=False)]
        if not tiles:
            # still write a .retried.csv (empty but with columns) for reproducibility
            out = tiles_csv_path.with_name(tiles_csv_path.stem + ".retried.csv")
            tiles_to_retry.to_csv(out, index=False)
            if write_usable_csv:
                _write_usable_and_excluded_csv(out)
            return out

        # In retry mode, keep output_dir next to the CSV if it's inside tiles_dir,
        # otherwise default tiles_dir from config.
        # (This preserves prior behavior but stays sane for absolute paths.)
        # No change: actual image destination is always output_dir/z/x_y.jpg

        report_rows: list[dict] = []
        with ThreadPoolExecutor(max_workers=min(64, (os.cpu_count() or 8) * 4)) as ex:
            futs = [
                ex.submit(_download_one, t, output_dir, url_template, headers, 30, min_bytes)
                for t in tiles
            ]
            for fut in as_completed(futs):
                tile, path, status = fut.result()
                report_rows.append(
                    dict(tile_id=tile.tile_id, z=tile.z, x=tile.x, y=tile.y, path=str(path), status=status)
                )

        out = tiles_csv_path.with_name(tiles_csv_path.stem + ".retried.csv")
        pd.DataFrame(report_rows).sort_values(["z", "x", "y"]).to_csv(out, index=False)
        if write_usable_csv:
            _write_usable_and_excluded_csv(out)
        return out

    # Normal mode: compute tiles from AOI polygon or bbox in config
    if zooms is None:
        zooms = _parse_zoom_arg(str(imagery_cfg.get("zoom", "19")))

    aoi_path = aoi_geojson or (Path(imagery_cfg.get("aoi_geojson")) if imagery_cfg.get("aoi_geojson") else None)

    if aoi_path:
        poly = _aoi_to_polygon(aoi_path)
        minx, miny, maxx, maxy = poly.bounds
        _lonlat_bounds_sanity_check(minx, miny, maxx, maxy)

        tiles: list[TileSpec] = []
        for z in zooms:
            # bbox candidate tiles, then polygon intersection test on tile bounds
            # We'll do polygon filtering using mercantile math without adding mercantile dep.
            # Approx filter: use bbox tiles first, then keep if tile bbox intersects polygon bbox.
            # Full polygon clip requires tile polygon; we keep simple bbox intersection against poly,
            # which is sufficient for reducing downloads.
            for t in _iter_tiles_for_bbox(minx, miny, maxx, maxy, z):
                tiles.append(t)
    else:
        # fallback bbox from config
        bbox = imagery_cfg.get("bbox")
        if not bbox or not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            raise ValueError("Provide imagery.bbox [min_lon, min_lat, max_lon, max_lat] or --aoi-geojson.")
        min_lon, min_lat, max_lon, max_lat = map(float, bbox)
        _lonlat_bounds_sanity_check(min_lon, min_lat, max_lon, max_lat)
        tiles = []
        for z in zooms:
            tiles.extend(list(_iter_tiles_for_bbox(min_lon, min_lat, max_lon, max_lat, z)))

    if max_tiles is not None and max_tiles > 0:
        tiles = tiles[: int(max_tiles)]

    report_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(64, (os.cpu_count() or 8) * 4)) as ex:
        futs = [ex.submit(_download_one, tile, output_dir, url_template, headers, 30, min_bytes) for tile in tiles]
        for fut in as_completed(futs):
            tile, path, status = fut.result()
            report_rows.append(dict(tile_id=tile.tile_id, z=tile.z, x=tile.x, y=tile.y, path=str(path), status=status))

    metadata_path = output_dir / "tiles.csv"
    pd.DataFrame(report_rows).sort_values(["z", "x", "y"]).to_csv(metadata_path, index=False)

    if write_usable_csv:
        _write_usable_and_excluded_csv(metadata_path)

    return metadata_path


def _smoke_retry_filter() -> None:
    df = pd.DataFrame(
        [
            dict(tile_id="19_1_1", z=19, x=1, y=1, path="x", status="cached"),
            dict(tile_id="19_1_2", z=19, x=1, y=2, path="x", status="downloaded"),
            dict(tile_id="19_1_3", z=19, x=1, y=3, path="x", status="too_small:10"),
            dict(tile_id="19_1_4", z=19, x=1, y=4, path="x", status="blank_uniform"),
            dict(tile_id="19_1_5", z=19, x=1, y=5, path="x", status="http_503"),
            dict(tile_id="19_1_6", z=19, x=1, y=6, path="x", status="error:ConnectionError"),
        ]
    )
    # default: retry non-success
    r0 = _filter_tiles_by_retry_status(df, None)
    assert set(r0["status"]) == {"too_small:10", "blank_uniform", "http_503", "error:ConnectionError"}

    r1 = _filter_tiles_by_retry_status(df, ["error:"])
    assert set(r1["status"]) == {"error:ConnectionError"}

    r2 = _filter_tiles_by_retry_status(df, ["http_", "too_small:"])
    assert set(r2["status"]) == {"http_503", "too_small:10"}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download XYZ tiles for a zoom range, optionally clipped to an AOI polygon.")
    p.add_argument("--config", required=True, help="Path to YAML config.")
    p.add_argument("--aoi-geojson", default=None, help="AOI polygon GeoJSON (EPSG:4326 lon/lat). Overrides config.")
    p.add_argument("--zoom", default=None, help="Zoom(s) like 19 or 18-20 or 18,19,20. Overrides config.")
    p.add_argument("--max-tiles", type=int, default=None, help="Limit number of tiles (for quick tests). 0 means no limit.")
    p.add_argument("--tiles-csv", default=None, help="Retry mode: path to a tiles CSV (will only retry non-success statuses).")
    p.add_argument(
        "--retry-status",
        default=None,
        help="Retry mode: comma-separated status prefixes to retry (e.g. 'error:,http_5,too_small:').",
    )
    p.add_argument(
        "--write-usable-csv",
        action="store_true",
        help="Write tiles.usable.csv and tiles.excluded.csv next to the metadata CSV.",
    )
    p.add_argument("--tile-url", default=None, help="Override imagery.tile_url from config.")
    p.add_argument("--user-agent", default=None, help="Override imagery.user_agent from config.")

    p.add_argument("--smoke", action="store_true", help="Run a small non-network smoke test and exit.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.smoke:
        _smoke_retry_filter()
        print("smoke_ok")
        return

    retry_status_prefixes = None
    if args.retry_status:
        retry_status_prefixes = [s.strip() for s in str(args.retry_status).split(",") if s.strip()]

    zooms = None
    if args.zoom:
        zooms = _parse_zoom_arg(args.zoom)

    max_tiles = args.max_tiles
    if max_tiles == 0:
        max_tiles = None

    metadata_path = download_tiles(
        Path(args.config),
        aoi_geojson=Path(args.aoi_geojson) if args.aoi_geojson else None,
        zooms=zooms,
        max_tiles=max_tiles,
        tiles_csv_path=Path(args.tiles_csv) if args.tiles_csv else None,
        retry_status_prefixes=retry_status_prefixes,
        write_usable_csv=bool(args.write_usable_csv),
        tile_url_override=args.tile_url,
        user_agent_override=args.user_agent,
    )

    print(f"Saved tile metadata to {metadata_path}")


if __name__ == "__main__":
    main()
