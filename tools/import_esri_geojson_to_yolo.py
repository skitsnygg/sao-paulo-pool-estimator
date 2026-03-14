#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import transform

IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Import georeferenced polygon labels into a YOLO segmentation dataset.")
    ap.add_argument("--labels", required=True, help="GeoJSON or GPKG path")
    ap.add_argument("--tiles-dir", required=True, help="Directory of georeferenced tile images")
    ap.add_argument("--dataset", required=True, help="YOLO dataset root")
    ap.add_argument("--split", choices=("train", "val"), default="train")
    ap.add_argument("--class-name", default="pool")
    ap.add_argument("--class-id", type=int, default=0)
    ap.add_argument("--prefix", required=True, help="Prefix to prepend to imported stems, e.g. vila_mariana_z18")
    ap.add_argument("--copy-images", action="store_true", help="Copy images instead of symlink")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def iter_tile_files(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def normalize_ring(coords: np.ndarray, width: int, height: int) -> str | None:
    if len(coords) < 3:
        return None
    parts: list[str] = []
    for x, y in coords:
        xn = min(max(float(x) / width, 0.0), 1.0)
        yn = min(max(float(y) / height, 0.0), 1.0)
        parts.append(f"{xn:.6f}")
        parts.append(f"{yn:.6f}")
    return " ".join(parts) if len(parts) >= 6 else None


def polygon_to_yolo_lines(geom: Polygon | MultiPolygon, transform_affine, width: int, height: int, class_id: int) -> list[str]:
    inv = ~transform_affine
    lines: list[str] = []

    def world_to_pixel(x, y, z=None):
        px, py = inv * (x, y)
        return px, py

    polys = [geom] if isinstance(geom, Polygon) else list(geom.geoms) if isinstance(geom, MultiPolygon) else []
    for poly in polys:
        if poly.is_empty:
            continue
        px_poly = transform(world_to_pixel, poly)
        ext = np.asarray(px_poly.exterior.coords)
        line = normalize_ring(ext, width, height)
        if line is not None:
            lines.append(f"{class_id} {line}")
    return lines


def main() -> int:
    args = parse_args()

    labels_path = Path(args.labels).expanduser().resolve()
    tiles_dir = Path(args.tiles_dir).expanduser().resolve()
    dataset = Path(args.dataset).expanduser().resolve()

    img_dst = dataset / "images" / args.split
    lbl_dst = dataset / "labels" / args.split
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(labels_path)
    if "class" in gdf.columns:
        gdf = gdf[gdf["class"].astype(str).str.lower() == args.class_name.lower()].copy()
    if gdf.empty:
        raise SystemExit("No matching features found.")
    if gdf.crs is None:
        raise SystemExit("Labels have no CRS.")

    imported = 0
    skipped_existing = 0
    tiles_seen = 0
    tiles_with_labels = 0

    for tile_path in iter_tile_files(tiles_dir):
        tiles_seen += 1
        with rasterio.open(tile_path) as src:
            if src.crs is None:
                continue

            tile_gdf = gdf if str(gdf.crs) == str(src.crs) else gdf.to_crs(src.crs)
            left, bottom, right, top = src.bounds
            hits = tile_gdf.cx[left:right, bottom:top]
            if hits.empty:
                continue

            width = src.width
            height = src.height
            tile_box = box(left, bottom, right, top)

            lines: list[str] = []
            for _, row in hits.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                clipped = geom.intersection(tile_box)
                if clipped.is_empty:
                    continue
                lines.extend(polygon_to_yolo_lines(clipped, src.transform, width, height, args.class_id))

            if not lines:
                continue

            tiles_with_labels += 1
            stem = f"{args.prefix}__{tile_path.stem}"
            out_img = img_dst / f"{stem}{tile_path.suffix.lower()}"
            out_lbl = lbl_dst / f"{stem}.txt"

            if out_img.exists() or out_lbl.exists():
                skipped_existing += 1
                print(f"Skipping existing: {stem}")
                continue

            if not args.dry_run:
                if args.copy_images:
                    shutil.copy2(tile_path, out_img)
                else:
                    if out_img.exists() or out_img.is_symlink():
                        out_img.unlink()
                    out_img.symlink_to(tile_path)
                out_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")

            imported += 1
            print(f"Imported: {stem} ({len(lines)} polygons)")

    print()
    print("tiles_seen:", tiles_seen)
    print("tiles_with_labels:", tiles_with_labels)
    print("imported:", imported)
    print("skipped_existing:", skipped_existing)
    print("dry_run:", args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
