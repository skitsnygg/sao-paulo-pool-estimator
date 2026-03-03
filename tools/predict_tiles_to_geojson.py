from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import Polygon
from ultralytics import YOLO

try:  # Shapely 2.x
    from shapely.validation import make_valid  # type: ignore
except Exception:  # pragma: no cover
    make_valid = None  # type: ignore


@dataclass
class InferenceStats:
    tiles_processed: int = 0
    tiles_with_masks: int = 0
    polys_total: int = 0
    polys_kept: int = 0
    polys_scaled_from_norm: int = 0
    polys_dropped_mask_area: int = 0
    polys_dropped_poly_area: int = 0
    polys_dropped_area_m2: int = 0


def mercator_lat_from_t(t: float) -> float:
    return math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * t))))


def lonlat_from_xyz_pixel(z: int, x: int, y: int, px: float, py: float, w: int, h: int) -> Tuple[float, float]:
    """
    Convert pixel coordinates within an XYZ tile image into lon/lat (WGS84).
    Assumes the image covers exactly one XYZ tile at zoom z, tile coords (x,y).
    """
    n = 2 ** z
    fx = x + (px / w)
    fy = y + (py / h)
    lon = (fx / n) * 360.0 - 180.0
    lat = mercator_lat_from_t(fy / n)
    return lon, lat


def parse_xy_from_path(p: Path) -> Tuple[int, int]:
    """
    Expect filenames like: 97086_148749.jpg -> x=97086, y=148749
    """
    stem = p.stem
    a, b = stem.split("_", 1)
    return int(a), int(b)


GeoTransform = Tuple[float, float, float, float, float, float]


def is_geotiff(p: Path) -> bool:
    return p.suffix.lower() in {".tif", ".tiff"}


def normalize_crs(crs: Optional[str]) -> Optional[str]:
    if not crs:
        return None
    try:
        return CRS.from_user_input(crs).to_string()
    except Exception:
        return crs


def read_geotiff_image(p: Path) -> Tuple[np.ndarray, GeoTransform, str]:
    try:
        import rasterio
    except Exception:
        rasterio = None

    if rasterio is not None:
        with rasterio.open(p) as src:
            arr = src.read()
            if arr.ndim == 2:
                arr = arr[:, :, None]
            else:
                arr = np.transpose(arr, (1, 2, 0))
            if arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            elif arr.shape[2] > 3:
                arr = arr[:, :, :3]
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
            transform = src.transform.to_gdal()
            crs = normalize_crs(src.crs.to_string() if src.crs else None)
            if not crs:
                raise RuntimeError("GeoTIFF missing CRS")
            return arr, transform, crs

    try:
        from osgeo import gdal, osr
    except Exception as exc:
        raise RuntimeError("rasterio or GDAL Python bindings are required to read GeoTIFFs") from exc

    ds = gdal.Open(str(p), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Could not open GeoTIFF: {p}")
    transform = ds.GetGeoTransform()
    proj = ds.GetProjection()
    if not transform or not proj:
        raise RuntimeError("GeoTIFF missing geotransform/CRS")
    try:
        crs = CRS.from_wkt(proj).to_string()
    except Exception:
        crs = proj

    arr = ds.ReadAsArray()
    if arr is None:
        raise RuntimeError(f"Empty GeoTIFF: {p}")
    if arr.ndim == 2:
        arr = arr[:, :, None]
    else:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] > 3:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr, transform, normalize_crs(crs) or crs


def pixel_to_projected(transform: GeoTransform, px: float, py: float) -> Tuple[float, float]:
    origin_x, pixel_w, rot_x, origin_y, rot_y, pixel_h = transform
    geo_x = origin_x + px * pixel_w + py * rot_x
    geo_y = origin_y + px * rot_y + py * pixel_h
    return geo_x, geo_y


_TRANSFORMERS: Dict[Tuple[str, str], Transformer] = {}


def get_transformer(src_crs: str, dst_crs: str) -> Transformer:
    key = (src_crs, dst_crs)
    if key not in _TRANSFORMERS:
        _TRANSFORMERS[key] = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return _TRANSFORMERS[key]


def transform_ring(ring: List[List[float]], src_crs: str, dst_crs: str) -> List[List[float]]:
    if src_crs == dst_crs:
        return ring
    transformer = get_transformer(src_crs, dst_crs)
    return [[float(x), float(y)] for x, y in transformer.itransform(ring)]


def poly_area_px(poly_xy: List[Tuple[float, float]]) -> float:
    """Shoelace polygon area in pixel space."""
    if len(poly_xy) < 3:
        return 0.0
    area = 0.0
    for i in range(len(poly_xy)):
        x1, y1 = poly_xy[i]
        x2, y2 = poly_xy[(i + 1) % len(poly_xy)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _signed_area(ring: List[List[float]]) -> float:
    if len(ring) < 3:
        return 0.0
    area2 = 0.0
    for i in range(len(ring) - 1):
        x1, y1 = ring[i]
        x2, y2 = ring[i + 1]
        area2 += x1 * y2 - x2 * y1
    return area2 / 2.0


def _ensure_closed(ring: List[List[float]]) -> List[List[float]]:
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0][:])
    return ring


def _orient_ring_ccw(ring: List[List[float]]) -> List[List[float]]:
    ring = _ensure_closed(ring)
    if _signed_area(ring) < 0:
        ring = list(reversed(ring))
        ring = _ensure_closed(ring)
    return ring


def _round_float(value: float, precision: int) -> float:
    v = float(round(float(value), precision))
    if v == -0.0:
        v = 0.0
    return v


def _round_ring(ring: List[List[float]], precision: int) -> List[List[float]]:
    rounded = [[_round_float(x, precision), _round_float(y, precision)] for x, y in ring]
    return _ensure_closed(rounded)


def _repair_ring_if_needed(ring: List[List[float]], precision: int) -> List[List[float]]:
    if len(ring) < 4:
        return ring
    try:
        poly = Polygon(ring)
    except Exception:
        return ring

    if poly.is_valid:
        return ring

    g = poly
    if make_valid is not None:
        try:
            g2 = make_valid(g)
            if not g2.is_empty:
                g = g2
        except Exception:
            pass

    try:
        if not g.is_valid:
            g2 = g.buffer(0)
            if not g2.is_empty:
                g = g2
    except Exception:
        pass

    if g.is_empty:
        return ring

    if g.geom_type == "Polygon":
        target = g
    elif g.geom_type == "MultiPolygon":
        target = max(g.geoms, key=lambda p: p.area)
    else:
        return ring

    coords = [[float(x), float(y)] for x, y in target.exterior.coords]
    coords = _round_ring(coords, precision)
    coords = _orient_ring_ccw(coords)
    return coords


def _normalize_ring(ring: List[List[float]], precision: int) -> List[List[float]]:
    ring = _round_ring(ring, precision)
    ring = _orient_ring_ccw(ring)
    ring = _repair_ring_if_needed(ring, precision)
    ring = _ensure_closed(ring)
    return ring


def looks_normalized(poly_xy: List[Tuple[float, float]]) -> bool:
    """
    Heuristic: if all coordinates are between [0, 1.5], treat as normalized coords.
    (Some implementations may return 0..1 or slightly outside due to rounding.)
    """
    if not poly_xy:
        return False
    xs = [p[0] for p in poly_xy]
    ys = [p[1] for p in poly_xy]
    return (max(xs) <= 1.5 and max(ys) <= 1.5 and min(xs) >= -0.5 and min(ys) >= -0.5)


def scale_norm_to_px(poly_xy: List[Tuple[float, float]], w: int, h: int) -> List[Tuple[float, float]]:
    return [(float(x) * w, float(y) * h) for x, y in poly_xy]


def iter_images(d: Path, recursive: bool = True) -> List[Path]:
    exts = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")
    if d.is_file():
        return [d] if d.suffix.lower() in exts else []
    out: List[Path] = []
    if recursive:
        for p in d.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                out.append(p)
    else:
        for p in d.glob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                out.append(p)
    return sorted(out)


def find_worldfile(img: Path) -> Optional[Path]:
    suffix = img.suffix.lower()
    candidates: List[Path] = []
    if suffix == ".png":
        candidates.extend([
            img.with_suffix(".pgw"),
            img.with_suffix(".wld"),
            img.with_suffix(".pngw"),
        ])
    elif suffix in (".jpg", ".jpeg"):
        candidates.extend([
            img.with_suffix(".jgw"),
            img.with_suffix(".wld"),
            img.with_suffix(".jpgw"),
        ])
    else:
        candidates.extend([
            img.with_suffix(".wld"),
        ])
    for p in candidates:
        if p.exists():
            return p
    return None


def read_worldfile(p: Path) -> GeoTransform:
    vals = [float(x.strip()) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(vals) != 6:
        raise ValueError(f"Worldfile must have 6 lines, got {len(vals)}: {p}")
    A, D, B, E, C, F = vals
    return (C, A, B, F, D, E)


def _mask_areas_from_masks(masks: Any) -> Optional[List[float]]:
    data = getattr(masks, "data", None)
    if data is None:
        return None
    try:
        if hasattr(data, "detach"):
            data = data.detach()
        if hasattr(data, "cpu"):
            data = data.cpu()
        if hasattr(data, "numpy"):
            arr = data.numpy()
        else:
            arr = data
    except Exception:
        return None

    try:
        if arr is None:
            return None
        if getattr(arr, "ndim", None) != 3:
            return None
        mask = arr > 0.5
        areas = mask.sum(axis=(1, 2))
        return [float(v) for v in areas]
    except Exception:
        return None


def run_inference(
    *,
    model_path: Path,
    tiles: Sequence[Path],
    z: Optional[int],
    imgsz: int,
    conf: float,
    iou: float,
    min_area_px: float,
    min_mask_area_px: float,
    min_area_m2: float,
    max_det: int,
    retina_masks: bool,
    device: Optional[str],
    verbose: bool,
    precision: int,
    worldfile_crs: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], InferenceStats]:
    from PIL import Image

    model = YOLO(str(model_path))

    features_3857: List[Dict[str, Any]] = []
    features_31983: List[Dict[str, Any]] = []
    stats = InferenceStats()

    for p in tiles:
        stats.tiles_processed += 1

        use_geotiff = is_geotiff(p)
        worldfile = None if use_geotiff else find_worldfile(p)
        use_worldfile = worldfile is not None
        img = None
        src_crs: Optional[str] = None
        transform: Optional[GeoTransform] = None
        x: Optional[int] = None
        y: Optional[int] = None

        if use_geotiff:
            try:
                img, transform, src_crs = read_geotiff_image(p)
            except Exception as e:
                if verbose:
                    print(f"[skip] {p.name}: cannot read GeoTIFF ({e})")
                continue
            h, w = img.shape[:2]
            try:
                x, y = parse_xy_from_path(p)
            except Exception:
                x, y = None, None
        else:
            try:
                w, h = Image.open(p).size
            except Exception as e:
                if verbose:
                    print(f"[skip] {p.name}: cannot open image ({e})")
                continue

            if use_worldfile:
                try:
                    transform = read_worldfile(worldfile)
                    src_crs = normalize_crs(worldfile_crs) or worldfile_crs
                except Exception as e:
                    if verbose:
                        print(f"[skip] {p.name}: cannot read worldfile ({e})")
                    continue
            else:
                try:
                    x, y = parse_xy_from_path(p)
                except Exception as e:
                    if verbose:
                        print(f"[skip] {p.name}: cannot parse x_y ({e})")
                    continue

        if use_geotiff:
            results = model.predict(
                source=img,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                max_det=max_det,
                retina_masks=bool(retina_masks),
                device=device,
                verbose=False,
            )
        else:
            if not use_worldfile and z is None:
                if verbose:
                    print(f"[skip] {p.name}: missing --z for XYZ tiles")
                continue
            results = model.predict(
                source=str(p),
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                max_det=max_det,
                retina_masks=bool(retina_masks),
                device=device,
                verbose=False,
            )

        r = results[0]
        if r.masks is None or r.masks.xy is None:
            continue

        stats.tiles_with_masks += 1
        mask_areas = _mask_areas_from_masks(r.masks)
        confs: List[float] = []
        try:
            if r.boxes is not None and getattr(r.boxes, "conf", None) is not None:
                confs = [float(v) for v in r.boxes.conf.tolist()]
        except Exception:
            confs = []

        for idx, poly in enumerate(r.masks.xy):
            poly_xy = [(float(px), float(py)) for px, py in poly]
            stats.polys_total += 1

            if looks_normalized(poly_xy):
                poly_xy = scale_norm_to_px(poly_xy, w, h)
                stats.polys_scaled_from_norm += 1

            mask_area_px = None
            if mask_areas is not None and idx < len(mask_areas):
                mask_area_px = mask_areas[idx]

            if mask_area_px is not None:
                if mask_area_px < min_mask_area_px:
                    stats.polys_dropped_mask_area += 1
                    continue
            else:
                if poly_area_px(poly_xy) < min_area_px:
                    stats.polys_dropped_poly_area += 1
                    continue

            src_ring: List[List[float]] = []
            if use_geotiff:
                if transform is None or src_crs is None:
                    if verbose:
                        print(f"[skip] {p.name}: missing geotransform/CRS")
                    continue
                for px, py in poly_xy:
                    X, Y = pixel_to_projected(transform, px, py)
                    src_ring.append([float(X), float(Y)])
                src_crs_eff = src_crs
            elif use_worldfile:
                if transform is None or src_crs is None:
                    if verbose:
                        print(f"[skip] {p.name}: missing worldfile transform/CRS")
                    continue
                for px, py in poly_xy:
                    X, Y = pixel_to_projected(transform, px, py)
                    src_ring.append([float(X), float(Y)])
                # Convert to EPSG:4326 first as requested
                src_ring = transform_ring(src_ring, src_crs, "EPSG:4326")
                src_crs_eff = "EPSG:4326"
            else:
                if z is None or x is None or y is None:
                    if verbose:
                        print(f"[skip] {p.name}: missing z/x/y for XYZ conversion")
                    continue
                for px, py in poly_xy:
                    lon, lat = lonlat_from_xyz_pixel(z, x, y, px, py, w, h)
                    src_ring.append([float(lon), float(lat)])
                src_crs_eff = "EPSG:4326"

            ring_3857 = transform_ring(src_ring, src_crs_eff, "EPSG:3857")
            ring_3857 = _normalize_ring(ring_3857, precision)
            if len(ring_3857) < 4:
                if verbose:
                    print(f"[skip] {p.name}: invalid ring in EPSG:3857")
                continue

            ring_31983 = transform_ring(ring_3857, "EPSG:3857", "EPSG:31983")
            ring_31983 = _normalize_ring(ring_31983, precision)
            if len(ring_31983) < 4:
                if verbose:
                    print(f"[skip] {p.name}: invalid ring in EPSG:31983")
                continue

            poly_31983 = Polygon(ring_31983)
            if poly_31983.is_empty or (not poly_31983.is_valid) or poly_31983.area == 0.0:
                stats.polys_dropped_area_m2 += 1
                continue

            area_m2 = float(poly_31983.area)
            if min_area_m2 > 0.0 and area_m2 < min_area_m2:
                stats.polys_dropped_area_m2 += 1
                continue

            stats.polys_kept += 1
            confidence = confs[idx] if idx < len(confs) else float(conf)

            props = {
                "tile": p.name,
                "z": z,
                "x": x,
                "y": y,
                "mask_idx": idx,
                "mask_area_px": mask_area_px,
                "area_m2": area_m2,
                "confidence": confidence,
            }

            features_3857.append(
                {
                    "type": "Feature",
                    "properties": props,
                    "geometry": {"type": "Polygon", "coordinates": [ring_3857]},
                }
            )
            features_31983.append(
                {
                    "type": "Feature",
                    "properties": props,
                    "geometry": {"type": "Polygon", "coordinates": [ring_31983]},
                }
            )

        if verbose and stats.tiles_processed % 50 == 0:
            print(
                f"[progress] tiles={stats.tiles_processed} features={len(features_3857)} "
                f"tiles_with_masks={stats.tiles_with_masks}"
            )

    return features_3857, features_31983, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--tiles-dir", required=True, type=Path)
    ap.add_argument("--out-geojson", required=True, type=Path)
    ap.add_argument("--out-geojson-3857", type=Path, default=None)

    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--min-area-px", type=float, default=120.0)
    ap.add_argument("--min-mask-area-px", type=float, default=120.0)
    ap.add_argument("--min-area-m2", type=float, default=6.0, help="Drop polygons smaller than this area in EPSG:31983")
    ap.add_argument("--max-tiles", type=int, default=0, help="0 = all tiles, else limit for testing")
    ap.add_argument("--precision", type=int, default=7, help="decimal places for output coordinate rounding")
    ap.add_argument("--z", type=int, default=None, help="XYZ zoom (required for non-GeoTIFF tiles)")
    ap.add_argument("--worldfile-crs", type=str, default="EPSG:31983")
    ap.add_argument("--no-recursive", action="store_true", default=False)

    # Inference knobs
    ap.add_argument(
        "--retina-masks",
        action="store_true",
        default=True,
        help="Use retina_masks=True (recommended for clean polygons). Default: on.",
    )
    ap.add_argument("--no-retina-masks", dest="retina_masks", action="store_false", help="Disable retina_masks.")
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--device", type=str, default=None, help="e.g. 'cpu', '0' for GPU 0")
    ap.add_argument("--verbose", action="store_true", default=False)

    args = ap.parse_args()

    tiles = iter_images(args.tiles_dir, recursive=not args.no_recursive)
    if not tiles:
        raise SystemExit(f"No images found in {args.tiles_dir}")

    if args.max_tiles and args.max_tiles > 0:
        tiles = tiles[: args.max_tiles]

    images_found_total = len(tiles)
    images_with_worldfile = 0
    images_geotiff = 0
    images_xyz = 0

    for p in tiles:
        if is_geotiff(p):
            images_geotiff += 1
            continue
        wf = find_worldfile(p)
        if wf is not None:
            images_with_worldfile += 1
        else:
            images_xyz += 1

    print("images_found_total:", images_found_total)
    print("images_with_worldfile:", images_with_worldfile)
    print("images_geotiff:", images_geotiff)
    print("images_xyz:", images_xyz)

    if images_xyz > 0 and args.z is None:
        raise SystemExit("--z is required when using non-GeoTIFF XYZ tiles")

    out_geojson_3857 = args.out_geojson_3857
    if out_geojson_3857 is None:
        if args.out_geojson.suffix.lower() == ".geojson":
            out_geojson_3857 = args.out_geojson.with_name(args.out_geojson.stem + "_3857.geojson")
        else:
            out_geojson_3857 = args.out_geojson.with_name(args.out_geojson.name + "_3857.geojson")

    features_3857, features_31983, stats = run_inference(
        model_path=args.model,
        tiles=tiles,
        z=args.z,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        min_area_px=args.min_area_px,
        min_mask_area_px=args.min_mask_area_px,
        min_area_m2=args.min_area_m2,
        max_det=args.max_det,
        retina_masks=bool(args.retina_masks),
        device=args.device,
        verbose=args.verbose,
        precision=args.precision,
        worldfile_crs=args.worldfile_crs,
    )

    def _sort_key(feat: Dict[str, Any]) -> Tuple[str, int]:
        props = feat.get("properties", {})
        tile = props.get("tile") or ""
        idx = props.get("mask_idx")
        try:
            idx_val = int(idx)
        except Exception:
            idx_val = 0
        return (tile, idx_val)

    features_3857 = sorted(features_3857, key=_sort_key)
    features_31983 = sorted(features_31983, key=_sort_key)

    out_3857 = {
        "type": "FeatureCollection",
        "features": features_3857,
        "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
    }
    out_31983 = {
        "type": "FeatureCollection",
        "features": features_31983,
        "crs": {"type": "name", "properties": {"name": "EPSG:31983"}},
    }

    out_geojson_3857.parent.mkdir(parents=True, exist_ok=True)
    out_geojson_3857.write_text(
        json.dumps(out_3857, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8"
    )
    args.out_geojson.parent.mkdir(parents=True, exist_ok=True)
    args.out_geojson.write_text(
        json.dumps(out_31983, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8"
    )

    print("Wrote:", out_geojson_3857)
    print("Wrote:", args.out_geojson)
    print("Tiles processed:", stats.tiles_processed)
    print("Tiles with masks:", stats.tiles_with_masks)
    print("Polys total:", stats.polys_total)
    print("Polys scaled_from_norm:", stats.polys_scaled_from_norm)
    print("Polys dropped_area_px:", stats.polys_dropped_mask_area + stats.polys_dropped_poly_area)
    print("Polys dropped_area_m2:", stats.polys_dropped_area_m2)
    print("Polys dropped_mask_area:", stats.polys_dropped_mask_area)
    print("Polys dropped_poly_area:", stats.polys_dropped_poly_area)
    print("Polys kept:", stats.polys_kept)
    print("Features (3857):", len(features_3857))
    print("Features (31983):", len(features_31983))


if __name__ == "__main__":
    main()
