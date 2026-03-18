
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
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
    tiles_inference_attempted: int = 0
    tiles_with_masks: int = 0
    tiles_without_masks: int = 0
    tiles_flagged_blank_white: int = 0
    tiles_skipped_geotiff_read_error: int = 0
    tiles_skipped_image_open_error: int = 0
    tiles_skipped_worldfile_read_error: int = 0
    tiles_skipped_parse_xy_error: int = 0
    tiles_skipped_missing_z_for_xyz: int = 0
    tiles_skipped_model_error: int = 0
    polys_total: int = 0
    polys_kept: int = 0
    polys_scaled_from_norm: int = 0
    polys_dropped_too_few_vertices: int = 0
    polys_dropped_mask_area: int = 0
    polys_dropped_poly_area: int = 0
    polys_dropped_missing_georef: int = 0
    polys_dropped_invalid_ring_3857: int = 0
    polys_dropped_invalid_ring_31983: int = 0
    polys_dropped_invalid_geom: int = 0
    polys_dropped_area_m2_threshold: int = 0
    polys_dropped_area_m2: int = 0


GeoTransform = Tuple[float, float, float, float, float, float]
_TRANSFORMERS: Dict[Tuple[str, str], Transformer] = {}


def mercator_lat_from_t(t: float) -> float:
    return math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * t))))


def lonlat_from_xyz_pixel(z: int, x: int, y: int, px: float, py: float, w: int, h: int) -> Tuple[float, float]:
    n = 2 ** z
    fx = x + (px / w)
    fy = y + (py / h)
    lon = (fx / n) * 360.0 - 180.0
    lat = mercator_lat_from_t(fy / n)
    return lon, lat


def parse_xy_from_path(p: Path) -> Tuple[int, int]:
    stem = p.stem
    a, b = stem.split("_", 1)
    return int(a), int(b)


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
        from osgeo import gdal
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
        candidates.extend([img.with_suffix(".pgw"), img.with_suffix(".wld"), img.with_suffix(".pngw")])
    elif suffix in (".jpg", ".jpeg"):
        candidates.extend([img.with_suffix(".jgw"), img.with_suffix(".wld"), img.with_suffix(".jpgw")])
    else:
        candidates.extend([img.with_suffix(".wld")])
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
        if arr is None or getattr(arr, "ndim", None) != 3:
            return None
        mask = arr > 0.5
        areas = mask.sum(axis=(1, 2))
        return [float(v) for v in areas]
    except Exception:
        return None


def path_parts_info(tiles_root: Path, p: Path) -> Tuple[str, str, str]:
    rel = p.relative_to(tiles_root)
    tile_rel = rel.as_posix()
    cell = ""
    for part in rel.parts:
        if part.startswith("cell_"):
            cell = part
            break
    return tile_rel, cell, rel.stem


def is_blank_white_tile(p: Path, mean_thresh: float, std_thresh: float) -> bool:
    try:
        from PIL import Image, ImageStat
        im = Image.open(p).convert("RGB")
        stat = ImageStat.Stat(im)
        mean = sum(stat.mean) / 3.0
        std = sum(stat.stddev) / 3.0
        return mean >= mean_thresh and std <= std_thresh
    except Exception:
        return False


def append_tile_summary(
    rows: List[Dict[str, Any]],
    *,
    tile_rel: str,
    tile_name: str,
    tile_stem: str,
    cell: str,
    tile_path_abs: str,
    blank_white: bool,
    num_preds: int,
    min_conf: Optional[float],
    mean_conf: Optional[float],
    max_conf: Optional[float],
    max_area_m2: Optional[float],
    sum_area_m2: float,
    max_mask_area_px: Optional[float],
    model_masks: int = 0,
    dropped_too_few_vertices: int = 0,
    dropped_mask_area: int = 0,
    dropped_poly_area: int = 0,
    dropped_invalid_geo: int = 0,
    dropped_area_m2: int = 0,
    skip_reason: str = "",
) -> None:
    rows.append(
        {
            "tile_rel": tile_rel,
            "tile": tile_name,
            "tile_stem": tile_stem,
            "cell": cell,
            "tile_path_abs": tile_path_abs,
            "blank_white": int(bool(blank_white)),
            "num_preds": int(num_preds),
            "min_conf": "" if min_conf is None else float(min_conf),
            "mean_conf": "" if mean_conf is None else float(mean_conf),
            "max_conf": "" if max_conf is None else float(max_conf),
            "max_area_m2": "" if max_area_m2 is None else float(max_area_m2),
            "sum_area_m2": float(sum_area_m2),
            "max_mask_area_px": "" if max_mask_area_px is None else float(max_mask_area_px),
            "model_masks": int(model_masks),
            "dropped_too_few_vertices": int(dropped_too_few_vertices),
            "dropped_mask_area": int(dropped_mask_area),
            "dropped_poly_area": int(dropped_poly_area),
            "dropped_invalid_geo": int(dropped_invalid_geo),
            "dropped_area_m2": int(dropped_area_m2),
            "skip_reason": str(skip_reason),
        }
    )


def run_inference(
    *,
    model_path: Path,
    tiles_root: Path,
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
    tile_summary_rows: Optional[List[Dict[str, Any]]],
    white_mean_threshold: float,
    white_std_threshold: float,
    progress_every: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], InferenceStats]:
    from PIL import Image

    model = YOLO(str(model_path))

    features_3857: List[Dict[str, Any]] = []
    features_31983: List[Dict[str, Any]] = []
    stats = InferenceStats()
    total_tiles = len(tiles)
    started_at = time.monotonic()

    for p in tiles:
        stats.tiles_processed += 1
        should_print_progress = False
        if verbose and stats.tiles_processed % 50 == 0:
            should_print_progress = True
        if progress_every > 0 and stats.tiles_processed % progress_every == 0:
            should_print_progress = True
        if should_print_progress:
            elapsed_s = max(time.monotonic() - started_at, 1.0)
            rate_tps = stats.tiles_processed / elapsed_s
            remaining_tiles = max(total_tiles - stats.tiles_processed, 0)
            eta_min = (remaining_tiles / rate_tps) / 60.0 if rate_tps > 0 else float("inf")
            print(
                f"[progress] tiles={stats.tiles_processed}/{total_tiles} "
                f"features={len(features_3857)} tiles_with_masks={stats.tiles_with_masks} "
                f"rate_tps={rate_tps:.2f} eta_min={eta_min:.1f}"
            )

        tile_rel, cell, tile_stem = path_parts_info(tiles_root, p)
        blank_white = is_blank_white_tile(p, white_mean_threshold, white_std_threshold)
        if blank_white:
            stats.tiles_flagged_blank_white += 1

        use_geotiff = is_geotiff(p)
        worldfile = None if use_geotiff else find_worldfile(p)
        use_worldfile = worldfile is not None
        img = None
        src_crs: Optional[str] = None
        transform: Optional[GeoTransform] = None
        x: Optional[int] = None
        y: Optional[int] = None

        tile_model_masks = 0
        tile_dropped_too_few_vertices = 0
        tile_dropped_mask_area = 0
        tile_dropped_poly_area = 0
        tile_dropped_invalid_geo = 0
        tile_dropped_area_m2 = 0
        kept_tile_confs: List[float] = []
        kept_tile_areas_m2: List[float] = []
        kept_tile_mask_areas_px: List[float] = []

        def append_summary_row(skip_reason: str = "") -> None:
            if tile_summary_rows is None:
                return
            if kept_tile_confs:
                min_conf = min(kept_tile_confs)
                mean_conf = float(sum(kept_tile_confs) / len(kept_tile_confs))
                max_conf = max(kept_tile_confs)
                max_area_m2 = max(kept_tile_areas_m2)
                sum_area_m2 = float(sum(kept_tile_areas_m2))
                max_mask_area_px = max(kept_tile_mask_areas_px) if kept_tile_mask_areas_px else None
            else:
                min_conf = None
                mean_conf = None
                max_conf = None
                max_area_m2 = None
                sum_area_m2 = 0.0
                max_mask_area_px = None

            append_tile_summary(
                tile_summary_rows,
                tile_rel=tile_rel,
                tile_name=p.name,
                tile_stem=tile_stem,
                cell=cell,
                tile_path_abs=str(p.resolve()),
                blank_white=blank_white,
                num_preds=len(kept_tile_confs),
                min_conf=min_conf,
                mean_conf=mean_conf,
                max_conf=max_conf,
                max_area_m2=max_area_m2,
                sum_area_m2=sum_area_m2,
                max_mask_area_px=max_mask_area_px,
                model_masks=tile_model_masks,
                dropped_too_few_vertices=tile_dropped_too_few_vertices,
                dropped_mask_area=tile_dropped_mask_area,
                dropped_poly_area=tile_dropped_poly_area,
                dropped_invalid_geo=tile_dropped_invalid_geo,
                dropped_area_m2=tile_dropped_area_m2,
                skip_reason=skip_reason,
            )

        if use_geotiff:
            try:
                img, transform, src_crs = read_geotiff_image(p)
            except Exception as e:
                if verbose:
                    print(f"[skip] {p.name}: cannot read GeoTIFF ({e})")
                stats.tiles_skipped_geotiff_read_error += 1
                append_summary_row(skip_reason="geotiff_read_error")
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
                stats.tiles_skipped_image_open_error += 1
                append_summary_row(skip_reason="image_open_error")
                continue

            if use_worldfile:
                try:
                    transform = read_worldfile(worldfile)
                    src_crs = normalize_crs(worldfile_crs) or worldfile_crs
                except Exception as e:
                    if verbose:
                        print(f"[skip] {p.name}: cannot read worldfile ({e})")
                    stats.tiles_skipped_worldfile_read_error += 1
                    append_summary_row(skip_reason="worldfile_read_error")
                    continue
            else:
                try:
                    x, y = parse_xy_from_path(p)
                except Exception as e:
                    if verbose:
                        print(f"[skip] {p.name}: cannot parse x_y ({e})")
                    stats.tiles_skipped_parse_xy_error += 1
                    append_summary_row(skip_reason="parse_xy_error")
                    continue

        if not use_geotiff and (not use_worldfile and z is None):
            if verbose:
                print(f"[skip] {p.name}: missing --z for XYZ tiles")
            stats.tiles_skipped_missing_z_for_xyz += 1
            append_summary_row(skip_reason="missing_z_for_xyz")
            continue

        try:
            stats.tiles_inference_attempted += 1
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
        except Exception as e:
            if verbose:
                print(f"[skip] {p.name}: model.predict failed ({e})")
            stats.tiles_skipped_model_error += 1
            append_summary_row(skip_reason="model_predict_error")
            continue

        r = results[0]
        if r.masks is None or r.masks.xy is None:
            stats.tiles_without_masks += 1
            append_summary_row(skip_reason="no_masks")
            continue

        stats.tiles_with_masks += 1
        tile_model_masks = len(r.masks.xy)
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

            if len(poly_xy) < 3:
                stats.polys_dropped_too_few_vertices += 1
                tile_dropped_too_few_vertices += 1
                continue

            if looks_normalized(poly_xy):
                poly_xy = scale_norm_to_px(poly_xy, w, h)
                stats.polys_scaled_from_norm += 1

            mask_area_px = None
            if mask_areas is not None and idx < len(mask_areas):
                mask_area_px = mask_areas[idx]

            if mask_area_px is not None:
                if mask_area_px < min_mask_area_px:
                    stats.polys_dropped_mask_area += 1
                    tile_dropped_mask_area += 1
                    continue
            else:
                if poly_area_px(poly_xy) < min_area_px:
                    stats.polys_dropped_poly_area += 1
                    tile_dropped_poly_area += 1
                    continue

            src_ring: List[List[float]] = []
            if use_geotiff:
                if transform is None or src_crs is None:
                    if verbose:
                        print(f"[skip] {p.name}: missing geotransform/CRS")
                    stats.polys_dropped_missing_georef += 1
                    tile_dropped_invalid_geo += 1
                    continue
                for px, py in poly_xy:
                    X, Y = pixel_to_projected(transform, px, py)
                    src_ring.append([float(X), float(Y)])
                src_crs_eff = src_crs
            elif use_worldfile:
                if transform is None or src_crs is None:
                    if verbose:
                        print(f"[skip] {p.name}: missing worldfile transform/CRS")
                    stats.polys_dropped_missing_georef += 1
                    tile_dropped_invalid_geo += 1
                    continue
                for px, py in poly_xy:
                    X, Y = pixel_to_projected(transform, px, py)
                    src_ring.append([float(X), float(Y)])
                src_ring = transform_ring(src_ring, src_crs, "EPSG:4326")
                src_crs_eff = "EPSG:4326"
            else:
                if z is None or x is None or y is None:
                    if verbose:
                        print(f"[skip] {p.name}: missing z/x/y for XYZ conversion")
                    stats.polys_dropped_missing_georef += 1
                    tile_dropped_invalid_geo += 1
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
                stats.polys_dropped_invalid_ring_3857 += 1
                tile_dropped_invalid_geo += 1
                continue

            ring_31983 = transform_ring(ring_3857, "EPSG:3857", "EPSG:31983")
            ring_31983 = _normalize_ring(ring_31983, precision)
            if len(ring_31983) < 4:
                if verbose:
                    print(f"[skip] {p.name}: invalid ring in EPSG:31983")
                stats.polys_dropped_invalid_ring_31983 += 1
                tile_dropped_invalid_geo += 1
                continue

            poly_31983 = Polygon(ring_31983)
            if poly_31983.is_empty or (not poly_31983.is_valid) or poly_31983.area == 0.0:
                stats.polys_dropped_invalid_geom += 1
                stats.polys_dropped_area_m2 += 1
                tile_dropped_invalid_geo += 1
                continue

            area_m2 = float(poly_31983.area)
            if min_area_m2 > 0.0 and area_m2 < min_area_m2:
                stats.polys_dropped_area_m2_threshold += 1
                stats.polys_dropped_area_m2 += 1
                tile_dropped_area_m2 += 1
                continue

            stats.polys_kept += 1
            confidence = confs[idx] if idx < len(confs) else float(conf)

            kept_tile_confs.append(float(confidence))
            kept_tile_areas_m2.append(float(area_m2))
            if mask_area_px is not None:
                kept_tile_mask_areas_px.append(float(mask_area_px))

            props = {
                "tile": p.name,
                "tile_rel": tile_rel,
                "tile_stem": tile_stem,
                "tile_path_abs": str(p.resolve()),
                "cell": cell,
                "z": z,
                "x": x,
                "y": y,
                "mask_idx": idx,
                "mask_area_px": mask_area_px,
                "area_m2": area_m2,
                "confidence": confidence,
                "blank_white_tile": bool(blank_white),
            }

            features_3857.append(
                {"type": "Feature", "properties": props, "geometry": {"type": "Polygon", "coordinates": [ring_3857]}}
            )
            features_31983.append(
                {"type": "Feature", "properties": props, "geometry": {"type": "Polygon", "coordinates": [ring_31983]}}
            )

        append_summary_row()

    return features_3857, features_31983, stats


def write_tile_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tile_rel",
        "tile",
        "tile_stem",
        "cell",
        "tile_path_abs",
        "blank_white",
        "num_preds",
        "min_conf",
        "mean_conf",
        "max_conf",
        "max_area_m2",
        "sum_area_m2",
        "max_mask_area_px",
        "model_masks",
        "dropped_too_few_vertices",
        "dropped_mask_area",
        "dropped_poly_area",
        "dropped_invalid_geo",
        "dropped_area_m2",
        "skip_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_tile_summary_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--tiles-dir", required=True, type=Path)
    ap.add_argument("--out-geojson", required=True, type=Path)
    ap.add_argument("--out-geojson-3857", type=Path, default=None)
    ap.add_argument("--out-tile-summary-csv", type=Path, default=None)
    ap.add_argument("--out-tile-summary-jsonl", type=Path, default=None)
    ap.add_argument("--out-stats-json", type=Path, default=None, help="Optional run-level stats JSON path")

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
    ap.add_argument("--white-mean-threshold", type=float, default=245.0)
    ap.add_argument("--white-std-threshold", type=float, default=8.0)

    ap.add_argument("--retina-masks", action="store_true", default=True, help="Use retina_masks=True (recommended for clean polygons). Default: on.")
    ap.add_argument("--no-retina-masks", dest="retina_masks", action="store_false", help="Disable retina_masks.")
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--device", type=str, default=None, help="e.g. 'cpu', '0' for GPU 0")
    ap.add_argument("--verbose", action="store_true", default=False)
    ap.add_argument("--progress-every", type=int, default=500, help="Print progress every N tiles (0 disables)")
    ap.add_argument(
        "--recall-profile-2024",
        action="store_true",
        default=False,
        help=(
            "Apply recall-first thresholds tuned for 2024 imagery: "
            "conf<=0.05, iou<=0.5, min-area-px<=30, min-mask-area-px<=30, "
            "min-area-m2<=3, max-det>=600"
        ),
    )

    args = ap.parse_args()

    if args.recall_profile_2024:
        args.conf = min(float(args.conf), 0.05)
        args.iou = min(float(args.iou), 0.5)
        args.min_area_px = min(float(args.min_area_px), 30.0)
        args.min_mask_area_px = min(float(args.min_mask_area_px), 30.0)
        args.min_area_m2 = min(float(args.min_area_m2), 3.0)
        args.max_det = max(int(args.max_det), 600)

    tiles_root = args.tiles_dir.resolve()
    tiles = [p.resolve() for p in iter_images(args.tiles_dir, recursive=not args.no_recursive)]
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

    out_tile_summary_csv = args.out_tile_summary_csv
    if out_tile_summary_csv is None:
        if args.out_geojson.suffix.lower() == ".geojson":
            out_tile_summary_csv = args.out_geojson.with_name(args.out_geojson.stem + "_tiles.csv")
        else:
            out_tile_summary_csv = args.out_geojson.with_name(args.out_geojson.name + "_tiles.csv")

    out_tile_summary_jsonl = args.out_tile_summary_jsonl
    if out_tile_summary_jsonl is None:
        if args.out_geojson.suffix.lower() == ".geojson":
            out_tile_summary_jsonl = args.out_geojson.with_name(args.out_geojson.stem + "_tiles.jsonl")
        else:
            out_tile_summary_jsonl = args.out_geojson.with_name(args.out_geojson.name + "_tiles.jsonl")

    out_stats_json = args.out_stats_json
    if out_stats_json is None:
        if args.out_geojson.suffix.lower() == ".geojson":
            out_stats_json = args.out_geojson.with_name(args.out_geojson.stem + "_stats.json")
        else:
            out_stats_json = args.out_geojson.with_name(args.out_geojson.name + "_stats.json")

    tile_summary_rows: List[Dict[str, Any]] = []

    features_3857, features_31983, stats = run_inference(
        model_path=args.model,
        tiles_root=tiles_root,
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
        tile_summary_rows=tile_summary_rows,
        white_mean_threshold=args.white_mean_threshold,
        white_std_threshold=args.white_std_threshold,
        progress_every=args.progress_every,
    )

    def _sort_key(feat: Dict[str, Any]) -> Tuple[str, int]:
        props = feat.get("properties", {})
        tile_rel = props.get("tile_rel") or props.get("tile") or ""
        idx = props.get("mask_idx")
        try:
            idx_val = int(idx)
        except Exception:
            idx_val = 0
        return (tile_rel, idx_val)

    features_3857 = sorted(features_3857, key=_sort_key)
    features_31983 = sorted(features_31983, key=_sort_key)
    tile_summary_rows = sorted(tile_summary_rows, key=lambda row: row["tile_rel"])

    out_3857 = {"type": "FeatureCollection", "features": features_3857, "crs": {"type": "name", "properties": {"name": "EPSG:3857"}}}
    out_31983 = {"type": "FeatureCollection", "features": features_31983, "crs": {"type": "name", "properties": {"name": "EPSG:31983"}}}

    out_geojson_3857.parent.mkdir(parents=True, exist_ok=True)
    out_geojson_3857.write_text(json.dumps(out_3857, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    args.out_geojson.parent.mkdir(parents=True, exist_ok=True)
    args.out_geojson.write_text(json.dumps(out_31983, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")

    write_tile_summary_csv(out_tile_summary_csv, tile_summary_rows)
    write_tile_summary_jsonl(out_tile_summary_jsonl, tile_summary_rows)

    dropped_invalid_geo = (
        stats.polys_dropped_missing_georef
        + stats.polys_dropped_invalid_ring_3857
        + stats.polys_dropped_invalid_ring_31983
        + stats.polys_dropped_invalid_geom
    )
    tiles_with_predictions = sum(1 for row in tile_summary_rows if int(row.get("num_preds", 0)) > 0)
    stats_payload = {
        "model": str(args.model.resolve()),
        "tiles_root": str(tiles_root),
        "images_found_total": images_found_total,
        "images_with_worldfile": images_with_worldfile,
        "images_geotiff": images_geotiff,
        "images_xyz": images_xyz,
        "inference_params": {
            "imgsz": int(args.imgsz),
            "conf": float(args.conf),
            "iou": float(args.iou),
            "min_area_px": float(args.min_area_px),
            "min_mask_area_px": float(args.min_mask_area_px),
            "min_area_m2": float(args.min_area_m2),
            "max_det": int(args.max_det),
            "retina_masks": bool(args.retina_masks),
            "worldfile_crs": str(args.worldfile_crs),
            "precision": int(args.precision),
            "z": args.z,
            "recall_profile_2024": bool(args.recall_profile_2024),
        },
        "inference_stats": asdict(stats),
        "derived": {
            "features_3857": len(features_3857),
            "features_31983": len(features_31983),
            "tiles_with_predictions": tiles_with_predictions,
            "tiles_without_predictions": len(tile_summary_rows) - tiles_with_predictions,
            "tiles_with_predictions_rate": (tiles_with_predictions / len(tile_summary_rows)) if tile_summary_rows else 0.0,
            "polys_dropped_area_px": stats.polys_dropped_mask_area + stats.polys_dropped_poly_area,
            "polys_dropped_invalid_geo": dropped_invalid_geo,
        },
    }
    out_stats_json.parent.mkdir(parents=True, exist_ok=True)
    out_stats_json.write_text(
        json.dumps(stats_payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("Wrote:", out_geojson_3857)
    print("Wrote:", args.out_geojson)
    print("Wrote:", out_tile_summary_csv)
    print("Wrote:", out_tile_summary_jsonl)
    print("Wrote:", out_stats_json)
    print("Tiles processed:", stats.tiles_processed)
    print("Tiles inference_attempted:", stats.tiles_inference_attempted)
    print("Tiles with masks:", stats.tiles_with_masks)
    print("Tiles without masks:", stats.tiles_without_masks)
    print("Tiles flagged_blank_white:", stats.tiles_flagged_blank_white)
    print("Tiles skipped_geotiff_read_error:", stats.tiles_skipped_geotiff_read_error)
    print("Tiles skipped_image_open_error:", stats.tiles_skipped_image_open_error)
    print("Tiles skipped_worldfile_read_error:", stats.tiles_skipped_worldfile_read_error)
    print("Tiles skipped_parse_xy_error:", stats.tiles_skipped_parse_xy_error)
    print("Tiles skipped_missing_z_for_xyz:", stats.tiles_skipped_missing_z_for_xyz)
    print("Tiles skipped_model_error:", stats.tiles_skipped_model_error)
    print("Polys total:", stats.polys_total)
    print("Polys scaled_from_norm:", stats.polys_scaled_from_norm)
    print("Polys dropped_too_few_vertices:", stats.polys_dropped_too_few_vertices)
    print("Polys dropped_area_px:", stats.polys_dropped_mask_area + stats.polys_dropped_poly_area)
    print("Polys dropped_invalid_geo:", dropped_invalid_geo)
    print("Polys dropped_area_m2:", stats.polys_dropped_area_m2)
    print("Polys dropped_area_m2_threshold:", stats.polys_dropped_area_m2_threshold)
    print("Polys dropped_mask_area:", stats.polys_dropped_mask_area)
    print("Polys dropped_poly_area:", stats.polys_dropped_poly_area)
    print("Polys kept:", stats.polys_kept)
    print("Features (3857):", len(features_3857))
    print("Features (31983):", len(features_31983))


if __name__ == "__main__":
    main()
