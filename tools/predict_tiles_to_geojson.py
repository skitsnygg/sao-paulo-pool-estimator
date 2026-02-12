from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from ultralytics import YOLO

from shapely.geometry import Polygon

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


def iter_images(d: Path) -> List[Path]:
    exts = (".jpg", ".jpeg", ".png", ".webp")
    out = []
    for e in exts:
        out.extend(d.glob(f"*{e}"))
        out.extend(d.glob(f"*{e.upper()}"))
    return sorted([p for p in out if p.is_file()])


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
    z: int,
    imgsz: int,
    conf: float,
    iou: float,
    min_area_px: float,
    max_det: int,
    retina_masks: bool,
    device: Optional[str],
    verbose: bool,
    precision: int,
) -> Tuple[List[Dict[str, Any]], InferenceStats]:
    from PIL import Image

    model = YOLO(str(model_path))

    features: List[Dict[str, Any]] = []
    stats = InferenceStats()

    for p in tiles:
        stats.tiles_processed += 1

        try:
            x, y = parse_xy_from_path(p)
        except Exception as e:
            if verbose:
                print(f"[skip] {p.name}: cannot parse x_y ({e})")
            continue

        try:
            w, h = Image.open(p).size
        except Exception as e:
            if verbose:
                print(f"[skip] {p.name}: cannot open image ({e})")
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
                if mask_area_px < min_area_px:
                    stats.polys_dropped_mask_area += 1
                    continue
            else:
                if poly_area_px(poly_xy) < min_area_px:
                    stats.polys_dropped_poly_area += 1
                    continue

            stats.polys_kept += 1

            ring_ll: List[List[float]] = []
            for px, py in poly_xy:
                lon, lat = lonlat_from_xyz_pixel(z, x, y, px, py, w, h)
                ring_ll.append([lon, lat])

            ring_ll = _normalize_ring(ring_ll, precision)
            if len(ring_ll) < 4:
                if verbose:
                    print(f"[skip] {p.name}: invalid ring after normalization")
                continue

            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "tile": p.name,
                        "z": z,
                        "x": x,
                        "y": y,
                        "mask_area_px": mask_area_px,
                    },
                    "geometry": {"type": "Polygon", "coordinates": [ring_ll]},
                }
            )

        if verbose and stats.tiles_processed % 50 == 0:
            print(
                f"[progress] tiles={stats.tiles_processed} features={len(features)} "
                f"tiles_with_masks={stats.tiles_with_masks}"
            )

    return features, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--tiles-dir", required=True, type=Path)
    ap.add_argument("--z", required=True, type=int)
    ap.add_argument("--out-geojson", required=True, type=Path)

    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--min-area-px", type=float, default=120.0)
    ap.add_argument("--max-tiles", type=int, default=0, help="0 = all tiles, else limit for testing")
    ap.add_argument("--precision", type=int, default=7, help="decimal places for output coordinate rounding")

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

    tiles = iter_images(args.tiles_dir)
    if not tiles:
        raise SystemExit(f"No images found in {args.tiles_dir}")

    if args.max_tiles and args.max_tiles > 0:
        tiles = tiles[: args.max_tiles]

    features, stats = run_inference(
        model_path=args.model,
        tiles=tiles,
        z=args.z,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        min_area_px=args.min_area_px,
        max_det=args.max_det,
        retina_masks=bool(args.retina_masks),
        device=args.device,
        verbose=args.verbose,
        precision=args.precision,
    )

    out = {"type": "FeatureCollection", "features": features}
    args.out_geojson.parent.mkdir(parents=True, exist_ok=True)
    args.out_geojson.write_text(
        json.dumps(out, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8"
    )

    print("Wrote:", args.out_geojson)
    print("Tiles processed:", stats.tiles_processed)
    print("Tiles with masks:", stats.tiles_with_masks)
    print("Polys total:", stats.polys_total)
    print("Polys scaled_from_norm:", stats.polys_scaled_from_norm)
    print("Polys dropped_area:", stats.polys_dropped_mask_area + stats.polys_dropped_poly_area)
    print("Polys dropped_mask_area:", stats.polys_dropped_mask_area)
    print("Polys dropped_poly_area:", stats.polys_dropped_poly_area)
    print("Polys kept:", stats.polys_kept)
    print("Features:", len(features))


if __name__ == "__main__":
    main()
