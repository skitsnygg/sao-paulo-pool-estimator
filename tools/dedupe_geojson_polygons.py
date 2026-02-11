#!/usr/bin/env python3
"""
Deduplicate overlapping polygons in a GeoJSON FeatureCollection via greedy NMS.

Guarantee: features_out <= features_parsed (we only suppress; no splitting/union output expansion).

Fixes common Shapely/GEOS TopologyException issues by:
- make_valid() when available (Shapely 2.x)
- fallback buffer(0) fix
- robust try/except around union/intersection

Usage:
  python tools/dedupe_geojson_polygons.py \
    --in-geojson input.geojson \
    --out-geojson output.geojson \
    --iou 0.35 \
    --buffer 0 \
    --grid-size 0
"""

import argparse
import heapq
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from shapely.strtree import STRtree

# Shapely 2.x
try:
    from shapely.validation import make_valid  # type: ignore
except Exception:  # pragma: no cover
    make_valid = None  # type: ignore

try:  # optional, for STRtree index typing
    import numpy as np  # type: ignore

    _INT_TYPES = (int, np.integer)
except Exception:  # pragma: no cover
    _INT_TYPES = (int,)


@dataclass
class ParsedItem:
    idx: int
    geom: BaseGeometry
    feature: Dict[str, Any]
    score: float
    area: float
    bbox: Tuple[float, float, float, float]


@dataclass
class DedupeStats:
    features_in: int = 0
    repaired: int = 0
    still_invalid: int = 0
    dropped_small: int = 0
    candidate_pairs_considered: int = 0
    suppressed: int = 0
    top_overlap_ious: List[float] = field(default_factory=list)


def _bbox_intersects(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def _safe_is_valid(g: BaseGeometry) -> bool:
    try:
        return bool(g.is_valid)
    except Exception:
        return False


def _fix_geom(g: BaseGeometry, tiny_buffer: float = 0.0) -> BaseGeometry:
    """
    Attempt to make geometry valid/robust for overlay ops.
    tiny_buffer: optional tiny buffer in degrees to resolve self-intersections.
    """
    if g.is_empty:
        return g

    # Optional tiny buffer pre-fix (sometimes helps ring orientation / tiny slivers)
    if tiny_buffer != 0.0:
        try:
            g = g.buffer(tiny_buffer)
        except Exception:
            pass

    # Shapely 2.x make_valid
    if make_valid is not None:
        try:
            g2 = make_valid(g)
            if not g2.is_empty:
                g = g2
        except Exception:
            pass

    # Fallback classic fix
    try:
        if not g.is_valid:
            g2 = g.buffer(0)
            if not g2.is_empty:
                g = g2
    except Exception:
        pass

    return g


def _collect_polygons(g: BaseGeometry) -> List[Polygon]:
    if g.is_empty:
        return []
    if g.geom_type == "Polygon":
        return [g]  # type: ignore[list-item]
    if g.geom_type == "MultiPolygon":
        return list(g.geoms)  # type: ignore[return-value]
    if g.geom_type == "GeometryCollection":
        polys: List[Polygon] = []
        for part in g.geoms:  # type: ignore[union-attr]
            polys.extend(_collect_polygons(part))
        return polys
    return []


def _polygonal_or_empty(g: BaseGeometry) -> BaseGeometry:
    polys = _collect_polygons(g)
    if not polys:
        return Polygon()
    if len(polys) == 1:
        return polys[0]
    try:
        return unary_union(polys)
    except Exception:
        return MultiPolygon(polys)


def _safe_area(g: BaseGeometry) -> float:
    try:
        return float(g.area)
    except Exception:
        return 0.0


def _intersection(a: BaseGeometry, b: BaseGeometry, grid_size: float) -> BaseGeometry:
    if grid_size > 0:
        try:
            return a.intersection(b, grid_size=grid_size)
        except TypeError:
            return a.intersection(b)
    return a.intersection(b)


def _union(a: BaseGeometry, b: BaseGeometry, grid_size: float) -> BaseGeometry:
    if grid_size > 0:
        try:
            return a.union(b, grid_size=grid_size)
        except TypeError:
            return a.union(b)
    return a.union(b)


def _safe_iou(a: BaseGeometry, b: BaseGeometry, grid_size: float = 0.0) -> float:
    """
    Robust IoU with defensive exception handling.
    grid_size: if >0, snap overlay operations to a precision grid (reduces topology errors).
    """
    if a.is_empty or b.is_empty:
        return 0.0

    # Quick reject: bbox
    try:
        if not _bbox_intersects(a.bounds, b.bounds):
            return 0.0
    except Exception:
        pass

    try:
        inter = _intersection(a, b, grid_size).area
        if inter <= 0:
            return 0.0
        union = _union(a, b, grid_size).area
        if union <= 0:
            return 0.0
        return float(inter / union)
    except Exception:
        # Last-ditch: try again after buffer(0) on both
        try:
            a2 = _fix_geom(a, tiny_buffer=0.0)
            b2 = _fix_geom(b, tiny_buffer=0.0)
            inter = _intersection(a2, b2, grid_size).area
            if inter <= 0:
                return 0.0
            union = _union(a2, b2, grid_size).area
            if union <= 0:
                return 0.0
            return float(inter / union)
        except Exception:
            return 0.0


def _parse_score(props: Dict[str, Any], score_key: Optional[str]) -> Optional[float]:
    if not score_key:
        return None
    if score_key not in props:
        return None
    try:
        return float(props.get(score_key))
    except Exception:
        return None


def _track_top_ious(heap: List[float], value: float, k: int) -> None:
    if value <= 0:
        return
    if len(heap) < k:
        heapq.heappush(heap, value)
        return
    if value > heap[0]:
        heapq.heapreplace(heap, value)


def _orient_polygonal(g: BaseGeometry) -> BaseGeometry:
    if g.geom_type == "Polygon":
        return orient(g, sign=1.0)
    if g.geom_type == "MultiPolygon":
        return MultiPolygon([orient(p, sign=1.0) for p in g.geoms])
    return g


def _round_float(value: float, precision: int) -> float:
    v = float(round(float(value), precision))
    if v == -0.0:
        v = 0.0
    return v


def _round_point(pt: Sequence[float], precision: int) -> List[float]:
    return [_round_float(pt[0], precision), _round_float(pt[1], precision)]


def _round_ring(ring: Sequence[Sequence[float]], precision: int) -> List[List[float]]:
    rounded = [_round_point(pt, precision) for pt in ring]
    if rounded and rounded[0] != rounded[-1]:
        rounded[-1] = rounded[0][:]
    return rounded


def _round_polygon_coords(coords: Sequence[Sequence[Sequence[float]]], precision: int) -> List[List[List[float]]]:
    if not coords:
        return []
    exterior = _round_ring(coords[0], precision)
    interiors = [_round_ring(ring, precision) for ring in coords[1:]]
    return [exterior] + interiors


def _round_multipolygon_coords(
    coords: Sequence[Sequence[Sequence[Sequence[float]]]], precision: int
) -> List[List[List[List[float]]]]:
    return [_round_polygon_coords(poly, precision) for poly in coords]


def _round_coords_generic(coords: Any, precision: int) -> Any:
    if isinstance(coords, (list, tuple)):
        if coords and isinstance(coords[0], (int, float)):
            return [_round_float(c, precision) for c in coords]
        return [_round_coords_generic(c, precision) for c in coords]
    return coords


def _round_geojson_coords(geom: Dict[str, Any], precision: int) -> Dict[str, Any]:
    if precision is None:
        return geom
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    if gtype == "Polygon":
        geom["coordinates"] = _round_polygon_coords(coords, precision)
    elif gtype == "MultiPolygon":
        geom["coordinates"] = _round_multipolygon_coords(coords, precision)
    else:
        geom["coordinates"] = _round_coords_generic(coords, precision)
    return geom


def _geometry_to_geojson(g: BaseGeometry, precision: int) -> Dict[str, Any]:
    g_oriented = _orient_polygonal(g)
    geom = mapping(g_oriented)
    if precision is None:
        return geom

    geom = _round_geojson_coords(geom, precision)
    try:
        g_round = shape(geom)
    except Exception:
        g_round = g_oriented

    if not _safe_is_valid(g_round):
        g_round = _fix_geom(g_round, tiny_buffer=0.0)
        g_round = _polygonal_or_empty(g_round)

    g_round = _orient_polygonal(g_round)
    geom = mapping(g_round)
    geom = _round_geojson_coords(geom, precision)

    try:
        if not _safe_is_valid(shape(geom)):
            print("Warning: output geometry still invalid after repair", file=sys.stderr, flush=True)
    except Exception:
        pass

    return geom


def _candidate_indices(
    candidates: Optional[Sequence[Any]],
    id_to_index: Dict[int, int],
) -> List[int]:
    if candidates is None:
        return []
    try:
        if len(candidates) == 0:
            return []
    except TypeError:
        return []

    first = candidates[0]
    if isinstance(first, _INT_TYPES):
        return [int(i) for i in candidates]

    indices: List[int] = []
    for geom in candidates:  # type: ignore[assignment]
        idx = id_to_index.get(id(geom))
        if idx is not None:
            indices.append(idx)
    return indices


def dedupe_feature_collection(
    fc: Dict[str, Any],
    *,
    iou: float = 0.35,
    buffer: float = 0.0,
    grid_size: float = 0.0,
    score_key: Optional[str] = None,
    min_area: float = 0.0,
    precision: int = 7,
    top_k: int = 5,
) -> Tuple[Dict[str, Any], DedupeStats]:
    feats = fc.get("features", [])
    if not isinstance(feats, list):
        raise ValueError("Input GeoJSON missing 'features' list.")

    stats = DedupeStats(features_in=len(feats))

    parsed: List[ParsedItem] = []

    for i, f in enumerate(feats):
        geom = f.get("geometry")
        if not geom:
            stats.still_invalid += 1
            continue

        try:
            g = shape(geom)
        except Exception:
            stats.still_invalid += 1
            continue

        orig_valid = _safe_is_valid(g)
        g = _fix_geom(g, tiny_buffer=buffer)
        g = _polygonal_or_empty(g)

        if g.is_empty or g.geom_type not in ("Polygon", "MultiPolygon"):
            stats.still_invalid += 1
            continue

        if not _safe_is_valid(g):
            stats.still_invalid += 1
            continue

        if not orig_valid:
            stats.repaired += 1

        area = _safe_area(g)
        if area <= 0:
            stats.still_invalid += 1
            continue
        if min_area > 0 and area < min_area:
            stats.dropped_small += 1
            continue

        props = f.get("properties") or {}
        score = _parse_score(props, score_key)
        if score is None:
            score = float(area)  # fallback

        bbox = g.bounds
        parsed.append(
            ParsedItem(idx=i, geom=g, feature=f, score=float(score), area=float(area), bbox=bbox)
        )

    # Sort: highest score first, tie-breaker bigger area
    parsed.sort(key=lambda t: (t.score, t.area), reverse=True)

    geoms = [item.geom for item in parsed]
    tree = STRtree(geoms) if geoms else None
    id_to_index = {id(g): i for i, g in enumerate(geoms)}

    kept_mask = [False] * len(parsed)
    kept_items: List[ParsedItem] = []
    top_heap: List[float] = []

    for i, item in enumerate(parsed):
        dup = False

        if tree is not None:
            candidates = tree.query(item.geom)
            cand_indices = _candidate_indices(candidates, id_to_index)
            cand_indices.sort()
        else:
            cand_indices = []

        for j in cand_indices:
            if j >= i:
                continue
            if not kept_mask[j]:
                continue
            if not _bbox_intersects(item.bbox, parsed[j].bbox):
                continue

            stats.candidate_pairs_considered += 1
            v = _safe_iou(item.geom, parsed[j].geom, grid_size=grid_size)
            _track_top_ious(top_heap, v, top_k)
            if v >= iou:
                dup = True
                break

        if dup:
            stats.suppressed += 1
            continue

        kept_mask[i] = True
        kept_items.append(item)

    out_features: List[Dict[str, Any]] = []
    for item in kept_items:
        of = dict(item.feature)
        of["geometry"] = _geometry_to_geojson(item.geom, precision)
        out_features.append(of)

    out_fc = dict(fc)
    out_fc["type"] = "FeatureCollection"
    out_fc["features"] = out_features

    stats.top_overlap_ious = sorted(top_heap, reverse=True)
    return out_fc, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-geojson", required=True)
    ap.add_argument("--out-geojson", required=True)
    ap.add_argument("--iou", type=float, default=0.35, help="IoU threshold to consider duplicates")
    ap.add_argument("--buffer", type=float, default=0.0, help="optional tiny buffer in degrees (e.g. 1e-9)")
    ap.add_argument("--grid-size", type=float, default=0.0, help="optional precision grid size for overlay ops (0 disables)")
    ap.add_argument("--score-key", default=None, help="optional properties key for confidence score (higher wins)")
    ap.add_argument("--min-area", type=float, default=0.0, help="drop polygons with area below this (CRS units)")
    ap.add_argument("--precision", type=int, default=7, help="decimal places for output coordinate rounding")
    ap.add_argument("--top-k", type=int, default=5, help="number of top IoUs to report in --stats")
    ap.add_argument("--stats", action="store_true", help="print summary stats to stdout")
    args = ap.parse_args()

    with open(args.in_geojson, "r", encoding="utf-8") as r:
        fc = json.load(r)

    out_fc, stats = dedupe_feature_collection(
        fc,
        iou=args.iou,
        buffer=args.buffer,
        grid_size=args.grid_size,
        score_key=args.score_key,
        min_area=args.min_area,
        precision=args.precision,
        top_k=args.top_k,
    )

    with open(args.out_geojson, "w", encoding="utf-8") as w:
        json.dump(out_fc, w, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

    print(f"Wrote: {args.out_geojson}")
    if args.stats:
        print(f"features_in: {stats.features_in}")
        print(f"repaired: {stats.repaired}")
        print(f"still_invalid: {stats.still_invalid}")
        print(f"dropped_small: {stats.dropped_small}")
        print(f"candidate_pairs_considered: {stats.candidate_pairs_considered}")
        print(f"suppressed: {stats.suppressed}")
        print(f"top_overlap_ious: {stats.top_overlap_ious}")


if __name__ == "__main__":
    main()
