#!/usr/bin/env python3
"""
Audit remaining overlaps in a GeoJSON FeatureCollection using IoU.

Exits nonzero if any pair exceeds the IoU threshold (default 0.35).
"""

import argparse
import heapq
import json
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.geometry.base import BaseGeometry
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
class AuditItem:
    idx: int
    geom: BaseGeometry
    bbox: Tuple[float, float, float, float]
    fid: str


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
    if g.is_empty:
        return g

    if tiny_buffer != 0.0:
        try:
            g = g.buffer(tiny_buffer)
        except Exception:
            pass

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
    if a.is_empty or b.is_empty:
        return 0.0

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
    for geom in candidates:
        idx = id_to_index.get(id(geom))
        if idx is not None:
            indices.append(idx)
    return indices


def _feature_id(feature: Dict[str, Any], idx: int) -> str:
    if "id" in feature and feature["id"] is not None:
        return str(feature["id"])
    props = feature.get("properties") or {}
    if "id" in props and props["id"] is not None:
        return str(props["id"])
    return f"idx:{idx}"


def _track_top_pairs(
    heap: List[Tuple[float, str, str]],
    iou: float,
    id_a: str,
    id_b: str,
    k: int,
) -> None:
    if iou <= 0:
        return
    entry = (iou, id_a, id_b)
    if len(heap) < k:
        heapq.heappush(heap, entry)
        return
    if iou > heap[0][0]:
        heapq.heapreplace(heap, entry)


def audit_feature_collection(
    fc: Dict[str, Any],
    *,
    iou: float = 0.35,
    top_k: int = 5,
    buffer: float = 0.0,
    grid_size: float = 0.0,
) -> Tuple[int, int, int, float, List[Tuple[float, str, str]]]:
    feats = fc.get("features", [])
    if not isinstance(feats, list):
        raise ValueError("Input GeoJSON missing 'features' list.")

    parsed: List[AuditItem] = []
    for idx, f in enumerate(feats):
        geom = f.get("geometry")
        if not geom:
            continue
        try:
            g = shape(geom)
        except Exception:
            continue
        g = _fix_geom(g, tiny_buffer=buffer)
        g = _polygonal_or_empty(g)
        if g.is_empty or g.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        if not _safe_is_valid(g):
            continue
        parsed.append(AuditItem(idx=idx, geom=g, bbox=g.bounds, fid=_feature_id(f, idx)))

    geoms = [item.geom for item in parsed]
    tree = STRtree(geoms) if geoms else None
    id_to_index = {id(g): i for i, g in enumerate(geoms)}

    candidate_pairs_considered = 0
    num_pairs_iou_ge_threshold = 0
    max_iou_remaining = 0.0
    top_heap: List[Tuple[float, str, str]] = []

    for i, item in enumerate(parsed):
        if tree is None:
            break
        candidates = tree.query(item.geom)
        cand_indices = _candidate_indices(candidates, id_to_index)
        for j in cand_indices:
            if j <= i:
                continue
            if not _bbox_intersects(item.bbox, parsed[j].bbox):
                continue
            candidate_pairs_considered += 1
            v = _safe_iou(item.geom, parsed[j].geom, grid_size=grid_size)
            if v > max_iou_remaining:
                max_iou_remaining = v
            if v >= iou:
                num_pairs_iou_ge_threshold += 1
            _track_top_pairs(top_heap, v, item.fid, parsed[j].fid, top_k)

    top_pairs = sorted(top_heap, key=lambda t: (-t[0], t[1], t[2]))
    return (
        len(feats),
        candidate_pairs_considered,
        num_pairs_iou_ge_threshold,
        max_iou_remaining,
        top_pairs,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-geojson", required=True)
    ap.add_argument("--iou", type=float, default=0.35)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--buffer", type=float, default=0.0)
    ap.add_argument("--grid-size", type=float, default=0.0)
    args = ap.parse_args()

    with open(args.in_geojson, "r", encoding="utf-8") as r:
        fc = json.load(r)

    (
        total_features,
        candidate_pairs_considered,
        num_pairs_iou_ge_threshold,
        max_iou_remaining,
        top_pairs,
    ) = audit_feature_collection(
        fc,
        iou=args.iou,
        top_k=args.top_k,
        buffer=args.buffer,
        grid_size=args.grid_size,
    )

    print(f"total_features: {total_features}")
    print(f"candidate_pairs_considered: {candidate_pairs_considered}")
    print(f"num_pairs_iou_ge_threshold: {num_pairs_iou_ge_threshold}")
    print(f"max_iou_remaining: {max_iou_remaining}")
    top_pairs_payload = [
        {"id_a": a, "id_b": b, "iou": round(v, 6)} for v, a, b in top_pairs
    ]
    print(f"top_k_remaining_overlaps: {json.dumps(top_pairs_payload, ensure_ascii=False)}")

    if num_pairs_iou_ge_threshold > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
