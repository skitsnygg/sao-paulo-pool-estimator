#!/usr/bin/env python3
"""
Deduplicate polygons in a GeoJSON using IoU threshold.
Keeps the larger polygon when IoU >= threshold.

Works with Shapely 1.8 and 2.x (STRtree.query may return geoms or indices).

Extras:
- --sweep "0.5,0.6,0.7,0.75,0.8,0.9" to print counts per threshold
- --write-dropped <path> to write removed features (exact keep-mask result)
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

import geopandas as gpd
from shapely.strtree import STRtree

try:
    import shapely
except Exception:  # pragma: no cover
    shapely = None  # type: ignore

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


def _fix_geom(g):
    if g is None or g.is_empty:
        return None
    if not g.is_valid:
        try:
            g = g.buffer(0)
        except Exception:
            return None
        if g is None or g.is_empty:
            return None
    return g


def _query_indices(tree: STRtree, id_to_idx: dict, geom) -> List[int]:
    """
    Return candidate indices for `geom`, compatible with Shapely 1.8 and 2.x.
    """
    cands = tree.query(geom)
    if len(cands) == 0:
        return []

    first = cands[0]

    # Shapely 2.x commonly returns indices (ints / numpy ints)
    if isinstance(first, int) or (np is not None and isinstance(first, np.integer)):
        return list(map(int, cands))

    # Shapely 1.8: may return geometry-like objects; map by Python id when possible
    out: List[int] = []
    for g in cands:
        j = id_to_idx.get(id(g))
        if j is not None:
            out.append(j)
    return out


def dedupe_iou_keep_mask(gdf: gpd.GeoDataFrame, iou_thresh: float) -> Tuple[gpd.GeoDataFrame, List[bool]]:
    """
    Returns (deduped_gdf, keep_mask_for_cleaned_gdf_rows).
    Note: keep_mask aligns to the cleaned/filtered gdf inside this function.
    """
    if "geometry" not in gdf:
        raise ValueError("No geometry column found.")

    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.apply(_fix_geom)
    gdf = gdf[~gdf.geometry.isna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    geoms: List = list(gdf.geometry.values)
    if len(geoms) == 0:
        return gdf.iloc[0:0].copy(), []

    tree = STRtree(geoms)
    id_to_idx = {id(g): i for i, g in enumerate(geoms)}

    keep = [True] * len(geoms)

    # big -> small so "keep larger" is stable
    order = sorted(range(len(geoms)), key=lambda i: geoms[i].area, reverse=True)

    for i in order:
        if not keep[i]:
            continue

        gi = geoms[i]
        if gi is None or gi.is_empty:
            keep[i] = False
            continue

        cand_idxs = _query_indices(tree, id_to_idx, gi)

        for j in cand_idxs:
            if j == i:
                continue
            if j < 0 or j >= len(geoms):
                continue
            if not keep[j]:
                continue

            gj = geoms[j]
            if gj is None or gj.is_empty:
                keep[j] = False
                continue

            if not gi.intersects(gj):
                continue

            try:
                inter = gi.intersection(gj).area
                if inter <= 0:
                    continue
                union = gi.union(gj).area
                if union <= 0:
                    continue
                iou = inter / union
            except Exception:
                continue

            if iou >= iou_thresh:
                # drop the smaller
                if gi.area >= gj.area:
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    out = gdf.iloc[[idx for idx, k in enumerate(keep) if k]].copy()
    return out, keep


def dedupe_iou(gdf: gpd.GeoDataFrame, iou_thresh: float) -> gpd.GeoDataFrame:
    out, _keep = dedupe_iou_keep_mask(gdf, iou_thresh)
    return out


def _parse_sweep(s: str) -> List[float]:
    vals = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-geojson", required=True, help="Input GeoJSON path")
    ap.add_argument("--out-geojson", help="Output GeoJSON path (optional; required unless --sweep)")
    ap.add_argument("--iou", type=float, default=0.75, help="IoU threshold (default: 0.75)")
    ap.add_argument("--sweep", default=None, help='Comma list, e.g. "0.5,0.6,0.7,0.75,0.8,0.9"')
    ap.add_argument("--write-dropped", default=None, help="Write dropped features to this GeoJSON path")
    args = ap.parse_args()

    gdf = gpd.read_file(args.in_geojson)

    if args.sweep:
        ths = _parse_sweep(args.sweep)
        # Print counts for each threshold using the same logic as the script
        print("shapely:", getattr(shapely, "__version__", "unknown"))
        print("in:", len(gdf))
        for th in ths:
            out = dedupe_iou(gdf, th)
            print(th, len(out))
        return

    if not args.out_geojson:
        raise SystemExit("--out-geojson is required unless you use --sweep")

    out, keep_mask = dedupe_iou_keep_mask(gdf, args.iou)

    out.to_file(args.out_geojson, driver="GeoJSON")
    print("in:", len(gdf))
    print("out:", len(out))
    print("wrote:", args.out_geojson)

    if args.write_dropped:
        # Recompute the cleaned gdf exactly as in dedupe function to align mask
        cleaned = gdf.copy()
        cleaned["geometry"] = cleaned.geometry.apply(_fix_geom)
        cleaned = cleaned[~cleaned.geometry.isna()].copy()
        cleaned = cleaned[~cleaned.geometry.is_empty].copy()

        dropped = cleaned.iloc[[i for i, k in enumerate(keep_mask) if not k]].copy()
        dropped.to_file(args.write_dropped, driver="GeoJSON")
        print("dropped:", len(dropped))
        print("wrote dropped:", args.write_dropped)


if __name__ == "__main__":
    main()