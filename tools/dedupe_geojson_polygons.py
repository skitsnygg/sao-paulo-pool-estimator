from __future__ import annotations

import argparse
import json
from pathlib import Path

from shapely.geometry import shape, mapping
from shapely.ops import unary_union


def iou(a, b) -> float:
    inter = a.intersection(b).area
    if inter == 0:
        return 0.0
    union = a.union(b).area
    return inter / union if union else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-geojson", required=True, type=Path)
    ap.add_argument("--out-geojson", required=True, type=Path)
    ap.add_argument("--iou", type=float, default=0.35)
    ap.add_argument("--buffer", type=float, default=0.0, help="optional small buffer in degrees (e.g. 0.000001)")
    args = ap.parse_args()

    fc = json.loads(args.in_geojson.read_text())
    feats = fc.get("features", [])

    geoms = []
    props = []

    for f in feats:
        g = shape(f["geometry"])
        if args.buffer:
            g = g.buffer(args.buffer)
        if not g.is_valid:
            g = g.buffer(0)
        if g.is_empty:
            continue
        geoms.append(g)
        props.append(f.get("properties", {}))

    kept = []
    used = [False] * len(geoms)

    for i, g in enumerate(geoms):
        if used[i]:
            continue
        group = [g]
        used[i] = True

        # naive O(n^2) grouping; OK for moderate sizes.
        for j in range(i + 1, len(geoms)):
            if used[j]:
                continue
            if not g.bounds[0] <= geoms[j].bounds[2] or not geoms[j].bounds[0] <= g.bounds[2]:
                pass
            if iou(g, geoms[j]) >= args.iou:
                group.append(geoms[j])
                used[j] = True

        merged = unary_union(group)
        if not merged.is_valid:
            merged = merged.buffer(0)

        # unary_union can return MultiPolygon
        if merged.geom_type == "Polygon":
            out_geoms = [merged]
        else:
            out_geoms = list(getattr(merged, "geoms", []))

        for mg in out_geoms:
            kept.append({
                "type": "Feature",
                "properties": {},
                "geometry": mapping(mg)
            })

    out = {"type": "FeatureCollection", "features": kept}
    args.out_geojson.parent.mkdir(parents=True, exist_ok=True)
    args.out_geojson.write_text(json.dumps(out))
    print("Wrote:", args.out_geojson)
    print("features_in:", len(feats))
    print("features_out:", len(kept))


if __name__ == "__main__":
    main()
