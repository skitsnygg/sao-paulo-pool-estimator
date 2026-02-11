#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass
import json

from shapely.geometry import Polygon, MultiPolygon, mapping
from shapely.ops import unary_union


@dataclass
class Det:
    poly: Polygon
    conf: float


def parse_ultralytics_seg_line(line: str, w: int, h: int):
    parts = line.strip().split()
    cls = int(float(parts[0]))

    nums = [float(x) for x in parts[1:]]
    conf = None

    # if odd number of values → first is confidence
    if len(nums) % 2 == 1:
        conf = nums[0]
        nums = nums[1:]

    pts = [(nums[i] * w, nums[i + 1] * h) for i in range(0, len(nums), 2)]
    poly = Polygon(pts).buffer(0)

    return cls, conf if conf is not None else 1.0, poly


def poly_iou(a: Polygon, b: Polygon):
    inter = a.intersection(b).area
    if inter <= 0:
        return 0.0
    union = a.union(b).area
    return inter / union if union > 0 else 0.0


def cluster_and_merge(dets, iou_thresh):
    used = [False] * len(dets)
    merged = []

    for i, di in enumerate(dets):
        if used[i]:
            continue

        cluster = [di]
        used[i] = True
        changed = True

        while changed:
            changed = False
            base_union = unary_union([d.poly for d in cluster])

            for j, dj in enumerate(dets):
                if used[j]:
                    continue
                if poly_iou(base_union, dj.poly) >= iou_thresh:
                    cluster.append(dj)
                    used[j] = True
                    changed = True

        u = unary_union([d.poly for d in cluster])
        conf = max([d.conf for d in cluster])

        if isinstance(u, Polygon):
            merged.append(Det(u, conf))
        elif isinstance(u, MultiPolygon):
            for p in u.geoms:
                merged.append(Det(p, conf))

    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, type=Path)
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--out-geojson", required=True, type=Path)
    ap.add_argument("--iou", type=float, default=0.35)
    ap.add_argument("--min-area-px", type=float, default=200.0)
    args = ap.parse_args()

    features = []

    for lab_path in sorted(args.pred_dir.glob("*.txt")):
        stem = lab_path.stem

        img_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            p = args.images_dir / f"{stem}{ext}"
            if p.exists():
                img_path = p
                break

        if img_path is None:
            continue

        from PIL import Image
        w, h = Image.open(img_path).size

        dets = []
        for line in lab_path.read_text().splitlines():
            if not line.strip():
                continue

            cls, conf, poly = parse_ultralytics_seg_line(line, w, h)

            if cls != 0:
                continue
            if poly.is_empty:
                continue
            if poly.area < args.min_area_px:
                continue

            dets.append(Det(poly, conf))

        merged = cluster_and_merge(dets, args.iou)

        for d in merged:
            features.append({
                "type": "Feature",
                "properties": {
                    "image": img_path.name,
                    "conf": float(d.conf),
                },
                "geometry": mapping(d.poly)
            })

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    args.out_geojson.parent.mkdir(parents=True, exist_ok=True)
    args.out_geojson.write_text(json.dumps(geojson))

    print("Wrote", args.out_geojson)
    print("Features:", len(features))


if __name__ == "__main__":
    main()
