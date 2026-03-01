#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image
from shapely.affinity import affine_transform
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union


# -----------------------------
# Worldfile parsing / transform
# -----------------------------

@dataclass(frozen=True)
class WorldFile:
    # Standard worldfile lines:
    # 1: A pixel size in x direction
    # 2: D rotation about y-axis
    # 3: B rotation about x-axis
    # 4: E pixel size in y direction (typically negative)
    # 5: C x of center of upper-left pixel
    # 6: F y of center of upper-left pixel
    A: float
    D: float
    B: float
    E: float
    C: float
    F: float

    @staticmethod
    def load(p: Path) -> "WorldFile":
        vals = [float(x.strip()) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        if len(vals) != 6:
            raise ValueError(f"Worldfile must have 6 lines, got {len(vals)}: {p}")
        return WorldFile(A=vals[0], D=vals[1], B=vals[2], E=vals[3], C=vals[4], F=vals[5])

    def px_to_world(self, x: float, y: float) -> Tuple[float, float]:
        # For continuous coordinates from segmentation (already not integer pixel indices),
        # using x,y directly is fine (worldfile uses pixel-center convention).
        X = self.A * x + self.B * y + self.C
        Y = self.D * x + self.E * y + self.F
        return (X, Y)


def worldfile_for_image(img: Path) -> Optional[Path]:
    # Prefer common worldfile extensions.
    # For PNG tiles you have *.pgw, but allow others too.
    candidates = [
        img.with_suffix(".pgw"),
        img.with_suffix(".wld"),
        img.with_suffix(".tfw"),
        img.with_suffix(".jgw"),
        img.with_suffix(".PGW"),
        img.with_suffix(".WLD"),
        img.with_suffix(".TFW"),
        img.with_suffix(".JGW"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# -----------------------------
# YOLO seg parsing
# -----------------------------

@dataclass
class Det:
    poly_px: Polygon
    conf: Optional[float]


def parse_ultralytics_seg_line(line: str, w: int, h: int) -> Optional[Tuple[int, Optional[float], Polygon]]:
    """
    Accepts both:
      - cls x1 y1 x2 y2 ... (normalized)
      - cls x1 y1 x2 y2 ... conf (normalized; Ultralytics save_conf)
    """
    parts = line.strip().split()
    if len(parts) < 3:
        return None

    try:
        cls = int(float(parts[0]))
    except Exception:
        return None

    try:
        nums = [float(x) for x in parts[1:]]
    except Exception:
        return None

    conf: Optional[float] = None
    if len(nums) % 2 == 1:
        # Ultralytics save_conf: conf at end of line
        conf = float(nums[-1])
        nums = nums[:-1]

    if len(nums) < 6 or (len(nums) % 2 != 0):
        return None

    pts_px = [(nums[i] * w, nums[i + 1] * h) for i in range(0, len(nums), 2)]
    poly = Polygon(pts_px)

    if poly.is_empty or poly.area <= 0:
        return None

    if not poly.is_valid:
        poly = poly.buffer(0)
        if poly.is_empty or poly.area <= 0:
            return None
        if poly.geom_type == "MultiPolygon":
            poly = max(list(poly.geoms), key=lambda g: g.area)

    return cls, conf, poly


# -----------------------------
# IoU merge (pixel space)
# -----------------------------

def poly_iou(a: Polygon, b: Polygon) -> float:
    if a.is_empty or b.is_empty:
        return 0.0
    inter = a.intersection(b).area
    if inter <= 0:
        return 0.0
    union = a.union(b).area
    return float(inter / union) if union > 0 else 0.0


def merge_by_iou(dets: List[Det], iou_thresh: float) -> List[Det]:
    """
    Simple greedy clustering:
      - group detections that overlap above threshold
      - union their polygons
      - keep max conf
    """
    merged: List[Det] = []
    used = [False] * len(dets)

    for i in range(len(dets)):
        if used[i]:
            continue
        cluster = [i]
        used[i] = True

        # grow cluster
        changed = True
        while changed:
            changed = False
            for j in range(len(dets)):
                if used[j]:
                    continue
                if any(poly_iou(dets[j].poly_px, dets[k].poly_px) >= iou_thresh for k in cluster):
                    used[j] = True
                    cluster.append(j)
                    changed = True

        polys = [dets[k].poly_px for k in cluster]
        confs = [dets[k].conf for k in cluster if dets[k].conf is not None]
        conf = max(confs) if confs else None
        u = unary_union(polys)

        if u.is_empty:
            continue

        if u.geom_type == "Polygon":
            merged.append(Det(u, conf))
        elif u.geom_type == "MultiPolygon":
            for g in u.geoms:
                if not g.is_empty and g.area > 0:
                    merged.append(Det(g, conf))
        else:
            # ignore other types
            continue

    return merged


def nms_by_iou(dets: List[Det], iou_thresh: float) -> List[Det]:
    """
    Greedy NMS in pixel space:
      - sort by confidence (None treated as 0)
      - suppress polygons with IoU >= threshold
    """
    if not dets:
        return []

    def _score(d: Det) -> float:
        return float(d.conf) if d.conf is not None else 0.0

    remaining = sorted(enumerate(dets), key=lambda t: (-_score(t[1]), t[0]))
    kept: List[Det] = []

    while remaining:
        _, current = remaining.pop(0)
        kept.append(current)
        filtered: List[Tuple[int, Det]] = []
        for idx, det in remaining:
            if poly_iou(current.poly_px, det.poly_px) < iou_thresh:
                filtered.append((idx, det))
        remaining = filtered

    return kept


# -----------------------------
# IO helpers
# -----------------------------

def iter_label_files(pred_dir: Path) -> List[Path]:
    # Ultralytics typical: <pred_dir>/labels/*.txt
    # Your stage matches this; also support deeper nesting.
    return sorted(pred_dir.rglob("labels/*.txt"))


def find_image(images_dir: Path, stem: str) -> Optional[Path]:
    # Fast path: direct in images_dir
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p

    # Slow path: recursive search for just this stem
    # (Useful if someone passes the whole sp_city_2020 root; avoid indexing everything.)
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        matches = list(images_dir.rglob(f"{stem}{ext}"))
        if matches:
            return matches[0]

    return None


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge YOLOv8-seg prediction labels into world-coordinate polygons.",
        epilog=(
            "Examples:\n"
            "  # baseline merge\n"
            "  python tools/merge_pred_polygons.py \\\n"
            "    --pred-dir runs/segment/<run> \\\n"
            "    --images-dir data/raw/geosampa_ortho/sp_city_2020/cell_0026_0007 \\\n"
            "    --out-geojson runs/segment/<run>/merged.geojson\n"
            "\n"
            "  # min confidence\n"
            "  python tools/merge_pred_polygons.py \\\n"
            "    --pred-dir runs/segment/<run> \\\n"
            "    --images-dir data/raw/geosampa_ortho/sp_city_2020/cell_0026_0007 \\\n"
            "    --out-geojson runs/segment/<run>/merged_conf25.geojson \\\n"
            "    --min-conf 0.25\n"
            "\n"
            "  # clip edge artifacts\n"
            "  python tools/merge_pred_polygons.py \\\n"
            "    --pred-dir runs/segment/<run> \\\n"
            "    --images-dir data/raw/geosampa_ortho/sp_city_2020/cell_0026_0007 \\\n"
            "    --out-geojson runs/segment/<run>/merged_clip3.geojson \\\n"
            "    --clip-edge-px 3\n"
            "\n"
            "  # min conf + clip edge\n"
            "  python tools/merge_pred_polygons.py \\\n"
            "    --pred-dir runs/segment/<run> \\\n"
            "    --images-dir data/raw/geosampa_ortho/sp_city_2020/cell_0026_0007 \\\n"
            "    --out-geojson runs/segment/<run>/merged_conf25_clip3.geojson \\\n"
            "    --min-conf 0.25 --clip-edge-px 3\n"
            "\n"
            "  # prefer NMS over union merge\n"
            "  python tools/merge_pred_polygons.py \\\n"
            "    --pred-dir runs/segment/<run> \\\n"
            "    --images-dir data/raw/geosampa_ortho/sp_city_2020/cell_0026_0007 \\\n"
            "    --out-geojson runs/segment/<run>/merged_nms.geojson \\\n"
            "    --prefer-nms\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out-geojson", required=True)
    ap.add_argument("--iou", type=float, default=0.35)
    ap.add_argument("--min-area-px", type=float, default=0.0)
    ap.add_argument("--min-conf", type=float, default=None, help="Drop detections with conf < min_conf (if conf present)")
    ap.add_argument(
        "--clip-edge-px",
        type=int,
        default=0,
        help="Drop detections touching within this many pixels of any image border (default: 0)",
    )
    ap.add_argument(
        "--prefer-nms",
        action="store_true",
        default=False,
        help="Use NMS instead of polygon union when IoU >= threshold",
    )
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    images_dir = Path(args.images_dir)
    out_geojson = Path(args.out_geojson)

    features: List[dict] = []

    txts = iter_label_files(pred_dir)

    for txt in txts:
        if not txt.is_file() or txt.stat().st_size == 0:
            continue

        stem = txt.stem

        img = find_image(images_dir, stem)
        if img is None:
            continue

        wf_path = worldfile_for_image(img)
        if wf_path is None:
            # If you want to allow “no worldfile” output, remove this continue.
            continue

        w, h = Image.open(img).size
        wf = WorldFile.load(wf_path)

        dets: List[Det] = []

        for ln in txt.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            parsed = parse_ultralytics_seg_line(ln, w, h)
            if parsed is None:
                continue
            cls, conf, poly_px = parsed

            if args.min_conf is not None and conf is not None and conf < args.min_conf:
                continue

            # Filter in pixel space
            if poly_px.area < args.min_area_px:
                continue

            if args.clip_edge_px > 0:
                k = args.clip_edge_px
                minx, miny, maxx, maxy = poly_px.bounds
                if minx <= k or miny <= k or maxx >= (w - 1 - k) or maxy >= (h - 1 - k):
                    continue

            dets.append(Det(poly_px, conf))

        if not dets:
            continue

        dets_merged = nms_by_iou(dets, args.iou) if args.prefer_nms else merge_by_iou(dets, args.iou)

        # Convert each merged polygon from pixel coords -> world coords and emit a feature
        for d in dets_merged:
            poly_world = affine_transform(d.poly_px, [wf.A, wf.B, wf.D, wf.E, wf.C, wf.F])

            if poly_world.is_empty or poly_world.area == 0:
                continue

            if not poly_world.is_valid:
                poly_world = poly_world.buffer(0)
                if poly_world.is_empty or poly_world.area == 0:
                    continue
                if poly_world.geom_type == "MultiPolygon":
                    # keep all pieces
                    for g in poly_world.geoms:
                        if not g.is_empty and g.area > 0:
                            features.append(
                                {
                                    "type": "Feature",
                                    "geometry": mapping(g),
                                    "properties": {
                                        "stem": stem,
                                        "conf": float(d.conf) if d.conf is not None else None,
                                        "img": str(img),
                                        "worldfile": str(wf_path),
                                    },
                                }
                            )
                    continue

            features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(poly_world),
                    "properties": {
                        "stem": stem,
                        "conf": float(d.conf) if d.conf is not None else None,
                        "img": str(img),
                        "worldfile": str(wf_path),
                    },
                }
            )

    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    out_geojson.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Wrote", out_geojson)
    print("Features:", len(features))


if __name__ == "__main__":
    main()
