#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Tuple

from PIL import Image
from pyproj import Transformer
from shapely.geometry import Point, Polygon, mapping
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union
from shapely.strtree import STRtree

CELL_RE = re.compile(r"^cell_(\d{4})_(\d{4})$")
TILE_RE = re.compile(r"^r(\d{4})_c(\d{4})$")


@dataclass
class TileRec:
    cell: str
    row: int
    col: int
    name: str
    poly: Polygon
    bounds: Tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Audit local GeoSampa 2020 tile coverage and export coverage/hole "
            "GeoJSON layers for Folium."
        )
    )
    ap.add_argument("--tiles-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--src-crs", default="EPSG:31983")
    ap.add_argument("--dst-crs", default="EPSG:4326")
    ap.add_argument("--point-lon", type=float, default=None)
    ap.add_argument("--point-lat", type=float, default=None)
    ap.add_argument("--viewer-meta", type=Path, default=None)
    ap.add_argument("--nearest-cells", type=int, default=12)
    ap.add_argument("--full-chip-count", type=int, default=50, help="Cell chip-count threshold used to classify 'full'")
    return ap.parse_args()


def read_worldfile(p: Path) -> Optional[Tuple[float, float, float, float, float, float]]:
    try:
        vals = [float(x.strip()) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    except Exception:
        return None
    if len(vals) != 6:
        return None
    return vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]


def tile_polygon(
    A: float,
    D: float,
    B: float,
    E: float,
    C: float,
    F: float,
    width: int,
    height: int,
) -> Polygon:
    # Worldfiles store UL pixel center; convert from pixel-corner coordinates.
    corners = [
        (-0.5, -0.5),
        (width - 0.5, -0.5),
        (width - 0.5, height - 0.5),
        (-0.5, height - 0.5),
    ]
    pts = []
    for x, y in corners:
        X = A * x + B * y + C
        Y = D * x + E * y + F
        pts.append((X, Y))
    return Polygon(pts)


def to_feature_collection(features: List[dict]) -> dict:
    return {"type": "FeatureCollection", "features": features}


def write_geojson(path: Path, fc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")


def row_col_summary(missing: List[Tuple[int, int]]) -> List[dict]:
    by_row: Dict[int, List[int]] = {}
    for r, c in missing:
        by_row.setdefault(r, []).append(c)
    out: List[dict] = []
    for r in sorted(by_row):
        cols = sorted(by_row[r])
        out.append({"row": r, "cols": cols, "count": len(cols)})
    return out


def bbox_from_geom(poly: Polygon) -> Tuple[float, float, float, float]:
    b = poly.bounds
    return float(b[0]), float(b[1]), float(b[2]), float(b[3])


def center_wh_from_bbox(b: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    minx, miny, maxx, maxy = b
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    w = maxx - minx
    h = maxy - miny
    return cx, cy, w, h


def classify_cell_status(chip_count: int, missing_chip_count: int, full_chip_count: int) -> str:
    if chip_count <= 0:
        return "empty"
    if chip_count >= full_chip_count:
        return "full"
    return "partial"


def main() -> int:
    args = parse_args()
    tiles_root = args.tiles_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not tiles_root.exists():
        raise SystemExit(f"tiles root not found: {tiles_root}")

    to_wgs84 = Transformer.from_crs(args.src_crs, args.dst_crs, always_xy=True)
    from_wgs84 = Transformer.from_crs(args.dst_crs, args.src_crs, always_xy=True)

    cells = sorted([p for p in tiles_root.iterdir() if p.is_dir() and CELL_RE.match(p.name)])
    if not cells:
        raise SystemExit(f"No cell_* folders found under: {tiles_root}")

    all_tiles: List[TileRec] = []
    cell_unions: Dict[str, Polygon] = {}
    cell_expected_geom_src: Dict[str, Polygon] = {}
    cell_missing_rows_cols: Dict[str, List[Tuple[int, int]]] = {}
    cell_chip_count: Dict[str, int] = {}
    cell_expected_chip_count: Dict[str, int] = {}
    cell_missing_chip_count: Dict[str, int] = {}
    cell_idx_by_name: Dict[str, Tuple[int, int]] = {}
    inferred_geom_cells = set()
    empty_cells: List[str] = []
    cell_idx_set = set()

    for cell_dir in cells:
        m = CELL_RE.match(cell_dir.name)
        assert m is not None
        ci, cj = int(m.group(1)), int(m.group(2))
        cell_idx_set.add((ci, cj))
        cell_idx_by_name[cell_dir.name] = (ci, cj)

        pngs = sorted([p for p in cell_dir.glob("*.png") if p.is_file()])
        cell_chip_count[cell_dir.name] = len(pngs)

        if not pngs:
            empty_cells.append(cell_dir.name)
            cell_expected_chip_count[cell_dir.name] = 0
            cell_missing_chip_count[cell_dir.name] = 0
            continue

        # Read a sample image size once per cell.
        try:
            with Image.open(pngs[0]) as im:
                width, height = im.size
        except Exception:
            continue

        cell_tile_polys: List[Polygon] = []
        rc_present = set()
        tile_affines: Dict[Tuple[int, int], Tuple[float, float, float, float, float, float]] = {}

        for png in pngs:
            tm = TILE_RE.match(png.stem)
            if tm is None:
                continue
            row, col = int(tm.group(1)), int(tm.group(2))
            rc_present.add((row, col))

            wf = png.with_suffix(".pgw")
            if not wf.exists():
                continue
            vals = read_worldfile(wf)
            if vals is None:
                continue
            A, D, B, E, C, F = vals
            tile_affines[(row, col)] = vals
            poly = tile_polygon(A, D, B, E, C, F, width, height)
            if poly.is_empty or not poly.is_valid:
                continue
            cell_tile_polys.append(poly)
            all_tiles.append(
                TileRec(
                    cell=cell_dir.name,
                    row=row,
                    col=col,
                    name=png.name,
                    poly=poly,
                    bounds=poly.bounds,
                )
            )

        if not cell_tile_polys:
            continue
        cell_unions[cell_dir.name] = unary_union(cell_tile_polys)

        rows = sorted({r for r, _ in rc_present})
        cols = sorted({c for _, c in rc_present})

        # Build complete expected rectangle from observed row/col span using affine anchor.
        if rows and cols and tile_affines:
            rmin, rmax = rows[0], rows[-1]
            cmin, cmax = cols[0], cols[-1]
            expected_count = (rmax - rmin + 1) * (cmax - cmin + 1)
            missing = []
            for r in range(rmin, rmax + 1):
                for c in range(cmin, cmax + 1):
                    if (r, c) not in rc_present:
                        missing.append((r, c))

            base_rc = min(tile_affines.keys())
            A, D, B, E, C0, F0 = tile_affines[base_rc]
            r0, c0 = base_rc

            expected_polys: List[Polygon] = []
            for r in range(rmin, rmax + 1):
                for c in range(cmin, cmax + 1):
                    C = C0 + (c - c0) * A * width + (r - r0) * B * height
                    F = F0 + (c - c0) * D * width + (r - r0) * E * height
                    ep = tile_polygon(A, D, B, E, C, F, width, height)
                    if ep.is_empty or not ep.is_valid:
                        continue
                    expected_polys.append(ep)

            if expected_polys:
                cell_expected_geom_src[cell_dir.name] = unary_union(expected_polys)
            else:
                cell_expected_geom_src[cell_dir.name] = cell_unions[cell_dir.name]

            if missing:
                cell_missing_rows_cols[cell_dir.name] = list(missing)
                for r, c in missing:
                    C = C0 + (c - c0) * A * width + (r - r0) * B * height
                    F = F0 + (c - c0) * D * width + (r - r0) * E * height
                    mp = tile_polygon(A, D, B, E, C, F, width, height)
                    if mp.is_empty or not mp.is_valid:
                        continue
                    all_tiles.append(
                        TileRec(
                            cell=cell_dir.name,
                            row=r,
                            col=c,
                            name=f"missing_r{r:04d}_c{c:04d}.png",
                            poly=mp,
                            bounds=mp.bounds,
                        )
                    )

            cell_expected_chip_count[cell_dir.name] = int(expected_count)
            cell_missing_chip_count[cell_dir.name] = int(len(missing))
        else:
            # Fallback when row/col parsing is unavailable.
            cell_expected_geom_src[cell_dir.name] = cell_unions[cell_dir.name]
            cell_expected_chip_count[cell_dir.name] = int(cell_chip_count[cell_dir.name])
            cell_missing_chip_count[cell_dir.name] = 0

    # Infer geometry for cells without valid footprints (typically empty cells).
    missing_geom_cells = [name for name in sorted(cell_idx_by_name.keys()) if name not in cell_expected_geom_src]
    if missing_geom_cells and cell_expected_geom_src:
        cell_name_by_idx = {idx: name for name, idx in cell_idx_by_name.items()}
        cell_centers: Dict[str, Tuple[float, float]] = {}
        widths: List[float] = []
        heights: List[float] = []
        for name, geom in cell_expected_geom_src.items():
            if geom is None or geom.is_empty:
                continue
            cx, cy, w, h = center_wh_from_bbox(bbox_from_geom(geom))
            cell_centers[name] = (float(cx), float(cy))
            widths.append(float(w))
            heights.append(float(h))

        vec_i_samples: List[Tuple[float, float]] = []
        vec_j_samples: List[Tuple[float, float]] = []
        for name, (ci, cj) in cell_idx_by_name.items():
            c0 = cell_centers.get(name)
            if c0 is None:
                continue
            ni = cell_name_by_idx.get((ci + 1, cj))
            if ni and ni in cell_centers:
                c1 = cell_centers[ni]
                vec_i_samples.append((c1[0] - c0[0], c1[1] - c0[1]))
            nj = cell_name_by_idx.get((ci, cj + 1))
            if nj and nj in cell_centers:
                c1 = cell_centers[nj]
                vec_j_samples.append((c1[0] - c0[0], c1[1] - c0[1]))

        vec_i = (
            float(median([v[0] for v in vec_i_samples])) if vec_i_samples else 0.0,
            float(median([v[1] for v in vec_i_samples])) if vec_i_samples else 0.0,
        )
        vec_j = (
            float(median([v[0] for v in vec_j_samples])) if vec_j_samples else 0.0,
            float(median([v[1] for v in vec_j_samples])) if vec_j_samples else 0.0,
        )
        width_med = float(median(widths)) if widths else 0.0
        height_med = float(median(heights)) if heights else 0.0

        for name in missing_geom_cells:
            ci, cj = cell_idx_by_name[name]
            center_candidates: List[Tuple[float, float]] = []

            for di, dj in [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ]:
                nname = cell_name_by_idx.get((ci + di, cj + dj))
                c = cell_centers.get(nname or "")
                if c is None:
                    continue
                cx = c[0] - di * vec_i[0] - dj * vec_j[0]
                cy = c[1] - di * vec_i[1] - dj * vec_j[1]
                center_candidates.append((cx, cy))

            if not center_candidates and cell_centers:
                nearest_name = min(
                    cell_centers.keys(),
                    key=lambda n: (cell_idx_by_name[n][0] - ci) ** 2 + (cell_idx_by_name[n][1] - cj) ** 2,
                )
                bi, bj = cell_idx_by_name[nearest_name]
                base_cx, base_cy = cell_centers[nearest_name]
                center_candidates.append(
                    (
                        base_cx + (ci - bi) * vec_i[0] + (cj - bj) * vec_j[0],
                        base_cy + (ci - bi) * vec_i[1] + (cj - bj) * vec_j[1],
                    )
                )

            if not center_candidates:
                continue

            cx = float(sum(c[0] for c in center_candidates) / len(center_candidates))
            cy = float(sum(c[1] for c in center_candidates) / len(center_candidates))

            if width_med > 0.0 and height_med > 0.0:
                inferred = Polygon(
                    [
                        (cx - width_med / 2.0, cy - height_med / 2.0),
                        (cx + width_med / 2.0, cy - height_med / 2.0),
                        (cx + width_med / 2.0, cy + height_med / 2.0),
                        (cx - width_med / 2.0, cy + height_med / 2.0),
                    ]
                )
            else:
                # Last-resort degenerate square centered at estimated location.
                inferred = Point(cx, cy).buffer(1.0, cap_style=3)

            if inferred.is_empty or not inferred.is_valid:
                continue
            cell_expected_geom_src[name] = inferred
            inferred_geom_cells.add(name)

            expected = int(cell_expected_chip_count.get(name, 0))
            if expected <= 0:
                expected = int(args.full_chip_count)
                cell_expected_chip_count[name] = expected
            chip_count = int(cell_chip_count.get(name, 0))
            cell_missing_chip_count[name] = max(0, expected - chip_count)

    # Split real vs synthesized records (missing placeholders are name-prefixed).
    real_tiles = [t for t in all_tiles if not t.name.startswith("missing_")]
    missing_tiles = [t for t in all_tiles if t.name.startswith("missing_")]

    if not real_tiles:
        raise SystemExit("No valid tiles found after parsing worldfiles.")

    # Coverage union from real tiles only.
    coverage_union = unary_union([t.poly for t in real_tiles])
    minx = min(t.bounds[0] for t in real_tiles)
    miny = min(t.bounds[1] for t in real_tiles)
    maxx = max(t.bounds[2] for t in real_tiles)
    maxy = max(t.bounds[3] for t in real_tiles)
    west, south = to_wgs84.transform(minx, miny)
    east, north = to_wgs84.transform(maxx, maxy)

    # Cell-grid gaps in index space.
    all_i = [i for i, _ in cell_idx_set]
    all_j = [j for _, j in cell_idx_set]
    imin, imax = min(all_i), max(all_i)
    jmin, jmax = min(all_j), max(all_j)
    present = cell_idx_set
    neighbors8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    missing_cells_all = []
    interior_holes = []
    edge_gaps = []
    for i in range(imin, imax + 1):
        for j in range(jmin, jmax + 1):
            if (i, j) in present:
                continue
            n = sum(((i + di, j + dj) in present) for di, dj in neighbors8)
            rec = {"cell": f"cell_{i:04d}_{j:04d}", "i": i, "j": j, "neighbors": n}
            missing_cells_all.append(rec)
            if n >= 5:
                interior_holes.append(rec)
            elif n >= 1:
                edge_gaps.append(rec)

    # Point coverage check.
    point_report = None
    if args.point_lon is not None and args.point_lat is not None:
        px, py = from_wgs84.transform(args.point_lon, args.point_lat)
        point = Point(px, py)
        covered = bool(coverage_union.contains(point) or coverage_union.touches(point))

        real_tree = STRtree([t.poly for t in real_tiles])
        nearest_idx = int(real_tree.nearest(point))
        nearest = real_tiles[nearest_idx]
        nearest_dist = float(nearest.poly.distance(point))

        nearest_cells = sorted(
            ((name, float(geom.distance(point))) for name, geom in cell_unions.items()),
            key=lambda item: item[1],
        )[: max(1, int(args.nearest_cells))]

        nearest_cells_out = []
        for name, dist in nearest_cells:
            miss = cell_missing_rows_cols.get(name, [])
            chip_count = int(cell_chip_count.get(name, 0))
            expected_chip_count = int(cell_expected_chip_count.get(name, chip_count))
            missing_chip_count = int(cell_missing_chip_count.get(name, max(0, expected_chip_count - chip_count)))
            nearest_cells_out.append(
                {
                    "cell": name,
                    "distance_m": dist,
                    "chip_count": chip_count,
                    "expected_chip_count": expected_chip_count,
                    "missing_chip_count": missing_chip_count,
                    "status": classify_cell_status(chip_count, missing_chip_count, int(args.full_chip_count)),
                    "missing_tile_count": len(miss),
                    "missing_rows_cols": row_col_summary(miss),
                }
            )

        point_report = {
            "lon": float(args.point_lon),
            "lat": float(args.point_lat),
            "x_src": float(px),
            "y_src": float(py),
            "covered_by_local_tiles": covered,
            "nearest_tile": {
                "cell": nearest.cell,
                "tile": nearest.name,
                "distance_m": nearest_dist,
                "bounds_src": list(nearest.bounds),
            },
            "nearest_cells": nearest_cells_out,
        }

    # Optional compare against current map bounds.
    map_check = None
    if args.viewer_meta and args.viewer_meta.exists():
        try:
            meta = json.loads(args.viewer_meta.read_text(encoding="utf-8"))
            view_bounds = meta.get("view_bounds_4326") or meta.get("bounds_4326")
            if point_report and isinstance(view_bounds, list) and len(view_bounds) == 4:
                vw, vs, ve, vn = [float(x) for x in view_bounds]
                lon = point_report["lon"]
                lat = point_report["lat"]
                in_view = vw <= lon <= ve and vs <= lat <= vn
                map_check = {
                    "viewer_meta": str(args.viewer_meta.resolve()),
                    "view_bounds_4326": [vw, vs, ve, vn],
                    "point_inside_view": in_view,
                }
        except Exception:
            map_check = None

    # GeoJSON outputs.
    coverage_31983_fc = to_feature_collection(
        [
            {
                "type": "Feature",
                "geometry": mapping(coverage_union),
                "properties": {"name": "local_tile_coverage", "crs": args.src_crs},
            }
        ]
    )
    coverage_4326 = shapely_transform(to_wgs84.transform, coverage_union)
    coverage_4326_fc = to_feature_collection(
        [
            {
                "type": "Feature",
                "geometry": mapping(coverage_4326),
                "properties": {"name": "local_tile_coverage", "crs": args.dst_crs},
            }
        ]
    )

    missing_31983_features = []
    missing_4326_features = []
    for t in missing_tiles:
        props = {"cell": t.cell, "row": t.row, "col": t.col, "reason": "missing_within_cell_rect"}
        missing_31983_features.append({"type": "Feature", "geometry": mapping(t.poly), "properties": props})
        g4326 = shapely_transform(to_wgs84.transform, t.poly)
        missing_4326_features.append({"type": "Feature", "geometry": mapping(g4326), "properties": props})

    cell_cov_31983_features = []
    cell_cov_4326_features = []
    cell_status_counts = {"full": 0, "partial": 0, "empty": 0}
    cell_records_for_report = []
    for name in sorted(cell_idx_by_name.keys()):
        geom_src = cell_expected_geom_src.get(name)
        if geom_src is None or geom_src.is_empty or not geom_src.is_valid:
            geom_src = cell_unions.get(name)
        if geom_src is None or geom_src.is_empty or not geom_src.is_valid:
            continue

        chip_count = int(cell_chip_count.get(name, 0))
        expected_chip_count = int(cell_expected_chip_count.get(name, chip_count))
        if expected_chip_count < chip_count:
            expected_chip_count = chip_count
        missing_chip_count = int(cell_missing_chip_count.get(name, max(0, expected_chip_count - chip_count)))
        if missing_chip_count < 0:
            missing_chip_count = 0
        status = classify_cell_status(chip_count, missing_chip_count, int(args.full_chip_count))
        cell_status_counts[status] = int(cell_status_counts.get(status, 0) + 1)
        ci, cj = cell_idx_by_name[name]
        completeness = (float(chip_count) / float(expected_chip_count)) if expected_chip_count > 0 else 0.0

        props = {
            "cell": name,
            "cell_i": int(ci),
            "cell_j": int(cj),
            "chip_count": chip_count,
            "expected_chip_count": expected_chip_count,
            "missing_chip_count": missing_chip_count,
            "status": status,
            "completeness_ratio": completeness,
            "full_chip_threshold": int(args.full_chip_count),
            "geometry_inferred": bool(name in inferred_geom_cells),
        }
        cell_records_for_report.append(dict(props))

        cell_cov_31983_features.append({"type": "Feature", "geometry": mapping(geom_src), "properties": props})
        g4326 = shapely_transform(to_wgs84.transform, geom_src)
        cell_cov_4326_features.append({"type": "Feature", "geometry": mapping(g4326), "properties": props})

    write_geojson(out_dir / "coverage_union_31983.geojson", coverage_31983_fc)
    write_geojson(out_dir / "coverage_union_4326.geojson", coverage_4326_fc)
    write_geojson(out_dir / "missing_tiles_31983.geojson", to_feature_collection(missing_31983_features))
    write_geojson(out_dir / "missing_tiles_4326.geojson", to_feature_collection(missing_4326_features))
    write_geojson(out_dir / "cell_coverage_31983.geojson", to_feature_collection(cell_cov_31983_features))
    write_geojson(out_dir / "cell_coverage_4326.geojson", to_feature_collection(cell_cov_4326_features))

    report = {
        "tiles_root": str(tiles_root),
        "cells_count": len(cells),
        "real_tiles_count": len(real_tiles),
        "missing_tile_polygons_count": len(missing_tiles),
        "empty_cells": empty_cells,
        "extent_src": {"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy},
        "extent_4326": {"west": west, "south": south, "east": east, "north": north},
        "cell_index_range": {"imin": imin, "imax": imax, "jmin": jmin, "jmax": jmax},
        "full_chip_count_threshold": int(args.full_chip_count),
        "cell_status_counts": cell_status_counts,
        "cells_with_inferred_geometry": sorted(inferred_geom_cells),
        "cells": cell_records_for_report,
        "missing_cells_in_index_bbox": len(missing_cells_all),
        "interior_hole_candidates": interior_holes,
        "edge_gap_candidates_count": len(edge_gaps),
        "point_check": point_report,
        "map_check": map_check,
        "outputs": {
            "coverage_union_31983": str((out_dir / "coverage_union_31983.geojson").resolve()),
            "coverage_union_4326": str((out_dir / "coverage_union_4326.geojson").resolve()),
            "missing_tiles_31983": str((out_dir / "missing_tiles_31983.geojson").resolve()),
            "missing_tiles_4326": str((out_dir / "missing_tiles_4326.geojson").resolve()),
            "cell_coverage_31983": str((out_dir / "cell_coverage_31983.geojson").resolve()),
            "cell_coverage_4326": str((out_dir / "cell_coverage_4326.geojson").resolve()),
        },
    }

    report_path = out_dir / "coverage_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"cells={len(cells)} real_tiles={len(real_tiles)}")
    print(f"extent_4326 west={west:.6f} south={south:.6f} east={east:.6f} north={north:.6f}")
    if point_report:
        print(
            "point_covered="
            f"{point_report['covered_by_local_tiles']} "
            f"nearest={point_report['nearest_tile']['cell']}/{point_report['nearest_tile']['tile']} "
            f"dist_m={point_report['nearest_tile']['distance_m']:.3f}"
        )
    print(f"wrote={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
