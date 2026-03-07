#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import requests
from PIL import Image
from pyproj import CRS, Transformer
from shapely.geometry import box, mapping, shape
from shapely.ops import transform as shapely_transform

try:
    import geopandas as gpd
except Exception as exc:
    raise SystemExit(f"Missing dependency geopandas: {exc}")

try:
    import rasterio
    from rasterio.crs import CRS as RasterioCRS
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject
except Exception as exc:
    raise SystemExit(f"Missing dependency rasterio: {exc}")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import rebuild_sp_city_geosampa_2020 as rebuild  # noqa: E402
from src.data.fetch_geosampa_ortho import _write_world_file  # noqa: E402


FAIXA_RE = re.compile(r"(?<!\d)(\d{4}-\d{3})(?!\d)")
DEFAULT_EXTENSIONS = ["tif", "tiff", "jp2", "j2k", "ecw"]
DEFAULT_WFS = "https://wfs.geosampa.prefeitura.sp.gov.br/geoserver/geoportal/wfs"
DEFAULT_ARTICULATION_LAYER = "geoportal:quadricula_orto_2020"
SUCCESS_STATUSES = set(rebuild.SUCCESS_STATUSES)

_thread_local = threading.local()


@dataclass(frozen=True)
class RenderJob:
    index: int
    path: Path
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    width: int
    height: int


@dataclass(frozen=True)
class RenderResult:
    index: int
    status: str
    http_status: str
    content_type: str
    attempts_made: int
    error: str


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def infer_faixa_code(path: Path) -> str:
    m = FAIXA_RE.search(path.stem)
    return m.group(1) if m else ""


def split_extensions(raw: str) -> List[str]:
    out = []
    for tok in (raw or "").split(","):
        t = tok.strip().lower().lstrip(".")
        if t:
            out.append(t)
    return out if out else list(DEFAULT_EXTENSIONS)


def find_rasters(root: Path, extensions: Sequence[str]) -> List[Path]:
    if not root.exists():
        raise SystemExit(f"Sources root not found: {root}")
    ext_set = {f".{e.lower().lstrip('.')}" for e in extensions}
    rasters: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in ext_set:
            rasters.append(p.resolve())
    rasters.sort()
    return rasters


def read_codes(path: Path, column: str = "") -> Set[str]:
    if not path.exists():
        raise SystemExit(f"Codes file not found: {path}")

    if path.suffix.lower() in {".txt", ".lst"}:
        return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}

    out: Set[str] = set()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        if not fields:
            return out
        col = column.strip() if column.strip() else ("faixa_code" if "faixa_code" in fields else fields[0])
        if col not in fields:
            raise SystemExit(f"Column '{col}' not found in {path}. Available: {fields}")
        for row in reader:
            code = str(row.get(col, "") or "").strip()
            if code:
                out.add(code)
    return out


def write_codes_txt(path: Path, codes: Iterable[str]) -> None:
    vals = sorted({c.strip() for c in codes if c and c.strip()})
    ensure_dir(path.parent)
    path.write_text("\n".join(vals) + ("\n" if vals else ""), encoding="utf-8")


def looks_like_degrees(bounds: Sequence[float]) -> bool:
    if len(bounds) != 4:
        return False
    minx, miny, maxx, maxy = [float(v) for v in bounds]
    if abs(minx) > 180 or abs(maxx) > 180:
        return False
    if abs(miny) > 90 or abs(maxy) > 90:
        return False
    if abs(maxx - minx) < 1e-12 and abs(maxy - miny) < 1e-12:
        return False
    return True


def set_or_fix_crs(gdf: "gpd.GeoDataFrame", assume_crs: str, force: bool = False) -> "gpd.GeoDataFrame":
    if force:
        return gdf.set_crs(assume_crs, allow_override=True)

    if gdf.crs is None:
        return gdf.set_crs(assume_crs, allow_override=True)

    try:
        crs_str = gdf.crs.to_string()
    except Exception:
        crs_str = ""

    if crs_str == "EPSG:4326":
        b = tuple(map(float, gdf.total_bounds))
        if not looks_like_degrees(b):
            return gdf.set_crs(assume_crs, allow_override=True)
    return gdf


def load_articulation_geojson(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Articulation GeoJSON not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to parse articulation GeoJSON: {path}: {exc}")
    if payload.get("type") != "FeatureCollection":
        raise SystemExit(f"Expected FeatureCollection in articulation file: {path}")
    return payload


def cmd_fetch_articulation(args: argparse.Namespace) -> int:
    out_geojson = Path(args.out_geojson).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_codes = Path(args.out_codes).resolve()

    params = {
        "service": "WFS",
        "version": "1.0.0",
        "request": "GetFeature",
        "typeName": args.layer,
        "outputFormat": "application/json",
        "srsName": args.crs,
    }

    resp = requests.get(args.wfs_url, params=params, timeout=args.timeout)
    resp.raise_for_status()

    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "json" not in ctype and resp.text.lstrip().startswith("<"):
        body = resp.text.replace("\n", " ")[:220]
        raise SystemExit(f"WFS did not return GeoJSON. content-type={ctype} body={body}")

    payload = resp.json()
    features = payload.get("features", [])
    if not isinstance(features, list):
        raise SystemExit("Invalid WFS GeoJSON: missing features list")

    ensure_dir(out_geojson.parent)
    out_geojson.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    rows = []
    codes = set()
    for feat in features:
        props = feat.get("properties") or {}
        code = str(props.get(args.code_field, "") or "").strip()
        if code:
            codes.add(code)
        rows.append(
            {
                "faixa_code": code,
                "cd_identificador": str(props.get("cd_identificador", "") or ""),
                "cd_levantamento": str(props.get("cd_levantamento", "") or ""),
                "cd_escala_quadricula": str(props.get("cd_escala_quadricula", "") or ""),
            }
        )

    ensure_dir(out_csv.parent)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["faixa_code", "cd_identificador", "cd_levantamento", "cd_escala_quadricula"],
        )
        writer.writeheader()
        writer.writerows(rows)

    write_codes_txt(out_codes, codes)

    print(
        json.dumps(
            {
                "features": len(features),
                "unique_faixas": len(codes),
                "out_geojson": str(out_geojson),
                "out_csv": str(out_csv),
                "out_codes": str(out_codes),
            }
        )
    )
    return 0


def cmd_select_faixas(args: argparse.Namespace) -> int:
    articulation = Path(args.articulation_geojson).resolve()
    aoi = Path(args.aoi).resolve()
    out_geojson = Path(args.out_geojson).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_codes = Path(args.out_codes).resolve()

    g_art = gpd.read_file(articulation)
    if g_art.empty:
        raise SystemExit(f"Articulation file is empty: {articulation}")
    g_art = set_or_fix_crs(g_art, args.articulation_crs, force=bool(args.force_articulation_crs))

    if args.code_field not in g_art.columns:
        raise SystemExit(f"Code field '{args.code_field}' not found in articulation columns: {list(g_art.columns)}")

    g_aoi = gpd.read_file(aoi)
    if g_aoi.empty:
        raise SystemExit(f"AOI file is empty: {aoi}")
    g_aoi = set_or_fix_crs(g_aoi, args.aoi_crs, force=bool(args.force_aoi_crs))

    work_crs = args.work_crs
    g_art = g_art.to_crs(work_crs)
    g_aoi = g_aoi.to_crs(work_crs)

    aoi_union = g_aoi.geometry.unary_union
    if aoi_union.is_empty:
        raise SystemExit("AOI geometry is empty after reprojection")

    g_sel = g_art[g_art.geometry.intersects(aoi_union)].copy()
    if g_sel.empty:
        print(json.dumps({"selected": 0, "message": "No faixa intersects AOI"}))
        ensure_dir(out_codes.parent)
        out_codes.write_text("", encoding="utf-8")
        ensure_dir(out_csv.parent)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["faixa_code", "intersection_area_m2"])
            writer.writeheader()
        ensure_dir(out_geojson.parent)
        g_sel.to_file(out_geojson, driver="GeoJSON")
        return 0

    g_sel["intersection_area_m2"] = g_sel.geometry.intersection(aoi_union).area
    if args.min_intersection_area_m2 > 0:
        g_sel = g_sel[g_sel["intersection_area_m2"] >= float(args.min_intersection_area_m2)].copy()

    g_sel = g_sel.sort_values(by=[args.code_field]).reset_index(drop=True)

    ensure_dir(out_geojson.parent)
    g_sel.to_file(out_geojson, driver="GeoJSON")

    rows = []
    codes = []
    for _, rec in g_sel.iterrows():
        code = str(rec.get(args.code_field, "") or "").strip()
        if not code:
            continue
        codes.append(code)
        rows.append(
            {
                "faixa_code": code,
                "intersection_area_m2": f"{float(rec.get('intersection_area_m2', 0.0)):.3f}",
            }
        )

    ensure_dir(out_csv.parent)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["faixa_code", "intersection_area_m2"])
        writer.writeheader()
        writer.writerows(rows)

    write_codes_txt(out_codes, codes)

    print(
        json.dumps(
            {
                "selected": len(rows),
                "out_geojson": str(out_geojson),
                "out_csv": str(out_csv),
                "out_codes": str(out_codes),
            }
        )
    )
    return 0


def cmd_index_sources(args: argparse.Namespace) -> int:
    src_root = Path(args.sources_root).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_geojson_31983 = Path(args.out_geojson_31983).resolve()
    out_geojson_4326 = Path(args.out_geojson_4326).resolve()
    exts = split_extensions(args.extensions)

    rasters = find_rasters(src_root, exts)
    if not rasters:
        raise SystemExit(f"No raster files found in {src_root} with extensions={exts}")

    rows: List[Dict[str, str]] = []
    features_31983: List[dict] = []
    features_4326: List[dict] = []
    to_31983_cache: Dict[str, Transformer] = {}
    to_4326_cache: Dict[str, Transformer] = {}
    read_errors = 0

    def _to_crs_transformer(crs_from: str, crs_to: str, cache: Dict[str, Transformer]) -> Transformer:
        key = f"{crs_from}->{crs_to}"
        tr = cache.get(key)
        if tr is None:
            tr = Transformer.from_crs(crs_from, crs_to, always_xy=True)
            cache[key] = tr
        return tr

    for p in rasters:
        try:
            with rasterio.open(p) as ds:
                ds_crs = ds.crs.to_string() if ds.crs is not None else ""
                b = ds.bounds
                res_x, res_y = ds.res
                row = {
                    "path": str(p),
                    "filename": p.name,
                    "faixa_code": infer_faixa_code(p),
                    "crs": ds_crs,
                    "width": str(ds.width),
                    "height": str(ds.height),
                    "bands": str(ds.count),
                    "dtype": str(ds.dtypes[0] if ds.dtypes else ""),
                    "res_x": f"{float(res_x):.8f}",
                    "res_y": f"{float(res_y):.8f}",
                    "minx": f"{float(b.left):.6f}",
                    "miny": f"{float(b.bottom):.6f}",
                    "maxx": f"{float(b.right):.6f}",
                    "maxy": f"{float(b.top):.6f}",
                }
                rows.append(row)

                if ds.crs is None:
                    continue

                geom_src = box(float(b.left), float(b.bottom), float(b.right), float(b.top))
                geom_31983 = geom_src
                geom_4326 = geom_src
                if ds_crs != "EPSG:31983":
                    tr31983 = _to_crs_transformer(ds_crs, "EPSG:31983", to_31983_cache)
                    geom_31983 = shapely_transform(tr31983.transform, geom_src)
                if ds_crs != "EPSG:4326":
                    tr4326 = _to_crs_transformer(ds_crs, "EPSG:4326", to_4326_cache)
                    geom_4326 = shapely_transform(tr4326.transform, geom_src)

                props = {
                    "path": str(p),
                    "filename": p.name,
                    "faixa_code": infer_faixa_code(p),
                    "crs": ds_crs,
                    "width": int(ds.width),
                    "height": int(ds.height),
                    "bands": int(ds.count),
                    "res_x": float(res_x),
                    "res_y": float(res_y),
                }
                features_31983.append({"type": "Feature", "geometry": mapping(geom_31983), "properties": props})
                features_4326.append({"type": "Feature", "geometry": mapping(geom_4326), "properties": props})
        except Exception:
            read_errors += 1
            rows.append(
                {
                    "path": str(p),
                    "filename": p.name,
                    "faixa_code": infer_faixa_code(p),
                    "crs": "",
                    "width": "",
                    "height": "",
                    "bands": "",
                    "dtype": "",
                    "res_x": "",
                    "res_y": "",
                    "minx": "",
                    "miny": "",
                    "maxx": "",
                    "maxy": "",
                }
            )

    ensure_dir(out_csv.parent)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "path",
            "filename",
            "faixa_code",
            "crs",
            "width",
            "height",
            "bands",
            "dtype",
            "res_x",
            "res_y",
            "minx",
            "miny",
            "maxx",
            "maxy",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ensure_dir(out_geojson_31983.parent)
    gdf_31983 = gpd.GeoDataFrame.from_features(features_31983, crs="EPSG:31983")
    gdf_31983.to_file(out_geojson_31983, driver="GeoJSON")

    ensure_dir(out_geojson_4326.parent)
    gdf_4326 = gpd.GeoDataFrame.from_features(features_4326, crs="EPSG:4326")
    gdf_4326.to_file(out_geojson_4326, driver="GeoJSON")

    unique_faixas = {r["faixa_code"] for r in rows if r.get("faixa_code")}
    print(
        json.dumps(
            {
                "rasters": len(rasters),
                "rows": len(rows),
                "read_errors": read_errors,
                "unique_faixas_in_filenames": len(unique_faixas),
                "out_csv": str(out_csv),
                "out_geojson_31983": str(out_geojson_31983),
                "out_geojson_4326": str(out_geojson_4326),
            }
        )
    )
    return 0


def cmd_compare_faixas(args: argparse.Namespace) -> int:
    expected = read_codes(Path(args.expected_codes).resolve(), column=args.expected_column)

    sources_index = Path(args.sources_index_csv).resolve()
    if not sources_index.exists():
        raise SystemExit(f"sources index CSV not found: {sources_index}")

    found: Set[str] = set()
    with sources_index.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        if args.found_column not in fields:
            raise SystemExit(f"Column '{args.found_column}' not found in {sources_index}. Available: {fields}")
        for row in reader:
            code = str(row.get(args.found_column, "") or "").strip()
            if code:
                found.add(code)

    missing = sorted(expected - found)
    extra = sorted(found - expected)

    out_missing = Path(args.out_missing_codes).resolve()
    out_extra = Path(args.out_extra_codes).resolve()
    out_report = Path(args.out_report_json).resolve()

    write_codes_txt(out_missing, missing)
    write_codes_txt(out_extra, extra)

    report = {
        "expected_count": len(expected),
        "found_count": len(found),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "out_missing_codes": str(out_missing),
        "out_extra_codes": str(out_extra),
    }
    ensure_dir(out_report.parent)
    out_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report))
    return 0


def cmd_compare_faixas_spatial(args: argparse.Namespace) -> int:
    articulation = Path(args.articulation_geojson).resolve()
    source_footprints = Path(args.sources_footprints_geojson).resolve()
    code_field = str(args.code_field)

    g_art = gpd.read_file(articulation)
    if g_art.empty:
        raise SystemExit(f"Articulation file is empty: {articulation}")
    g_art = set_or_fix_crs(g_art, args.articulation_crs, force=bool(args.force_articulation_crs))
    if code_field not in g_art.columns:
        raise SystemExit(f"Code field '{code_field}' not found in articulation columns: {list(g_art.columns)}")

    g_src = gpd.read_file(source_footprints)
    if g_src.empty:
        raise SystemExit(f"Source footprints file is empty: {source_footprints}")
    g_src = set_or_fix_crs(g_src, args.sources_crs, force=bool(args.force_sources_crs))

    work_crs = args.work_crs
    g_art = g_art.to_crs(work_crs)
    g_src = g_src.to_crs(work_crs)

    expected_codes: Optional[Set[str]] = None
    if str(args.expected_codes).strip():
        expected_codes = read_codes(Path(args.expected_codes).resolve(), column=args.expected_column)
        g_art = g_art[g_art[code_field].astype(str).isin(expected_codes)].copy()

    src_union = g_src.geometry.union_all() if hasattr(g_src.geometry, "union_all") else g_src.geometry.unary_union
    if src_union.is_empty:
        raise SystemExit("Source footprints union is empty")

    g_art["faixa_code"] = g_art[code_field].astype(str)
    g_art["faixa_area_m2"] = g_art.geometry.area
    g_art["overlap_area_m2"] = g_art.geometry.intersection(src_union).area
    g_art["overlap_ratio"] = g_art["overlap_area_m2"] / g_art["faixa_area_m2"].replace(0, np.nan)
    g_art["overlap_ratio"] = g_art["overlap_ratio"].fillna(0.0)
    g_art["matched"] = (
        (g_art["overlap_area_m2"] >= float(args.min_overlap_area_m2))
        & (g_art["overlap_ratio"] >= float(args.min_overlap_ratio))
    )

    matched_codes = set(g_art[g_art["matched"]]["faixa_code"].astype(str).tolist())
    if expected_codes is None:
        expected_codes = set(g_art["faixa_code"].astype(str).tolist())
    missing_codes = sorted(expected_codes - matched_codes)
    extra_codes = sorted(matched_codes - expected_codes)

    out_report = Path(args.out_report_json).resolve()
    out_missing = Path(args.out_missing_codes).resolve()
    out_extra = Path(args.out_extra_codes).resolve()
    out_detail_csv = Path(args.out_detail_csv).resolve()

    ensure_dir(out_detail_csv.parent)
    cols = ["faixa_code", "faixa_area_m2", "overlap_area_m2", "overlap_ratio", "matched"]
    g_art[cols].sort_values(by=["faixa_code"]).to_csv(out_detail_csv, index=False)

    write_codes_txt(out_missing, missing_codes)
    write_codes_txt(out_extra, extra_codes)

    report = {
        "expected_count": len(expected_codes),
        "matched_count": len(matched_codes),
        "missing_count": len(missing_codes),
        "extra_count": len(extra_codes),
        "min_overlap_ratio": float(args.min_overlap_ratio),
        "min_overlap_area_m2": float(args.min_overlap_area_m2),
        "out_detail_csv": str(out_detail_csv),
        "out_missing_codes": str(out_missing),
        "out_extra_codes": str(out_extra),
    }
    ensure_dir(out_report.parent)
    out_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report))
    return 0


def _paths_from_sources_index(path: Path, allowed_codes: Optional[Set[str]]) -> List[Path]:
    if not path.exists():
        raise SystemExit(f"sources index CSV not found: {path}")
    out: List[Path] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        if "path" not in fields:
            raise SystemExit(f"Column 'path' not found in {path}")
        for row in reader:
            p = Path(str(row.get("path", "") or "")).expanduser().resolve()
            if not p.exists():
                continue
            if allowed_codes is not None:
                code = str(row.get("faixa_code", "") or "").strip()
                if code not in allowed_codes:
                    continue
            out.append(p)
    out = sorted(set(out))
    return out


def _build_vrt(out_vrt: Path, paths: Sequence[Path], overwrite: bool) -> None:
    if not paths:
        raise SystemExit("No raster inputs to build VRT")
    ensure_dir(out_vrt.parent)

    file_list = out_vrt.with_suffix(".inputs.txt")
    file_list.write_text("\n".join(str(p) for p in paths) + "\n", encoding="utf-8")

    cmd = ["gdalbuildvrt"]
    if overwrite:
        cmd.append("-overwrite")
    cmd.extend(["-input_file_list", str(file_list), str(out_vrt)])

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode == 0:
        return

    # Fallback for environments where gdalbuildvrt binary is broken/missing but rasterio works.
    # This fallback supports north-up rasters with uniform CRS/resolution.
    _build_vrt_fallback(out_vrt=out_vrt, paths=paths)


def _build_vrt_fallback(*, out_vrt: Path, paths: Sequence[Path]) -> None:
    if not paths:
        raise SystemExit("No raster inputs for fallback VRT build")

    src_meta: List[Dict[str, object]] = []
    base_crs: Optional[str] = None
    base_res: Optional[Tuple[float, float]] = None
    base_count: Optional[int] = None
    base_dtype: Optional[str] = None

    minx = float("inf")
    miny = float("inf")
    maxx = float("-inf")
    maxy = float("-inf")

    for p in paths:
        with rasterio.open(p) as ds:
            if ds.crs is None:
                raise SystemExit(f"Fallback VRT requires CRS on all sources. Missing CRS: {p}")
            crs = ds.crs.to_string()
            if base_crs is None:
                base_crs = crs
            elif crs != base_crs:
                raise SystemExit(
                    f"Fallback VRT requires a single CRS. Found mixed CRS: {base_crs} vs {crs} ({p})"
                )

            if ds.transform.b != 0.0 or ds.transform.d != 0.0:
                raise SystemExit(f"Fallback VRT only supports north-up rasters (no rotation/skew): {p}")

            resx = float(abs(ds.transform.a))
            resy = float(abs(ds.transform.e))
            if base_res is None:
                base_res = (resx, resy)
            else:
                if abs(resx - base_res[0]) > 1e-9 or abs(resy - base_res[1]) > 1e-9:
                    raise SystemExit(
                        f"Fallback VRT requires uniform resolution. Found {base_res} vs {(resx, resy)} ({p})"
                    )

            count = int(ds.count)
            dtype = str(ds.dtypes[0] if ds.dtypes else "Byte")
            if base_count is None:
                base_count = count
            elif count != base_count:
                raise SystemExit(
                    f"Fallback VRT requires same band count in all sources. Found {base_count} vs {count} ({p})"
                )

            if base_dtype is None:
                base_dtype = dtype
            elif dtype != base_dtype:
                raise SystemExit(
                    f"Fallback VRT requires same datatype in all sources. Found {base_dtype} vs {dtype} ({p})"
                )

            b = ds.bounds
            sx0, sy0, sx1, sy1 = float(b.left), float(b.bottom), float(b.right), float(b.top)
            minx = min(minx, sx0)
            miny = min(miny, sy0)
            maxx = max(maxx, sx1)
            maxy = max(maxy, sy1)

            src_meta.append(
                {
                    "path": str(p),
                    "width": int(ds.width),
                    "height": int(ds.height),
                    "minx": sx0,
                    "miny": sy0,
                    "maxx": sx1,
                    "maxy": sy1,
                }
            )

    if base_crs is None or base_res is None or base_count is None:
        raise SystemExit("Fallback VRT failed: no valid source metadata")

    resx, resy = base_res
    raster_x_size = int(round((maxx - minx) / resx))
    raster_y_size = int(round((maxy - miny) / resy))

    root = ET.Element("VRTDataset", rasterXSize=str(raster_x_size), rasterYSize=str(raster_y_size))
    ET.SubElement(root, "SRS").text = base_crs
    ET.SubElement(root, "GeoTransform").text = f"{minx}, {resx}, 0.0, {maxy}, 0.0, {-resy}"

    dtype_to_vrt = {
        "uint8": "Byte",
        "int8": "Byte",
        "uint16": "UInt16",
        "int16": "Int16",
        "uint32": "UInt32",
        "int32": "Int32",
        "float32": "Float32",
        "float64": "Float64",
    }
    vrt_dtype = dtype_to_vrt.get((base_dtype or "uint8").lower(), "Byte")

    color_names = {1: "Red", 2: "Green", 3: "Blue", 4: "Alpha"}
    for band_idx in range(1, int(base_count) + 1):
        band_el = ET.SubElement(root, "VRTRasterBand", dataType=vrt_dtype, band=str(band_idx))
        if band_idx in color_names:
            ET.SubElement(band_el, "ColorInterp").text = color_names[band_idx]
        ET.SubElement(band_el, "NoDataValue").text = "0"

        for rec in src_meta:
            src_path = str(rec["path"])
            src_w = int(rec["width"])
            src_h = int(rec["height"])
            src_minx = float(rec["minx"])
            src_maxy = float(rec["maxy"])

            x_off = int(round((src_minx - minx) / resx))
            y_off = int(round((maxy - src_maxy) / resy))

            src_el = ET.SubElement(band_el, "SimpleSource")
            ET.SubElement(src_el, "SourceFilename", relativeToVRT="0").text = src_path
            ET.SubElement(src_el, "SourceBand").text = str(band_idx)
            ET.SubElement(src_el, "SrcRect", xOff="0", yOff="0", xSize=str(src_w), ySize=str(src_h))
            ET.SubElement(
                src_el,
                "DstRect",
                xOff=str(x_off),
                yOff=str(y_off),
                xSize=str(src_w),
                ySize=str(src_h),
            )

    tree = ET.ElementTree(root)
    tree.write(out_vrt, encoding="UTF-8", xml_declaration=True)


def cmd_build_vrt(args: argparse.Namespace) -> int:
    out_vrt = Path(args.out_vrt).resolve()
    allowed_codes: Optional[Set[str]] = None
    if args.filter_codes:
        allowed_codes = read_codes(Path(args.filter_codes).resolve(), column=args.filter_codes_column)

    if args.sources_index_csv:
        paths = _paths_from_sources_index(Path(args.sources_index_csv).resolve(), allowed_codes)
    else:
        src_root = Path(args.sources_root).resolve()
        paths = find_rasters(src_root, split_extensions(args.extensions))
        if allowed_codes is not None:
            paths = [p for p in paths if infer_faixa_code(p) in allowed_codes]

    if not paths:
        raise SystemExit("No source rasters selected for VRT")

    _build_vrt(out_vrt, paths, overwrite=bool(args.overwrite))

    print(
        json.dumps(
            {
                "inputs": len(paths),
                "out_vrt": str(out_vrt),
                "inputs_list": str(out_vrt.with_suffix('.inputs.txt')),
            }
        )
    )
    return 0


def _parse_resampling(name: str) -> Resampling:
    v = name.strip().lower()
    if v == "nearest":
        return Resampling.nearest
    if v == "bilinear":
        return Resampling.bilinear
    if v == "cubic":
        return Resampling.cubic
    raise SystemExit(f"Unsupported resampling: {name}")


def _get_source_dataset(source_vrt: Path):
    current = getattr(_thread_local, "source_ds", None)
    current_path = getattr(_thread_local, "source_path", "")
    target_path = str(source_vrt)
    if current is not None and current_path == target_path:
        return current

    if current is not None:
        try:
            current.close()
        except Exception:
            pass

    ds = rasterio.open(source_vrt)
    _thread_local.source_ds = ds
    _thread_local.source_path = target_path
    return ds


def render_one_job(
    *,
    job: RenderJob,
    source_vrt: Path,
    dst_crs: str,
    assume_source_crs: str,
    render_retries: int,
    retry_delay: float,
    resampling: Resampling,
    blank_max_value: int,
) -> RenderResult:
    attempts = 0
    last_err = ""

    for attempt in range(render_retries + 1):
        attempts += 1
        try:
            ds = _get_source_dataset(source_vrt)
            src_crs = ds.crs
            if src_crs is None and assume_source_crs.strip():
                src_crs = RasterioCRS.from_string(assume_source_crs.strip())
            if src_crs is None:
                raise RuntimeError("source CRS missing and --assume-source-crs not provided")

            band_count = max(1, min(3, int(ds.count)))
            dst = np.zeros((3, job.height, job.width), dtype=np.uint8)
            dst_transform = from_bounds(job.xmin, job.ymin, job.xmax, job.ymax, job.width, job.height)

            src_nodata = ds.nodata
            for band_idx in range(1, band_count + 1):
                out_band = np.zeros((job.height, job.width), dtype=np.float32)
                reproject(
                    source=rasterio.band(ds, band_idx),
                    destination=out_band,
                    src_transform=ds.transform,
                    src_crs=src_crs,
                    src_nodata=src_nodata,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    dst_nodata=0,
                    resampling=resampling,
                )
                dst[band_idx - 1] = np.clip(np.rint(out_band), 0, 255).astype(np.uint8)

            if band_count == 1:
                dst[1] = dst[0]
                dst[2] = dst[0]
            elif band_count == 2:
                dst[2] = dst[1]

            if int(dst.max()) <= int(blank_max_value):
                return RenderResult(
                    index=job.index,
                    status="missing",
                    http_status="local_blank",
                    content_type="image/png",
                    attempts_made=attempts,
                    error=f"blank_max_le_{blank_max_value}",
                )

            ensure_dir(job.path.parent)
            tmp_path = job.path.with_suffix(".tmp.png")
            rgb = np.moveaxis(dst, 0, 2)
            Image.fromarray(rgb, mode="RGB").save(tmp_path, format="PNG")
            tmp_path.replace(job.path)

            _write_world_file(job.path, job.xmin, job.ymin, job.xmax, job.ymax, job.width, job.height)
            return RenderResult(
                index=job.index,
                status="downloaded",
                http_status="local_vrt",
                content_type="image/png",
                attempts_made=attempts,
                error="",
            )
        except Exception as exc:
            last_err = f"error:{type(exc).__name__}:{exc}"
            if attempt < render_retries:
                time.sleep(retry_delay * (2 ** attempt))
                continue

    return RenderResult(
        index=job.index,
        status="failed",
        http_status="local_error",
        content_type="",
        attempts_made=attempts,
        error=last_err,
    )


def run_render_round(
    *,
    rows: List[Dict[str, str]],
    statuses: Set[str],
    source_vrt: Path,
    dst_crs: str,
    assume_source_crs: str,
    workers: int,
    render_retries: int,
    retry_delay: float,
    resampling: Resampling,
    blank_max_value: int,
    overwrite_existing: bool,
    max_jobs: int,
) -> Counter:
    jobs: List[RenderJob] = []
    counts = Counter()

    for i, row in enumerate(rows):
        status = row.get("status", "pending")
        if status not in statuses:
            continue

        path = Path(row["path"])
        if not overwrite_existing and path.exists() and path.with_suffix(".pgw").exists():
            row["status"] = "downloaded"
            row["http_status"] = "local_cached"
            row["content_type"] = "image/png"
            row["last_error"] = ""
            row["last_update"] = rebuild.now_iso()
            counts["cached"] += 1
            continue

        jobs.append(
            RenderJob(
                index=i,
                path=path,
                xmin=rebuild.normalize_float(row.get("xmin", "0"), 0.0),
                ymin=rebuild.normalize_float(row.get("ymin", "0"), 0.0),
                xmax=rebuild.normalize_float(row.get("xmax", "0"), 0.0),
                ymax=rebuild.normalize_float(row.get("ymax", "0"), 0.0),
                width=rebuild.normalize_int(row.get("width_px", "1024"), 1024),
                height=rebuild.normalize_int(row.get("height_px", "1024"), 1024),
            )
        )

    if max_jobs > 0:
        jobs = jobs[:max_jobs]

    if jobs:
        print(f"render_jobs={len(jobs)} workers={workers} statuses={sorted(statuses)}")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(
                render_one_job,
                job=job,
                source_vrt=source_vrt,
                dst_crs=dst_crs,
                assume_source_crs=assume_source_crs,
                render_retries=render_retries,
                retry_delay=retry_delay,
                resampling=resampling,
                blank_max_value=blank_max_value,
            )
            for job in jobs
        ]

        for fut in as_completed(futs):
            res = fut.result()
            row = rows[res.index]
            row["status"] = res.status
            row["http_status"] = res.http_status
            row["content_type"] = res.content_type
            row["attempts"] = str(rebuild.normalize_int(row.get("attempts", "0"), 0) + int(res.attempts_made))
            row["last_error"] = res.error
            row["last_update"] = rebuild.now_iso()
            counts[res.status] += 1

    return counts


def _manifest_path(args: argparse.Namespace, out_root: Path) -> Path:
    return Path(args.manifest_csv).resolve() if str(args.manifest_csv).strip() else (out_root / "chips_manifest.csv")


def cmd_render_from_vrt(args: argparse.Namespace) -> int:
    out_root = Path(args.out_root).resolve()
    source_vrt = Path(args.source_vrt).resolve()
    if not source_vrt.exists():
        raise SystemExit(f"Source VRT not found: {source_vrt}")

    manifest_csv = _manifest_path(args, out_root)
    rows = rebuild.load_manifest(manifest_csv)

    counts = run_render_round(
        rows=rows,
        statuses=rebuild.parse_statuses(args.statuses),
        source_vrt=source_vrt,
        dst_crs=args.crs,
        assume_source_crs=args.assume_source_crs,
        workers=args.workers,
        render_retries=args.render_retries,
        retry_delay=args.retry_delay,
        resampling=_parse_resampling(args.resampling),
        blank_max_value=args.blank_max_value,
        overwrite_existing=bool(args.overwrite_existing),
        max_jobs=args.max_jobs,
    )

    rebuild.write_manifest_and_cells(out_root, rows)
    unresolved = sum(1 for r in rows if r.get("status", "") not in SUCCESS_STATUSES)

    print(
        json.dumps(
            {
                "render_round": dict(counts),
                "unresolved": unresolved,
                "manifest": str(out_root / "chips_manifest.csv"),
            }
        )
    )
    return 0


def cmd_retry_render_until_complete(args: argparse.Namespace) -> int:
    out_root = Path(args.out_root).resolve()
    source_vrt = Path(args.source_vrt).resolve()
    if not source_vrt.exists():
        raise SystemExit(f"Source VRT not found: {source_vrt}")

    manifest_csv = _manifest_path(args, out_root)
    rows = rebuild.load_manifest(manifest_csv)

    statuses = rebuild.parse_statuses(args.statuses)
    unresolved_prev = None
    for round_idx in range(1, args.max_rounds + 1):
        unresolved_before = sum(1 for r in rows if r.get("status", "") not in SUCCESS_STATUSES)
        if unresolved_before == 0:
            print(json.dumps({"round": round_idx, "message": "already_complete"}))
            break

        counts = run_render_round(
            rows=rows,
            statuses=statuses,
            source_vrt=source_vrt,
            dst_crs=args.crs,
            assume_source_crs=args.assume_source_crs,
            workers=args.workers,
            render_retries=args.render_retries,
            retry_delay=args.retry_delay,
            resampling=_parse_resampling(args.resampling),
            blank_max_value=args.blank_max_value,
            overwrite_existing=bool(args.overwrite_existing),
            max_jobs=args.max_jobs,
        )

        rebuild.write_manifest_and_cells(out_root, rows)
        unresolved_after = sum(1 for r in rows if r.get("status", "") not in SUCCESS_STATUSES)

        print(
            json.dumps(
                {
                    "round": round_idx,
                    "counts": dict(counts),
                    "unresolved_before": unresolved_before,
                    "unresolved_after": unresolved_after,
                }
            )
        )

        if unresolved_after == 0:
            break

        if unresolved_prev is not None and unresolved_after >= unresolved_prev:
            print(json.dumps({"round": round_idx, "message": "no_progress_stopping"}))
            break

        unresolved_prev = unresolved_after
        if args.round_sleep > 0:
            time.sleep(args.round_sleep)

    return 0


def cmd_full_rebuild_from_vrt(args: argparse.Namespace) -> int:
    out_root = Path(args.out_root).resolve()
    if args.clean and out_root.exists():
        import shutil

        shutil.rmtree(out_root)

    source_vrt = Path(args.source_vrt).resolve()
    if not source_vrt.exists():
        raise SystemExit(f"Source VRT not found: {source_vrt}")

    manifest_csv = rebuild.build_manifest(
        grid_dir=Path(args.grid_dir).resolve(),
        out_root=out_root,
        crs=args.crs,
        chip_size=args.chip_size,
        meters_per_pixel=args.meters_per_pixel,
        reuse_existing_files=bool(args.reuse_existing_files),
        preserve_status=bool(args.preserve_status),
    )

    rows = rebuild.load_manifest(manifest_csv)
    statuses = rebuild.parse_statuses(args.statuses)

    unresolved_prev = None
    for round_idx in range(1, args.max_rounds + 1):
        unresolved_before = sum(1 for r in rows if r.get("status", "") not in SUCCESS_STATUSES)
        if unresolved_before == 0:
            break

        counts = run_render_round(
            rows=rows,
            statuses=statuses,
            source_vrt=source_vrt,
            dst_crs=args.crs,
            assume_source_crs=args.assume_source_crs,
            workers=args.workers,
            render_retries=args.render_retries,
            retry_delay=args.retry_delay,
            resampling=_parse_resampling(args.resampling),
            blank_max_value=args.blank_max_value,
            overwrite_existing=bool(args.overwrite_existing),
            max_jobs=args.max_jobs,
        )

        rebuild.write_manifest_and_cells(out_root, rows)
        unresolved_after = sum(1 for r in rows if r.get("status", "") not in SUCCESS_STATUSES)
        print(
            json.dumps(
                {
                    "round": round_idx,
                    "counts": dict(counts),
                    "unresolved_before": unresolved_before,
                    "unresolved_after": unresolved_after,
                }
            )
        )

        if unresolved_after == 0:
            break
        if unresolved_prev is not None and unresolved_after >= unresolved_prev:
            print(json.dumps({"round": round_idx, "message": "no_progress_stopping"}))
            break

        unresolved_prev = unresolved_after
        if args.round_sleep > 0:
            time.sleep(args.round_sleep)

    coverage_out = Path(args.coverage_out).resolve() if args.coverage_out else (out_root / "_coverage")
    rebuild.validate_manifest(
        rows=rows,
        out_dir=coverage_out,
        src_crs=args.crs,
        dst_crs=args.dst_crs,
        full_chip_count=args.full_chip_count,
    )
    return 0


def add_render_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--manifest-csv", default="")
    p.add_argument("--source-vrt", required=True)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--render-retries", type=int, default=1)
    p.add_argument("--retry-delay", type=float, default=1.0)
    p.add_argument("--statuses", default="pending,failed,missing")
    p.add_argument("--max-jobs", type=int, default=0, help="Optional debug limit")
    p.add_argument("--overwrite-existing", action="store_true")
    p.add_argument("--assume-source-crs", default="")
    p.add_argument("--resampling", choices=["nearest", "bilinear", "cubic"], default="nearest")
    p.add_argument("--blank-max-value", type=int, default=0, help="Mark chip as missing when max pixel value <= this threshold")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "GeoSampa 2020 official-download workflow: fetch articulation/faixas, "
            "index local official rasters, build VRT, and chip into existing manifest format."
        )
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch-articulation", help="Download official articulation index (quadricula_orto_2020) via WFS")
    p_fetch.add_argument("--wfs-url", default=DEFAULT_WFS)
    p_fetch.add_argument("--layer", default=DEFAULT_ARTICULATION_LAYER)
    p_fetch.add_argument("--crs", default="EPSG:31983")
    p_fetch.add_argument("--timeout", type=int, default=120)
    p_fetch.add_argument("--code-field", default="cd_quadricula")
    p_fetch.add_argument("--out-geojson", default="data/external/geosampa_2020/quadricula_orto_2020.geojson")
    p_fetch.add_argument("--out-csv", default="data/external/geosampa_2020/quadricula_orto_2020.csv")
    p_fetch.add_argument("--out-codes", default="data/external/geosampa_2020/quadricula_orto_2020_codes.txt")
    p_fetch.set_defaults(func=cmd_fetch_articulation)

    p_sel = sub.add_parser("select-faixas", help="Select faixa codes intersecting AOI using articulation GeoJSON")
    p_sel.add_argument("--articulation-geojson", required=True)
    p_sel.add_argument("--articulation-crs", default="EPSG:31983")
    p_sel.add_argument("--force-articulation-crs", action="store_true")
    p_sel.add_argument("--aoi", required=True, help="AOI geometry file (e.g., city boundary / cell union)")
    p_sel.add_argument("--aoi-crs", default="EPSG:31983")
    p_sel.add_argument("--force-aoi-crs", action="store_true")
    p_sel.add_argument("--work-crs", default="EPSG:31983")
    p_sel.add_argument("--code-field", default="cd_quadricula")
    p_sel.add_argument("--min-intersection-area-m2", type=float, default=0.0)
    p_sel.add_argument("--out-geojson", default="data/external/geosampa_2020/selected_faixas.geojson")
    p_sel.add_argument("--out-csv", default="data/external/geosampa_2020/selected_faixas.csv")
    p_sel.add_argument("--out-codes", default="data/external/geosampa_2020/selected_faixas_codes.txt")
    p_sel.set_defaults(func=cmd_select_faixas)

    p_index = sub.add_parser("index-sources", help="Index locally downloaded official ortho rasters")
    p_index.add_argument("--sources-root", required=True)
    p_index.add_argument("--extensions", default=",".join(DEFAULT_EXTENSIONS))
    p_index.add_argument("--out-csv", default="data/external/geosampa_2020/sources_index.csv")
    p_index.add_argument("--out-geojson-31983", default="data/external/geosampa_2020/sources_footprints_31983.geojson")
    p_index.add_argument("--out-geojson-4326", default="data/external/geosampa_2020/sources_footprints_4326.geojson")
    p_index.set_defaults(func=cmd_index_sources)

    p_cmp = sub.add_parser("compare-faixas", help="Compare expected faixa codes against indexed local sources")
    p_cmp.add_argument("--expected-codes", required=True, help="Expected faixa codes file (.txt/.csv)")
    p_cmp.add_argument("--expected-column", default="")
    p_cmp.add_argument("--sources-index-csv", required=True)
    p_cmp.add_argument("--found-column", default="faixa_code")
    p_cmp.add_argument("--out-missing-codes", default="data/external/geosampa_2020/missing_faixas.txt")
    p_cmp.add_argument("--out-extra-codes", default="data/external/geosampa_2020/extra_faixas.txt")
    p_cmp.add_argument("--out-report-json", default="data/external/geosampa_2020/faixa_comparison_report.json")
    p_cmp.set_defaults(func=cmd_compare_faixas)

    p_cmp_sp = sub.add_parser(
        "compare-faixas-spatial",
        help="Compare expected faixa coverage using spatial overlap between articulation and source footprints",
    )
    p_cmp_sp.add_argument("--articulation-geojson", required=True)
    p_cmp_sp.add_argument("--articulation-crs", default="EPSG:31983")
    p_cmp_sp.add_argument("--force-articulation-crs", action="store_true")
    p_cmp_sp.add_argument("--sources-footprints-geojson", required=True)
    p_cmp_sp.add_argument("--sources-crs", default="EPSG:31983")
    p_cmp_sp.add_argument("--force-sources-crs", action="store_true")
    p_cmp_sp.add_argument("--work-crs", default="EPSG:31983")
    p_cmp_sp.add_argument("--code-field", default="cd_quadricula")
    p_cmp_sp.add_argument("--expected-codes", default="")
    p_cmp_sp.add_argument("--expected-column", default="")
    p_cmp_sp.add_argument("--min-overlap-ratio", type=float, default=0.10)
    p_cmp_sp.add_argument("--min-overlap-area-m2", type=float, default=10.0)
    p_cmp_sp.add_argument("--out-detail-csv", default="data/external/geosampa_2020/faixa_spatial_detail.csv")
    p_cmp_sp.add_argument("--out-missing-codes", default="data/external/geosampa_2020/missing_faixas_spatial.txt")
    p_cmp_sp.add_argument("--out-extra-codes", default="data/external/geosampa_2020/extra_faixas_spatial.txt")
    p_cmp_sp.add_argument("--out-report-json", default="data/external/geosampa_2020/faixa_spatial_report.json")
    p_cmp_sp.set_defaults(func=cmd_compare_faixas_spatial)

    p_vrt = sub.add_parser("build-vrt", help="Build a VRT mosaic from local official rasters")
    p_vrt.add_argument("--out-vrt", required=True)
    p_vrt.add_argument("--overwrite", action="store_true")
    p_vrt.add_argument("--sources-index-csv", default="")
    p_vrt.add_argument("--sources-root", default="")
    p_vrt.add_argument("--extensions", default=",".join(DEFAULT_EXTENSIONS))
    p_vrt.add_argument("--filter-codes", default="", help="Optional faixa codes file (.txt/.csv)")
    p_vrt.add_argument("--filter-codes-column", default="")
    p_vrt.set_defaults(func=cmd_build_vrt)

    p_render = sub.add_parser("render-from-vrt", help="Render one chip-generation pass from local VRT into manifest paths")
    rebuild.add_shared_rebuild_args(p_render)
    add_render_args(p_render)
    p_render.set_defaults(func=cmd_render_from_vrt)

    p_retry = sub.add_parser(
        "retry-render-until-complete",
        help="Run multiple local render passes from VRT until complete or no progress",
    )
    rebuild.add_shared_rebuild_args(p_retry)
    add_render_args(p_retry)
    p_retry.add_argument("--max-rounds", type=int, default=6)
    p_retry.add_argument("--round-sleep", type=float, default=1.0)
    p_retry.set_defaults(func=cmd_retry_render_until_complete)

    p_full = sub.add_parser("full-rebuild-from-vrt", help="Build manifest, render from local VRT, and validate coverage")
    rebuild.add_shared_rebuild_args(p_full)
    add_render_args(p_full)
    p_full.add_argument("--max-rounds", type=int, default=6)
    p_full.add_argument("--round-sleep", type=float, default=1.0)
    rebuild.add_validate_args(p_full, include_manifest_csv=False)
    p_full.set_defaults(func=cmd_full_rebuild_from_vrt)

    return ap


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
