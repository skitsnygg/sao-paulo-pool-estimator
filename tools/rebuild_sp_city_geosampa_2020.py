#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import threading
import time
from io import BytesIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import requests
from PIL import Image
from pyproj import Transformer
from shapely.geometry import box, mapping, shape
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union

from src.data.fetch_geosampa_ortho import _write_world_file
from src.data.geosampa_config import ORTHO_LAYER, WMS_BASE_URL

CHIP_FIELDNAMES = [
    "cell_id",
    "chip_id",
    "row",
    "col",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "crs",
    "width_px",
    "height_px",
    "path",
    "status",
    "http_status",
    "content_type",
    "attempts",
    "last_error",
    "last_update",
]

SUCCESS_STATUSES = {"downloaded", "cached"}
TARGET_RETRY_STATUSES = {"pending", "failed", "missing"}
RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class DownloadJob:
    index: int
    cell_id: str
    chip_id: str
    path: Path
    row: int
    col: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    width: int
    height: int


@dataclass(frozen=True)
class BlockDownloadJob:
    chips: Tuple[DownloadJob, ...]
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    width: int
    height: int
    min_row: int
    max_row: int
    min_col: int
    max_col: int


@dataclass(frozen=True)
class DownloadResult:
    index: int
    status: str
    http_status: str
    content_type: str
    attempts_made: int
    error: str


_thread_local = threading.local()


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CHIP_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            fixed = {k: str(row.get(k, "")) for k in CHIP_FIELDNAMES}
            rows.append(fixed)
    return rows


def geojson_bbox(path: Path) -> Tuple[float, float, float, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    gtype = payload.get("type")
    geoms = []

    if gtype == "FeatureCollection":
        for feat in payload.get("features", []):
            geom = feat.get("geometry")
            if geom:
                geoms.append(shape(geom))
    elif gtype == "Feature":
        geom = payload.get("geometry")
        if geom:
            geoms.append(shape(geom))
    else:
        geoms.append(shape(payload))

    geoms = [g for g in geoms if not g.is_empty]
    if not geoms:
        raise ValueError(f"No geometry found in {path}")

    minx = min(g.bounds[0] for g in geoms)
    miny = min(g.bounds[1] for g in geoms)
    maxx = max(g.bounds[2] for g in geoms)
    maxy = max(g.bounds[3] for g in geoms)
    return float(minx), float(miny), float(maxx), float(maxy)


def normalize_int(v: str, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def normalize_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def build_manifest(
    *,
    grid_dir: Path,
    out_root: Path,
    crs: str,
    chip_size: int,
    meters_per_pixel: float,
    reuse_existing_files: bool,
    preserve_status: bool,
) -> Path:
    chip_m = chip_size * meters_per_pixel
    cell_geojsons = sorted([p for p in grid_dir.glob("cell_*.geojson") if p.is_file()])
    if not cell_geojsons:
        raise SystemExit(f"No cell_*.geojson found in {grid_dir}")

    ensure_dir(out_root)

    all_rows: List[Dict[str, str]] = []
    per_cell_count: Dict[str, int] = {}

    for cell_path in cell_geojsons:
        cell_id = cell_path.stem
        cell_dir = out_root / cell_id
        ensure_dir(cell_dir)

        existing_rows = {}
        if preserve_status:
            for row in read_csv_rows(cell_dir / "chips.csv"):
                existing_rows[row.get("chip_id", "")] = row

        minx, miny, maxx, maxy = geojson_bbox(cell_path)
        cols = int(math.ceil((maxx - minx) / chip_m))
        rows = int(math.ceil((maxy - miny) / chip_m))

        cell_rows: List[Dict[str, str]] = []
        for r in range(rows):
            for c in range(cols):
                x0 = minx + c * chip_m
                y0 = miny + r * chip_m
                x1 = x0 + chip_m
                y1 = y0 + chip_m
                chip_id = f"r{r:04d}_c{c:04d}"
                png_path = cell_dir / f"{chip_id}.png"

                status = "pending"
                http_status = ""
                content_type = ""
                attempts = "0"
                last_error = ""
                last_update = ""

                prev = existing_rows.get(chip_id)
                if prev:
                    status = prev.get("status", status)
                    http_status = prev.get("http_status", "")
                    content_type = prev.get("content_type", "")
                    attempts = prev.get("attempts", attempts)
                    last_error = prev.get("last_error", "")
                    last_update = prev.get("last_update", "")

                if reuse_existing_files and png_path.exists() and png_path.with_suffix(".pgw").exists():
                    status = "downloaded"
                    last_error = ""
                    last_update = now_iso()

                row = {
                    "cell_id": cell_id,
                    "chip_id": chip_id,
                    "row": str(r),
                    "col": str(c),
                    "xmin": f"{x0:.3f}",
                    "ymin": f"{y0:.3f}",
                    "xmax": f"{x1:.3f}",
                    "ymax": f"{y1:.3f}",
                    "crs": crs,
                    "width_px": str(chip_size),
                    "height_px": str(chip_size),
                    "path": str(png_path.resolve()),
                    "status": status,
                    "http_status": http_status,
                    "content_type": content_type,
                    "attempts": str(normalize_int(attempts, 0)),
                    "last_error": last_error,
                    "last_update": last_update,
                }
                cell_rows.append(row)

        write_csv(cell_dir / "chips.csv", cell_rows)
        all_rows.extend(cell_rows)
        per_cell_count[cell_id] = len(cell_rows)

    manifest_csv = out_root / "chips_manifest.csv"
    write_csv(manifest_csv, all_rows)

    summary = {
        "grid_dir": str(grid_dir.resolve()),
        "out_root": str(out_root.resolve()),
        "crs": crs,
        "chip_size": chip_size,
        "meters_per_pixel": meters_per_pixel,
        "cells": len(per_cell_count),
        "expected_chips_total": len(all_rows),
        "expected_chips_per_cell": per_cell_count,
        "manifest_csv": str(manifest_csv.resolve()),
    }
    (out_root / "manifest_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "cells": len(per_cell_count),
                "expected_chips_total": len(all_rows),
                "manifest_csv": str(manifest_csv),
            }
        )
    )
    return manifest_csv


def load_manifest(manifest_csv: Path) -> List[Dict[str, str]]:
    rows = read_csv_rows(manifest_csv)
    if not rows:
        raise SystemExit(f"Manifest is empty or missing: {manifest_csv}")
    return rows


def group_rows_by_cell(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["cell_id"]].append(row)
    for cell_rows in grouped.values():
        cell_rows.sort(key=lambda r: (normalize_int(r.get("row", "0")), normalize_int(r.get("col", "0"))))
    return grouped


def write_manifest_and_cells(out_root: Path, rows: List[Dict[str, str]]) -> None:
    manifest_csv = out_root / "chips_manifest.csv"
    write_csv(manifest_csv, rows)

    grouped = group_rows_by_cell(rows)
    for cell_id, cell_rows in grouped.items():
        write_csv(out_root / cell_id / "chips.csv", cell_rows)

    unresolved_paths = [r["path"] for r in rows if r.get("status", "") not in SUCCESS_STATUSES]
    (out_root / "_retry_paths.txt").write_text("\n".join(unresolved_paths) + ("\n" if unresolved_paths else ""), encoding="utf-8")


def _get_session() -> requests.Session:
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        sess.headers.update({"User-Agent": "geosampa-2020-rebuild/1.0"})
        _thread_local.session = sess
    return sess


def _verify_image_content(data: bytes, min_bytes: int) -> Tuple[bool, str]:
    if len(data) < min_bytes:
        return False, f"too_small:{len(data)}"

    try:
        with Image.open(BytesIO(data)) as im:
            im.verify()
        return True, ""
    except Exception as exc:
        return False, f"invalid_image:{type(exc).__name__}"


def _build_wms_params(
    *,
    layer: str,
    crs: str,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    width: int,
    height: int,
) -> Dict[str, str]:
    return {
        "service": "WMS",
        "request": "GetMap",
        "version": "1.3.0",
        "layers": layer,
        "styles": "",
        "crs": crs,
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "width": str(width),
        "height": str(height),
        "format": "image/png",
        "transparent": "false",
    }


def build_download_jobs(
    *,
    rows: List[Dict[str, str]],
    statuses: set[str],
    max_jobs: int,
) -> List[DownloadJob]:
    jobs: List[DownloadJob] = []
    for i, row in enumerate(rows):
        status = row.get("status", "pending")
        if status not in statuses:
            continue
        jobs.append(
            DownloadJob(
                index=i,
                cell_id=row["cell_id"],
                chip_id=row["chip_id"],
                path=Path(row["path"]),
                row=normalize_int(row.get("row", "0"), 0),
                col=normalize_int(row.get("col", "0"), 0),
                xmin=normalize_float(row["xmin"]),
                ymin=normalize_float(row["ymin"]),
                xmax=normalize_float(row["xmax"]),
                ymax=normalize_float(row["ymax"]),
                width=normalize_int(row["width_px"], 1024),
                height=normalize_int(row["height_px"], 1024),
            )
        )

    if max_jobs > 0:
        jobs = jobs[:max_jobs]
    return jobs


def build_block_jobs(
    *,
    jobs: List[DownloadJob],
    block_rows: int,
    block_cols: int,
) -> List[BlockDownloadJob]:
    if not jobs:
        return []

    br = max(1, int(block_rows))
    bc = max(1, int(block_cols))

    grouped: Dict[Tuple[str, int, int, int, int], List[DownloadJob]] = defaultdict(list)
    for job in jobs:
        key = (job.cell_id, job.width, job.height, job.row // br, job.col // bc)
        grouped[key].append(job)

    block_jobs: List[BlockDownloadJob] = []
    for key in sorted(grouped.keys()):
        group = grouped[key]
        min_row = min(j.row for j in group)
        max_row = max(j.row for j in group)
        min_col = min(j.col for j in group)
        max_col = max(j.col for j in group)
        width = (max_col - min_col + 1) * group[0].width
        height = (max_row - min_row + 1) * group[0].height
        block_jobs.append(
            BlockDownloadJob(
                chips=tuple(sorted(group, key=lambda j: (j.row, j.col, j.index))),
                xmin=min(j.xmin for j in group),
                ymin=min(j.ymin for j in group),
                xmax=max(j.xmax for j in group),
                ymax=max(j.ymax for j in group),
                width=width,
                height=height,
                min_row=min_row,
                max_row=max_row,
                min_col=min_col,
                max_col=max_col,
            )
        )

    block_jobs.sort(key=lambda b: min(j.index for j in b.chips))
    return block_jobs


def _block_results(
    *,
    block: BlockDownloadJob,
    status: str,
    http_status: str,
    content_type: str,
    attempts_made: int,
    error: str,
) -> List[DownloadResult]:
    return [
        DownloadResult(
            index=chip.index,
            status=status,
            http_status=http_status,
            content_type=content_type,
            attempts_made=attempts_made,
            error=error,
        )
        for chip in block.chips
    ]


def _slice_block_to_chips(
    *,
    image_data: bytes,
    block: BlockDownloadJob,
    attempts_made: int,
    http_status: str,
    content_type: str,
) -> List[DownloadResult]:
    results: List[DownloadResult] = []

    with Image.open(BytesIO(image_data)) as im:
        if im.size != (block.width, block.height):
            return _block_results(
                block=block,
                status="failed",
                http_status=http_status,
                content_type=content_type,
                attempts_made=attempts_made,
                error=f"unexpected_image_size:{im.size[0]}x{im.size[1]}",
            )

        for chip in block.chips:
            x_offset = (chip.col - block.min_col) * chip.width
            y_offset = (block.max_row - chip.row) * chip.height
            x_end = x_offset + chip.width
            y_end = y_offset + chip.height

            if x_offset < 0 or y_offset < 0 or x_end > block.width or y_end > block.height:
                results.append(
                    DownloadResult(
                        index=chip.index,
                        status="failed",
                        http_status=http_status,
                        content_type=content_type,
                        attempts_made=attempts_made,
                        error=f"slice_oob:x={x_offset}:{x_end},y={y_offset}:{y_end}",
                    )
                )
                continue

            try:
                ensure_dir(chip.path.parent)
                tile = im.crop((x_offset, y_offset, x_end, y_end))
                tmp_path = chip.path.with_suffix(".tmp")
                tile.save(tmp_path, format="PNG")
                tmp_path.replace(chip.path)
                _write_world_file(chip.path, chip.xmin, chip.ymin, chip.xmax, chip.ymax, chip.width, chip.height)
                results.append(
                    DownloadResult(
                        index=chip.index,
                        status="downloaded",
                        http_status=http_status,
                        content_type=content_type,
                        attempts_made=attempts_made,
                        error="",
                    )
                )
            except Exception as exc:
                results.append(
                    DownloadResult(
                        index=chip.index,
                        status="failed",
                        http_status=http_status,
                        content_type=content_type,
                        attempts_made=attempts_made,
                        error=f"slice_write_error:{type(exc).__name__}",
                    )
                )

    return results


def download_one_block(
    *,
    block: BlockDownloadJob,
    wms_url: str,
    layer: str,
    crs: str,
    timeout: int,
    request_retries: int,
    retry_delay: float,
    min_bytes: int,
) -> List[DownloadResult]:
    attempts_made = 0
    last_http = ""
    last_content_type = ""
    last_error = ""

    for attempt in range(request_retries + 1):
        attempts_made += 1
        try:
            sess = _get_session()
            resp = sess.get(
                wms_url,
                params=_build_wms_params(
                    layer=layer,
                    crs=crs,
                    xmin=block.xmin,
                    ymin=block.ymin,
                    xmax=block.xmax,
                    ymax=block.ymax,
                    width=block.width,
                    height=block.height,
                ),
                timeout=timeout,
            )
            last_http = str(resp.status_code)
            last_content_type = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()

            if resp.status_code == 200 and last_content_type.startswith("image"):
                ok, err = _verify_image_content(resp.content, min_bytes=min_bytes)
                if not ok:
                    last_error = err
                    if attempt < request_retries:
                        time.sleep(retry_delay * (2 ** attempt))
                        continue
                    return _block_results(
                        block=block,
                        status="failed",
                        http_status=last_http,
                        content_type=last_content_type,
                        attempts_made=attempts_made,
                        error=last_error,
                    )

                try:
                    return _slice_block_to_chips(
                        image_data=resp.content,
                        block=block,
                        attempts_made=attempts_made,
                        http_status=last_http,
                        content_type=last_content_type,
                    )
                except Exception as exc:
                    last_error = f"slice_error:{type(exc).__name__}"
                    return _block_results(
                        block=block,
                        status="failed",
                        http_status=last_http,
                        content_type=last_content_type,
                        attempts_made=attempts_made,
                        error=last_error,
                    )

            if resp.status_code in RETRYABLE_HTTP_CODES and attempt < request_retries:
                last_error = f"http_{resp.status_code}"
                time.sleep(retry_delay * (2 ** attempt))
                continue

            if resp.status_code >= 400 and resp.status_code < 500 and resp.status_code != 429:
                last_error = f"http_{resp.status_code}"
                return _block_results(
                    block=block,
                    status="missing",
                    http_status=last_http,
                    content_type=last_content_type,
                    attempts_made=attempts_made,
                    error=last_error,
                )

            if resp.status_code == 200 and not last_content_type.startswith("image"):
                body = (resp.text or "").strip().replace("\n", " ")[:160]
                last_error = f"non_image_response:{body}" if body else "non_image_response"
                return _block_results(
                    block=block,
                    status="missing",
                    http_status=last_http,
                    content_type=last_content_type,
                    attempts_made=attempts_made,
                    error=last_error,
                )

            last_error = f"http_{resp.status_code}"
            if attempt < request_retries:
                time.sleep(retry_delay * (2 ** attempt))
                continue
        except requests.RequestException as exc:
            last_error = f"error:{type(exc).__name__}"
            if attempt < request_retries:
                time.sleep(retry_delay * (2 ** attempt))
                continue

    return _block_results(
        block=block,
        status="failed",
        http_status=last_http,
        content_type=last_content_type,
        attempts_made=attempts_made,
        error=last_error,
    )


def run_download_round(
    *,
    rows: List[Dict[str, str]],
    statuses: set[str],
    wms_url: str,
    layer: str,
    crs: str,
    workers: int,
    timeout: int,
    request_retries: int,
    retry_delay: float,
    min_bytes: int,
    max_jobs: int,
    block_rows: int,
    block_cols: int,
) -> Counter:
    jobs = build_download_jobs(rows=rows, statuses=statuses, max_jobs=max_jobs)
    if not jobs:
        return Counter()

    blocks = build_block_jobs(jobs=jobs, block_rows=block_rows, block_cols=block_cols)
    req_reduction = (float(len(jobs)) / float(len(blocks))) if blocks else 1.0
    print(
        f"download_jobs={len(jobs)} block_jobs={len(blocks)} block_size={max(1, block_rows)}x{max(1, block_cols)} "
        f"workers={workers} statuses={sorted(statuses)} req_reduction={req_reduction:.2f}x"
    )

    counts = Counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(
                download_one_block,
                block=block,
                wms_url=wms_url,
                layer=layer,
                crs=crs,
                timeout=timeout,
                request_retries=request_retries,
                retry_delay=retry_delay,
                min_bytes=min_bytes,
            )
            for block in blocks
        ]

        for fut in as_completed(futs):
            results = fut.result()
            for res in results:
                row = rows[res.index]
                row["status"] = res.status
                row["http_status"] = res.http_status
                row["content_type"] = res.content_type
                row["attempts"] = str(normalize_int(row.get("attempts", "0"), 0) + res.attempts_made)
                row["last_error"] = res.error
                row["last_update"] = now_iso()
                counts[res.status] += 1

    return counts


def parse_statuses(raw: str) -> set[str]:
    vals = {s.strip() for s in raw.split(",") if s.strip()}
    return vals if vals else set(TARGET_RETRY_STATUSES)


def classify_cell_status(downloaded: int, expected: int) -> str:
    if downloaded <= 0:
        return "empty"
    if downloaded >= expected:
        return "full"
    return "partial"


def to_feature_collection(features: List[dict]) -> dict:
    return {"type": "FeatureCollection", "features": features}


def write_geojson(path: Path, fc: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")


def validate_manifest(
    *,
    rows: List[Dict[str, str]],
    out_dir: Path,
    src_crs: str,
    dst_crs: str,
    full_chip_count: int,
) -> Path:
    ensure_dir(out_dir)

    grouped = group_rows_by_cell(rows)
    to_wgs84 = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    downloaded_polys_src = []
    unresolved_features_src = []
    cell_features_src = []
    summary_rows = []
    stripe_pattern_counts = Counter()
    stripe_cells = []

    for cell_id, cell_rows in sorted(grouped.items()):
        expected = len(cell_rows)
        downloaded = 0
        missing = 0
        failed = 0
        pending = 0

        unresolved_by_row: Dict[int, List[int]] = defaultdict(list)

        minx = float("inf")
        miny = float("inf")
        maxx = float("-inf")
        maxy = float("-inf")
        max_row = 0
        max_col = 0

        for row in cell_rows:
            r = normalize_int(row.get("row", "0"), 0)
            c = normalize_int(row.get("col", "0"), 0)
            x0 = normalize_float(row.get("xmin", "0"), 0.0)
            y0 = normalize_float(row.get("ymin", "0"), 0.0)
            x1 = normalize_float(row.get("xmax", "0"), 0.0)
            y1 = normalize_float(row.get("ymax", "0"), 0.0)
            status = row.get("status", "pending")

            minx = min(minx, x0)
            miny = min(miny, y0)
            maxx = max(maxx, x1)
            maxy = max(maxy, y1)
            max_row = max(max_row, r)
            max_col = max(max_col, c)

            poly = box(x0, y0, x1, y1)
            if status in SUCCESS_STATUSES:
                downloaded += 1
                downloaded_polys_src.append(poly)
            else:
                unresolved_features_src.append(
                    {
                        "type": "Feature",
                        "geometry": mapping(poly),
                        "properties": {
                            "cell": cell_id,
                            "chip_id": row.get("chip_id", ""),
                            "row": r,
                            "col": c,
                            "status": status,
                        },
                    }
                )

                if status == "missing":
                    missing += 1
                    unresolved_by_row[r].append(c)
                elif status == "failed":
                    failed += 1
                    unresolved_by_row[r].append(c)
                else:
                    pending += 1

        nrows = max_row + 1
        ncols = max_col + 1

        stripe_signatures = []
        for r in sorted(unresolved_by_row):
            cols = sorted(set(unresolved_by_row[r]))
            if not cols:
                continue
            if len(cols) < max(3, int(math.ceil(ncols * 0.4))):
                continue

            prefix = cols == list(range(0, cols[-1] + 1))
            suffix = cols == list(range(cols[0], ncols))
            if prefix:
                sig = f"row_{r:04d}_prefix_to_{cols[-1]:04d}"
                stripe_pattern_counts[sig] += 1
                stripe_signatures.append(sig)
            elif suffix:
                sig = f"row_{r:04d}_suffix_from_{cols[0]:04d}"
                stripe_pattern_counts[sig] += 1
                stripe_signatures.append(sig)

        cell_status = classify_cell_status(downloaded, expected)
        is_striped = len(stripe_signatures) > 0
        if is_striped:
            stripe_cells.append(cell_id)

        completeness = (float(downloaded) / float(expected)) if expected > 0 else 0.0
        missing_total = expected - downloaded

        cell_poly = box(minx, miny, maxx, maxy)
        cell_features_src.append(
            {
                "type": "Feature",
                "geometry": mapping(cell_poly),
                "properties": {
                    "cell": cell_id,
                    "expected_chip_count": expected,
                    "chip_count": downloaded,
                    "missing_chip_count": missing_total,
                    "status": cell_status,
                    "completeness_ratio": completeness,
                    "pending_count": pending,
                    "failed_count": failed,
                    "missing_count": missing,
                    "full_chip_threshold": full_chip_count,
                    "striped": is_striped,
                    "stripe_patterns": ",".join(stripe_signatures),
                    "rows": nrows,
                    "cols": ncols,
                },
            }
        )

        summary_rows.append(
            {
                "cell": cell_id,
                "expected_chip_count": str(expected),
                "downloaded_chip_count": str(downloaded),
                "missing_chip_count": str(missing_total),
                "pending_count": str(pending),
                "failed_count": str(failed),
                "missing_status_count": str(missing),
                "status": cell_status,
                "completeness_ratio": f"{completeness:.6f}",
                "striped": "1" if is_striped else "0",
                "stripe_patterns": ",".join(stripe_signatures),
            }
        )

    summary_csv = out_dir / "cell_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "cell",
            "expected_chip_count",
            "downloaded_chip_count",
            "missing_chip_count",
            "pending_count",
            "failed_count",
            "missing_status_count",
            "status",
            "completeness_ratio",
            "striped",
            "stripe_patterns",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    # Coverage union (downloaded chips only)
    if downloaded_polys_src:
        coverage_union_src = unary_union(downloaded_polys_src)
        coverage_union_31983_fc = to_feature_collection(
            [
                {
                    "type": "Feature",
                    "geometry": mapping(coverage_union_src),
                    "properties": {"name": "local_tile_coverage", "crs": src_crs},
                }
            ]
        )

        coverage_union_wgs = shapely_transform(to_wgs84.transform, coverage_union_src)
        coverage_union_4326_fc = to_feature_collection(
            [
                {
                    "type": "Feature",
                    "geometry": mapping(coverage_union_wgs),
                    "properties": {"name": "local_tile_coverage", "crs": dst_crs},
                }
            ]
        )
    else:
        coverage_union_31983_fc = to_feature_collection([])
        coverage_union_4326_fc = to_feature_collection([])

    unresolved_4326_features = []
    for feat in unresolved_features_src:
        geom_src = shape(feat["geometry"])
        geom_wgs = shapely_transform(to_wgs84.transform, geom_src)
        unresolved_4326_features.append(
            {"type": "Feature", "geometry": mapping(geom_wgs), "properties": feat.get("properties", {})}
        )

    cell_cov_4326_features = []
    for feat in cell_features_src:
        geom_src = shape(feat["geometry"])
        geom_wgs = shapely_transform(to_wgs84.transform, geom_src)
        cell_cov_4326_features.append(
            {"type": "Feature", "geometry": mapping(geom_wgs), "properties": feat.get("properties", {})}
        )

    write_geojson(out_dir / "coverage_union_31983.geojson", coverage_union_31983_fc)
    write_geojson(out_dir / "coverage_union_4326.geojson", coverage_union_4326_fc)
    write_geojson(out_dir / "missing_tiles_31983.geojson", to_feature_collection(unresolved_features_src))
    write_geojson(out_dir / "missing_tiles_4326.geojson", to_feature_collection(unresolved_4326_features))
    write_geojson(out_dir / "cell_coverage_31983.geojson", to_feature_collection(cell_features_src))
    write_geojson(out_dir / "cell_coverage_4326.geojson", to_feature_collection(cell_cov_4326_features))

    status_counts = Counter(r.get("status", "") for r in rows)
    cell_status_counts = Counter(r.get("status", "") for r in summary_rows)

    report = {
        "generated_at": now_iso(),
        "cells_count": len(grouped),
        "chips_total": len(rows),
        "status_counts": dict(status_counts),
        "cell_status_counts": dict(cell_status_counts),
        "striped_cells_count": len(stripe_cells),
        "striped_cells": stripe_cells,
        "stripe_pattern_counts": dict(stripe_pattern_counts.most_common()),
        "outputs": {
            "cell_summary_csv": str(summary_csv.resolve()),
            "coverage_union_31983": str((out_dir / "coverage_union_31983.geojson").resolve()),
            "coverage_union_4326": str((out_dir / "coverage_union_4326.geojson").resolve()),
            "missing_tiles_31983": str((out_dir / "missing_tiles_31983.geojson").resolve()),
            "missing_tiles_4326": str((out_dir / "missing_tiles_4326.geojson").resolve()),
            "cell_coverage_31983": str((out_dir / "cell_coverage_31983.geojson").resolve()),
            "cell_coverage_4326": str((out_dir / "cell_coverage_4326.geojson").resolve()),
        },
    }

    report_path = out_dir / "coverage_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "status_counts": report["status_counts"], "striped_cells_count": len(stripe_cells)}))
    return report_path


def cmd_build_manifest(args: argparse.Namespace) -> int:
    out_root = Path(args.out_root).resolve()
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)

    build_manifest(
        grid_dir=Path(args.grid_dir).resolve(),
        out_root=out_root,
        crs=args.crs,
        chip_size=args.chip_size,
        meters_per_pixel=args.meters_per_pixel,
        reuse_existing_files=bool(args.reuse_existing_files),
        preserve_status=bool(args.preserve_status),
    )
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    out_root = Path(args.out_root).resolve()
    manifest_csv = Path(args.manifest_csv).resolve() if args.manifest_csv else (out_root / "chips_manifest.csv")

    rows = load_manifest(manifest_csv)
    statuses = parse_statuses(args.statuses)

    counts = run_download_round(
        rows=rows,
        statuses=statuses,
        wms_url=args.wms,
        layer=args.layer,
        crs=args.crs,
        workers=args.workers,
        timeout=args.timeout,
        request_retries=args.request_retries,
        retry_delay=args.retry_delay,
        min_bytes=args.min_bytes,
        max_jobs=args.max_jobs,
        block_rows=args.block_rows,
        block_cols=args.block_cols,
    )

    write_manifest_and_cells(out_root, rows)
    unresolved = sum(1 for r in rows if r.get("status", "") not in SUCCESS_STATUSES)
    print(json.dumps({"download_round": dict(counts), "unresolved": unresolved, "manifest": str((out_root / 'chips_manifest.csv'))}))
    return 0


def cmd_retry_until_complete(args: argparse.Namespace) -> int:
    out_root = Path(args.out_root).resolve()
    manifest_csv = Path(args.manifest_csv).resolve() if args.manifest_csv else (out_root / "chips_manifest.csv")

    rows = load_manifest(manifest_csv)
    statuses = parse_statuses(args.statuses)

    unresolved_prev = None
    for round_idx in range(1, args.max_rounds + 1):
        unresolved_before = sum(1 for r in rows if r.get("status", "") not in SUCCESS_STATUSES)
        if unresolved_before == 0:
            print(json.dumps({"round": round_idx, "message": "already_complete"}))
            break

        counts = run_download_round(
            rows=rows,
            statuses=statuses,
            wms_url=args.wms,
            layer=args.layer,
            crs=args.crs,
            workers=args.workers,
            timeout=args.timeout,
            request_retries=args.request_retries,
            retry_delay=args.retry_delay,
            min_bytes=args.min_bytes,
            max_jobs=args.max_jobs,
            block_rows=args.block_rows,
            block_cols=args.block_cols,
        )

        write_manifest_and_cells(out_root, rows)
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


def cmd_validate(args: argparse.Namespace) -> int:
    out_root = Path(args.out_root).resolve()
    manifest_csv = Path(args.manifest_csv).resolve() if args.manifest_csv else (out_root / "chips_manifest.csv")
    rows = load_manifest(manifest_csv)

    coverage_out = Path(args.coverage_out).resolve() if args.coverage_out else (out_root / "_coverage")
    validate_manifest(
        rows=rows,
        out_dir=coverage_out,
        src_crs=args.crs,
        dst_crs=args.dst_crs,
        full_chip_count=args.full_chip_count,
    )
    return 0


def cmd_full_rebuild(args: argparse.Namespace) -> int:
    out_root = Path(args.out_root).resolve()
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)

    manifest_csv = build_manifest(
        grid_dir=Path(args.grid_dir).resolve(),
        out_root=out_root,
        crs=args.crs,
        chip_size=args.chip_size,
        meters_per_pixel=args.meters_per_pixel,
        reuse_existing_files=bool(args.reuse_existing_files),
        preserve_status=bool(args.preserve_status),
    )

    rows = load_manifest(manifest_csv)
    statuses = parse_statuses(args.statuses)

    unresolved_prev = None
    for round_idx in range(1, args.max_rounds + 1):
        unresolved_before = sum(1 for r in rows if r.get("status", "") not in SUCCESS_STATUSES)
        if unresolved_before == 0:
            break

        counts = run_download_round(
            rows=rows,
            statuses=statuses,
            wms_url=args.wms,
            layer=args.layer,
            crs=args.crs,
            workers=args.workers,
            timeout=args.timeout,
            request_retries=args.request_retries,
            retry_delay=args.retry_delay,
            min_bytes=args.min_bytes,
            max_jobs=args.max_jobs,
            block_rows=args.block_rows,
            block_cols=args.block_cols,
        )

        write_manifest_and_cells(out_root, rows)
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
    validate_manifest(
        rows=rows,
        out_dir=coverage_out,
        src_crs=args.crs,
        dst_crs=args.dst_crs,
        full_chip_count=args.full_chip_count,
    )
    return 0


def add_shared_rebuild_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--grid-dir", default="data/external/sp_city_grid_2km_epsg31983")
    p.add_argument("--out-root", default="data/raw/geosampa_ortho/sp_city_2020_rebuild")
    p.add_argument("--crs", default="EPSG:31983")
    p.add_argument("--chip-size", type=int, default=1024)
    p.add_argument("--meters-per-pixel", type=float, default=0.10)
    p.add_argument("--reuse-existing-files", action="store_true", help="Reuse existing PNG+PGW files in out-root as downloaded")
    p.add_argument("--preserve-status", action="store_true", help="If chips.csv exists, preserve prior status/attempt metadata")
    p.add_argument("--clean", action="store_true", help="Delete out-root before rebuilding manifest")


def add_download_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--manifest-csv", default="")
    p.add_argument("--wms", default=WMS_BASE_URL)
    p.add_argument("--layer", default=ORTHO_LAYER)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--block-rows", type=int, default=4, help="Rows per WMS block request (1 disables row grouping)")
    p.add_argument("--block-cols", type=int, default=4, help="Cols per WMS block request (1 disables col grouping)")
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--request-retries", type=int, default=4)
    p.add_argument("--retry-delay", type=float, default=2.0)
    p.add_argument("--min-bytes", type=int, default=4096)
    p.add_argument("--statuses", default="pending,failed,missing")
    p.add_argument("--max-jobs", type=int, default=0, help="Optional debug limit")


def add_validate_args(p: argparse.ArgumentParser, *, include_manifest_csv: bool = True) -> None:
    if include_manifest_csv:
        p.add_argument("--manifest-csv", default="")
    p.add_argument("--coverage-out", default="")
    p.add_argument("--dst-crs", default="EPSG:4326")
    p.add_argument("--full-chip-count", type=int, default=50)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Rebuild GeoSampa 2020 citywide chips with manifest-first completeness guarantees: "
            "build-manifest -> download/retry -> validate."
        )
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build-manifest", help="Create full expected chip manifest (per-cell chips.csv + master manifest)")
    add_shared_rebuild_args(p_build)
    p_build.set_defaults(func=cmd_build_manifest)

    p_dl = sub.add_parser("download", help="Run one download pass for target statuses")
    add_shared_rebuild_args(p_dl)
    add_download_args(p_dl)
    p_dl.set_defaults(func=cmd_download)

    p_retry = sub.add_parser("retry-until-complete", help="Run multiple download passes until complete or no progress")
    add_shared_rebuild_args(p_retry)
    add_download_args(p_retry)
    p_retry.add_argument("--max-rounds", type=int, default=12)
    p_retry.add_argument("--round-sleep", type=float, default=2.0)
    p_retry.set_defaults(func=cmd_retry_until_complete)

    p_val = sub.add_parser("validate", help="Summarize expected vs downloaded and emit coverage GeoJSON outputs")
    add_shared_rebuild_args(p_val)
    add_validate_args(p_val)
    p_val.set_defaults(func=cmd_validate)

    p_full = sub.add_parser("full-rebuild", help="Build manifest, retry downloads, and validate coverage")
    add_shared_rebuild_args(p_full)
    add_download_args(p_full)
    p_full.add_argument("--max-rounds", type=int, default=12)
    p_full.add_argument("--round-sleep", type=float, default=2.0)
    add_validate_args(p_full, include_manifest_csv=False)
    p_full.set_defaults(func=cmd_full_rebuild)

    return ap


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
