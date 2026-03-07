#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import requests

DEFAULT_WFS = "https://wfs.geosampa.prefeitura.sp.gov.br/geoserver/geoportal/wfs"
DEFAULT_WMS = "https://raster.geosampa.prefeitura.sp.gov.br/geoserver/geoportal/wms"
DEFAULT_LAYER = "geoportal:ORTO_RGB_2020"
DEFAULT_ARTICULATION_TYPENAME = "geoportal:quadricula_orto_2020"

REQUEST_RETRIES = 4
RETRY_DELAY = 1.0
TIMEOUT = 60

# GeoServer reports MaxMemoryExceeded around > 65MB render mem. 4096x4096 RGB is safe.
MAX_DIM_PX = 4096

_thread_local = threading.local()


@dataclass(frozen=True)
class Faixa:
    code: str
    escala: str
    bbox: Tuple[float, float, float, float]


@dataclass(frozen=True)
class DownloadJob:
    idx: int
    faixa_code: str
    escala: str
    chunk_row: int
    chunk_col: int
    bbox: Tuple[float, float, float, float]
    width: int
    height: int
    mpp: float
    out_path: Path


@dataclass(frozen=True)
class DownloadResult:
    idx: int
    status: str
    http_status: str
    bytes_written: int
    attempts_made: int
    error: str


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_codes(path: Path) -> List[str]:
    if not path.exists():
        raise SystemExit(f"Codes file not found: {path}")
    codes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        code = line.strip()
        if not code or code.startswith("#"):
            continue
        codes.append(code)
    if not codes:
        raise SystemExit(f"No codes found in: {path}")
    return sorted(set(codes))


def fetch_articulation_geojson(*, wfs_url: str, typename: str, crs: str, timeout: int) -> dict:
    params = {
        "service": "WFS",
        "version": "1.0.0",
        "request": "GetFeature",
        "typeName": typename,
        "outputFormat": "application/json",
        "srsName": crs,
    }
    r = requests.get(wfs_url, params=params, timeout=timeout)
    r.raise_for_status()
    ctype = (r.headers.get("Content-Type") or "").lower()
    if "json" not in ctype and r.text.lstrip().startswith("<"):
        body = r.text[:220].replace("\n", " ")
        raise SystemExit(f"WFS did not return GeoJSON. content-type={ctype} body={body}")
    payload = r.json()
    if payload.get("type") != "FeatureCollection":
        raise SystemExit("Invalid WFS payload: expected FeatureCollection")
    return payload


def _bounds_from_polygon_coords(coords: Sequence) -> Optional[Tuple[float, float, float, float]]:
    minx = miny = maxx = maxy = None

    def ingest_ring(ring: Sequence) -> None:
        nonlocal minx, miny, maxx, maxy
        for pt in ring:
            if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                continue
            x = float(pt[0])
            y = float(pt[1])
            minx = x if minx is None else min(minx, x)
            miny = y if miny is None else min(miny, y)
            maxx = x if maxx is None else max(maxx, x)
            maxy = y if maxy is None else max(maxy, y)

    for poly in coords:
        if not isinstance(poly, (list, tuple)):
            continue
        for ring in poly:
            if isinstance(ring, (list, tuple)):
                ingest_ring(ring)

    if minx is None:
        return None
    return (minx, miny, maxx, maxy)


def geometry_bbox(geometry: dict) -> Tuple[float, float, float, float]:
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")
    if not coords:
        raise ValueError("missing geometry coordinates")

    if gtype == "Polygon":
        # Normalize to MultiPolygon-like nesting
        bounds = _bounds_from_polygon_coords([coords])
    elif gtype == "MultiPolygon":
        bounds = _bounds_from_polygon_coords(coords)
    else:
        raise ValueError(f"unsupported geometry type: {gtype}")

    if bounds is None:
        raise ValueError("failed to compute geometry bbox")
    return bounds


def collect_faixas(
    *,
    articulation_geojson: dict,
    wanted_codes: Sequence[str],
    code_field: str,
    escala_field: str,
) -> List[Faixa]:
    wanted = set(wanted_codes)
    found: Dict[str, Faixa] = {}

    for feat in articulation_geojson.get("features", []):
        props = feat.get("properties") or {}
        code = str(props.get(code_field, "") or "").strip()
        if not code or code not in wanted:
            continue

        geom = feat.get("geometry") or {}
        try:
            bbox = geometry_bbox(geom)
        except Exception:
            continue

        escala = str(props.get(escala_field, "") or "").strip()
        found[code] = Faixa(code=code, escala=escala, bbox=bbox)

    missing = sorted(wanted - set(found.keys()))
    if missing:
        print(json.dumps({"warning": "codes_not_found_in_articulation", "count": len(missing), "sample": missing[:20]}))

    return [found[k] for k in sorted(found.keys())]


def mpp_for_faixa(escala: str, override_mpp: float) -> float:
    if override_mpp > 0:
        return override_mpp
    e = escala.replace(" ", "")
    if "1:5000" in e:
        return 0.2
    # urban/peripheral true ortho sheets are 0.10m
    return 0.1


def build_jobs(
    *,
    faixas: Sequence[Faixa],
    out_root: Path,
    max_dim_px: int,
    override_mpp: float,
) -> List[DownloadJob]:
    jobs: List[DownloadJob] = []
    idx = 0

    for faixa in faixas:
        minx, miny, maxx, maxy = faixa.bbox
        mpp = mpp_for_faixa(faixa.escala, override_mpp)
        width_px = int(math.ceil((maxx - minx) / mpp))
        height_px = int(math.ceil((maxy - miny) / mpp))

        # split to keep each GetMap request under safe render limits
        chunk_cols = max(1, int(math.ceil(width_px / max_dim_px)))
        chunk_rows = max(1, int(math.ceil(height_px / max_dim_px)))

        faixa_dir = out_root / faixa.code
        ensure_dir(faixa_dir)

        dx = (maxx - minx) / chunk_cols
        dy = (maxy - miny) / chunk_rows

        for r in range(chunk_rows):
            for c in range(chunk_cols):
                x0 = minx + c * dx
                y0 = miny + r * dy
                x1 = minx + (c + 1) * dx
                y1 = miny + (r + 1) * dy
                w = int(math.ceil((x1 - x0) / mpp))
                h = int(math.ceil((y1 - y0) / mpp))
                w = max(1, min(max_dim_px, w))
                h = max(1, min(max_dim_px, h))

                out_path = faixa_dir / f"{faixa.code}__r{r:02d}_c{c:02d}.tif"
                jobs.append(
                    DownloadJob(
                        idx=idx,
                        faixa_code=faixa.code,
                        escala=faixa.escala,
                        chunk_row=r,
                        chunk_col=c,
                        bbox=(x0, y0, x1, y1),
                        width=w,
                        height=h,
                        mpp=mpp,
                        out_path=out_path,
                    )
                )
                idx += 1

    return jobs


def _session() -> requests.Session:
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        sess.headers.update({"User-Agent": "geosampa-faixa-downloader/1.0"})
        _thread_local.session = sess
    return sess


def _is_service_exception_xml(content: bytes, content_type: str) -> bool:
    ctype = (content_type or "").lower()
    if "xml" not in ctype and "text" not in ctype:
        return False
    s = content[:300].decode("utf-8", errors="ignore").lower()
    return "serviceexception" in s or "exceptionreport" in s


def download_one(
    *,
    job: DownloadJob,
    wms_url: str,
    layer: str,
    crs: str,
    image_format: str,
    timeout: int,
    retries: int,
    retry_delay: float,
    overwrite: bool,
) -> DownloadResult:
    if job.out_path.exists() and not overwrite:
        return DownloadResult(
            idx=job.idx,
            status="cached",
            http_status="cached",
            bytes_written=job.out_path.stat().st_size,
            attempts_made=0,
            error="",
        )

    params = {
        "service": "WMS",
        "version": "1.3.0",
        "request": "GetMap",
        "layers": layer,
        "styles": "",
        "crs": crs,
        "bbox": f"{job.bbox[0]:.3f},{job.bbox[1]:.3f},{job.bbox[2]:.3f},{job.bbox[3]:.3f}",
        "width": str(job.width),
        "height": str(job.height),
        "format": image_format,
        "transparent": "false",
    }

    sess = _session()
    attempts = 0
    last_err = ""

    for attempt in range(retries + 1):
        attempts += 1
        try:
            r = sess.get(wms_url, params=params, timeout=timeout)
            http_status = str(r.status_code)
            ctype = (r.headers.get("Content-Type") or "").lower()

            if r.status_code != 200:
                last_err = f"http_{r.status_code}"
                if attempt < retries and r.status_code in {429, 500, 502, 503, 504}:
                    time.sleep(retry_delay * (2**attempt))
                    continue
                return DownloadResult(job.idx, "failed", http_status, 0, attempts, last_err)

            if _is_service_exception_xml(r.content, ctype):
                txt = r.content.decode("utf-8", errors="ignore")
                if "maxmemoryexceeded" in txt.lower():
                    last_err = "wms_MaxMemoryExceeded"
                else:
                    last_err = "wms_ServiceException"
                if attempt < retries:
                    time.sleep(retry_delay * (2**attempt))
                    continue
                return DownloadResult(job.idx, "failed", http_status, 0, attempts, last_err)

            if "tiff" not in ctype and "geotiff" not in ctype and image_format == "image/geotiff":
                last_err = f"unexpected_content_type:{ctype}"
                if attempt < retries:
                    time.sleep(retry_delay * (2**attempt))
                    continue
                return DownloadResult(job.idx, "failed", http_status, 0, attempts, last_err)

            ensure_dir(job.out_path.parent)
            tmp = job.out_path.with_suffix(job.out_path.suffix + ".tmp")
            tmp.write_bytes(r.content)
            tmp.replace(job.out_path)
            return DownloadResult(job.idx, "downloaded", http_status, len(r.content), attempts, "")

        except requests.RequestException as exc:
            last_err = f"request:{type(exc).__name__}"
            if attempt < retries:
                time.sleep(retry_delay * (2**attempt))
                continue
            return DownloadResult(job.idx, "failed", "", 0, attempts, f"{last_err}:{exc}")

    return DownloadResult(job.idx, "failed", "", 0, attempts, last_err)


def write_manifest(path: Path, rows: List[Dict[str, str]]) -> None:
    ensure_dir(path.parent)
    fields = [
        "faixa_code",
        "escala",
        "chunk_row",
        "chunk_col",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "width_px",
        "height_px",
        "meters_per_pixel",
        "path",
        "status",
        "http_status",
        "bytes",
        "attempts",
        "error",
        "last_update",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def run(args: argparse.Namespace) -> int:
    out_root = Path(args.out_root).resolve()
    ensure_dir(out_root)

    codes = read_codes(Path(args.codes_file).resolve())
    print(json.dumps({"codes_requested": len(codes)}))

    art_path = Path(args.articulation_geojson).resolve() if args.articulation_geojson else None
    if art_path and art_path.exists():
        articulation = json.loads(art_path.read_text(encoding="utf-8"))
    else:
        articulation = fetch_articulation_geojson(
            wfs_url=args.wfs_url,
            typename=args.articulation_typename,
            crs=args.crs,
            timeout=args.timeout,
        )
        if art_path:
            ensure_dir(art_path.parent)
            art_path.write_text(json.dumps(articulation, ensure_ascii=False), encoding="utf-8")

    faixas = collect_faixas(
        articulation_geojson=articulation,
        wanted_codes=codes,
        code_field=args.code_field,
        escala_field=args.escala_field,
    )
    print(json.dumps({"faixas_found": len(faixas)}))

    jobs = build_jobs(
        faixas=faixas,
        out_root=out_root,
        max_dim_px=args.max_dim_px,
        override_mpp=args.meters_per_pixel,
    )

    rows: List[Dict[str, str]] = []
    for job in jobs:
        rows.append(
            {
                "faixa_code": job.faixa_code,
                "escala": job.escala,
                "chunk_row": str(job.chunk_row),
                "chunk_col": str(job.chunk_col),
                "xmin": f"{job.bbox[0]:.3f}",
                "ymin": f"{job.bbox[1]:.3f}",
                "xmax": f"{job.bbox[2]:.3f}",
                "ymax": f"{job.bbox[3]:.3f}",
                "width_px": str(job.width),
                "height_px": str(job.height),
                "meters_per_pixel": f"{job.mpp:.3f}",
                "path": str(job.out_path),
                "status": "pending",
                "http_status": "",
                "bytes": "0",
                "attempts": "0",
                "error": "",
                "last_update": "",
            }
        )

    manifest = out_root / "manifest.csv"
    write_manifest(manifest, rows)

    if args.dry_run:
        print(json.dumps({"dry_run": True, "jobs": len(jobs), "manifest": str(manifest)}))
        return 0

    print(json.dumps({"jobs": len(jobs), "workers": args.workers, "wms": args.wms_url, "layer": args.layer}))

    counts = Counter()
    total_jobs = len(jobs)
    completed = 0
    progress_step = max(100, total_jobs // 100) if total_jobs > 0 else 100
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(
                download_one,
                job=job,
                wms_url=args.wms_url,
                layer=args.layer,
                crs=args.crs,
                image_format=args.image_format,
                timeout=args.timeout,
                retries=args.request_retries,
                retry_delay=args.retry_delay,
                overwrite=bool(args.overwrite),
            )
            for job in jobs
        ]

        for fut in as_completed(futs):
            res = fut.result()
            row = rows[res.idx]
            row["status"] = res.status
            row["http_status"] = res.http_status
            row["bytes"] = str(res.bytes_written)
            row["attempts"] = str(int(row["attempts"]) + res.attempts_made)
            row["error"] = res.error
            row["last_update"] = now_iso()
            counts[res.status] += 1
            completed += 1
            if completed % progress_step == 0 or completed == total_jobs:
                elapsed = max(1e-6, time.time() - t0)
                rate = completed / elapsed
                print(
                    json.dumps(
                        {
                            "progress": {
                                "completed": completed,
                                "total": total_jobs,
                                "pct": round(100.0 * completed / max(1, total_jobs), 2),
                                "rate_jobs_per_s": round(rate, 2),
                                "status_counts": dict(counts),
                            }
                        }
                    )
                )

    write_manifest(manifest, rows)

    unresolved = sum(1 for r in rows if r["status"] not in {"downloaded", "cached"})
    by_faixa = Counter(r["faixa_code"] for r in rows if r["status"] in {"downloaded", "cached"})

    summary = {
        "codes_requested": len(codes),
        "faixas_found": len(faixas),
        "jobs": len(rows),
        "status_counts": dict(counts),
        "unresolved_jobs": unresolved,
        "downloaded_faixas": len(by_faixa),
        "manifest": str(manifest),
        "out_root": str(out_root),
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Download missing GeoSampa 2020 faixa chunks via WMS (no UI/captcha), "
            "using articulation index from WFS."
        )
    )
    p.add_argument("--codes-file", required=True, help="Text file with missing faixa codes (one per line)")
    p.add_argument("--out-root", default="data/raw/geosampa_ortho/faixas_missing_2020", help="Output directory")

    p.add_argument("--articulation-geojson", default="", help="Optional local articulation GeoJSON cache path")
    p.add_argument("--wfs-url", default=DEFAULT_WFS)
    p.add_argument("--articulation-typename", default=DEFAULT_ARTICULATION_TYPENAME)
    p.add_argument("--code-field", default="cd_quadricula")
    p.add_argument("--escala-field", default="cd_escala_quadricula")

    p.add_argument("--wms-url", default=DEFAULT_WMS)
    p.add_argument("--layer", default=DEFAULT_LAYER)
    p.add_argument("--crs", default="EPSG:31983")
    p.add_argument("--image-format", default="image/geotiff")

    p.add_argument("--meters-per-pixel", type=float, default=-1.0, help="Override m/px. <=0 means auto (0.1 urban, 0.2 rural)")
    p.add_argument("--max-dim-px", type=int, default=MAX_DIM_PX, help="Max WMS width/height per request")

    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--timeout", type=int, default=TIMEOUT)
    p.add_argument("--request-retries", type=int, default=REQUEST_RETRIES)
    p.add_argument("--retry-delay", type=float, default=RETRY_DELAY)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
