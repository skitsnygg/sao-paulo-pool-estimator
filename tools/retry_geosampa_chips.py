#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import threading

import requests

from src.data.geosampa_config import WMS_BASE_URL, ORTHO_LAYER
from src.data.fetch_geosampa_ortho import _write_world_file


@dataclass(frozen=True)
class Job:
    cell_dir: Path
    chip_id: str
    path: Path
    minx: float
    miny: float
    maxx: float
    maxy: float
    width: int
    height: int


def _iter_retry_paths(path: Path) -> Iterable[Path]:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        yield Path(line)


def _parse_jobs(retry_paths: Path) -> tuple[list[Job], dict[Path, list[dict]], dict[Path, list[str]]]:
    by_cell: dict[Path, list[str]] = {}
    for p in _iter_retry_paths(retry_paths):
        cell_dir = p.parent
        by_cell.setdefault(cell_dir, []).append(p.stem)

    jobs: list[Job] = []
    rows_by_cell: dict[Path, list[dict]] = {}

    for cell_dir, chip_ids in sorted(by_cell.items()):
        chips_csv = cell_dir / "chips.csv"
        if not chips_csv.exists():
            continue
        with chips_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        rows_by_cell[cell_dir] = rows
        rows_by_id = {r.get("chip_id", ""): r for r in rows}

        for chip_id in sorted(set(chip_ids)):
            row = rows_by_id.get(chip_id)
            if not row:
                continue
            try:
                job = Job(
                    cell_dir=cell_dir,
                    chip_id=chip_id,
                    path=Path(row["path"]),
                    minx=float(row["xmin"]),
                    miny=float(row["ymin"]),
                    maxx=float(row["xmax"]),
                    maxy=float(row["ymax"]),
                    width=int(float(row.get("width_px", "1024"))),
                    height=int(float(row.get("height_px", "1024"))),
                )
            except Exception:
                continue
            jobs.append(job)

    return jobs, rows_by_cell, by_cell


def _build_params(job: Job, layer: str, crs: str) -> dict[str, str]:
    return {
        "service": "WMS",
        "request": "GetMap",
        "version": "1.3.0",
        "layers": layer,
        "styles": "",
        "crs": crs,
        "bbox": f"{job.minx},{job.miny},{job.maxx},{job.maxy}",
        "width": str(job.width),
        "height": str(job.height),
        "format": "image/png",
        "transparent": "false",
    }


_thread_local = threading.local()


def _get_session() -> requests.Session:
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        sess.headers.update({"User-Agent": "geosampa-retry/1.0"})
        _thread_local.session = sess
    return sess


def _download_job(
    job: Job,
    wms_url: str,
    layer: str,
    crs: str,
    timeout: int,
    retries: int,
    retry_delay: float,
) -> tuple[Job, str, str, str]:
    last_status = ""
    last_content_type = ""
    last_error = ""

    for attempt in range(retries + 1):
        try:
            sess = _get_session()
            resp = sess.get(wms_url, params=_build_params(job, layer, crs), timeout=timeout)
            last_status = str(resp.status_code)
            last_content_type = resp.headers.get("Content-Type", "")
            if resp.status_code == 200 and last_content_type.startswith("image"):
                job.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = job.path.with_suffix(".tmp")
                tmp.write_bytes(resp.content)
                tmp.replace(job.path)
                _write_world_file(job.path, job.minx, job.miny, job.maxx, job.maxy, job.width, job.height)
                return job, "downloaded", last_status, last_content_type
            last_error = f"http_{resp.status_code}"
        except Exception as exc:
            last_error = type(exc).__name__

        if attempt < retries:
            time.sleep(retry_delay * (2 ** attempt))

    return job, "error", last_status, last_content_type or last_error


def main() -> int:
    ap = argparse.ArgumentParser(description="Retry GeoSampa chip downloads from a retry path list.")
    ap.add_argument("--retry-paths", default="data/raw/geosampa_ortho/sp_city_2020/_retry_paths.txt")
    ap.add_argument("--wms", default=WMS_BASE_URL)
    ap.add_argument("--layer", default=ORTHO_LAYER)
    ap.add_argument("--crs", default="EPSG:31983")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--retry-delay", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-retry-paths", default=None)
    args = ap.parse_args()

    retry_paths = Path(args.retry_paths)
    if not retry_paths.exists():
        raise SystemExit(f"Missing retry list: {retry_paths}")

    jobs, rows_by_cell, by_cell = _parse_jobs(retry_paths)
    if not jobs:
        print("No jobs to retry.")
        return 0

    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(
                _download_job,
                job,
                args.wms,
                args.layer,
                args.crs,
                args.timeout,
                args.retries,
                args.retry_delay,
            )
            for job in jobs
        ]
        for fut in as_completed(futs):
            results.append(fut.result())

    # Update chips.csv per cell
    updates: dict[Path, dict[str, tuple[str, str, str]]] = {}
    for job, status, http_status, content_type in results:
        updates.setdefault(job.cell_dir, {})[job.chip_id] = (status, http_status, content_type)

    for cell_dir, rows in rows_by_cell.items():
        if cell_dir not in updates:
            continue
        upd = updates[cell_dir]
        for row in rows:
            chip_id = row.get("chip_id", "")
            if chip_id in upd:
                status, http_status, content_type = upd[chip_id]
                row["status"] = status
                row["http_status"] = http_status
                row["content_type"] = content_type

        chips_csv = cell_dir / "chips.csv"
        fieldnames = list(rows[0].keys()) if rows else []
        with chips_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # Optionally write remaining failures
    if args.out_retry_paths:
        remaining = []
        for job, status, _, _ in results:
            if status != "downloaded":
                remaining.append(str(job.path))
        Path(args.out_retry_paths).write_text("\n".join(sorted(set(remaining))) + "\n", encoding="utf-8")

    total = len(results)
    ok = sum(1 for _, status, _, _ in results if status == "downloaded")
    err = total - ok
    print(json.dumps({"total": total, "downloaded": ok, "error": err}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
