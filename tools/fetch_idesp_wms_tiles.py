#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse
from xml.etree import ElementTree as ET

import requests
from PIL import Image
from pyproj import Transformer

WMS_URL_DEFAULT = "https://www.idesp.sp.gov.br/geoimage/idesp_raster/wms"
WMS_URL_FALLBACK_DEFAULT = "https://datageo.ambiente.sp.gov.br/geoimage/datageoimg/ows?SERVICE=WMS"
WMS_LAYER_DEFAULT = "ORTOFOTOS_SP_2023_2024"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")


@dataclass(frozen=True)
class Worldfile:
    # Worldfile 6 lines: A, D, B, E, C, F
    A: float
    D: float
    B: float
    E: float
    C: float
    F: float

    def bbox(self, width: int, height: int) -> Tuple[float, float, float, float]:
        if abs(self.B) > 1e-12 or abs(self.D) > 1e-12:
            raise ValueError(f"Worldfile has rotation terms (B={self.B}, D={self.D}); not supported")

        minx = self.C - (self.A / 2.0)
        maxy = self.F - (self.E / 2.0)
        maxx = minx + (width * self.A)
        miny = maxy + (height * self.E)

        if minx > maxx:
            minx, maxx = maxx, minx
        if miny > maxy:
            miny, maxy = maxy, miny
        return (minx, miny, maxx, maxy)


@dataclass(frozen=True)
class Endpoint:
    name: str
    wms_url: str
    layer: str
    layer_bbox: Tuple[float, float, float, float]


@dataclass
class CellPlan:
    cell: str
    src_dir: Path
    expected_total: int
    existing_total: int
    missing_candidates: List[Path]


@dataclass
class FetchResult:
    status: str  # success | blank | failed
    endpoint_name: str
    content: Optional[bytes] = None
    blank_kind: str = ""
    blank_payload: Optional[bytes] = None
    error_kind: str = ""  # http | wms_exception | other
    last_err: str = ""
    error_payload: Optional[bytes] = None
    request_url: str = ""
    request_bbox: str = ""
    request_srs: str = ""
    request_version: str = ""
    classification: str = ""
    wms_exception_text: str = ""


def log_json(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=True))


def normalize_wms_url(url: str) -> str:
    u = urlparse(url)
    host = (u.hostname or "").lower()
    if host in {"idesp.sp.gov.br", "www.idesp.sp.gov.br"}:
        u = u._replace(scheme="https", netloc="www.idesp.sp.gov.br")
    # We always pass SERVICE/REQUEST via params; strip inline query to avoid duplicates.
    u = u._replace(query="", fragment="")
    return urlunparse(u)


@contextmanager
def resolve_ip_context(host_to_ip: Dict[str, str]):
    if not host_to_ip:
        yield
        return
    try:
        from urllib3.util import connection  # type: ignore
    except Exception:
        yield
        return

    orig_create = connection.create_connection

    def patched(address, *args, **kwargs):
        host, port = address
        mapped = host_to_ip.get(host)
        if mapped:
            host = mapped
        return orig_create((host, port), *args, **kwargs)

    connection.create_connection = patched
    try:
        yield
    finally:
        connection.create_connection = orig_create


def output_image_suffix_for_format(fmt: str) -> str:
    return ".jpg" if fmt == "image/jpeg" else ".png"


def worldfile_suffix_for_image_suffix(image_suffix: str) -> str:
    sfx = image_suffix.lower()
    if sfx == ".png":
        return ".pgw"
    if sfx in (".jpg", ".jpeg"):
        return ".jgw"
    return ".wld"


def find_worldfile(img_path: Path) -> Optional[Path]:
    sfx = img_path.suffix.lower()
    base = img_path.with_suffix("")
    candidates: List[Path] = []
    if sfx == ".png":
        candidates += [base.with_suffix(".pgw"), base.with_suffix(".wld"), img_path.with_suffix(".pngw")]
    elif sfx in (".jpg", ".jpeg"):
        candidates += [base.with_suffix(".jgw"), base.with_suffix(".wld"), img_path.with_suffix(".jpgw")]
    else:
        candidates += [base.with_suffix(".wld")]
    for c in candidates:
        if c.exists():
            return c
    return None


def read_worldfile(p: Path) -> Worldfile:
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) != 6:
        raise ValueError(f"Worldfile {p} must have 6 non-empty lines; got {len(lines)}")
    vals = list(map(float, lines))
    return Worldfile(*vals)  # type: ignore[arg-type]


def transform_bbox(
    bbox: Tuple[float, float, float, float],
    src_crs: str,
    dst_crs: str,
) -> Tuple[float, float, float, float]:
    if src_crs.strip().upper() == dst_crs.strip().upper():
        return bbox
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    minx, miny, maxx, maxy = bbox
    corners = [
        transformer.transform(minx, miny),
        transformer.transform(minx, maxy),
        transformer.transform(maxx, miny),
        transformer.transform(maxx, maxy),
    ]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return (min(xs), min(ys), max(xs), max(ys))


def bbox_to_str_xy(minx: float, miny: float, maxx: float, maxy: float) -> str:
    return f"{minx:.8f},{miny:.8f},{maxx:.8f},{maxy:.8f}"


def build_wms_getmap_request(
    wms_url: str,
    layer: str,
    bbox_xy: Tuple[float, float, float, float],
    width: int,
    height: int,
    srs: str,
    fmt: str,
    version: str = "1.1.1",
) -> Tuple[str, Dict[str, str], str]:
    minx, miny, maxx, maxy = bbox_xy
    bbox_str = bbox_to_str_xy(minx, miny, maxx, maxy)
    params: Dict[str, str] = {
        "SERVICE": "WMS",
        "REQUEST": "GetMap",
        "VERSION": version,
        "LAYERS": layer,
        "STYLES": "",
        "FORMAT": fmt,
        "TRANSPARENT": "FALSE",
        "WIDTH": str(width),
        "HEIGHT": str(height),
        "BBOX": bbox_str,
    }
    if version == "1.3.0":
        params["CRS"] = srs
    else:
        params["SRS"] = srs
    return (wms_url, params, bbox_str)


def build_request_url(url: str, params: Dict[str, str]) -> str:
    try:
        req = requests.Request("GET", url, params=params)
        prep = req.prepare()
        return prep.url or url
    except Exception:
        return url


def is_wms_exception(content: bytes, content_type: str) -> bool:
    ct = (content_type or "").lower()
    if "xml" in ct or "text" in ct:
        return True
    head = content[:800].lower()
    return (
        b"<serviceexception" in head
        or b"<serviceexceptionreport" in head
        or b"<exceptionreport" in head
        or b"internalerror" in head
    )


def decode_bytes_best_effort(b: bytes) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            return b.decode(enc, errors="replace")
        except Exception:
            continue
    return b.decode("utf-8", errors="replace")


def wms_exception_is_nodata(xml_bytes: bytes, patterns: List[re.Pattern]) -> bool:
    txt = decode_bytes_best_effort(xml_bytes).lower()
    return any(p.search(txt) is not None for p in patterns)


def summarize_wms_exception(xml_bytes: bytes, max_len: int = 800) -> str:
    txt = decode_bytes_best_effort(xml_bytes)
    cleaned = " ".join(txt.split())

    try:
        root = ET.fromstring(txt)
        messages: List[str] = []
        for el in root.iter():
            tag = el.tag.split("}")[-1]
            if tag in {"ServiceException", "ExceptionText"}:
                msg = " ".join((el.text or "").split())
                if msg:
                    messages.append(msg)
        if messages:
            cleaned = " | ".join(messages)
    except Exception:
        pass

    if len(cleaned) > max_len:
        return cleaned[:max_len] + "..."
    return cleaned


def get_rgb_extrema(content: bytes) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
    try:
        with Image.open(BytesIO(content)) as im:
            im = im.convert("RGB")
            extrema = im.getextrema()
            if not isinstance(extrema, (list, tuple)) or len(extrema) != 3:
                return None
            return (extrema[0], extrema[1], extrema[2])
    except Exception:
        return None


def is_uniform_white_rgb(extrema: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]) -> bool:
    (rmin, rmax), (gmin, gmax), (bmin, bmax) = extrema
    return rmin == rmax == 255 and gmin == gmax == 255 and bmin == bmax == 255


def bboxes_intersect(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    minx1, miny1, maxx1, maxy1 = a
    minx2, miny2, maxx2, maxy2 = b
    if maxx1 < minx2 or maxx2 < minx1:
        return False
    if maxy1 < miny2 or maxy2 < miny1:
        return False
    return True


def _tag_name(el: ET.Element) -> str:
    return el.tag.split("}")[-1]


def _text_child(el: ET.Element, child_name: str) -> Optional[str]:
    for c in list(el):
        if _tag_name(c) == child_name:
            return (c.text or "").strip()
    return None


def _parse_bbox_attrs(el: ET.Element) -> Optional[Tuple[float, float, float, float]]:
    try:
        minx = float(el.attrib.get("minx"))
        miny = float(el.attrib.get("miny"))
        maxx = float(el.attrib.get("maxx"))
        maxy = float(el.attrib.get("maxy"))
        return (minx, miny, maxx, maxy)
    except Exception:
        return None


def _parse_ex_geographic_bbox(el: ET.Element) -> Optional[Tuple[float, float, float, float]]:
    vals: Dict[str, float] = {}
    for c in list(el):
        name = _tag_name(c)
        if c.text is None:
            continue
        try:
            vals[name] = float(c.text.strip())
        except Exception:
            pass
    keys = {"westBoundLongitude", "eastBoundLongitude", "southBoundLatitude", "northBoundLatitude"}
    if not keys <= set(vals.keys()):
        return None
    return (vals["westBoundLongitude"], vals["southBoundLatitude"], vals["eastBoundLongitude"], vals["northBoundLatitude"])


def _collect_layer_tree(root: ET.Element) -> Optional[ET.Element]:
    for el in root.iter():
        if _tag_name(el) == "Capability":
            for c in list(el):
                if _tag_name(c) == "Layer":
                    return c
    for el in root.iter():
        if _tag_name(el) == "Layer":
            return el
    return None


def _layer_direct_children(layer_el: ET.Element, name: str) -> List[ET.Element]:
    return [c for c in list(layer_el) if _tag_name(c) == name]


def _find_layer_with_inherited_bboxes(
    top_layer: ET.Element,
    target_name: str,
) -> Optional[Dict[str, Any]]:
    def extract_layer_info(layer_el: ET.Element) -> Dict[str, Any]:
        out: Dict[str, Any] = {"bboxes": [], "geo_bbox": None}
        for bb_el in _layer_direct_children(layer_el, "BoundingBox"):
            crs = (bb_el.attrib.get("SRS") or bb_el.attrib.get("CRS") or "").strip().upper()
            bbox = _parse_bbox_attrs(bb_el)
            if crs and bbox is not None:
                out["bboxes"].append((crs, bbox))

        for ll_el in _layer_direct_children(layer_el, "LatLonBoundingBox"):
            bbox = _parse_bbox_attrs(ll_el)
            if bbox is not None:
                out["geo_bbox"] = bbox
                break

        if out["geo_bbox"] is None:
            for ex_el in _layer_direct_children(layer_el, "EX_GeographicBoundingBox"):
                bbox = _parse_ex_geographic_bbox(ex_el)
                if bbox is not None:
                    out["geo_bbox"] = bbox
                    break
        return out

    stack: List[Tuple[ET.Element, Dict[str, Tuple[float, float, float, float]], Optional[Tuple[float, float, float, float]]]] = []
    top_info = extract_layer_info(top_layer)
    stack.append((top_layer, {crs: bb for crs, bb in top_info["bboxes"]}, top_info["geo_bbox"]))

    while stack:
        layer_el, inherited_map, inherited_geo = stack.pop()
        info = extract_layer_info(layer_el)
        merged = dict(inherited_map)
        for crs, bb in info["bboxes"]:
            merged[crs] = bb
        merged_geo = info["geo_bbox"] if info["geo_bbox"] is not None else inherited_geo

        name = (_text_child(layer_el, "Name") or "").strip()
        if name == target_name:
            return {"bboxes": [(crs, merged[crs]) for crs in sorted(merged.keys())], "geo_bbox": merged_geo}

        for child in reversed(_layer_direct_children(layer_el, "Layer")):
            stack.append((child, merged, merged_geo))
    return None


def _fetch_getcapabilities(session: requests.Session, wms_url: str, timeout: float, version: str) -> bytes:
    params = {"SERVICE": "WMS", "REQUEST": "GetCapabilities", "VERSION": version}
    r = session.get(wms_url, params=params, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"GetCapabilities {version} failed: HTTP {r.status_code}")
    return r.content


def get_layer_bbox_from_capabilities(
    session: requests.Session,
    wms_url: str,
    layer: str,
    request_srs: str,
    timeout: float,
    prefer_versions: Tuple[str, ...],
) -> Tuple[float, float, float, float]:
    last_err: Optional[str] = None
    for ver in prefer_versions:
        try:
            xml = _fetch_getcapabilities(session, wms_url, timeout, ver)
            root = ET.fromstring(xml)
            top = _collect_layer_tree(root)
            if top is None:
                raise RuntimeError("Could not locate Capability/Layer in GetCapabilities XML")
            found = _find_layer_with_inherited_bboxes(top, layer)
            if found is None:
                raise RuntimeError(f"Layer not found: {layer}")

            req = request_srs.strip().upper()
            for crs, bb in found["bboxes"]:
                if crs.strip().upper() == req:
                    return bb

            geo = found["geo_bbox"]
            if geo is None:
                raise RuntimeError(f"No usable bbox found for layer {layer}")
            if req in {"EPSG:4326", "EPSG:4674"}:
                return geo
            return transform_bbox(geo, "EPSG:4326", request_srs)
        except Exception as exc:
            last_err = f"{ver}: {exc}"
    raise RuntimeError(f"Failed to parse GetCapabilities for layer bbox. Last error: {last_err}")


def _sleep_backoff(base: float, attempt: int, jitter: float = 0.15, max_sleep: float = 10.0) -> None:
    t = base * (2 ** (attempt - 1))
    t = min(t, max_sleep)
    if jitter > 0:
        t *= (1.0 + random.uniform(-jitter, jitter))
    time.sleep(max(0.0, t))


def parse_cells_arg(raw: str) -> set[str]:
    out: set[str] = set()
    txt = str(raw or "").strip()
    if not txt:
        return out
    for token in re.split(r"[\s,]+", txt):
        t = token.strip()
        if t:
            out.add(t)
    return out


def parse_cells_file(path: Path) -> set[str]:
    if not path.exists():
        raise SystemExit(f"cells file not found: {path}")
    out: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.add(line)
    return out


def parse_blank_patterns(user_patterns: Iterable[str]) -> List[re.Pattern]:
    defaults = [
        r"specified\s+sourceurl\s+doesn.?t\s+refer\s+to\s+an\s+existing\s+file",
        r"rendering\s+process\s+failed",
    ]
    pats = defaults + [p for p in user_patterns if str(p).strip()]
    return [re.compile(p, re.IGNORECASE) for p in pats]


def list_cell_dirs(src_root: Path, selected_cells: set[str]) -> List[Path]:
    out: List[Path] = []
    for d in sorted(src_root.glob("cell_*_*")):
        if not d.is_dir():
            continue
        if selected_cells and d.name not in selected_cells:
            continue
        out.append(d)
    return out


def list_cell_images(cell_dir: Path) -> List[Path]:
    imgs = [p for p in cell_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    imgs.sort()
    return imgs


def output_paths_for_src(
    src_root: Path,
    dst_root: Path,
    src_img: Path,
    out_img_suffix: str,
) -> Tuple[Path, Path]:
    rel = src_img.relative_to(src_root)
    out_img = dst_root / rel.with_suffix(out_img_suffix)
    out_wf = out_img.with_suffix(worldfile_suffix_for_image_suffix(out_img.suffix))
    return out_img, out_wf


def repair_missing_worldfile_if_possible(src_img: Path, out_wf: Path) -> Tuple[bool, bool]:
    if out_wf.exists():
        return (False, False)
    wf = find_worldfile(src_img)
    if wf is None:
        return (False, True)
    try:
        out_wf.parent.mkdir(parents=True, exist_ok=True)
        out_wf.write_text(wf.read_text(encoding="utf-8"), encoding="utf-8")
        return (True, False)
    except Exception:
        return (False, False)


def run_audit(
    src_root: Path,
    dst_root: Path,
    out_img_suffix: str,
    selected_cells: set[str],
    audit_csv: Optional[Path],
    audit_top: int,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    cells = list_cell_dirs(src_root, selected_cells)
    total_expected = 0
    total_downloaded = 0

    for cell_dir in cells:
        expected = len(list_cell_images(cell_dir))
        got = 0
        dst_cell = dst_root / cell_dir.name
        if dst_cell.exists():
            got = sum(1 for p in dst_cell.iterdir() if p.is_file() and p.suffix.lower() == out_img_suffix)
        missing = max(0, expected - got)
        pct = 0.0 if expected == 0 else (100.0 * got / expected)
        rows.append(
            {
                "cell": cell_dir.name,
                "expected": expected,
                "downloaded": got,
                "missing": missing,
                "pct_complete": round(pct, 4),
            }
        )
        total_expected += expected
        total_downloaded += got

    missing_total = max(0, total_expected - total_downloaded)
    incomplete = [r for r in rows if int(r["missing"]) > 0]
    complete = [r for r in rows if int(r["missing"]) == 0]

    if audit_csv:
        audit_csv.parent.mkdir(parents=True, exist_ok=True)
        with audit_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["cell", "expected", "downloaded", "missing", "pct_complete"])
            w.writeheader()
            w.writerows(rows)

    summary = {
        "cells_scanned": len(rows),
        "cells_complete": len(complete),
        "cells_incomplete": len(incomplete),
        "expected_tiles_total": total_expected,
        "downloaded_tiles_total": total_downloaded,
        "missing_tiles_total": missing_total,
        "audit_csv": str(audit_csv.resolve()) if audit_csv else "",
    }
    log_json({"stage": "audit_summary", **summary})

    if incomplete and audit_top > 0:
        worst = sorted(incomplete, key=lambda r: (-int(r["missing"]), r["cell"]))[: int(audit_top)]
        log_json({"stage": "audit_worst_cells", "count": len(worst), "rows": worst})
    return summary


def build_cell_plans(
    src_root: Path,
    dst_root: Path,
    out_img_suffix: str,
    selected_cells: set[str],
    selected_tile_stems: set[str],
    missing_only: bool,
    overwrite: bool,
    repair_worldfiles: bool,
) -> Tuple[List[CellPlan], Dict[str, int]]:
    cells = list_cell_dirs(src_root, selected_cells)
    plans: List[CellPlan] = []
    stats = {
        "cells_scanned": 0,
        "tiles_seen": 0,
        "tiles_existing": 0,
        "missing_tiles_found": 0,
        "repaired_worldfile": 0,
        "missing_worldfile_while_repairing": 0,
    }

    for cell_dir in cells:
        imgs = list_cell_images(cell_dir)
        if selected_tile_stems:
            imgs = [p for p in imgs if p.stem in selected_tile_stems]
        stats["cells_scanned"] += 1
        stats["tiles_seen"] += len(imgs)
        missing_candidates: List[Path] = []
        existing = 0

        for src_img in imgs:
            out_img, out_wf = output_paths_for_src(src_root, dst_root, src_img, out_img_suffix)
            exists_and_kept = out_img.exists() and (missing_only or not overwrite)
            if exists_and_kept:
                existing += 1
                stats["tiles_existing"] += 1
                if repair_worldfiles:
                    repaired, missing_wf = repair_missing_worldfile_if_possible(src_img, out_wf)
                    if repaired:
                        stats["repaired_worldfile"] += 1
                    if missing_wf:
                        stats["missing_worldfile_while_repairing"] += 1
                continue
            missing_candidates.append(src_img)

        stats["missing_tiles_found"] += len(missing_candidates)
        plans.append(
            CellPlan(
                cell=cell_dir.name,
                src_dir=cell_dir,
                expected_total=len(imgs),
                existing_total=existing,
                missing_candidates=missing_candidates,
            )
        )
    return plans, stats


def build_endpoints(
    session: requests.Session,
    *,
    primary_url: str,
    primary_layer: str,
    fallback_url: str,
    fallback_layer: str,
    disable_fallback: bool,
    request_srs: str,
    timeout: float,
    capabilities_version: str,
) -> List[Endpoint]:
    if capabilities_version == "auto":
        prefer = ("1.1.1", "1.3.0")
    else:
        prefer = (capabilities_version,)

    endpoints_cfg: List[Tuple[str, str, str]] = [("primary", normalize_wms_url(primary_url), primary_layer)]
    fb_norm = normalize_wms_url(fallback_url) if fallback_url.strip() else ""
    fb_layer = fallback_layer.strip() or primary_layer
    if (not disable_fallback) and fb_norm and (fb_norm != endpoints_cfg[0][1] or fb_layer != primary_layer):
        endpoints_cfg.append(("fallback", fb_norm, fb_layer))

    endpoints: List[Endpoint] = []
    for name, wms_url, layer in endpoints_cfg:
        try:
            bbox = get_layer_bbox_from_capabilities(
                session=session,
                wms_url=wms_url,
                layer=layer,
                request_srs=request_srs,
                timeout=timeout,
                prefer_versions=prefer,
            )
            endpoints.append(Endpoint(name=name, wms_url=wms_url, layer=layer, layer_bbox=bbox))
        except Exception as exc:
            if name == "primary":
                raise SystemExit(f"Failed to read primary endpoint capabilities: {exc}")
            print(f"WARN: fallback endpoint disabled due to capabilities failure: {exc}", file=sys.stderr)

    if not endpoints:
        raise SystemExit("No usable WMS endpoint available")
    return endpoints


def fetch_tile_with_fallback(
    session: requests.Session,
    endpoints: List[Endpoint],
    *,
    bbox_dst: Tuple[float, float, float, float],
    bbox_31983: Tuple[float, float, float, float],
    bbox_4674: Tuple[float, float, float, float],
    width: int,
    height: int,
    request_srs: str,
    image_format: str,
    getmap_version: str,
    timeout: float,
    retries: int,
    retry_backoff_base: float,
    treat_wms_internalerror_blank: bool,
    treat_white_blank: bool,
    blank_patterns: List[re.Pattern],
    debug: bool = False,
    tile_debug_meta: Optional[Dict[str, Any]] = None,
) -> FetchResult:
    last_err = ""
    last_kind = "other"
    last_payload: Optional[bytes] = None
    last_endpoint = endpoints[0].name
    last_url = ""
    last_bbox = ""
    last_wms_text = ""

    def _log_debug(stage: str, extra: Dict[str, Any]) -> None:
        if not debug:
            return
        payload: Dict[str, Any] = {"stage": stage}
        if tile_debug_meta:
            payload.update(tile_debug_meta)
        payload.update(extra)
        log_json(payload)

    for attempt in range(1, retries + 1):
        for ep in endpoints:
            last_endpoint = ep.name
            url, params, bbox_str = build_wms_getmap_request(
                wms_url=ep.wms_url,
                layer=ep.layer,
                bbox_xy=bbox_dst,
                width=width,
                height=height,
                srs=request_srs,
                fmt=image_format,
                version=getmap_version,
            )
            full_url = build_request_url(url, params)
            last_url = full_url
            last_bbox = bbox_str

            _log_debug(
                "tile_getmap_request",
                {
                    "attempt": attempt,
                    "endpoint": ep.name,
                    "wms_url": ep.wms_url,
                    "layer": ep.layer,
                    "version": getmap_version,
                    "srs": request_srs,
                    "bbox_31983": list(bbox_31983),
                    "bbox_4674": list(bbox_4674),
                    "bbox_request": bbox_str,
                    "url": full_url,
                    "width": width,
                    "height": height,
                    "format": image_format,
                },
            )
            try:
                r = session.get(url, params=params, timeout=timeout)
            except Exception as exc:
                last_err = f"{ep.name}: {type(exc).__name__}: {exc}"
                last_kind = "other"
                last_payload = None
                _log_debug(
                    "tile_getmap_error",
                    {
                        "attempt": attempt,
                        "endpoint": ep.name,
                        "classification": "request_exception",
                        "error_kind": "other",
                        "error": last_err,
                        "url": full_url,
                    },
                )
                continue

            ct = (r.headers.get("Content-Type") or "").lower()
            if r.status_code != 200:
                last_err = f"{ep.name}: HTTP {r.status_code} content-type={ct}"
                last_kind = "http"
                last_payload = None
                _log_debug(
                    "tile_getmap_error",
                    {
                        "attempt": attempt,
                        "endpoint": ep.name,
                        "classification": "http_error",
                        "error_kind": "http",
                        "status_code": r.status_code,
                        "content_type": ct,
                        "url": full_url,
                    },
                )
                continue

            if is_wms_exception(r.content, ct):
                ex_text = summarize_wms_exception(r.content)
                last_wms_text = ex_text
                if treat_wms_internalerror_blank and wms_exception_is_nodata(r.content, blank_patterns):
                    _log_debug(
                        "tile_getmap_result",
                        {
                            "attempt": attempt,
                            "endpoint": ep.name,
                            "classification": "blank_wms_exception_nodata",
                            "content_type": ct,
                            "url": full_url,
                            "wms_exception": ex_text,
                        },
                    )
                    return FetchResult(
                        status="blank",
                        endpoint_name=ep.name,
                        blank_kind="wms_exception_nodata",
                        blank_payload=r.content,
                        request_url=full_url,
                        request_bbox=bbox_str,
                        request_srs=request_srs,
                        request_version=getmap_version,
                        classification="blank_wms_exception_nodata",
                        wms_exception_text=ex_text,
                    )
                last_err = f"{ep.name}: WMS ServiceException/internalError"
                last_kind = "wms_exception"
                last_payload = r.content
                _log_debug(
                    "tile_getmap_error",
                    {
                        "attempt": attempt,
                        "endpoint": ep.name,
                        "classification": "wms_exception",
                        "error_kind": "wms_exception",
                        "content_type": ct,
                        "url": full_url,
                        "wms_exception": ex_text,
                    },
                )
                continue

            if treat_white_blank and image_format == "image/png":
                extrema = get_rgb_extrema(r.content)
                if extrema is not None and is_uniform_white_rgb(extrema):
                    _log_debug(
                        "tile_getmap_result",
                        {
                            "attempt": attempt,
                            "endpoint": ep.name,
                            "classification": "blank_uniform_white_png",
                            "content_type": ct,
                            "url": full_url,
                            "extrema": str(extrema),
                        },
                    )
                    return FetchResult(
                        status="blank",
                        endpoint_name=ep.name,
                        blank_kind=f"uniform_white_png(extrema={extrema})",
                        blank_payload=r.content,
                        request_url=full_url,
                        request_bbox=bbox_str,
                        request_srs=request_srs,
                        request_version=getmap_version,
                        classification="blank_uniform_white_png",
                    )

            _log_debug(
                "tile_getmap_result",
                {
                    "attempt": attempt,
                    "endpoint": ep.name,
                    "classification": "success",
                    "content_type": ct,
                    "url": full_url,
                    "bytes": len(r.content),
                },
            )
            return FetchResult(
                status="success",
                endpoint_name=ep.name,
                content=r.content,
                request_url=full_url,
                request_bbox=bbox_str,
                request_srs=request_srs,
                request_version=getmap_version,
                classification="success",
            )

        if attempt < retries:
            _sleep_backoff(retry_backoff_base, attempt)

    _log_debug(
        "tile_getmap_result",
        {
            "classification": "failed",
            "error_kind": last_kind,
            "last_err": last_err,
            "last_url": last_url,
            "last_bbox": last_bbox,
            "wms_exception": last_wms_text,
        },
    )
    return FetchResult(
        status="failed",
        endpoint_name=last_endpoint,
        error_kind=last_kind,
        last_err=last_err,
        error_payload=last_payload,
        request_url=last_url,
        request_bbox=last_bbox,
        request_srs=request_srs,
        request_version=getmap_version,
        classification=f"failed_{last_kind}" if last_kind else "failed",
        wms_exception_text=last_wms_text,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Download 2024 IDESP orthophoto tiles by reusing template tile footprints (cell/r/c naming) "
            "and writing fetched imagery into destination cell folders."
        )
    )
    ap.add_argument(
        "--src-tiles-dir",
        "--template-tiles-dir",
        dest="src_tiles_dir",
        required=True,
        help="Template tile root used only for tile footprints (size + worldfile). No imagery is copied.",
    )
    ap.add_argument("--dst-tiles-dir", required=True, help="Destination root for downloaded WMS tiles")
    ap.add_argument("--wms-url", default=WMS_URL_DEFAULT, help="Primary IDESP WMS endpoint")
    ap.add_argument(
        "--fallback-wms-url",
        default=WMS_URL_FALLBACK_DEFAULT,
        help="Fallback WMS endpoint used when primary requests fail",
    )
    ap.add_argument("--no-fallback", action="store_true", help="Disable fallback endpoint usage")
    ap.add_argument("--layer", default=WMS_LAYER_DEFAULT)
    ap.add_argument("--fallback-layer", default="", help="Fallback layer name (default: same as --layer)")
    ap.add_argument("--worldfile-crs", default="EPSG:31983")
    ap.add_argument("--request-srs", default="EPSG:4674")
    ap.add_argument("--format", default="image/png", choices=["image/png", "image/jpeg"])
    ap.add_argument("--missing-only", action="store_true", help="Only fetch missing destination tiles")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing destination tiles")
    ap.add_argument("--cells", default="", help="Optional comma/space-separated cell IDs")
    ap.add_argument("--cells-file", default="", help="Optional text file with one cell ID per line")
    ap.add_argument("--tile-stems", default="", help="Optional comma/space-separated tile stems (e.g., r0003_c0004)")
    ap.add_argument("--sleep", type=float, default=0.2, help="Sleep seconds after each successful request")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--retry-backoff-base", type=float, default=0.6)
    ap.add_argument(
        "--treat-uniform-white-as-blank",
        action="store_true",
        help="Treat uniform white PNG responses as blank/no-data",
    )
    ap.add_argument("--no-treat-uniform-white-as-blank", action="store_true")
    ap.add_argument(
        "--treat-wms-internalerror-as-blank",
        action="store_true",
        help="Treat matching WMS internalError exceptions as blank/no-data",
    )
    ap.add_argument("--no-treat-wms-internalerror-as-blank", action="store_true")
    ap.add_argument("--wms-blank-pattern", action="append", default=[])
    ap.add_argument("--dump-blank", action="store_true", help="Write blank image payloads for debugging")
    ap.add_argument("--dump-errors", action="store_true", help="Write WMS error XML payloads")
    ap.add_argument("--resolve-ip", default="", help="Optional IP override for primary endpoint hostname")
    ap.add_argument("--capabilities-version", default="auto", choices=["auto", "1.1.1", "1.3.0"])
    ap.add_argument("--getmap-version", default="1.1.1", choices=["1.1.1", "1.3.0"])
    ap.add_argument("--progress-every", type=int, default=200, help="Emit progress every N targeted requests")
    ap.add_argument("--debug", action="store_true", help="Verbose per-tile debug logs (bboxs + full GetMap URL)")
    ap.add_argument("--audit-only", action="store_true", help="Only audit expected vs downloaded counts and exit")
    ap.add_argument("--audit-csv", default="", help="Optional CSV output path for audit results")
    ap.add_argument("--audit-top", type=int, default=25, help="Print top N most incomplete cells in audit mode")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.missing_only and args.overwrite:
        raise SystemExit("--missing-only cannot be combined with --overwrite")
    if str(args.worldfile_crs).strip().upper() != "EPSG:31983":
        raise SystemExit("--worldfile-crs must be EPSG:31983 for this template grid")
    if str(args.request_srs).strip().upper() != "EPSG:4674":
        raise SystemExit("--request-srs must be EPSG:4674 for 2023/2024 WMS requests")

    src_root = Path(args.src_tiles_dir).resolve()
    dst_root = Path(args.dst_tiles_dir).resolve()
    if not src_root.exists():
        raise SystemExit(f"Template source directory not found: {src_root}")
    dst_root.mkdir(parents=True, exist_ok=True)

    out_img_suffix = output_image_suffix_for_format(args.format)
    selected_cells = parse_cells_arg(args.cells)
    if str(args.cells_file).strip():
        selected_cells |= parse_cells_file(Path(args.cells_file).resolve())
    selected_tile_stems = parse_cells_arg(args.tile_stems)

    if args.audit_only:
        run_audit(
            src_root=src_root,
            dst_root=dst_root,
            out_img_suffix=out_img_suffix,
            selected_cells=selected_cells,
            audit_csv=Path(args.audit_csv).resolve() if str(args.audit_csv).strip() else None,
            audit_top=int(args.audit_top),
        )
        return

    treat_white_blank = not bool(args.no_treat_uniform_white_as_blank)
    if args.treat_uniform_white_as_blank:
        treat_white_blank = True

    treat_wms_internalerror_blank = not bool(args.no_treat_wms_internalerror_as_blank)
    if args.treat_wms_internalerror_as_blank:
        treat_wms_internalerror_blank = True

    blank_patterns = parse_blank_patterns(args.wms_blank_pattern or [])
    debug = bool(args.debug) or os.environ.get("IDESP_DEBUG") == "1"

    plans, scan_stats = build_cell_plans(
        src_root=src_root,
        dst_root=dst_root,
        out_img_suffix=out_img_suffix,
        selected_cells=selected_cells,
        selected_tile_stems=selected_tile_stems,
        missing_only=bool(args.missing_only),
        overwrite=bool(args.overwrite),
        repair_worldfiles=True,
    )
    log_json({"stage": "scan", **scan_stats})

    total_missing_candidates = int(scan_stats["missing_tiles_found"])
    if total_missing_candidates == 0:
        log_json({"stage": "done", "message": "no_missing_tiles_to_fetch", "dst": str(dst_root)})
        return

    host_to_ip: Dict[str, str] = {}
    primary_norm = normalize_wms_url(args.wms_url)
    primary_host = urlparse(primary_norm).hostname or ""
    if args.resolve_ip.strip() and primary_host:
        host_to_ip[primary_host] = args.resolve_ip.strip()

    with resolve_ip_context(host_to_ip):
        sess = requests.Session()
        endpoints = build_endpoints(
            session=sess,
            primary_url=args.wms_url,
            primary_layer=args.layer,
            fallback_url=args.fallback_wms_url,
            fallback_layer=args.fallback_layer,
            disable_fallback=bool(args.no_fallback),
            request_srs=args.request_srs,
            timeout=float(args.timeout),
            capabilities_version=args.capabilities_version,
        )
        log_json(
            {
                "stage": "endpoints",
                "count": len(endpoints),
                "items": [
                    {"name": e.name, "url": e.wms_url, "layer": e.layer, "bbox": list(e.layer_bbox)}
                    for e in endpoints
                ],
            }
        )

        total_cells_with_missing = sum(1 for p in plans if p.missing_candidates)
        processed_cells = 0
        total_seen = int(scan_stats["tiles_seen"])
        targeted = 0
        fetched = 0
        skipped_existing = int(scan_stats["tiles_existing"])
        skipped_outside = 0
        skipped_blank = 0
        missing_wf = int(scan_stats["missing_worldfile_while_repairing"])
        repaired_wf = int(scan_stats["repaired_worldfile"])
        failed_wms_exception = 0
        failed_http = 0
        failed_other = 0
        endpoint_success: Dict[str, int] = {e.name: 0 for e in endpoints}

        for plan in plans:
            if not plan.missing_candidates:
                continue
            processed_cells += 1
            cell_done_fetch = 0
            cell_done_skip = 0
            cell_done_fail = 0
            log_json(
                {
                    "stage": "cell_start",
                    "cell_index": processed_cells,
                    "cells_with_missing_total": total_cells_with_missing,
                    "cell": plan.cell,
                    "expected_total": plan.expected_total,
                    "existing_total": plan.existing_total,
                    "missing_candidates": len(plan.missing_candidates),
                }
            )

            for src_img in plan.missing_candidates:
                out_img, out_wf = output_paths_for_src(src_root, dst_root, src_img, out_img_suffix)
                out_img.parent.mkdir(parents=True, exist_ok=True)
                tile_meta = {
                    "cell": plan.cell,
                    "tile": src_img.stem,
                    "src_img": str(src_img),
                    "dst_img": str(out_img),
                }

                if out_img.exists() and (args.missing_only or not args.overwrite):
                    skipped_existing += 1
                    cell_done_skip += 1
                    repaired, missing_wf_local = repair_missing_worldfile_if_possible(src_img, out_wf)
                    if repaired:
                        repaired_wf += 1
                    if missing_wf_local:
                        missing_wf += 1
                    if debug:
                        log_json({"stage": "tile_skip", **tile_meta, "classification": "skip_existing"})
                    continue

                wf = find_worldfile(src_img)
                if wf is None:
                    missing_wf += 1
                    failed_other += 1
                    cell_done_fail += 1
                    if debug:
                        log_json({"stage": "tile_fail", **tile_meta, "classification": "missing_worldfile"})
                    continue

                try:
                    with Image.open(src_img) as im:
                        width, height = im.size
                    bbox_src = read_worldfile(wf).bbox(width, height)
                    bbox_31983 = transform_bbox(bbox_src, args.worldfile_crs, "EPSG:31983")
                    bbox_4674 = transform_bbox(bbox_src, args.worldfile_crs, "EPSG:4674")
                    bbox_dst = bbox_4674
                except Exception as exc:
                    failed_other += 1
                    cell_done_fail += 1
                    print(f"WARN: could not build bbox for {src_img}: {exc}", file=sys.stderr)
                    if debug:
                        log_json(
                            {
                                "stage": "tile_fail",
                                **tile_meta,
                                "classification": "bbox_build_failed",
                                "error": str(exc),
                            }
                        )
                    continue

                if debug:
                    log_json(
                        {
                            "stage": "tile_bbox",
                            **tile_meta,
                            "width": width,
                            "height": height,
                            "worldfile_crs": str(args.worldfile_crs),
                            "request_srs": str(args.request_srs),
                            "bbox_31983": list(bbox_31983),
                            "bbox_4674": list(bbox_4674),
                        }
                    )

                if not any(bboxes_intersect(bbox_dst, ep.layer_bbox) for ep in endpoints):
                    skipped_outside += 1
                    cell_done_skip += 1
                    if debug:
                        log_json(
                            {
                                "stage": "tile_skip",
                                **tile_meta,
                                "classification": "skip_outside_layer_bbox",
                                "bbox_31983": list(bbox_31983),
                                "bbox_4674": list(bbox_4674),
                            }
                        )
                    continue

                targeted += 1
                res = fetch_tile_with_fallback(
                    session=sess,
                    endpoints=endpoints,
                    bbox_dst=bbox_dst,
                    bbox_31983=bbox_31983,
                    bbox_4674=bbox_4674,
                    width=width,
                    height=height,
                    request_srs=args.request_srs,
                    image_format=args.format,
                    getmap_version=args.getmap_version,
                    timeout=float(args.timeout),
                    retries=int(args.retries),
                    retry_backoff_base=float(args.retry_backoff_base),
                    treat_wms_internalerror_blank=bool(treat_wms_internalerror_blank),
                    treat_white_blank=bool(treat_white_blank),
                    blank_patterns=blank_patterns,
                    debug=debug,
                    tile_debug_meta=tile_meta,
                )

                if res.status == "success" and res.content is not None:
                    out_img.write_bytes(res.content)
                    out_wf.write_text(wf.read_text(encoding="utf-8"), encoding="utf-8")
                    endpoint_success[res.endpoint_name] = endpoint_success.get(res.endpoint_name, 0) + 1
                    fetched += 1
                    cell_done_fetch += 1
                    if debug:
                        log_json(
                            {
                                "stage": "tile_done",
                                **tile_meta,
                                "classification": res.classification or "success",
                                "endpoint": res.endpoint_name,
                                "url": res.request_url,
                                "bbox_request": res.request_bbox,
                            }
                        )
                elif res.status == "blank":
                    skipped_blank += 1
                    cell_done_skip += 1
                    if debug:
                        log_json(
                            {
                                "stage": "tile_skip",
                                **tile_meta,
                                "classification": res.classification or f"blank_{res.blank_kind}",
                                "endpoint": res.endpoint_name,
                                "blank_kind": res.blank_kind,
                                "url": res.request_url,
                                "bbox_request": res.request_bbox,
                                "wms_exception": res.wms_exception_text,
                            }
                        )
                    if args.dump_blank and res.blank_payload is not None and res.blank_kind.startswith("uniform_white_png"):
                        try:
                            out_img.with_suffix(out_img.suffix + ".blank.png").write_bytes(res.blank_payload)
                        except Exception:
                            pass
                    if (args.dump_errors or debug) and res.blank_payload is not None and res.blank_kind.startswith("wms_exception"):
                        try:
                            out_img.with_suffix(out_img.suffix + ".wms_error.xml").write_bytes(res.blank_payload)
                        except Exception:
                            pass
                else:
                    cell_done_fail += 1
                    if res.error_kind == "http":
                        failed_http += 1
                    elif res.error_kind == "wms_exception":
                        failed_wms_exception += 1
                    else:
                        failed_other += 1

                    if res.last_err:
                        print(f"WARN: failed {src_img} -> {out_img}. last_err={res.last_err}", file=sys.stderr)
                    if debug:
                        log_json(
                            {
                                "stage": "tile_fail",
                                **tile_meta,
                                "classification": res.classification or "failed",
                                "endpoint": res.endpoint_name,
                                "error_kind": res.error_kind,
                                "last_err": res.last_err,
                                "url": res.request_url,
                                "bbox_request": res.request_bbox,
                                "wms_exception": res.wms_exception_text,
                            }
                        )
                    if (args.dump_errors or debug) and res.error_payload is not None and res.error_kind == "wms_exception":
                        try:
                            out_img.with_suffix(out_img.suffix + ".wms_error.xml").write_bytes(res.error_payload)
                        except Exception:
                            pass

                if args.sleep > 0 and res.status == "success":
                    time.sleep(args.sleep)

                if args.progress_every > 0 and targeted % int(args.progress_every) == 0:
                    log_json(
                        {
                            "stage": "progress",
                            "current_cell": plan.cell,
                            "cells_done_with_missing": processed_cells,
                            "cells_with_missing_total": total_cells_with_missing,
                            "tiles_seen_total": total_seen,
                            "missing_tiles_found": total_missing_candidates,
                            "tiles_targeted": targeted,
                            "fetched": fetched,
                            "skipped_existing": skipped_existing,
                            "skipped_outside": skipped_outside,
                            "skipped_blank": skipped_blank,
                            "failed_http": failed_http,
                            "failed_wms_exception": failed_wms_exception,
                            "failed_other": failed_other,
                        }
                    )

            log_json(
                {
                    "stage": "cell_done",
                    "cell": plan.cell,
                    "fetched": cell_done_fetch,
                    "skipped": cell_done_skip,
                    "failed": cell_done_fail,
                }
            )

        summary = {
            "stage": "done",
            "cells_scanned": int(scan_stats["cells_scanned"]),
            "cells_with_missing": total_cells_with_missing,
            "tiles_seen_total": total_seen,
            "missing_tiles_found": total_missing_candidates,
            "tiles_targeted": targeted,
            "fetched": fetched,
            "skipped_existing": skipped_existing,
            "skipped_outside": skipped_outside,
            "skipped_blank": skipped_blank,
            "failed_http": failed_http,
            "failed_wms_exception": failed_wms_exception,
            "failed_other": failed_other,
            "missing_worldfile": missing_wf,
            "repaired_worldfile": repaired_wf,
            "endpoint_success_counts": endpoint_success,
            "dst": str(dst_root),
        }
        log_json(summary)


if __name__ == "__main__":
    main()
