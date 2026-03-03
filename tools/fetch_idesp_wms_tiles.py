#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from urllib.parse import urlparse, urlunparse
from xml.etree import ElementTree as ET

import requests
from PIL import Image
from pyproj import Transformer

WMS_URL_DEFAULT = "https://www.idesp.sp.gov.br/geoimage/idesp_raster/wms"
WMS_LAYER_DEFAULT = "ORTOFOTOS_SP_2023_2024"


@dataclass(frozen=True)
class Worldfile:
    # Worldfile 6 lines: A, D, B, E, C, F
    A: float  # pixel size in x
    D: float  # rotation about y
    B: float  # rotation about x
    E: float  # pixel size in y (typically negative)
    C: float  # x coord of center of upper-left pixel
    F: float  # y coord of center of upper-left pixel

    def bbox(self, width: int, height: int) -> Tuple[float, float, float, float]:
        # Compute bbox of image in projected coordinates (assuming no rotation).
        if abs(self.B) > 1e-12 or abs(self.D) > 1e-12:
            raise ValueError(f"Worldfile has rotation terms (B={self.B}, D={self.D}); not supported.")

        minx = self.C - (self.A / 2.0)
        maxy = self.F - (self.E / 2.0)  # E is usually negative
        maxx = minx + (width * self.A)
        miny = maxy + (height * self.E)  # E negative => miny < maxy

        if minx > maxx:
            minx, maxx = maxx, minx
        if miny > maxy:
            miny, maxy = maxy, miny
        return (minx, miny, maxx, maxy)


def normalize_wms_url(url: str) -> str:
    try:
        u = urlparse(url)
        host = (u.hostname or "").lower()
        if host in {"idesp.sp.gov.br", "www.idesp.sp.gov.br"}:
            u = u._replace(scheme="https", netloc="www.idesp.sp.gov.br")
            return urlunparse(u)
    except Exception:
        pass
    return url


@contextmanager
def resolve_ip_context(hostname: str, ip: Optional[str]):
    """
    Curl --resolve style:
      - Connect TCP to `ip`
      - Keep URL hostname for Host header and TLS SNI
    """
    if not ip:
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
        if host == hostname:
            host = ip
        return orig_create((host, port), *args, **kwargs)

    connection.create_connection = patched
    try:
        yield
    finally:
        connection.create_connection = orig_create


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
    # WMS 1.1.1 uses X,Y order. We enforce always_xy for geographic CRSs too.
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
) -> Tuple[str, dict, str]:
    minx, miny, maxx, maxy = bbox_xy
    bbox_str = bbox_to_str_xy(minx, miny, maxx, maxy)

    params = {
        "SERVICE": "WMS",
        "REQUEST": "GetMap",
        "VERSION": version,
        "LAYERS": layer,
        "STYLES": "",
        "FORMAT": fmt,
        "TRANSPARENT": "FALSE",
        "WIDTH": str(width),
        "HEIGHT": str(height),
    }
    if version == "1.3.0":
        params["CRS"] = srs
    else:
        params["SRS"] = srs
    params["BBOX"] = bbox_str
    return (wms_url, params, bbox_str)


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
    """
    DFS over layer tree with inherited bboxes:
      - inherited_bboxes: CRS->bbox
      - inherited_geo_bbox: lon/lat bbox
    """
    def extract_layer_info(layer_el: ET.Element) -> Dict[str, Any]:
        out: Dict[str, Any] = {"bboxes": [], "geo_bbox": None}

        # BoundingBox elements (direct children)
        for bb_el in _layer_direct_children(layer_el, "BoundingBox"):
            crs = (bb_el.attrib.get("SRS") or bb_el.attrib.get("CRS") or "").strip().upper()
            bbox = _parse_bbox_attrs(bb_el)
            if crs and bbox is not None:
                out["bboxes"].append((crs, bbox))

        # LatLonBoundingBox (WMS 1.1.1)
        for ll_el in _layer_direct_children(layer_el, "LatLonBoundingBox"):
            bbox = _parse_bbox_attrs(ll_el)
            if bbox is not None:
                out["geo_bbox"] = bbox
                break

        # EX_GeographicBoundingBox (WMS 1.3.0)
        if out["geo_bbox"] is None:
            for ex_el in _layer_direct_children(layer_el, "EX_GeographicBoundingBox"):
                bbox = _parse_ex_geographic_bbox(ex_el)
                if bbox is not None:
                    out["geo_bbox"] = bbox
                    break

        return out

    stack: List[Tuple[ET.Element, Dict[str, Tuple[float, float, float, float]], Optional[Tuple[float, float, float, float]]]] = []
    top_info = extract_layer_info(top_layer)
    top_map = {crs: bb for crs, bb in top_info["bboxes"]}
    stack.append((top_layer, top_map, top_info["geo_bbox"]))

    while stack:
        layer_el, inherited_map, inherited_geo = stack.pop()
        info = extract_layer_info(layer_el)

        merged = dict(inherited_map)
        for crs, bb in info["bboxes"]:
            merged[crs] = bb

        merged_geo = info["geo_bbox"] if info["geo_bbox"] is not None else inherited_geo

        name = (_text_child(layer_el, "Name") or "").strip()
        if name == target_name:
            merged_bboxes = [(crs, merged[crs]) for crs in sorted(merged.keys())]
            return {"layer_el": layer_el, "bboxes": merged_bboxes, "geo_bbox": merged_geo}

        # children layers
        children = _layer_direct_children(layer_el, "Layer")
        for c in reversed(children):
            stack.append((c, merged, merged_geo))

    return None


def _fetch_getcapabilities(
    session: requests.Session,
    wms_url: str,
    timeout: float,
    version: str,
) -> bytes:
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
    prefer_versions: Tuple[str, ...] = ("1.1.1", "1.3.0"),
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

            # Prefer explicit matching CRS bbox
            for crs, bb in found["bboxes"]:
                if crs.strip().upper() == req:
                    return bb

            # Fallback: transform geographic bbox (lon/lat)
            geo = found["geo_bbox"]
            if geo is None:
                raise RuntimeError(f"No usable bbox found for layer {layer} (no BoundingBox + no geographic bbox)")

            if req in {"EPSG:4326", "EPSG:4674"}:
                return geo

            return transform_bbox(geo, "EPSG:4326", request_srs)

        except Exception as e:
            last_err = f"{ver}: {e}"
            continue

    raise RuntimeError(f"Failed to parse GetCapabilities for layer bbox. Last error: {last_err}")


def debug_warn_swapped(minx: float, miny: float) -> None:
    # São Paulo area heuristic: lon ~ -46, lat ~ -23.
    if abs(minx) < 30 and abs(miny) > 30:
        print("DEBUG WARN: bbox minx/miny look swapped for SP area (minx~lat, miny~lon)", file=sys.stderr)


def _sleep_backoff(base: float, attempt: int, jitter: float = 0.15, max_sleep: float = 10.0) -> None:
    t = base * (2 ** (attempt - 1))
    t = min(t, max_sleep)
    if jitter > 0:
        t *= (1.0 + random.uniform(-jitter, jitter))
    time.sleep(max(0.0, t))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-tiles-dir", required=True, help="Existing GeoSampa tile root (has images + worldfiles)")
    ap.add_argument("--dst-tiles-dir", required=True, help="Destination root for fetched WMS tiles (mirrors src structure)")
    ap.add_argument("--wms-url", default=WMS_URL_DEFAULT)
    ap.add_argument("--layer", default=WMS_LAYER_DEFAULT)
    ap.add_argument("--worldfile-crs", default="EPSG:31983")
    ap.add_argument(
        "--request-srs",
        default="EPSG:3857",
        help="SRS/CRS for GetMap requests (XY order enforced even for geographic).",
    )
    ap.add_argument("--format", default="image/png", choices=["image/png", "image/jpeg"])
    ap.add_argument("--max-tiles", type=int, default=0, help="0 = no limit")
    ap.add_argument("--sleep", type=float, default=0.2, help="Seconds to sleep between *successful* requests")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--retry-backoff-base", type=float, default=0.6)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--treat-uniform-white-as-blank",
        action="store_true",
        help="If set, uniform white PNG responses are counted as skipped_blank (recommended).",
    )
    ap.add_argument(
        "--no-treat-uniform-white-as-blank",
        action="store_true",
        help="Disable uniform-white blank skipping.",
    )
    ap.add_argument(
        "--treat-wms-internalerror-as-blank",
        action="store_true",
        help="If set, WMS internalError ServiceExceptions matching patterns are counted as skipped_blank.",
    )
    ap.add_argument(
        "--no-treat-wms-internalerror-as-blank",
        action="store_true",
        help="Disable treating certain WMS internalError exceptions as blank.",
    )
    ap.add_argument(
        "--wms-blank-pattern",
        action="append",
        default=[],
        help="Regex (case-insensitive) that, if found in ServiceException XML, will be treated as blank. Can be repeated.",
    )
    ap.add_argument(
        "--dump-blank",
        action="store_true",
        help="If set, write blank images to disk with .blank.png suffix (useful for debugging).",
    )
    ap.add_argument(
        "--dump-errors",
        action="store_true",
        help="If set, always write .wms_error.xml files (otherwise only on IDESP_DEBUG=1).",
    )
    ap.add_argument(
        "--resolve-ip",
        default="",
        help="Optional IP to resolve the WMS hostname to (curl --resolve style).",
    )
    ap.add_argument(
        "--capabilities-version",
        default="auto",
        choices=["auto", "1.1.1", "1.3.0"],
        help="Which GetCapabilities VERSION to request. auto tries 1.1.1 then 1.3.0.",
    )
    ap.add_argument(
        "--getmap-version",
        default="1.1.1",
        choices=["1.1.1", "1.3.0"],
        help="GetMap VERSION to use. 1.1.1 recommended.",
    )
    args = ap.parse_args()

    debug = os.environ.get("IDESP_DEBUG") == "1"

    # Default behavior requested: treat uniform white PNG as skipped_blank
    treat_white_blank = True
    if args.no_treat_uniform_white_as_blank:
        treat_white_blank = False
    if args.treat_uniform_white_as_blank:
        treat_white_blank = True

    # Default behavior for the server you’re hitting: treat sourceURL-missing internalError as blank
    treat_wms_internalerror_blank = True
    if args.no_treat_wms_internalerror_as_blank:
        treat_wms_internalerror_blank = False
    if args.treat_wms_internalerror_as_blank:
        treat_wms_internalerror_blank = True

    # Default patterns (case-insensitive)
    default_blank_patterns = [
        r"specified\s+sourceurl\s+doesn[’']?t\s+refer\s+to\s+an\s+existing\s+file",
        r"rendering\s+process\s+failed",
    ]
    user_patterns = args.wms_blank_pattern or []
    patterns = default_blank_patterns + user_patterns
    compiled_patterns = [re.compile(p, re.IGNORECASE) for p in patterns]

    wms_url = normalize_wms_url(args.wms_url)
    request_srs = args.request_srs.strip()

    host = urlparse(wms_url).hostname or ""
    ctx = resolve_ip_context(host, args.resolve_ip) if args.resolve_ip else nullcontext()

    with ctx:
        sess = requests.Session()

        # Get layer bbox from GetCapabilities
        try:
            if args.capabilities_version == "auto":
                prefer = ("1.1.1", "1.3.0")
            else:
                prefer = (args.capabilities_version,)
            layer_bbox = get_layer_bbox_from_capabilities(
                session=sess,
                wms_url=wms_url,
                layer=args.layer,
                request_srs=request_srs,
                timeout=args.timeout,
                prefer_versions=prefer,
            )
        except Exception as e:
            raise SystemExit(f"Failed to read layer bbox from GetCapabilities: {e}")

        src_root = Path(args.src_tiles_dir)
        dst_root = Path(args.dst_tiles_dir)
        dst_root.mkdir(parents=True, exist_ok=True)

        exts = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")
        src_imgs = sorted([p for p in src_root.rglob("*") if p.is_file() and p.suffix.lower() in exts])
        if not src_imgs:
            raise SystemExit(f"No images found under {src_root}")

        total_seen = 0
        targeted = 0
        fetched = 0
        skipped_existing = 0
        skipped_outside = 0
        skipped_blank = 0
        missing_wf = 0
        failed_wms_exception = 0
        failed_http = 0
        failed_other = 0

        for img in src_imgs:
            total_seen += 1
            if args.max_tiles and targeted >= args.max_tiles:
                break

            wf = find_worldfile(img)
            if wf is None:
                missing_wf += 1
                continue

            rel = img.relative_to(src_root)
            out_img = dst_root / rel

            if out_img.suffix.lower() == ".png":
                out_wf = out_img.with_suffix(".pgw")
            elif out_img.suffix.lower() in (".jpg", ".jpeg"):
                out_wf = out_img.with_suffix(".jgw")
            else:
                out_wf = out_img.with_suffix(".wld")

            out_img.parent.mkdir(parents=True, exist_ok=True)

            if out_img.exists() and not args.overwrite:
                skipped_existing += 1
                continue

            try:
                with Image.open(img) as im:
                    width, height = im.size
            except Exception as e:
                failed_other += 1
                print(f"WARN: could not read image size for {img}: {e}", file=sys.stderr)
                continue

            try:
                w = read_worldfile(wf)
                bbox_src = w.bbox(width, height)
                bbox_dst = transform_bbox(bbox_src, args.worldfile_crs, request_srs)
            except Exception as e:
                failed_other += 1
                print(f"WARN: worldfile/bbox error for {img} ({wf}): {e}", file=sys.stderr)
                continue

            # Coverage filter
            if not bboxes_intersect(bbox_dst, layer_bbox):
                skipped_outside += 1
                continue

            targeted += 1

            url, params, bbox_str = build_wms_getmap_request(
                wms_url=wms_url,
                layer=args.layer,
                bbox_xy=bbox_dst,
                width=width,
                height=height,
                srs=request_srs,
                fmt=args.format,
                version=args.getmap_version,
            )

            req = requests.Request("GET", url, params=params)
            final_url = req.prepare().url

            if debug:
                print("DEBUG REQUEST_SRS:", request_srs)
                print("DEBUG BBOX SRC:", bbox_src)
                print("DEBUG BBOX DST:", bbox_dst)
                print("DEBUG LAYER BBOX:", layer_bbox)
                print("DEBUG BBOX STR:", bbox_str)
                print("DEBUG URL :", final_url)
                minx, miny, _maxx, _maxy = bbox_dst
                debug_warn_swapped(minx, miny)

            ok = False
            treated_blank = False
            last_err: Optional[str] = None

            for attempt in range(1, args.retries + 1):
                try:
                    r = sess.get(url, params=params, timeout=args.timeout)
                    ct = (r.headers.get("Content-Type") or "").lower()

                    if r.status_code != 200:
                        last_err = f"HTTP {r.status_code} content-type={ct}"
                        if attempt < args.retries:
                            _sleep_backoff(args.retry_backoff_base, attempt)
                            continue
                        failed_http += 1
                        break

                    if is_wms_exception(r.content, ct):
                        # Potentially treat certain internalError ServiceExceptions as "no data"
                        if treat_wms_internalerror_blank and wms_exception_is_nodata(r.content, compiled_patterns):
                            skipped_blank += 1
                            treated_blank = True
                            last_err = "WMS internalError treated as blank (matched pattern)"
                            err_path = out_img.with_suffix(out_img.suffix + ".wms_error.xml")
                            if args.dump_errors or debug:
                                try:
                                    err_path.write_bytes(r.content)
                                except Exception:
                                    pass
                            ok = True
                            break

                        failed_wms_exception += 1
                        last_err = "WMS ServiceException/internalError"
                        err_path = out_img.with_suffix(out_img.suffix + ".wms_error.xml")
                        if args.dump_errors or debug:
                            try:
                                err_path.write_bytes(r.content)
                            except Exception:
                                pass
                        break

                    # Uniform-white PNG as blank
                    if treat_white_blank and args.format == "image/png":
                        extrema = get_rgb_extrema(r.content)
                        if extrema is not None and is_uniform_white_rgb(extrema):
                            skipped_blank += 1
                            treated_blank = True
                            last_err = f"uniform_white_png(extrema={extrema})"
                            if args.dump_blank or debug:
                                blank_path = out_img.with_suffix(out_img.suffix + ".blank.png")
                                try:
                                    blank_path.write_bytes(r.content)
                                except Exception:
                                    pass
                            ok = True
                            break

                    # Write image and worldfile
                    out_img.write_bytes(r.content)
                    out_wf.write_text(wf.read_text(encoding="utf-8"), encoding="utf-8")
                    ok = True
                    break

                except Exception as e:
                    last_err = repr(e)
                    if attempt < args.retries:
                        _sleep_backoff(args.retry_backoff_base, attempt)
                        continue
                    failed_other += 1
                    break

            if ok:
                if not treated_blank:
                    fetched += 1
            else:
                if last_err is not None:
                    print(f"WARN: failed {img} -> {out_img}. last_err={last_err}", file=sys.stderr)

            if args.sleep > 0:
                time.sleep(args.sleep)

            if targeted % 200 == 0:
                print(
                    "progress:",
                    f"total_seen={total_seen}",
                    f"targeted={targeted}",
                    f"fetched={fetched}",
                    f"skipped_outside={skipped_outside}",
                    f"skipped_blank={skipped_blank}",
                    f"failed_wms_exception={failed_wms_exception}",
                    f"failed_http={failed_http}",
                    f"failed_other={failed_other}",
                    file=sys.stderr,
                )

        print("done")
        print(f"tiles_seen:              {total_seen}")
        print(f"tiles_targeted:          {targeted}")
        print(f"fetched:                 {fetched}")
        print(f"skipped_existing:        {skipped_existing}")
        print(f"skipped_outside:         {skipped_outside}")
        print(f"skipped_blank:           {skipped_blank}")
        print(f"failed_wms_exception:    {failed_wms_exception}")
        print(f"failed_http:             {failed_http}")
        print(f"failed_other:            {failed_other}")
        print(f"missing_worldfile:       {missing_wf}")
        print(f"layer_bbox({request_srs}): {layer_bbox}")
        print(f"dst:                     {dst_root}")


if __name__ == "__main__":
    main()