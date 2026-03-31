#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Set, Tuple

TILE_ID_PATTERN = r"cell_\d+_\d+__r\d+_c\d+"
TILE_ID_RE = re.compile(rf"({TILE_ID_PATTERN})")


def repo_root_from_tools() -> Path:
    return Path(__file__).resolve().parents[1]


def default_existing_image_roots(repo_root: Optional[Path] = None) -> Tuple[Path, Path]:
    root = repo_root or repo_root_from_tools()
    base = root / "data" / "datasets" / "geosampa_z21_v1" / "images"
    return base / "train", base / "val"


def compose_tile_id(cell: object, tile_stem: object) -> Optional[str]:
    c = str(cell or "").strip()
    s = str(tile_stem or "").strip()
    s = Path(s).with_suffix("").name
    if not c or not s:
        return None
    cand = f"{c}__{s}"
    return cand if re.fullmatch(TILE_ID_PATTERN, cand) else None


def extract_tile_id(value: object) -> Optional[str]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    noext = Path(raw).with_suffix("").as_posix()
    for cand in (Path(noext).name, noext):
        m = TILE_ID_RE.search(cand)
        if m:
            return m.group(1)
        pm = re.search(r"(cell_\d+_\d+)[/\\](r\d+_c\d+)", cand)
        if pm:
            tid = f"{pm.group(1)}__{pm.group(2)}"
            if re.fullmatch(TILE_ID_PATTERN, tid):
                return tid
    return None


def extract_tile_id_from_row(
    row: Mapping[str, object],
    *,
    keys: Sequence[str] = ("tile_id", "tile_rel", "tile_path_abs", "tile", "path", "filename"),
    cell_key: str = "cell",
    tile_stem_key: str = "tile_stem",
) -> Optional[str]:
    for k in keys:
        if k in row:
            tid = extract_tile_id(row.get(k))
            if tid:
                return tid
    return compose_tile_id(row.get(cell_key), row.get(tile_stem_key))


def canonical_rel_from_tile_id(tile_id: str, *, ext: str = ".png") -> Optional[str]:
    tid = str(tile_id).strip()
    if not re.fullmatch(TILE_ID_PATTERN, tid):
        return None
    cell, rc = tid.split("__", 1)
    return f"{cell}/{rc}{ext}"


def collect_tile_ids_from_roots(roots: Iterable[Path]) -> Tuple[Set[str], int]:
    out: Set[str] = set()
    files_scanned = 0
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            files_scanned += 1
            tid = extract_tile_id(p.name) or extract_tile_id(p.as_posix())
            if tid:
                out.add(tid)
    return out, files_scanned
