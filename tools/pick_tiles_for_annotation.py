#!/usr/bin/env python3
"""
Pick IDESP tiles for annotation using a prior 2020 run as a sampling heuristic.

Key idea: tile names (r####_c####) repeat per cell, so everything must be keyed by:
    (cell_xxxx_yyyy, r####_c####)

Expected inputs:
- IDESP tiles live under: .../<cell_xxxx_yyyy>/r####_c####.(jpg|png|jpeg)
- "Flat" 2020 labels live under: <flat_root>/<cell_xxxx_yyyy>/labels/r####_c####.txt

Sampling strategy:
- Positives: (cell, tile) where label file exists and is non-empty
- Hard negatives: unlabeled tiles adjacent to positives within the same cell (r/c +/- radius)
- Random negatives: unlabeled tiles sampled randomly

Outputs:
- dst/images_idesp/     (images to upload to CVAT)
- dst/manifest.jsonl    (one JSON per selected tile with provenance)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

IMG_EXTS = (".jpg", ".jpeg", ".png")
RC_RE = re.compile(r"^(r\d+_c\d+)$")
CELL_RE = re.compile(r"^cell_\d{4}_\d{4}$")


@dataclass(frozen=True)
class TileKey:
    cell: str
    stem: str
    r: int
    c: int


def parse_tile_stem(stem: str) -> Optional[Tuple[int, int]]:
    # stem is like r0002_c0011
    m = re.match(r"^r(\d+)_c(\d+)$", stem)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def iter_idesp_tiles(idesp_root: Path) -> Iterable[Tuple[str, Path]]:
    """
    Yield (cell, image_path) for images under .../cell_xxxx_yyyy/
    """
    for cell_dir in idesp_root.glob("cell_*_*"):
        if not cell_dir.is_dir():
            continue
        cell = cell_dir.name
        if not CELL_RE.match(cell):
            continue
        for ext in IMG_EXTS:
            yield from ((cell, p) for p in cell_dir.glob(f"r*_c*{ext}"))


def pick_preferred_image(paths: List[Path]) -> Path:
    """
    Prefer .jpg if duplicates exist.
    """
    if len(paths) == 1:
        return paths[0]
    paths_sorted = sorted(paths, key=lambda p: (p.suffix.lower() != ".jpg", p.suffix.lower()))
    return paths_sorted[0]


def label_is_positive(label_path: Path) -> bool:
    try:
        return label_path.exists() and label_path.stat().st_size > 0
    except OSError:
        return False


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--idesp-tiles-dir", required=True, type=Path,
                    help="Root of IDESP tiles (contains cell_xxxx_yyyy folders).")
    ap.add_argument("--flat-2020-labels-root", required=True, type=Path,
                    help="Flat labels root containing <cell>/labels/r*_c*.txt")
    ap.add_argument("--dst-dir", required=True, type=Path,
                    help="Output directory for annotation set.")
    ap.add_argument("--n-pos", type=int, default=500)
    ap.add_argument("--n-hard-neg", type=int, default=500)
    ap.add_argument("--n-rand-neg", type=int, default=500)
    ap.add_argument("--neighbor-radius", type=int, default=1)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    random.seed(args.seed)

    idesp_root = args.idesp_tiles_dir
    flat_labels_root = args.flat_2020_labels_root
    dst_dir = args.dst_dir

    # Index IDESP images by (cell, stem) and keep a preferred ext if duplicates exist
    idx: Dict[Tuple[str, str], List[Path]] = {}
    for cell, p in iter_idesp_tiles(idesp_root):
        stem = p.stem
        if not RC_RE.match(stem):
            continue
        idx.setdefault((cell, stem), []).append(p)

    if not idx:
        raise SystemExit(f"ERROR: no IDESP tiles found under {idesp_root}")

    idesp_img: Dict[Tuple[str, str], Path] = {
        k: pick_preferred_image(v) for k, v in idx.items()
    }

    # Build TileKeys
    keys: List[TileKey] = []
    for (cell, stem), p in idesp_img.items():
        rc = parse_tile_stem(stem)
        if not rc:
            continue
        r, c = rc
        keys.append(TileKey(cell=cell, stem=stem, r=r, c=c))

    # Fast lookup for neighbors
    by_cell: Dict[str, Dict[str, TileKey]] = {}
    for k in keys:
        by_cell.setdefault(k.cell, {})[k.stem] = k

    def label_path(k: TileKey) -> Path:
        return flat_labels_root / k.cell / "labels" / f"{k.stem}.txt"

    positives: List[TileKey] = []
    negatives: List[TileKey] = []
    for k in keys:
        if label_is_positive(label_path(k)):
            positives.append(k)
        else:
            negatives.append(k)

    # Hard negative candidates: neighbors of positives within same cell
    pos_set: Set[Tuple[str, str]] = {(k.cell, k.stem) for k in positives}
    hard_seen: Set[Tuple[str, str]] = set()
    hard_candidates: List[TileKey] = []

    for pk in positives:
        cell_map = by_cell.get(pk.cell, {})
        for dr in range(-args.neighbor_radius, args.neighbor_radius + 1):
            for dc in range(-args.neighbor_radius, args.neighbor_radius + 1):
                if dr == 0 and dc == 0:
                    continue
                nr = pk.r + dr
                nc = pk.c + dc
                nstem = f"r{nr:04d}_c{nc:04d}"
                nk = cell_map.get(nstem)
                if not nk:
                    continue
                kk = (nk.cell, nk.stem)
                if kk in pos_set:
                    continue
                if kk in hard_seen:
                    continue
                hard_seen.add(kk)
                hard_candidates.append(nk)

    def sample(lst: List[TileKey], k: int) -> List[TileKey]:
        if k <= 0:
            return []
        if len(lst) <= k:
            return lst[:]
        return random.sample(lst, k)

    sel_pos = sample(positives, args.n_pos)
    sel_hard = sample(hard_candidates, args.n_hard_neg)

    hard_keys = {(k.cell, k.stem) for k in sel_hard}
    neg_pool = [k for k in negatives if (k.cell, k.stem) not in hard_keys]
    sel_rand = sample(neg_pool, args.n_rand_neg)

    selected: List[Tuple[str, TileKey]] = (
        [("pos", k) for k in sel_pos] +
        [("hard_neg", k) for k in sel_hard] +
        [("rand_neg", k) for k in sel_rand]
    )

    out_imgs = dst_dir / "images_idesp"
    manifest = dst_dir / "manifest.jsonl"
    dst_dir.mkdir(parents=True, exist_ok=True)

    wrote = 0
    with manifest.open("w", encoding="utf-8") as f:
        for kind, k in selected:
            src = idesp_img.get((k.cell, k.stem))
            if not src:
                continue

            # Keep cell in filename so CVAT exports stay unique
            dst_name = f"{k.cell}__{src.name}"
            dst = out_imgs / dst_name
            safe_copy(src, dst)

            rec = {
                "kind": kind,
                "cell": k.cell,
                "stem": k.stem,
                "r": k.r,
                "c": k.c,
                "idesp_src": str(src),
                "idesp_dst": str(dst),
                "label_2020": str(label_path(k)),
                "label_2020_non_empty": label_is_positive(label_path(k)),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            wrote += 1

    print("IDESPs indexed:", len(keys))
    print("positives:", len(positives))
    print("negatives:", len(negatives))
    print("hard_neg_candidates:", len(hard_candidates))
    print("selected_total:", len(selected))
    print("copied_total:", wrote)
    print("out:", str(dst_dir))


if __name__ == "__main__":
    main()