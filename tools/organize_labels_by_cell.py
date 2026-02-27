#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def iter_paths_from_index(path: Path) -> Iterable[Path]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return []
            candidates = ["path", "image_path", "source", "image", "img_path"]
            col = None
            for c in candidates:
                if c in reader.fieldnames:
                    col = c
                    break
            if col is None:
                raise SystemExit(f"Index CSV missing path column. Found: {reader.fieldnames}")
            for row in reader:
                val = row.get(col)
                if val:
                    yield Path(val)
    else:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                yield Path(line)


def find_cell_part(path: Path, tiles_root: Path) -> str | None:
    try:
        rel = path.relative_to(tiles_root)
    except ValueError:
        rel = path
    for part in rel.parts:
        if part.startswith("cell_"):
            return part
    return None


def build_index(tiles_root: Path, index_path: Path | None) -> dict[str, list[Path]]:
    stem_to_paths: dict[str, list[Path]] = {}
    if index_path:
        for p in iter_paths_from_index(index_path):
            stem_to_paths.setdefault(p.stem, []).append(p)
        return stem_to_paths

    for p in tiles_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        stem_to_paths.setdefault(p.stem, []).append(p)
    return stem_to_paths


def target_path(out_root: Path, cell: str, stem: str, layout: str) -> Path:
    if layout == "cell_labels":
        return out_root / cell / "labels" / f"{stem}.txt"
    if layout == "cell_root":
        return out_root / cell / f"{stem}.txt"
    if layout == "prefix":
        return out_root / f"{cell}__{stem}.txt"
    raise ValueError(f"Unknown layout: {layout}")


def ensure_unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    i = 1
    while True:
        candidate = parent / f"{stem}__dup{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Organize YOLO label files by cell folder.")
    ap.add_argument("--tiles-root", required=True, type=Path, help="Root directory containing cell_* folders with tiles.")
    ap.add_argument("--labels-dir", required=True, type=Path, help="Directory containing label .txt files.")
    ap.add_argument("--out-root", required=True, type=Path, help="Output root for organized labels.")
    ap.add_argument("--index", default=None, type=Path, help="Optional CSV or text list of tile paths used for prediction.")
    ap.add_argument("--layout", default="cell_labels", choices=["cell_labels", "cell_root", "prefix"])
    ap.add_argument("--mode", default="copy", choices=["copy", "move"])
    ap.add_argument("--allow-duplicate-matches", action="store_true", help="If a label stem matches multiple tiles, write to all.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tiles_root = args.tiles_root.expanduser().resolve()
    labels_dir = args.labels_dir.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    index_path = args.index.expanduser().resolve() if args.index else None

    if not labels_dir.is_dir():
        raise SystemExit(f"labels-dir not found: {labels_dir}")
    if not tiles_root.is_dir():
        raise SystemExit(f"tiles-root not found: {tiles_root}")
    if index_path and not index_path.exists():
        raise SystemExit(f"index not found: {index_path}")

    stem_to_paths = build_index(tiles_root, index_path)

    labels = sorted(p for p in labels_dir.glob("*.txt") if p.is_file())
    if not labels:
        raise SystemExit(f"No labels found in {labels_dir}")

    unresolved = []
    ambiguous = []
    written = 0

    for label_path in labels:
        stem = label_path.stem
        matches = stem_to_paths.get(stem, [])
        if not matches:
            unresolved.append(label_path)
            continue

        if len(matches) > 1 and not args.allow_duplicate_matches:
            ambiguous.append(label_path)
            continue

        for match in matches:
            cell = find_cell_part(match, tiles_root)
            if not cell:
                ambiguous.append(label_path)
                continue
            target = target_path(out_root, cell, stem, args.layout)
            target.parent.mkdir(parents=True, exist_ok=True)
            target = ensure_unique_path(target)
            if args.dry_run:
                written += 1
                continue
            if args.mode == "move":
                label_path.replace(target)
            else:
                target.write_bytes(label_path.read_bytes())
            written += 1

    print("labels_total:", len(labels))
    print("written:", written)
    print("unresolved:", len(unresolved))
    if unresolved[:10]:
        print("  unresolved_examples:", [str(p) for p in unresolved[:10]])
    print("ambiguous:", len(ambiguous))
    if ambiguous[:10]:
        print("  ambiguous_examples:", [str(p) for p in ambiguous[:10]])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
