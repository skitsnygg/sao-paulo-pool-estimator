#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
PATH_COL_CANDIDATES = ("tile_rel", "relative_path", "path", "tile_path_abs", "image_path", "image")


@dataclass(frozen=True)
class TileRecord:
    rel: Path
    abs_path: Path
    cell: str
    tile_stem: str

    @property
    def unique_name(self) -> str:
        stem_parts = list(self.rel.with_suffix("").parts)
        return "__".join(stem_parts) + self.rel.suffix.lower()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Prepare a CVAT upload image set from nested tile trees by creating deterministic "
            "unique filenames and writing a mapping manifest."
        )
    )
    ap.add_argument("--tiles-root", required=True, type=Path, help="Nested tile root, e.g. .../cell_xxxx_yyyy")
    ap.add_argument("--out-dir", required=True, type=Path, help="Output root for CVAT package.")
    ap.add_argument("--tiles-csv", type=Path, default=None, help="Optional tile summary CSV to select specific tiles.")
    ap.add_argument(
        "--cells-file",
        type=Path,
        default=None,
        help="Optional text file with one cell_xxxx_yyyy per line to filter tiles.",
    )
    ap.add_argument(
        "--require-num-preds-gt",
        type=int,
        default=-1,
        help="When --tiles-csv is used, keep only rows with num_preds > this value. Default: -1 (no filter).",
    )
    ap.add_argument("--max-tiles", type=int, default=0, help="Optional max number of tiles after filtering.")
    ap.add_argument("--images-subdir", default="JPEGImages", help="Subdirectory name for exported images.")
    ap.add_argument("--symlink-images", action="store_true", help="Symlink images instead of copying.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files in output.")
    ap.add_argument("--dry-run", action="store_true", help="Preview only.")
    return ap.parse_args()


def load_cells_filter(path: Optional[Path]) -> Optional[set[str]]:
    if path is None:
        return None
    p = path.expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"--cells-file not found: {p}")
    cells = {
        ln.strip()
        for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    }
    return cells


def iter_all_tiles(root: Path, cells_filter: Optional[set[str]]) -> Iterable[TileRecord]:
    for img in sorted(root.rglob("*")):
        if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
            continue
        rel = img.relative_to(root)
        cell = rel.parts[0] if len(rel.parts) > 1 else ""
        if cells_filter is not None and cell not in cells_filter:
            continue
        yield TileRecord(rel=rel, abs_path=img.resolve(), cell=cell, tile_stem=img.stem)


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def choose_path_column(rows: Sequence[dict]) -> Optional[str]:
    if not rows:
        return None
    cols = set(rows[0].keys())
    for c in PATH_COL_CANDIDATES:
        if c in cols:
            return c
    return None


def row_num_preds(row: dict) -> Optional[int]:
    raw = (row.get("num_preds") or "").strip()
    if raw == "":
        return None
    try:
        return int(float(raw))
    except Exception:
        return None


def parse_rel_or_abs(path_text: str) -> Path:
    return Path(path_text.strip().replace("\\", "/"))


def derive_rel_from_abs(abs_path: Path) -> Path:
    parts = abs_path.parts
    for i, token in enumerate(parts):
        if token.startswith("cell_") and i + 1 < len(parts):
            return Path(token) / parts[i + 1]
    return Path(abs_path.name)


def collect_from_csv(
    root: Path,
    csv_path: Path,
    cells_filter: Optional[set[str]],
    require_num_preds_gt: int,
) -> list[TileRecord]:
    rows = read_csv_rows(csv_path)
    if not rows:
        return []
    path_col = choose_path_column(rows)
    if path_col is None:
        raise SystemExit(
            f"Could not find a path column in {csv_path}. "
            f"Tried: {', '.join(PATH_COL_CANDIDATES)}"
        )

    out: list[TileRecord] = []
    for row in rows:
        if require_num_preds_gt >= 0:
            n = row_num_preds(row)
            if n is None or n <= require_num_preds_gt:
                continue

        raw = (row.get(path_col) or "").strip()
        if not raw:
            continue
        p = parse_rel_or_abs(raw)
        if p.is_absolute():
            abs_path = p
            try:
                rel = abs_path.relative_to(root)
            except ValueError:
                # Not under tiles_root: derive a stable rel key from path tokens.
                rel = derive_rel_from_abs(abs_path)
        else:
            rel = p
            abs_path = (root / rel).resolve()

        if not abs_path.exists():
            continue
        if abs_path.suffix.lower() not in IMG_EXTS:
            continue

        cell = rel.parts[0] if len(rel.parts) > 1 else ""
        if cells_filter is not None and cell not in cells_filter:
            continue

        out.append(TileRecord(rel=rel, abs_path=abs_path, cell=cell, tile_stem=abs_path.stem))

    out.sort(key=lambda r: r.rel.as_posix())
    return out


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_manifest_csv(path: Path, rows: Sequence[dict]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "unique_name",
        "tile_rel",
        "tile_path_abs",
        "cell",
        "tile_stem",
        "out_image_abs",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_manifest_jsonl(path: Path, rows: Sequence[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def copy_or_link(src: Path, dst: Path, symlink_images: bool) -> None:
    ensure_dir(dst.parent)
    if symlink_images:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        rel_target = os.path.relpath(src, start=dst.parent)
        dst.symlink_to(rel_target)
        return
    shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    tiles_root = args.tiles_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    images_dir = out_dir / args.images_subdir
    manifest_csv = out_dir / "manifest.csv"
    manifest_jsonl = out_dir / "manifest.jsonl"

    if not tiles_root.exists():
        raise SystemExit(f"--tiles-root not found: {tiles_root}")

    cells_filter = load_cells_filter(args.cells_file)

    if args.tiles_csv is not None:
        tiles_csv = args.tiles_csv.expanduser().resolve()
        if not tiles_csv.exists():
            raise SystemExit(f"--tiles-csv not found: {tiles_csv}")
        selected = collect_from_csv(
            root=tiles_root,
            csv_path=tiles_csv,
            cells_filter=cells_filter,
            require_num_preds_gt=int(args.require_num_preds_gt),
        )
    else:
        selected = list(iter_all_tiles(tiles_root, cells_filter))

    if args.max_tiles and args.max_tiles > 0:
        selected = selected[: int(args.max_tiles)]

    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory is not empty: {out_dir} (use --overwrite)")
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)

    ensure_dir(images_dir)

    manifest_rows: list[dict] = []
    seen_unique: set[str] = set()
    collisions = 0
    written = 0
    missing = 0

    for rec in selected:
        unique_name = rec.unique_name
        if unique_name in seen_unique:
            collisions += 1
            continue
        seen_unique.add(unique_name)

        if not rec.abs_path.exists():
            missing += 1
            continue

        out_img = images_dir / unique_name
        row = {
            "unique_name": unique_name,
            "tile_rel": rec.rel.as_posix(),
            "tile_path_abs": str(rec.abs_path),
            "cell": rec.cell,
            "tile_stem": rec.tile_stem,
            "out_image_abs": str(out_img),
        }
        manifest_rows.append(row)

        if not args.dry_run:
            copy_or_link(rec.abs_path, out_img, symlink_images=bool(args.symlink_images))
            written += 1

    if not args.dry_run:
        write_manifest_csv(manifest_csv, manifest_rows)
        write_manifest_jsonl(manifest_jsonl, manifest_rows)

    print("tiles_root:", tiles_root)
    print("out_dir:", out_dir)
    print("images_dir:", images_dir)
    print("selected_total:", len(selected))
    print("manifest_rows:", len(manifest_rows))
    print("written_images:", written)
    print("missing_images:", missing)
    print("name_collisions:", collisions)
    print("dry_run:", bool(args.dry_run))
    if not args.dry_run:
        print("manifest_csv:", manifest_csv)
        print("manifest_jsonl:", manifest_jsonl)


if __name__ == "__main__":
    main()
