from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.predict_tiles_to_geojson import iter_images, run_inference  # noqa: E402


def load_tiles(tiles_dir: Path, tiles_list: Optional[Path]) -> List[Path]:
    if tiles_list is None:
        return iter_images(tiles_dir)

    if not tiles_list.exists():
        raise SystemExit(f"tiles-list not found: {tiles_list}")

    tiles: List[Path] = []
    for line in tiles_list.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        p = Path(raw)
        if not p.is_absolute():
            p = tiles_dir / p
        if p.is_file():
            tiles.append(p)
        else:
            print(f"[warn] tile missing: {p}")

    return tiles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--tiles-dir", required=True, type=Path)
    ap.add_argument("--tiles-list", type=Path, default=None, help="Optional file of tile filenames (one per line)")
    ap.add_argument("--z", required=True, type=int)
    ap.add_argument("--out-geojson", required=True, type=Path)

    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--min-area-px", type=float, default=30.0)
    ap.add_argument("--max-tiles", type=int, default=0, help="0 = all tiles, else limit for testing")
    ap.add_argument("--precision", type=int, default=7, help="decimal places for output coordinate rounding")

    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--device", type=str, default=None, help="e.g. 'cpu', '0' for GPU 0")
    ap.add_argument("--verbose", action="store_true", default=False)

    args = ap.parse_args()

    tiles = load_tiles(args.tiles_dir, args.tiles_list)
    if not tiles:
        raise SystemExit("No tiles found to evaluate")

    if args.max_tiles and args.max_tiles > 0:
        tiles = tiles[: args.max_tiles]

    features, stats = run_inference(
        model_path=args.model,
        tiles=tiles,
        z=args.z,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        min_area_px=args.min_area_px,
        max_det=args.max_det,
        retina_masks=True,
        device=args.device,
        verbose=args.verbose,
        precision=args.precision,
    )

    args.out_geojson.parent.mkdir(parents=True, exist_ok=True)
    args.out_geojson.write_text(
        json.dumps(
            {"type": "FeatureCollection", "features": features},
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("Wrote:", args.out_geojson)
    print("tiles_processed:", stats.tiles_processed)
    print("tiles_with_masks:", stats.tiles_with_masks)
    print("polys_total:", stats.polys_total)
    print("polys_kept:", stats.polys_kept)
    print("features_written:", len(features))


if __name__ == "__main__":
    main()
