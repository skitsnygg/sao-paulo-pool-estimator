#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from shapely.geometry import shape, mapping


def load_fc(p: Path) -> Dict:
    return json.loads(p.read_text())


def infer_neighborhood(run_dir: Path) -> str:
    # run_dir like runs/predict/moema_2020_conf015 -> neighborhood "moema"
    name = run_dir.name
    base = name.split("_2020_", 1)[0]
    return base


def clean_features(features: List[Dict]) -> Tuple[List[Dict], int]:
    kept: List[Dict] = []
    dropped = 0
    for f in features:
        try:
            g = shape(f["geometry"])
        except Exception:
            dropped += 1
            continue
        if g.is_empty or (not g.is_valid):
            dropped += 1
            continue
        f["geometry"] = mapping(g)
        kept.append(f)
    return kept, dropped


def run_dedupe(python_exe: str, dedupe_script: Path, in_geojson: Path, out_geojson: Path, iou: float, buffer: float) -> None:
    cmd = [
        python_exe,
        str(dedupe_script),
        "--in-geojson", str(in_geojson),
        "--out-geojson", str(out_geojson),
        "--iou", str(iou),
        "--buffer", str(buffer),
    ]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Merge neighborhood pools_dedup_*_clean.geojson files into a city-wide inventory and run global dedupe."
    )
    ap.add_argument("--predict-root", default="runs/predict")
    ap.add_argument("--pattern", default="pools_dedup_iou035_clean.geojson")
    ap.add_argument("--out-raw", default="runs/predict/sao_paulo_inventory_raw.geojson")
    ap.add_argument("--out-dedup", default="runs/predict/sao_paulo_inventory_dedup.geojson")
    ap.add_argument("--out-final", default="runs/predict/sao_paulo_inventory_final.geojson")
    ap.add_argument("--dedupe-script", default="tools/dedupe_geojson_polygons.py")
    ap.add_argument("--dedupe-iou", type=float, default=0.35)
    ap.add_argument("--dedupe-buffer", type=float, default=0.0)
    args = ap.parse_args()

    predict_root = Path(args.predict_root)
    if not predict_root.exists():
        raise SystemExit(f"predict root not found: {predict_root}")

    run_dirs = sorted([p for p in predict_root.iterdir() if p.is_dir()])
    files: List[Path] = []
    for d in run_dirs:
        files.extend(sorted(d.glob(args.pattern)))

    if not files:
        raise SystemExit(f"No files matched {args.pattern} under {predict_root}/*")

    merged_features: List[Dict] = []
    per_run_counts: List[Tuple[str, int]] = []

    for f in files:
        run_dir = f.parent
        run_name = run_dir.name
        neighborhood = infer_neighborhood(run_dir)
        fc = load_fc(f)
        feats = fc.get("features", [])
        per_run_counts.append((run_name, len(feats)))

        for feat in feats:
            props = feat.get("properties") or {}
            props = dict(props)
            props.setdefault("neighborhood", neighborhood)
            props.setdefault("run_name", run_name)
            props.setdefault("source_file", str(f))
            feat["properties"] = props
            merged_features.append(feat)

    merged_features, dropped0 = clean_features(merged_features)

    out_raw = Path(args.out_raw)
    out_raw.parent.mkdir(parents=True, exist_ok=True)
    out_raw.write_text(json.dumps({"type": "FeatureCollection", "features": merged_features}))
    print("merged runs:", len(per_run_counts))
    print("merged features (cleaned):", len(merged_features))
    if dropped0:
        print("dropped invalid during merge:", dropped0)
    print("Wrote:", out_raw)

    dedupe_script = Path(args.dedupe_script)
    if not dedupe_script.exists():
        raise SystemExit(f"dedupe script not found: {dedupe_script}")

    out_dedup = Path(args.out_dedup)
    run_dedupe(sys.executable, dedupe_script, out_raw, out_dedup, float(args.dedupe_iou), float(args.dedupe_buffer))
    print("Wrote:", out_dedup)

    dedup_fc = load_fc(out_dedup)
    feats2 = dedup_fc.get("features", [])
    feats2, dropped2 = clean_features(feats2)

    out_final = Path(args.out_final)
    out_final.write_text(json.dumps({"type": "FeatureCollection", "features": feats2}))
    print("final features:", len(feats2))
    if dropped2:
        print("dropped invalid after global dedupe:", dropped2)
    print("Wrote:", out_final)

    per_run_counts.sort(key=lambda x: x[1], reverse=True)
    print("runs included:")
    for name, cnt in per_run_counts:
        print(f"  {name}: {cnt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
