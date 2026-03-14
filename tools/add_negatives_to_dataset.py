#!/usr/bin/env python3
import shutil
from pathlib import Path

NEG_DIR = Path("/Users/admin/sao-paulo-pool-estimator/runs/review/hard_negatives_batch_auto/negs/neg")
DATASET = Path("/Users/admin/sao-paulo-pool-estimator/data/datasets/geosampa_master_2020_with_reviewed_empties_v1_blacklist_pruned")

IMG_DST = DATASET / "images/train"
LBL_DST = DATASET / "labels/train"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}

IMG_DST.mkdir(parents=True, exist_ok=True)
LBL_DST.mkdir(parents=True, exist_ok=True)

added = 0
skipped_existing = 0

for img in sorted(NEG_DIR.iterdir()):
    if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
        continue

    stem = img.stem
    dst_img = IMG_DST / img.name
    dst_lbl = LBL_DST / f"{stem}.txt"

    if dst_img.exists() or dst_lbl.exists():
        skipped_existing += 1
        continue

    shutil.copy2(img, dst_img)
    dst_lbl.write_text("", encoding="utf-8")
    added += 1

print("Added negatives:", added)
print("Skipped existing:", skipped_existing)