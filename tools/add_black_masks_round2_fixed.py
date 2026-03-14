from pathlib import Path
import cv2
import numpy as np

default_txt = Path("data/datasets/geosampa_active_learning_round2/ImageSets/Segmentation/default.txt")
images_root = Path("runs/annotate_round2")
masks_root = Path("data/datasets/geosampa_active_learning_round2/SegmentationObject")

exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"]

scanned = 0
present = 0
created = 0
missing_images = 0

for line in default_txt.read_text(encoding="utf-8").splitlines():
    rel = line.strip()
    if not rel:
        continue

    # default.txt entries start with annotate_round2/
    rel_no_prefix = rel.removeprefix("annotate_round2/")

    stem = Path(rel_no_prefix)
    img_path = None
    for ext in exts:
        p = images_root / stem.with_suffix(ext)
        if p.exists():
            img_path = p
            break

    scanned += 1

    if img_path is None:
        print("missing image:", rel_no_prefix)
        missing_images += 1
        continue

    mask_path = masks_root / stem.with_suffix(".png")

    if mask_path.exists():
        present += 1
        continue

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        print("could not read image:", img_path)
        missing_images += 1
        continue

    h, w = img.shape[:2]
    blank = np.zeros((h, w), dtype=np.uint8)

    mask_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask_path), blank)
    created += 1

print("images scanned:", scanned)
print("masks already present:", present)
print("black masks created:", created)
print("missing images:", missing_images)
