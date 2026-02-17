from ultralytics import YOLO
from pathlib import Path
import cv2
import json
import numpy as np
from shapely.geometry import Polygon, mapping
from pyproj import Transformer

MODEL_PATH = "checkpoints/pools_moema_ft_best.pt"
TILES_DIR = Path("data/raw/geosampa_ortho/vila_mariana_2020")
OUT_PATH = "runs/predict/vila_mariana_2020_conf015/pools.geojson"
CONF = 0.15
IOU = 0.7
IMGSZ = 1024

model = YOLO(MODEL_PATH)
transformer = Transformer.from_crs("EPSG:31983", "EPSG:4326", always_xy=True)

features = []

def read_pgw(pgw_path):
    with open(pgw_path) as f:
        lines = [float(x.strip()) for x in f.readlines()]
    A, D, B, E, C, F = lines
    return A, D, B, E, C, F

for img_path in sorted(TILES_DIR.glob("*.png")):
    pgw_path = img_path.with_suffix(".pgw")
    if not pgw_path.exists():
        continue

    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    results = model.predict(
        source=img,
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        verbose=False
    )[0]

    if results.masks is None:
        continue

    A, D, B, E, C, F = read_pgw(pgw_path)

    for idx, mask in enumerate(results.masks.xy):
        coords = []
        for px, py in mask:
            X = A * px + B * py + C
            Y = D * px + E * py + F
            lon, lat = transformer.transform(X, Y)
            coords.append((lon, lat))

        if len(coords) < 3:
            continue

        poly = Polygon(coords)
        if not poly.is_valid or poly.area == 0:
            continue

        features.append({
            "type": "Feature",
            "geometry": mapping(poly),
            "properties": {
                "confidence": float(results.boxes.conf[idx])
            }
        })

geojson = {
    "type": "FeatureCollection",
    "features": features
}

Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(geojson, f)

print("Features:", len(features))
print("Wrote:", OUT_PATH)

