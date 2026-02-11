from ultralytics import YOLO
import cv2
import numpy as np
import json
import mercantile
from shapely.geometry import Polygon, mapping

model = YOLO("runs/segment/sao-paulo-pools-seg-v4/weights/best.pt")

def pixel_to_latlon(tile_x, tile_y, zoom, px, py):
    bounds = mercantile.bounds(tile_x, tile_y, zoom)
    lon = bounds.west + (px / 256) * (bounds.east - bounds.west)
    lat = bounds.north - (py / 256) * (bounds.north - bounds.south)
    return lon, lat

features = []

for tile in tiles:
    img = cv2.imread(tile.path)
    results = model(img)

    for mask in results[0].masks.xy:
        coords = []
        for x, y in mask:
            lon, lat = pixel_to_latlon(tile.x, tile.y, tile.z, x, y)
            coords.append((lon, lat))

        polygon = Polygon(coords)
        features.append({
            "type": "Feature",
            "geometry": mapping(polygon),
            "properties": {
                "confidence": float(results[0].boxes.conf[0])
            }
        })

geojson = {
    "type": "FeatureCollection",
    "features": features
}

with open("pools.geojson", "w") as f:
    json.dump(geojson, f)
