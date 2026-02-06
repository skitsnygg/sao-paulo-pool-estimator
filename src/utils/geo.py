from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple
import random

import mercantile
from shapely.geometry import Polygon
from shapely.ops import transform
from pyproj import Transformer


WGS84 = "EPSG:4326"
WEB_MERCATOR = "EPSG:3857"

_TRANSFORMER_4326_TO_3857 = Transformer.from_crs(WGS84, WEB_MERCATOR, always_xy=True)


@dataclass(frozen=True)
class TileSpec:
    z: int
    x: int
    y: int

    @property
    def id(self) -> str:
        return f"{self.z}_{self.x}_{self.y}"

    def to_mercantile(self) -> mercantile.Tile:
        return mercantile.Tile(self.x, self.y, self.z)


def tiles_from_bbox(bbox: List[float], zoom: int) -> List[TileSpec]:
    west, south, east, north = bbox
    tiles = mercantile.tiles(west, south, east, north, zoom)
    return [TileSpec(t.z, t.x, t.y) for t in tiles]


def sample_tiles(tiles: Iterable[TileSpec], max_tiles: Optional[int], seed: int) -> List[TileSpec]:
    tiles = list(tiles)
    if max_tiles is None or max_tiles <= 0 or len(tiles) <= max_tiles:
        return tiles
    rng = random.Random(seed)
    return rng.sample(tiles, max_tiles)


def tile_polygon(tile: TileSpec) -> Polygon:
    bounds = mercantile.bounds(tile.x, tile.y, tile.z)
    return Polygon(
        [
            (bounds.west, bounds.south),
            (bounds.east, bounds.south),
            (bounds.east, bounds.north),
            (bounds.west, bounds.north),
        ]
    )


def tile_xy_bounds(tile: TileSpec) -> mercantile.Bbox:
    return mercantile.xy_bounds(tile.x, tile.y, tile.z)


def geom_to_3857(geom):
    return transform(_TRANSFORMER_4326_TO_3857.transform, geom)


def bbox_in_tile_pixels(geom_4326, tile: TileSpec, tile_size: int) -> Optional[Tuple[float, float, float, float]]:
    geom_3857 = geom_to_3857(geom_4326)
    minx, miny, maxx, maxy = geom_3857.bounds
    tile_bounds = tile_xy_bounds(tile)

    denom_x = tile_bounds.right - tile_bounds.left
    denom_y = tile_bounds.top - tile_bounds.bottom
    if denom_x == 0 or denom_y == 0:
        return None

    x1 = (minx - tile_bounds.left) / denom_x * tile_size
    x2 = (maxx - tile_bounds.left) / denom_x * tile_size
    y1 = (tile_bounds.top - maxy) / denom_y * tile_size
    y2 = (tile_bounds.top - miny) / denom_y * tile_size

    x1 = max(0.0, min(tile_size, x1))
    x2 = max(0.0, min(tile_size, x2))
    y1 = max(0.0, min(tile_size, y1))
    y2 = max(0.0, min(tile_size, y2))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def yolo_box_from_pixels(box_px: Tuple[float, float, float, float], tile_size: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box_px
    cx = (x1 + x2) / 2.0 / tile_size
    cy = (y1 + y2) / 2.0 / tile_size
    w = (x2 - x1) / tile_size
    h = (y2 - y1) / tile_size
    return cx, cy, w, h
