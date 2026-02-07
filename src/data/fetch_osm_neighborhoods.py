from __future__ import annotations

import argparse
from pathlib import Path

import requests
from shapely.geometry import LineString, Polygon, mapping
from shapely.ops import polygonize, unary_union

from src.utils.config import load_config, project_root
from src.utils.io import ensure_dir, write_json


def _build_query(names: list[str], bbox: list[float], timeout: int) -> str:
    west, south, east, north = bbox
    parts = []
    for name in names:
        name = name.strip()
        if not name:
            continue
        for key, value in (("place", "neighbourhood"), ("boundary", "administrative")):
            parts.append(f'way["name"="{name}"]["{key}"="{value}"]({south},{west},{north},{east});')
            parts.append(f'relation["name"="{name}"]["{key}"="{value}"]({south},{west},{north},{east});')
    joined = "\n  ".join(parts)
    return (
        f"[out:json][timeout:{timeout}];\n"
        f"(\n  {joined}\n);\n"
        "out body;\n>;\n"
        "out skel qt;"
    )


def _parse_elements(elements: list[dict]) -> tuple[dict, dict, list]:
    nodes = {}
    ways = {}
    relations = []
    for element in elements:
        if element.get("type") == "node":
            nodes[element["id"]] = (element["lon"], element["lat"])
        elif element.get("type") == "way":
            ways[element["id"]] = element.get("nodes", [])
        elif element.get("type") == "relation":
            relations.append(element)
    return nodes, ways, relations


def _way_to_polygon(node_ids: list[int], nodes: dict[int, tuple[float, float]]) -> Polygon | None:
    coords = [nodes.get(node_id) for node_id in node_ids]
    coords = [coord for coord in coords if coord is not None]
    if len(coords) < 4:
        return None
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    polygon = Polygon(coords)
    if not polygon.is_valid or polygon.area == 0:
        return None
    return polygon


def _relation_to_polygons(relation: dict, ways: dict[int, list[int]], nodes: dict[int, tuple[float, float]]):
    outer_way_ids = [
        member["ref"]
        for member in relation.get("members", [])
        if member.get("type") == "way" and member.get("role") in ("outer", "")
    ]
    lines = []
    for way_id in outer_way_ids:
        node_ids = ways.get(way_id)
        if not node_ids:
            continue
        coords = [nodes.get(node_id) for node_id in node_ids]
        coords = [coord for coord in coords if coord is not None]
        if len(coords) < 2:
            continue
        lines.append(LineString(coords))

    if not lines:
        return []
    polygons = [poly for poly in polygonize(lines) if poly.is_valid and poly.area > 0]
    return polygons


def fetch_neighborhood_union(config_path: str, names: list[str], output: str) -> Path:
    config = load_config(config_path)
    bbox = config["aoi"]["bbox"]
    timeout = config["osm"].get("timeout", 180)
    endpoint = config["osm"]["endpoint"]

    query = _build_query(names, bbox, timeout)
    response = requests.post(endpoint, data={"data": query}, timeout=timeout + 30)
    response.raise_for_status()

    payload = response.json()
    elements = payload.get("elements", [])
    nodes, ways, relations = _parse_elements(elements)

    polygons = []
    for way_id, node_ids in ways.items():
        polygon = _way_to_polygon(node_ids, nodes)
        if polygon is None:
            continue
        polygons.append(polygon)

    for relation in relations:
        polygons.extend(_relation_to_polygons(relation, ways, nodes))

    if not polygons:
        raise ValueError("No neighborhood polygons found for the requested names.")

    union_geom = unary_union(polygons)
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = project_root() / output_path
    ensure_dir(output_path.parent)
    write_json(
        output_path,
        {
            "type": "FeatureCollection",
            "name": "neighborhood_union",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"names": names},
                    "geometry": mapping(union_geom),
                }
            ],
        },
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch OSM neighborhood polygons and union them.")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument(
        "--names",
        default="Jardim America,Jardim Europa,Jardim Paulista,Jardim Paulistano",
        help="Comma-separated neighborhood names.",
    )
    parser.add_argument("--output", default="data/external/jardins_union.geojson")
    args = parser.parse_args()

    names = [name.strip() for name in args.names.split(",") if name.strip()]
    output_path = fetch_neighborhood_union(args.config, names, args.output)
    print(f"Saved neighborhood union to {output_path}")


if __name__ == "__main__":
    main()
