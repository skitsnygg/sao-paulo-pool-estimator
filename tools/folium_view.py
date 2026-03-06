#!/usr/bin/env python3
"""
Folium viewer for Sao Paulo pool predictions with optional imagery overlay.

Key fixes:
- Handles mislabeled CRS: GeoJSON may CLAIM EPSG:4326 but actually be UTM meters (EPSG:31983).
- Always sets view: fit_bounds from overlay bounds if available, otherwise from predictions bounds.
- Writes richer viewer_meta.json including pred/overlay/view bounds and CRS decisions.

Outputs:
  runs/folium/<name>/viewer.html
  runs/folium/<name>/viewer_meta.json
  (optional) expects existing overlay artifacts if you generated them elsewhere:
     runs/folium/<name>/<name>_3857.png
     runs/folium/<name>/<name>_3857.tif
     runs/folium/<name>/<name>.vrt
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import folium
from folium.plugins import MousePosition

try:
    import geopandas as gpd
except Exception as e:
    raise SystemExit(f"Missing dependency geopandas: {e}")

try:
    from shapely.geometry import shape
except Exception:
    shape = None  # type: ignore

try:
    import rasterio
    from rasterio.warp import transform_bounds
except Exception:
    rasterio = None  # type: ignore
    transform_bounds = None  # type: ignore


Bounds4 = Tuple[float, float, float, float]  # minx, miny, maxx, maxy


def looks_like_degrees(bounds: Bounds4) -> bool:
    minx, miny, maxx, maxy = bounds
    # Lon/lat sanity
    if any(map(lambda v: v != v, bounds)):  # NaN check
        return False
    if abs(minx) > 180 or abs(maxx) > 180:
        return False
    if abs(miny) > 90 or abs(maxy) > 90:
        return False
    # Avoid trivial near-zero boxes that could be bogus defaults
    if (abs(maxx - minx) < 1e-9) and (abs(maxy - miny) < 1e-9):
        return False
    return True


def center_from_bounds(bounds_4326: Bounds4) -> Tuple[float, float]:
    minx, miny, maxx, maxy = bounds_4326
    return ((miny + maxy) / 2.0, (minx + maxx) / 2.0)


def read_bounds_from_raster(raster_path: Path) -> Optional[Bounds4]:
    """
    Reads raster bounds and returns bounds in EPSG:4326.
    Requires rasterio.
    """
    if rasterio is None or transform_bounds is None:
        return None
    if not raster_path.exists():
        return None
    try:
        with rasterio.open(raster_path) as src:
            b = src.bounds
            src_crs = src.crs
            if src_crs is None:
                return None
            out = transform_bounds(src_crs, "EPSG:4326", b.left, b.bottom, b.right, b.top, densify_pts=21)
            # transform_bounds returns (minx, miny, maxx, maxy) in target CRS
            return (float(out[0]), float(out[1]), float(out[2]), float(out[3]))
    except Exception:
        return None


def load_predictions(pred_geojson: Path, assume_crs: str) -> Tuple["gpd.GeoDataFrame", Dict[str, Any]]:
    """
    Load predictions GeoJSON and return a GeoDataFrame in EPSG:4326 plus metadata.

    Handles the critical case: file has crs EPSG:4326 but coordinates are UTM meters.
    """
    gdf = gpd.read_file(pred_geojson)

    meta: Dict[str, Any] = {}
    meta["feature_count"] = int(len(gdf))

    crs_read = None
    try:
        crs_read = gdf.crs.to_string() if gdf.crs is not None else None
    except Exception:
        crs_read = None
    meta["pred_crs_read"] = crs_read

    raw_bounds = tuple(map(float, gdf.total_bounds))  # type: ignore
    meta["pred_raw_bounds"] = list(raw_bounds)

    # Decide CRS
    pred_crs_decision = None
    crs_overridden = False

    if gdf.crs is None:
        gdf = gdf.set_crs(assume_crs, allow_override=True)
        pred_crs_decision = f"missing:set:{assume_crs}"
        crs_overridden = True
    else:
        # If it claims 4326 but bounds do not look like degrees, treat as mislabeled
        crs_str = None
        try:
            crs_str = gdf.crs.to_string()
        except Exception:
            crs_str = None

        if crs_str == "EPSG:4326" and not looks_like_degrees(raw_bounds):  # type: ignore[arg-type]
            gdf = gdf.set_crs(assume_crs, allow_override=True)
            pred_crs_decision = f"mislabeled_4326:override_to:{assume_crs}"
            crs_overridden = True
        else:
            pred_crs_decision = f"keep:{crs_str or 'unknown'}"

    # Reproject to 4326 for Folium
    gdf_4326 = gdf.to_crs("EPSG:4326")
    bounds_4326 = tuple(map(float, gdf_4326.total_bounds))  # type: ignore

    meta["pred_crs_decision"] = pred_crs_decision
    meta["crs_overridden"] = bool(crs_overridden)
    meta["pred_final_crs"] = "EPSG:4326"
    meta["pred_bounds_4326"] = list(bounds_4326)

    return gdf_4326, meta


def add_google_satellite(m: folium.Map) -> None:
    folium.TileLayer(
        tiles="https://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Satellite",
        subdomains=["mt0", "mt1", "mt2", "mt3"],
        overlay=False,
        control=True,
        max_zoom=20,
    ).add_to(m)


def add_geosampa_overlay(m: folium.Map, out_dir: Path, name: str, opacity: float) -> Optional[Bounds4]:
    """
    Adds a local ImageOverlay if the expected PNG exists.

    We try to compute bounds from:
      - viewer_meta.json (if it already contains bounds_4326)
      - <name>_3857.tif, <name>.vrt via rasterio
    """
    png = out_dir / f"{name}_3857.png"
    if not png.exists():
        return None

    bounds_4326: Optional[Bounds4] = None

    meta_path = out_dir / "viewer_meta.json"
    if meta_path.exists():
        try:
            prev = json.loads(meta_path.read_text(encoding="utf-8"))
            b = prev.get("bounds_4326") or prev.get("overlay_bounds_4326")
            if isinstance(b, (list, tuple)) and len(b) == 4:
                bounds_4326 = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        except Exception:
            pass

    if bounds_4326 is None:
        # Try tif then vrt
        tif = out_dir / f"{name}_3857.tif"
        vrt = out_dir / f"{name}.vrt"
        bounds_4326 = read_bounds_from_raster(tif) or read_bounds_from_raster(vrt)

    if bounds_4326 is None:
        return None

    minx, miny, maxx, maxy = bounds_4326
    folium.raster_layers.ImageOverlay(
        image=str(png),
        bounds=[[miny, minx], [maxy, maxx]],
        opacity=float(opacity),
        name="GeoSampa 2020",
        interactive=False,
        cross_origin=False,
        zindex=1,
    ).add_to(m)

    return bounds_4326


def add_predictions_layer(
    m: folium.Map,
    preds_4326: "gpd.GeoDataFrame",
    centroids: bool,
    popup_fields: Optional[List[str]] = None,
) -> None:
    if centroids:
        # Render points for speed
        if shape is None:
            raise SystemExit("shapely is required for centroid rendering")
        for geom in preds_4326.geometry:
            if geom is None or geom.is_empty:
                continue
            c = geom.centroid
            folium.CircleMarker(
                location=[float(c.y), float(c.x)],
                radius=3,
                weight=1,
                fill=True,
                fill_opacity=0.6,
                opacity=0.9,
                tooltip=None,
            ).add_to(m)
        return

    # Render polygons as GeoJSON
    popup = None
    if popup_fields:
        valid_popup_fields = [f for f in popup_fields if f in preds_4326.columns]
        if valid_popup_fields:
            popup = folium.GeoJsonPopup(
                fields=valid_popup_fields,
                labels=True,
                localize=True,
                max_width=900,
            )

    gj = folium.GeoJson(
        data=json.loads(preds_4326.to_json()),
        name="Pools",
        style_function=lambda _: {
            "color": "#00FFFF",
            "weight": 1,
            "fillColor": "#00FFFF",
            "fillOpacity": 0.25,
        },
        highlight_function=lambda _: {
            "color": "#FFFF00",
            "weight": 2,
            "fillOpacity": 0.35,
        },
        popup=popup,
    )
    gj.add_to(m)


def add_geojson_layer(
    m: folium.Map,
    geojson_path: Path,
    layer_name: str,
    color: str,
    weight: float,
    fill_opacity: float,
) -> None:
    if not geojson_path.exists():
        raise SystemExit(f"GeoJSON layer not found: {geojson_path}")

    gdf = gpd.read_file(geojson_path)
    if gdf.empty:
        return

    # Coverage/hole helper layers are expected in WGS84. If no CRS is present, assume 4326.
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    gdf_4326 = gdf.to_crs("EPSG:4326")

    folium.GeoJson(
        data=json.loads(gdf_4326.to_json()),
        name=layer_name,
        style_function=lambda _: {
            "color": color,
            "weight": float(weight),
            "fillColor": color,
            "fillOpacity": float(fill_opacity),
        },
    ).add_to(m)


def add_cell_coverage_layer(
    m: folium.Map,
    geojson_path: Path,
    layer_name: str,
) -> None:
    if not geojson_path.exists():
        raise SystemExit(f"Cell coverage GeoJSON not found: {geojson_path}")

    gdf = gpd.read_file(geojson_path)
    if gdf.empty:
        return

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    gdf_4326 = gdf.to_crs("EPSG:4326")

    # Normalize expected status values.
    status_colors = {
        "full": "#18A558",
        "partial": "#FF8F00",
        "empty": "#D32F2F",
    }

    tooltip_fields = [f for f in ["cell", "chip_count", "expected_chip_count", "missing_chip_count", "status"] if f in gdf_4326.columns]

    folium.GeoJson(
        data=json.loads(gdf_4326.to_json()),
        name=layer_name,
        style_function=lambda feat: {
            "color": status_colors.get(str((feat.get("properties") or {}).get("status", "partial")), "#607D8B"),
            "weight": 1.2,
            "fillColor": status_colors.get(str((feat.get("properties") or {}).get("status", "partial")), "#607D8B"),
            "fillOpacity": 0.18,
        },
        tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, labels=True) if tooltip_fields else None,
    ).add_to(m)


def add_cell_coverage_legend(m: folium.Map) -> None:
    legend_html = """
    <div style="
        position: fixed;
        bottom: 24px;
        right: 24px;
        z-index: 9999;
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid #cccccc;
        border-radius: 8px;
        padding: 10px 12px;
        font-size: 12px;
        line-height: 1.4;
    ">
      <div style="font-weight: 600; margin-bottom: 6px;">2020 Tile Coverage</div>
      <div><span style="display:inline-block;width:10px;height:10px;background:#18A558;margin-right:6px;"></span>Full cell</div>
      <div><span style="display:inline-block;width:10px;height:10px;background:#FF8F00;margin-right:6px;"></span>Partial cell</div>
      <div><span style="display:inline-block;width:10px;height:10px;background:#D32F2F;margin-right:6px;"></span>Empty cell</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--pred-geojson", required=True)
    ap.add_argument("--assume-pred-crs", required=True)
    ap.add_argument("--base", choices=["google_sat", "geosampa_png", "both"], default="google_sat")
    ap.add_argument("--centroids", action="store_true")
    ap.add_argument("--simplify-m", type=float, default=0.0)  # kept for compatibility; no-op unless you add it
    ap.add_argument("--max-features", type=int, default=0)
    ap.add_argument("--opacity", type=float, default=1.0)
    ap.add_argument("--cell-coverage-geojson", default="", help="Optional WGS84 GeoJSON with per-cell chip counts/status")
    ap.add_argument("--cell-coverage-label", default="Cell Coverage (Full/Partial/Empty)")
    ap.add_argument("--coverage-geojson", default="", help="Optional WGS84 GeoJSON for local imagery coverage footprint")
    ap.add_argument("--coverage-label", default="Local Imagery Coverage")
    ap.add_argument("--coverage-color", default="#00FF66")
    ap.add_argument("--coverage-weight", type=float, default=2.0)
    ap.add_argument("--coverage-fill-opacity", type=float, default=0.06)
    ap.add_argument("--holes-geojson", default="", help="Optional WGS84 GeoJSON for missing-tile hole polygons")
    ap.add_argument("--holes-label", default="Missing Imagery Tiles")
    ap.add_argument("--holes-color", default="#FF3B30")
    ap.add_argument("--holes-weight", type=float, default=1.0)
    ap.add_argument("--holes-fill-opacity", type=float, default=0.22)
    ap.add_argument(
        "--popup-fields",
        default="stem,conf,neighborhood,run_name",
        help="Comma-separated properties to show when clicking polygons",
    )
    args = ap.parse_args()

    name = args.name
    out_dir = Path("runs/folium") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_geojson = Path(args.pred_geojson)
    if not pred_geojson.exists():
        raise SystemExit(f"Predictions GeoJSON not found: {pred_geojson}")

    preds_4326, pred_meta = load_predictions(pred_geojson, args.assume_pred_crs)

    if args.max_features and args.max_features > 0 and len(preds_4326) > args.max_features:
        preds_4326 = preds_4326.iloc[: args.max_features].copy()
        pred_meta["feature_count"] = int(len(preds_4326))
        pred_meta["max_features_applied"] = int(args.max_features)

    # Create map with canvas for performance
    # Start at 0,0; we will fit_bounds after we determine view_bounds_4326.
    m = folium.Map(location=[0.0, 0.0], zoom_start=2, prefer_canvas=True, control_scale=True)
    MousePosition(
        position="topright",
        separator=" | ",
        prefix="Lat/Lon:",
    ).add_to(m)

    if args.base in ("google_sat", "both"):
        add_google_satellite(m)

    overlay_bounds_4326: Optional[Bounds4] = None
    if args.base in ("geosampa_png", "both"):
        overlay_bounds_4326 = add_geosampa_overlay(m, out_dir=out_dir, name=name, opacity=float(args.opacity))

    popup_fields = [s.strip() for s in str(args.popup_fields).split(",") if s.strip()]

    if str(args.cell_coverage_geojson).strip():
        add_cell_coverage_layer(
            m=m,
            geojson_path=Path(args.cell_coverage_geojson),
            layer_name=str(args.cell_coverage_label),
        )
        add_cell_coverage_legend(m)

    if str(args.coverage_geojson).strip():
        add_geojson_layer(
            m=m,
            geojson_path=Path(args.coverage_geojson),
            layer_name=str(args.coverage_label),
            color=str(args.coverage_color),
            weight=float(args.coverage_weight),
            fill_opacity=float(args.coverage_fill_opacity),
        )

    if str(args.holes_geojson).strip():
        add_geojson_layer(
            m=m,
            geojson_path=Path(args.holes_geojson),
            layer_name=str(args.holes_label),
            color=str(args.holes_color),
            weight=float(args.holes_weight),
            fill_opacity=float(args.holes_fill_opacity),
        )

    # Keep predictions on top for easier visual + click inspection.
    add_predictions_layer(
        m,
        preds_4326,
        centroids=bool(args.centroids),
        popup_fields=popup_fields,
    )

    # Decide view bounds
    pred_bounds_4326 = tuple(pred_meta["pred_bounds_4326"])  # type: ignore
    view_bounds_4326: Bounds4 = overlay_bounds_4326 if overlay_bounds_4326 is not None else (  # type: ignore
        float(pred_bounds_4326[0]),
        float(pred_bounds_4326[1]),
        float(pred_bounds_4326[2]),
        float(pred_bounds_4326[3]),
    )

    # Fit map to bounds ALWAYS
    minx, miny, maxx, maxy = view_bounds_4326
    m.fit_bounds([[miny, minx], [maxy, maxx]])

    # Add title
    title_html = f"""
    <div style="position: fixed; top: 10px; left: 10px; z-index: 9999;
                background: rgba(255,255,255,0.85); padding: 6px 10px;
                border-radius: 6px; font-size: 14px;">
      São Paulo Pool Detection — {int(pred_meta.get("feature_count", len(preds_4326)))} features
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    folium.LayerControl(collapsed=False).add_to(m)

    # Meta
    meta: Dict[str, Any] = {
        "tiles_dir": os.path.abspath(args.tiles_dir),
        "pred_geojson": os.path.abspath(str(pred_geojson)),
        "base": args.base,
        "opacity": float(args.opacity),
        "cell_coverage_geojson": os.path.abspath(args.cell_coverage_geojson) if str(args.cell_coverage_geojson).strip() else "",
        "coverage_geojson": os.path.abspath(args.coverage_geojson) if str(args.coverage_geojson).strip() else "",
        "holes_geojson": os.path.abspath(args.holes_geojson) if str(args.holes_geojson).strip() else "",
        "simplify_m": float(args.simplify_m),
        "centroids": bool(args.centroids),
        "max_features": int(args.max_features),
        "popup_fields": popup_fields,
        # Back-compat key (your old file had bounds_4326):
        "bounds_4326": list(view_bounds_4326),
        # New explicit keys:
        "overlay_bounds_4326": list(overlay_bounds_4326) if overlay_bounds_4326 is not None else None,
        "pred_bounds_4326": pred_meta.get("pred_bounds_4326"),
        "view_bounds_4326": list(view_bounds_4326),
        # CRS info:
        "pred_crs_read": pred_meta.get("pred_crs_read"),
        "pred_crs_decision": pred_meta.get("pred_crs_decision"),
        "crs_overridden": pred_meta.get("crs_overridden"),
        "pred_raw_bounds": pred_meta.get("pred_raw_bounds"),
        "feature_count": int(pred_meta.get("feature_count", len(preds_4326))),
    }

    meta_path = out_dir / "viewer_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    html_path = out_dir / "viewer.html"
    m.save(str(html_path))

    print(f"Wrote: {html_path}")
    print(f"Wrote: {meta_path}")
    print(
        "pred_crs_decision="
        f"{meta.get('pred_crs_decision')} "
        f"pred_bounds_4326={meta.get('pred_bounds_4326')} "
        f"overlay_bounds_4326={meta.get('overlay_bounds_4326')} "
        f"view_bounds_4326={meta.get('view_bounds_4326')}"
    )
    


if __name__ == "__main__":
    main()
