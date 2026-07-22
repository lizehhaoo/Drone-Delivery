from __future__ import annotations

import contextlib
import io
import json
import math
import re
import shutil
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import payload_drone_backend_interactive_fixed_only as backend
from payload_drone_backend_interactive_fixed_only import SimulationConfig


BASE_DIR = Path(__file__).resolve().parent
CANDIDATE_CSV = BASE_DIR / "manitoba_household_candidates_ALL_points.csv"
RUNS_DIR = BASE_DIR / "interactive_runs"

# Make the backend use the data file next to this Streamlit app, even if the app
# is launched from another working directory.
backend.INPUT_ALL_POINTS_CSV = CANDIDATE_CSV


st.set_page_config(
    page_title="Manitoba Drone Delivery Interactive Model",
    page_icon="🚁",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {font-size: 2.2rem; font-weight: 800; margin-bottom: .2rem;}
    .subtle {color: #667085; font-size: .95rem;}
    .small-note {background:#f8fafc; border:1px solid #e5e7eb; padding:12px 14px; border-radius:12px; color:#475467;}
    .legend-row {display:flex; gap:10px; flex-wrap:wrap; margin: 6px 0 12px 0;}
    .legend-item {border:1px solid #e5e7eb; border-radius:10px; padding:8px 10px; background:white; font-size:.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🚁 Manitoba Drone Delivery Interactive Model</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtle">Interactive backend for random household sampling, station-origin drone routing, charger relay planning, runtime benchmarking, and permanent charger selection.</div>',
    unsafe_allow_html=True,
)




def read_csv_if_exists(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def read_text_if_exists(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def first_value(df: pd.DataFrame, col: str, default=None):
    if df.empty or col not in df.columns:
        return default
    try:
        return df.iloc[0][col]
    except Exception:
        return default


def show_map(map_path: str | Path, height: int = 760):
    map_path = Path(map_path)
    if not map_path.exists():
        st.warning("Map HTML was not found. Please check the backend output folder.")
        return
    html = map_path.read_text(encoding="utf-8", errors="ignore")
    components.html(html, height=height, scrolling=True)




def parse_household_counts(raw: str) -> List[int]:
    counts: List[int] = []
    for piece in raw.replace(";", ",").split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            value = int(piece)
        except ValueError:
            continue
        if value > 0 and value not in counts:
            counts.append(value)
    return counts or [10, 20, 40, 60, 80, 100]



def download_file_button(label: str, path: str | Path, mime: str = "text/csv"):
    path = Path(path)
    if path.exists():
        with path.open("rb") as f:
            st.download_button(
                label=f"Download {label}",
                data=f.read(),
                file_name=path.name,
                mime=mime,
                use_container_width=True,
            )




# Permanent charger optimization + final-plan map overlay helpers
def find_lat_lon_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Find latitude/longitude columns even if the backend output names change."""
    if df.empty:
        return None, None
    lowered = {str(c).lower().strip(): c for c in df.columns}
    lat_candidates = [
        "lat", "latitude", "charger_lat", "selected_lat", "recommended_lat",
        "centroid_lat", "building_lat", "y_lat",
    ]
    lon_candidates = [
        "lon", "lng", "longitude", "charger_lon", "charger_lng", "selected_lon",
        "selected_lng", "recommended_lon", "recommended_lng", "centroid_lon",
        "centroid_lng", "building_lon", "building_lng", "x_lon",
    ]
    lat_col = next((lowered[c] for c in lat_candidates if c in lowered), None)
    lon_col = next((lowered[c] for c in lon_candidates if c in lowered), None)
    if lat_col and lon_col:
        return lat_col, lon_col
    lat_like = [c for c in df.columns if "lat" in str(c).lower()]
    lon_like = [c for c in df.columns if "lon" in str(c).lower() or "lng" in str(c).lower()]
    if lat_like and lon_like:
        return lat_like[0], lon_like[0]
    return None, None




def find_score_columns(df: pd.DataFrame) -> List[str]:
    """Find columns useful for ranking permanent charger importance."""
    if df.empty:
        return []
    keywords = [
        "score", "weighted", "frequency", "scenario_count", "selected_count",
        "route_use", "actual_use", "used_count", "total_uses", "uses", "count",
    ]
    cols: List[str] = []
    for keyword in keywords:
        for col in df.columns:
            if keyword in str(col).lower() and col not in cols:
                numeric = pd.to_numeric(df[col], errors="coerce")
                if numeric.notna().any():
                    cols.append(col)
    return cols


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    d_phi = math.radians(float(lat2) - float(lat1))
    d_lambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return radius_km * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def clean_point_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str], Optional[str]]:
    lat_col, lon_col = find_lat_lon_columns(df)
    if df.empty or not lat_col or not lon_col:
        return pd.DataFrame(), None, None
    out = df.copy()
    out[lat_col] = pd.to_numeric(out[lat_col], errors="coerce")
    out[lon_col] = pd.to_numeric(out[lon_col], errors="coerce")
    out = out.dropna(subset=[lat_col, lon_col]).copy()
    return out, lat_col, lon_col



def optimize_chargers_by_min_spacing(
    candidates: pd.DataFrame,
    *,
    top_n: int,
    min_spacing_km: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Greedily keep high-score chargers while avoiding overlapping permanent chargers."""
    candidates, lat_col, lon_col = clean_point_df(candidates)
    if candidates.empty or not lat_col or not lon_col:
        return pd.DataFrame(), pd.DataFrame(), "No valid charger coordinates were found."

    df = candidates.copy()
    score_cols = find_score_columns(df)
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    if score_cols:
        df = df.sort_values(score_cols, ascending=[False] * len(score_cols), kind="mergesort").reset_index(drop=True)
        ranking_note = "ranked by " + ", ".join(score_cols)
    else:
        df = df.reset_index(drop=True)
        ranking_note = "ranked by backend output order"

    selected_rows = []
    skipped_rows = []
    selected_coords: List[Tuple[float, float]] = []

    for _, row in df.iterrows():
        lat = float(row[lat_col])
        lon = float(row[lon_col])
        distances = [haversine_km(lat, lon, s_lat, s_lon) for s_lat, s_lon in selected_coords]
        nearest_selected = min(distances) if distances else None
        row_out = row.copy()
        row_out["minimum_spacing_rule_km"] = float(min_spacing_km)
        row_out["nearest_selected_charger_km"] = nearest_selected

        if min_spacing_km <= 0 or nearest_selected is None or nearest_selected >= float(min_spacing_km):
            row_out["optimized_rank"] = len(selected_rows) + 1
            selected_rows.append(row_out)
            selected_coords.append((lat, lon))
            if len(selected_rows) >= int(top_n):
                break
        else:
            row_out["skip_reason"] = f"Too close to a higher-ranked charger ({nearest_selected:.2f} km < {min_spacing_km:.2f} km)"
            skipped_rows.append(row_out)

    selected_df = pd.DataFrame(selected_rows)
    skipped_df = pd.DataFrame(skipped_rows)
    if len(selected_df) < int(top_n):
        message = (
            f"Selected {len(selected_df)} optimized permanent chargers instead of {int(top_n)} because "
            f"the {min_spacing_km:.2f} km spacing rule was strict. Candidates were {ranking_note}."
        )
    else:
        message = f"Selected {len(selected_df)} optimized permanent chargers with at least {min_spacing_km:.2f} km spacing; {ranking_note}."
    return selected_df, skipped_df, message



def write_optimized_permanent_charger_map(
    selected_df: pd.DataFrame,
    skipped_df: pd.DataFrame,
    *,
    output_path: Path,
    min_spacing_km: float,
) -> bool:
    """Create a simple map for the optimized permanent charger study result."""
    selected_df, lat_col, lon_col = clean_point_df(selected_df)
    if selected_df.empty or not lat_col or not lon_col:
        return False
    try:
        import folium
        from folium.plugins import MarkerCluster
    except Exception:
        return False

    fmap = folium.Map(
        location=[float(selected_df[lat_col].mean()), float(selected_df[lon_col].mean())],
        zoom_start=6,
        tiles="CartoDB positron",
    )

    selected_layer = folium.FeatureGroup(name="Optimized fixed permanent chargers - GREEN", show=True)
    for _, row in selected_df.iterrows():
        popup_lines = ["<b>Optimized fixed permanent charger</b>"]
        if "optimized_rank" in row:
            popup_lines.append(f"rank: {row.get('optimized_rank', '')}")
        for col in find_score_columns(selected_df)[:5]:
            popup_lines.append(f"{col}: {row.get(col, '')}")
        folium.CircleMarker(
            location=[float(row[lat_col]), float(row[lon_col])],
            radius=7,
            color="green",
            fill=True,
            fill_color="green",
            fill_opacity=0.9,
            popup=folium.Popup("<br>".join(popup_lines), max_width=360),
        ).add_to(selected_layer)
        if min_spacing_km > 0:
            folium.Circle(
                location=[float(row[lat_col]), float(row[lon_col])],
                radius=float(min_spacing_km) * 1000,
                color="green",
                weight=1,
                opacity=0.22,
                fill=False,
            ).add_to(selected_layer)
    selected_layer.add_to(fmap)

    skipped_df, sk_lat, sk_lon = clean_point_df(skipped_df)
    if not skipped_df.empty and sk_lat and sk_lon:
        skipped_layer = folium.FeatureGroup(name="Skipped: too close to selected permanent chargers", show=False)
        cluster = MarkerCluster(name="Skipped close candidates").add_to(skipped_layer)
        for _, row in skipped_df.iterrows():
            folium.CircleMarker(
                location=[float(row[sk_lat]), float(row[sk_lon])],
                radius=4,
                color="gray",
                fill=True,
                fill_color="gray",
                fill_opacity=0.55,
                popup=folium.Popup(str(row.get("skip_reason", "Skipped by spacing rule")), max_width=360),
            ).add_to(cluster)
        skipped_layer.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(output_path))
    return True


def optimize_permanent_charger_outputs(study_outputs: dict, *, top_n_chargers: int, min_spacing_km: float) -> dict:
    """Add optimized permanent charger CSV/map files to backend permanent-study outputs."""
    output_dir = Path(study_outputs.get("output_dir", Path(study_outputs.get("recommended_csv", ".")).parent))
    output_dir.mkdir(parents=True, exist_ok=True)
    frequency_csv = Path(study_outputs.get("frequency_csv", ""))
    recommended_csv = Path(study_outputs.get("recommended_csv", ""))
    candidates = read_csv_if_exists(frequency_csv) if frequency_csv.exists() else read_csv_if_exists(recommended_csv)

    optimized_df, skipped_df, message = optimize_chargers_by_min_spacing(
        candidates,
        top_n=int(top_n_chargers),
        min_spacing_km=float(min_spacing_km),
    )

    optimized_csv = output_dir / "recommended_permanent_chargers_optimized.csv"
    skipped_csv = output_dir / "permanent_chargers_skipped_too_close.csv"
    summary_csv = output_dir / "permanent_charger_spacing_summary.csv"
    optimized_map = output_dir / "permanent_charger_frequency_map_optimized.html"

    optimized_df.to_csv(optimized_csv, index=False)
    skipped_df.to_csv(skipped_csv, index=False)
    map_created = write_optimized_permanent_charger_map(
        optimized_df,
        skipped_df,
        output_path=optimized_map,
        min_spacing_km=float(min_spacing_km),
    )

    pd.DataFrame([
        {
            "requested_top_n_chargers": int(top_n_chargers),
            "optimized_selected_chargers": int(len(optimized_df)),
            "skipped_too_close_chargers": int(len(skipped_df)),
            "minimum_spacing_km": float(min_spacing_km),
            "optimized_map_created": bool(map_created),
            "message": message,
        }
    ]).to_csv(summary_csv, index=False)

    out = dict(study_outputs)
    out["optimized_recommended_csv"] = str(optimized_csv)
    out["optimized_skipped_csv"] = str(skipped_csv)
    out["optimized_spacing_summary_csv"] = str(summary_csv)
    out["optimized_frequency_map_html"] = str(optimized_map) if map_created else out.get("frequency_map_html", "")
    out["optimized_message"] = message
    return out



def find_latest_permanent_charger_csv(permanent_outputs: Optional[dict] = None) -> Optional[Path]:
    """Find the optimized permanent charger CSV from this session or previous runs."""
    candidates: List[Path] = []
    if permanent_outputs:
        for key in ["optimized_recommended_csv", "recommended_csv"]:
            value = permanent_outputs.get(key)
            if value:
                candidates.append(Path(value))
    candidates.extend([
        BASE_DIR / "recommended_permanent_chargers_optimized.csv",
        BASE_DIR / "recommended_permanent_chargers.csv",
    ])
    for p in candidates:
        if p.exists():
            return p
    if RUNS_DIR.exists():
        matches = list(RUNS_DIR.rglob("recommended_permanent_chargers_optimized.csv"))
        matches += list(RUNS_DIR.rglob("recommended_permanent_chargers.csv"))
        if matches:
            return max(matches, key=lambda p: p.stat().st_mtime)
    return None




def detect_folium_map_variable(html: str) -> Optional[str]:
    """Find the Leaflet/Folium map variable name inside backend-generated map HTML."""
    patterns = [
        r"var\s+(map_[A-Za-z0-9_]+)\s*=\s*L\.map\(",
        r"(map_[A-Za-z0-9_]+)\s*=\s*L\.map\(",
    ]
    for pattern in patterns:
        match = re.search(pattern, html)
        if match:
            return match.group(1)
    return None




def make_permanent_charger_js(
    permanent_df: pd.DataFrame,
    *,
    map_var: str,
    drone_range_km: float,
    show_coverage_circles: bool,
) -> str:
    """Create a JavaScript overlay so the final plan uses the exact backend routing map type."""
    df, lat_col, lon_col = clean_point_df(permanent_df)
    if df.empty or not lat_col or not lon_col:
        return ""

    score_cols = find_score_columns(df)[:4]
    points = []
    for idx, row in df.reset_index(drop=True).iterrows():
        popup_lines = ["<b>Fixed permanent charger</b>", "Type: fixed permanent"]
        if "optimized_rank" in row and pd.notna(row.get("optimized_rank")):
            popup_lines.append(f"Rank: {row.get('optimized_rank')}")
        else:
            popup_lines.append(f"Number: {idx + 1}")
        for col in score_cols:
            popup_lines.append(f"{col}: {row.get(col, '')}")
        points.append({
            "lat": float(row[lat_col]),
            "lon": float(row[lon_col]),
            "popup": "<br>".join(str(x) for x in popup_lines),
        })

    points_json = json.dumps(points, ensure_ascii=False)
    range_m = float(drone_range_km) * 1000.0
    show_circles_js = "true" if show_coverage_circles else "false"

    return f"""
<script>
(function() {{
  var fixedPermanentLayer = L.layerGroup();
  var fixedPermanentCircleLayer = L.layerGroup();
  var fixedPermanentPoints = {points_json};

  fixedPermanentPoints.forEach(function(p) {{
    L.circleMarker([p.lat, p.lon], {{
      radius: 7,
      color: "green",
      fillColor: "green",
      fillOpacity: 0.9,
      weight: 2
    }}).bindPopup(p.popup).addTo(fixedPermanentLayer);

    if ({show_circles_js}) {{
      L.circle([p.lat, p.lon], {{
        radius: {range_m},
        color: "green",
        weight: 1,
        opacity: 0.25,
        fill: false
      }}).addTo(fixedPermanentCircleLayer);
    }}
  }});

  fixedPermanentLayer.addTo({map_var});
  if ({show_circles_js}) {{
    fixedPermanentCircleLayer.addTo({map_var});
  }}

  L.control.layers(null, {{
    "Fixed permanent chargers - GREEN": fixedPermanentLayer,
    "Fixed charger coverage circles": fixedPermanentCircleLayer
  }}, {{collapsed: false}}).addTo({map_var});
}})();
</script>
"""



def inject_permanent_chargers_into_backend_map(
    *,
    backend_map_html: Path,
    permanent_csv: Path,
    output_path: Path,
    drone_range_km: float,
    show_coverage_circles: bool,
) -> bool:
    """
    Reuse the exact map generated by backend.run_model, then overlay green fixed permanent chargers.

    This keeps the final plan result visually the same type as "Run one routing model":
    same route lines, same households, same stations, same route popups, and the same base-map style.
    """
    if not backend_map_html.exists() or not permanent_csv.exists():
        return False
    html = backend_map_html.read_text(encoding="utf-8", errors="ignore")
    map_var = detect_folium_map_variable(html)
    if not map_var:
        shutil.copyfile(backend_map_html, output_path)
        return False

    permanent_df = read_csv_if_exists(permanent_csv)
    overlay_js = make_permanent_charger_js(
        permanent_df,
        map_var=map_var,
        drone_range_km=float(drone_range_km),
        show_coverage_circles=bool(show_coverage_circles),
    )
    if not overlay_js:
        shutil.copyfile(backend_map_html, output_path)
        return False

    if "</body>" in html:
        html = html.replace("</body>", overlay_js + "\n</body>", 1)
    else:
        html = html + overlay_js

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return True



def build_same_style_final_plan_outputs(
    *,
    routing_outputs: dict,
    permanent_outputs: Optional[dict],
    output_dir: Path,
    drone_range_km: float,
    charger_unit_cost_cad: float,
    show_coverage_circles: bool,
) -> dict:
    """Final plan uses the backend's normal route map and adds the permanent charger layer."""
    output_dir.mkdir(parents=True, exist_ok=True)
    permanent_csv = find_latest_permanent_charger_csv(permanent_outputs)
    if not permanent_csv or not permanent_csv.exists():
        raise FileNotFoundError(
            "No permanent charger CSV was found. Run 'Permanent charger study' first, "
            "or put recommended_permanent_chargers_optimized.csv in the same folder as this app."
        )

    permanent_df = read_csv_if_exists(permanent_csv)
    permanent_clean, _, _ = clean_point_df(permanent_df)
    status_df = read_csv_if_exists(routing_outputs.get("status_csv", ""))
    routes_df = read_csv_if_exists(routing_outputs.get("routes_csv", ""))
    legs_df = read_csv_if_exists(routing_outputs.get("legs_csv", ""))
    chargers_df = read_csv_if_exists(routing_outputs.get("chargers_csv", ""))
    summary_df = read_csv_if_exists(routing_outputs.get("summary_csv", ""))

    fixed_csv = output_dir / "final_fixed_permanent_chargers.csv"
    empty_temp_csv = output_dir / "final_added_temporary_chargers.csv"
    all_csv = output_dir / "final_all_chargers.csv"
    summary_csv = output_dir / "final_plan_summary.csv"
    routes_csv = output_dir / "final_routes.csv"
    legs_csv = output_dir / "final_route_legs.csv"
    status_csv = output_dir / "final_household_status.csv"
    route_chargers_csv = output_dir / "final_backend_route_selected_chargers.csv"
    final_map_html = output_dir / "final_plan_route_map_same_as_run_one.html"

    permanent_clean.to_csv(fixed_csv, index=False)
    pd.DataFrame().to_csv(empty_temp_csv, index=False)
    permanent_clean.to_csv(all_csv, index=False)
    routes_df.to_csv(routes_csv, index=False)
    legs_df.to_csv(legs_csv, index=False)
    status_df.to_csv(status_csv, index=False)
    chargers_df.to_csv(route_chargers_csv, index=False)

    map_created = inject_permanent_chargers_into_backend_map(
        backend_map_html=Path(routing_outputs.get("map_html", "")),
        permanent_csv=permanent_csv,
        output_path=final_map_html,
        drone_range_km=float(drone_range_km),
        show_coverage_circles=bool(show_coverage_circles),
    )

    selected_households = first_value(summary_df, "selected_households", len(status_df)) if not summary_df.empty else len(status_df)
    served_households = first_value(summary_df, "served_households", None) if not summary_df.empty else None
    if served_households is None and not status_df.empty and "service_status" in status_df.columns:
        served_households = int((status_df["service_status"] == "served").sum())
    unserved_households = first_value(summary_df, "unserved_households", None) if not summary_df.empty else None
    if unserved_households is None and not status_df.empty and "service_status" in status_df.columns:
        unserved_households = int((status_df["service_status"] == "unserved").sum())
    drone_routes = first_value(summary_df, "drone_routes", len(routes_df)) if not summary_df.empty else len(routes_df)
    total_distance = first_value(summary_df, "total_route_distance_km", None) if not summary_df.empty else None
    if total_distance is None and not legs_df.empty and "distance_km" in legs_df.columns:
        total_distance = float(pd.to_numeric(legs_df["distance_km"], errors="coerce").fillna(0).sum())

    pd.DataFrame([
        {
            "selected_households": selected_households,
            "served_households": served_households,
            "unserved_households": unserved_households,
            "drone_routes": drone_routes,
            "total_route_distance_km": total_distance,
            "fixed_permanent_chargers_green": int(len(permanent_clean)),
            "new_temporary_chargers_blue": 0,
            "backend_route_selected_relay_chargers": int(len(chargers_df)),
            "charger_unit_cost_cad": float(charger_unit_cost_cad),
            "estimated_fixed_permanent_charger_asset_cost_cad": float(len(permanent_clean)) * float(charger_unit_cost_cad),
            "permanent_charger_csv_used": str(permanent_csv),
            "backend_routing_map_used": str(routing_outputs.get("map_html", "")),
            "final_map_is_same_type_as_run_one_routing_model": bool(map_created),
            "method_note": "This final map reuses the backend map generated by run_model, then overlays fixed permanent chargers in green. No blue temporary chargers are added in this version.",
        }
    ]).to_csv(summary_csv, index=False)

    return {
        "output_dir": str(output_dir),
        "map_html": str(final_map_html),
        "summary_csv": str(summary_csv),
        "fixed_permanent_csv": str(fixed_csv),
        "temporary_chargers_csv": str(empty_temp_csv),
        "all_chargers_csv": str(all_csv),
        "routes_csv": str(routes_csv),
        "legs_csv": str(legs_csv),
        "status_csv": str(status_csv),
        "route_chargers_csv": str(route_chargers_csv),
        "permanent_charger_csv_used": str(permanent_csv),
        "map_created": bool(map_created),
    }


with st.sidebar:
    st.header("⚙️ Simulation Settings")

    model_names = list(backend.DRONE_MODEL_CATALOG.keys())
    default_model = "DJI FlyCart 30"
    model_name = st.selectbox(
        "Drone model",
        model_names,
        index=model_names.index(default_model) if default_model in model_names else 0,
    )
    model_info = backend.DRONE_MODEL_CATALOG[model_name]
    st.caption(model_info.get("note", ""))

    use_catalog_defaults = st.checkbox("Use selected drone model range/payload defaults", value=True)

    if use_catalog_defaults:
        default_range = float(model_info["range_km"])
        default_payload = float(model_info["payload_kg"])
        default_speed = float(model_info["speed_kmh"])
    else:
        default_range = 40.0
        default_payload = 5.0
        default_speed = float(model_info.get("speed_kmh", 80.0))

    number_of_households = st.slider("Number of random households", 10, 100, 40, 1)
    random_seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)
    number_of_agents = st.slider("Number of drones", 1, 50, 15, 1)

    drone_range_km = st.slider("Drone range / battery cycle (km)", 10.0, 220.0, float(default_range), 1.0)
    drone_payload_kg = st.slider("Drone payload capacity (kg)", 1.0, 100.0, float(default_payload), 0.1)
    drone_speed_kmh = st.slider("Drone speed (km/h)", 20.0, 140.0, float(default_speed), 1.0)

    st.divider()
    st.subheader("Food demand")
    food_kg_min = st.slider("Minimum food demand per household (kg)", 0.5, 10.0, 1.0, 0.1)
    food_kg_max = st.slider("Maximum food demand per household (kg)", 0.5, 15.0, 5.0, 0.1)
    if food_kg_min > food_kg_max:
        st.warning("Minimum food demand is larger than maximum. The backend will automatically swap them.")

    max_households_per_route = st.slider(
        "Max households per drone route",
        1,
        2,
        2,
        1,
        help="This backend currently supports one-household or two-household route merging.",
    )

    st.divider()
    st.subheader("Charging network")
    max_selected_chargers = st.slider("Max selected relay chargers", 10, 150, 90, 1)
    charger_unit_cost = st.number_input("Charger unit cost (CAD)", min_value=0.0, value=5000.0, step=500.0)
    show_charger_circles = st.checkbox("Show charger coverage circles", value=True)
    use_overpass = st.checkbox("Use OSM Overpass fallback if local buildings cannot fill gaps", value=False)

    st.divider()
    run_button = st.button("🚀 Run one routing model", type="primary", use_container_width=True)

    with st.expander("📊  Model evaluation experiments", expanded=False):
        st.markdown("**Runtime graph**")
        runtime_counts_raw = st.text_input("Household counts", value="10,20,40,60,80,100")
        runtime_seed_start = st.number_input("Benchmark seed start", min_value=0, max_value=999999, value=42, step=1)
        runtime_repeats = st.slider("Repeats per household count", 1, 5, 1, 1)
        runtime_include_map = st.checkbox("Include map drawing time", value=False)
        run_runtime_button = st.button("⏱️ Run runtime benchmark", use_container_width=True)

        st.markdown("**Permanent charger study**")
        permanent_scenarios = st.slider("Number of random scenarios", 2, 20, 5, 1)
        permanent_seed_start = st.number_input("Permanent-study seed start", min_value=0, max_value=999999, value=1000, step=1)
        permanent_top_n = st.slider("Recommended permanent chargers", 5, 80, 30, 1)
        permanent_min_spacing_km = st.slider(
            "Minimum distance between recommended permanent chargers (km)",
            0.0,
            40.0,
            5.0,
            0.5,
            help="Use this to prevent overlapping permanent chargers. Try 5–10 km first.",
        )
        run_permanent_button = st.button("📍 Run permanent charger study", use_container_width=True)

    st.divider()
    st.subheader("✅ Final plan with fixed permanent chargers")
    st.caption(
        "Uses the same drone model and routing parameters above, but only the fixed permanent chargers are available. "
        "The backend will not add blue temporary chargers, connector chargers, or rescue chargers."
    )
    run_final_plan_button = st.button("✅ Run final plan with fixed permanent chargers", use_container_width=True)




def make_config(
    output_dir: Path,
    *,
    generate_map: bool = True,
    use_fixed_permanent_chargers: bool = False,
    fixed_permanent_chargers_csv: str | Path = "",
    fixed_permanent_chargers_only: bool = False,
) -> SimulationConfig:
    return SimulationConfig(
        drone_model_name=model_name,
        drone_range_km=float(drone_range_km),
        drone_payload_kg=float(drone_payload_kg),
        drone_speed_kmh=float(drone_speed_kmh),
        number_of_agents=int(number_of_agents),
        number_of_households=int(number_of_households),
        random_seed=int(random_seed),
        food_kg_min=float(food_kg_min),
        food_kg_max=float(food_kg_max),
        max_households_per_route=int(max_households_per_route),
        max_selected_chargers=int(max_selected_chargers),
        charger_unit_cost_cad=float(charger_unit_cost),
        show_charger_coverage_circles=bool(show_charger_circles),
        use_overpass_if_local_fails=bool(use_overpass),
        use_fixed_permanent_chargers=bool(use_fixed_permanent_chargers),
        fixed_permanent_chargers_csv=str(fixed_permanent_chargers_csv or ""),
        fixed_permanent_chargers_only=bool(fixed_permanent_chargers_only),
        generate_map=bool(generate_map),
        output_dir=str(output_dir),
    )


st.markdown("### Required local data")
if CANDIDATE_CSV.exists():
    try:
        candidate_count = sum(1 for _ in CANDIDATE_CSV.open("r", encoding="utf-8", errors="ignore")) - 1
    except Exception:
        candidate_count = "available"
    st.success(
        f"Found {CANDIDATE_CSV.name} ({candidate_count:,} points)"
        if isinstance(candidate_count, int)
        else f"Found {CANDIDATE_CSV.name}"
    )
else:
    st.error(f"Missing {CANDIDATE_CSV.name}")
    st.markdown(
        "Put `manitoba_household_candidates_ALL_points.csv` in the same folder as this app. "
        "If you only have the Microsoft building GeoJSON, run `python plot_map.py` first to generate the CSV."
    )

RUNS_DIR.mkdir(exist_ok=True)

if run_button:
    if not CANDIDATE_CSV.exists():
        st.stop()

    run_id = time.strftime("%Y%m%d_%H%M%S")
    outdir = RUNS_DIR / f"run_{run_id}_H{number_of_households}_R{int(drone_range_km)}_seed{int(random_seed)}"
    config = make_config(outdir, generate_map=True)

    progress = st.progress(0, text="Running backend model...")
    stdout = io.StringIO()

    try:
        with st.spinner("Recomputing households, chargers, routes, and map..."):
            progress.progress(10, text="Backend started")
            with contextlib.redirect_stdout(stdout):
                outputs = backend.run_model(config)
            progress.progress(100, text="Done")
        st.session_state["last_outputs"] = outputs
        st.session_state["last_config"] = config
        st.session_state["last_log"] = stdout.getvalue()
        st.success("Interactive model run completed.")
    except Exception as exc:
        st.session_state["last_log"] = stdout.getvalue()
        st.error(f"Backend run failed: {exc}")
        with st.expander("Backend log before error", expanded=True):
            st.code(st.session_state.get("last_log", ""))
        st.stop()

if run_runtime_button:
    if not CANDIDATE_CSV.exists():
        st.stop()

    counts = parse_household_counts(runtime_counts_raw)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    outdir = RUNS_DIR / f"runtime_benchmark_{run_id}"
    config = make_config(outdir, generate_map=False)
    stdout = io.StringIO()

    try:
        with st.spinner("Running runtime benchmark. This may take several minutes..."):
            with contextlib.redirect_stdout(stdout):
                benchmark_outputs = backend.run_runtime_benchmark(
                    config=config,
                    household_counts=counts,
                    seeds_per_count=int(runtime_repeats),
                    seed_start=int(runtime_seed_start),
                    output_dir=outdir,
                    include_map_drawing=bool(runtime_include_map),
                )
        st.session_state["benchmark_outputs"] = benchmark_outputs
        st.session_state["benchmark_log"] = stdout.getvalue()
        st.success("Runtime benchmark completed.")
    except Exception as exc:
        st.session_state["benchmark_log"] = stdout.getvalue()
        st.error(f"Runtime benchmark failed: {exc}")
        with st.expander("Benchmark log before error", expanded=True):
            st.code(st.session_state.get("benchmark_log", ""))
        st.stop()

if run_permanent_button:
    if not CANDIDATE_CSV.exists():
        st.stop()

    run_id = time.strftime("%Y%m%d_%H%M%S")
    outdir = RUNS_DIR / f"permanent_charger_study_{run_id}"
    config = make_config(outdir, generate_map=False)
    stdout = io.StringIO()

    try:
        with st.spinner("Running multiple random scenarios for permanent charger selection..."):
            with contextlib.redirect_stdout(stdout):
                study_outputs = backend.run_permanent_charger_study(
                    config=config,
                    num_scenarios=int(permanent_scenarios),
                    seed_start=int(permanent_seed_start),
                    top_n_chargers=int(permanent_top_n),
                    output_dir=outdir,
                )
        study_outputs = optimize_permanent_charger_outputs(
            study_outputs,
            top_n_chargers=int(permanent_top_n),
            min_spacing_km=float(permanent_min_spacing_km),
        )
        st.session_state["permanent_outputs"] = study_outputs
        st.session_state["permanent_log"] = stdout.getvalue()
        st.success("Permanent charger study completed and optimized permanent chargers were generated.")
    except Exception as exc:
        st.session_state["permanent_log"] = stdout.getvalue()
        st.error(f"Permanent charger study failed: {exc}")
        with st.expander("Permanent charger study log before error", expanded=True):
            st.code(st.session_state.get("permanent_log", ""))
        st.stop()


if run_final_plan_button:
    if not CANDIDATE_CSV.exists():
        st.stop()

    permanent_csv = find_latest_permanent_charger_csv(st.session_state.get("permanent_outputs"))
    if not permanent_csv:
        st.error("Please run Permanent charger study first, or put recommended_permanent_chargers_optimized.csv in the same folder as this app.")
        st.stop()

    run_id = time.strftime("%Y%m%d_%H%M%S")
    outdir = RUNS_DIR / f"final_plan_fixed_only_{run_id}_H{number_of_households}_R{int(drone_range_km)}_seed{int(random_seed)}"
    config = make_config(
        outdir,
        generate_map=True,
        use_fixed_permanent_chargers=True,
        fixed_permanent_chargers_csv=permanent_csv,
        fixed_permanent_chargers_only=True,
    )
    stdout = io.StringIO()

    try:
        with st.spinner("Running final plan using ONLY fixed permanent chargers..."):
            with contextlib.redirect_stdout(stdout):
                final_outputs = backend.run_model(config)

        # Alias the backend outputs into final-plan names. The backend map itself
        # is the same type/code path as Run one routing model; the only difference
        # is that the charging network is loaded from the fixed permanent CSV.
        final_outputs = dict(final_outputs)
        final_output_dir = Path(final_outputs.get("output_dir", outdir))
        final_outputs["fixed_permanent_csv_used"] = str(permanent_csv)
        final_outputs["fixed_permanent_csv"] = final_outputs.get("chargers_csv", "")
        final_outputs["temporary_chargers_csv"] = ""
        final_outputs["diagnostics_csv"] = str(final_output_dir / "unserved_household_diagnostics_after_rescue.csv")
        final_outputs["diagnostics_before_rescue_csv"] = str(final_output_dir / "unserved_household_diagnostics_before_rescue.csv")
        final_outputs["sampling_diagnostics_csv"] = str(final_output_dir / "sampling_diagnostics.csv")
        final_outputs["base_graph_diagnostics_csv"] = str(final_output_dir / "charging_base_graph_diagnostics.csv")

        st.session_state["final_plan_outputs"] = final_outputs
        st.session_state["final_plan_log"] = stdout.getvalue()
        st.success("Final plan completed. This run used ONLY the fixed permanent chargers; no blue temporary chargers were added.")
    except Exception as exc:
        st.session_state["final_plan_log"] = stdout.getvalue()
        st.error(f"Final plan failed: {exc}")
        with st.expander("Final plan log before error", expanded=True):
            st.code(st.session_state.get("final_plan_log", ""), language="text")
        st.stop()


outputs = st.session_state.get("last_outputs")
benchmark_outputs = st.session_state.get("benchmark_outputs")
permanent_outputs = st.session_state.get("permanent_outputs")
final_plan_outputs = st.session_state.get("final_plan_outputs")


if final_plan_outputs:
    st.markdown("---")
    st.markdown("## ✅ Final plan result map")
    st.caption(
        "This is intentionally the same map type as Run one routing model. "
        "The backend itself uses ONLY the fixed permanent charger CSV as the charger network. "
        "No blue temporary chargers, connector chargers, or rescue chargers are added."
    )

    final_summary = read_csv_if_exists(final_plan_outputs.get("summary_csv", ""))
    if not final_summary.empty:
        f1, f2, f3, f4, f5, f6 = st.columns(6)
        f1.metric("Selected households", first_value(final_summary, "selected_households", ""))
        f2.metric("Served", first_value(final_summary, "served_households", ""))
        f3.metric("Unserved", first_value(final_summary, "unserved_households", ""))
        f4.metric("Drone routes", first_value(final_summary, "drone_routes", ""))
        f5.metric("Fixed chargers", first_value(final_summary, "selected_relay_chargers", ""))
        f6.metric("Blue temporary", first_value(final_summary, "blue_temporary_chargers_added", 0))

    st.markdown("### Map Legend")

    st.markdown(
        """
        <div class="legend-row">

          <div class="legend-item">
            🟢 Fixed permanent charger used in the final plan
          </div>

          <div class="legend-item">
            🔵 Train station and charging base
          </div>

          <div class="legend-item">
            🟠 Household or delivery route stop
          </div>

          <div class="legend-item">
            <span style="
                display:inline-block;
                width:18px;
                height:18px;
                border:2px solid #2563eb;
                border-radius:50%;
                margin-right:6px;
                vertical-align:middle;
            "></span>
            Operating range around a train station
          </div>

          <div class="legend-item">
            <span style="
                display:inline-block;
                width:18px;
                height:18px;
                border:2px solid #16a34a;
                border-radius:50%;
                margin-right:6px;
                vertical-align:middle;
            "></span>
            Operating range around a fixed permanent charger
          </div>

          <div class="legend-item">
            <span style="
                display:inline-block;
                width:26px;
                border-top:3px solid #111827;
                margin-right:6px;
                vertical-align:middle;
            "></span>
            Railway corridor
          </div>

          <div class="legend-item">
            <span style="
                display:inline-block;
                width:18px;
                height:18px;
                border:2px solid #6b7280;
                border-radius:50%;
                margin-right:6px;
                vertical-align:middle;
            "></span>
            300 km study-area buffer around the railway network
          </div>

          <div class="legend-item">
            <span style="
                display:inline-block;
                width:28px;
                border-top:4px solid;
                border-image:linear-gradient(
                    to right,
                    #dc2626,
                    #16a34a,
                    #2563eb,
                    #9333ea
                ) 1;
                margin-right:6px;
                vertical-align:middle;
            "></span>
            Each line color represents a different drone delivery route
          </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Use the layer control in the upper-right corner of the map to show "
        "or hide households, permanent chargers, coverage circles, railway "
        "layers, and drone route legs."
    )

    show_map(final_plan_outputs["map_html"], height=780)



    final_status = read_csv_if_exists(final_plan_outputs.get("status_csv", ""))
    final_diag = read_csv_if_exists(final_plan_outputs.get("diagnostics_csv", ""))
    if not final_status.empty:
        with st.expander("Final household service status", expanded=False):
            cols = [c for c in [
                "delivery_household_id", "service_status", "food_kg", "demand_packages",
                "nearest_station_name", "nearest_station_distance_km",
                "nearest_base_id", "nearest_base_type", "nearest_base_distance_km",
                "unserved_reason"
            ] if c in final_status.columns]
            st.dataframe(final_status[cols], use_container_width=True, height=360)

    if not final_diag.empty:
        with st.expander("Why households are served/unserved: charge-cycle diagnostics", expanded=True):
            st.caption(
                "This table is the most important check: inside one charger circle is not enough. "
                "A household is feasible only when a connected chargeable node -> household -> connected chargeable node fits within one battery cycle."
            )
            diag_cols = [c for c in [
                "delivery_household_id", "service_status", "nearest_base_id", "nearest_base_distance_km",
                "num_bases_within_40km", "looks_reachable_by_single_leg",
                "best_entry_base_id", "best_exit_base_id",
                "best_single_household_charge_cycle_km", "charge_cycle_feasible",
                "pre_path_connected_to_station", "post_path_connected_to_station", "diagnosis"
            ] if c in final_diag.columns]
            st.dataframe(final_diag[diag_cols], use_container_width=True, height=420)

    with st.expander("Final plan files", expanded=False):
        st.dataframe(final_summary, use_container_width=True, height=180)
        download_file_button("Final plan map HTML", final_plan_outputs["map_html"], "text/html")
        download_file_button("Final plan summary CSV", final_plan_outputs["summary_csv"], "text/csv")
        download_file_button("Fixed permanent chargers actually used CSV", final_plan_outputs["chargers_csv"], "text/csv")
        download_file_button("Final routes CSV", final_plan_outputs["routes_csv"], "text/csv")
        download_file_button("Final route legs CSV", final_plan_outputs["legs_csv"], "text/csv")
        download_file_button("Final household status CSV", final_plan_outputs["status_csv"], "text/csv")
        download_file_button("Final household charge-cycle diagnostics CSV", final_plan_outputs.get("diagnostics_csv", ""), "text/csv")
        download_file_button("Sampling diagnostics CSV", final_plan_outputs.get("sampling_diagnostics_csv", ""), "text/csv")
        download_file_button("Charging base graph diagnostics CSV", final_plan_outputs.get("base_graph_diagnostics_csv", ""), "text/csv")
        with st.expander("Final plan backend log"):
            st.code(st.session_state.get("final_plan_log", ""), language="text")


if not outputs and not final_plan_outputs:
    st.markdown("---")
    st.info("Choose settings on the left and press **Run one routing model**. After a run, the new map and CSV summaries will appear here.")

if outputs:
    output_dir = Path(outputs["output_dir"])
    summary = read_csv_if_exists(outputs["summary_csv"])
    routes = read_csv_if_exists(outputs["routes_csv"])
    status = read_csv_if_exists(outputs["status_csv"])
    chargers = read_csv_if_exists(outputs["chargers_csv"])
    legs = read_csv_if_exists(outputs["legs_csv"])

    st.markdown("---")
    st.markdown("## Latest routing result")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Selected households", first_value(summary, "selected_households", len(status)))
    m2.metric("Served", first_value(summary, "served_households", int((status.get("service_status", pd.Series(dtype=str)) == "served").sum()) if not status.empty else 0))
    m3.metric("Unserved", first_value(summary, "unserved_households", int((status.get("service_status", pd.Series(dtype=str)) == "unserved").sum()) if not status.empty else 0))
    m4.metric("Drone routes", first_value(summary, "drone_routes", len(routes)))
    m5.metric("Relay chargers", first_value(summary, "selected_relay_chargers", len(chargers)))
    m6.metric("Total distance km", first_value(summary, "total_route_distance_km", round(float(legs.get("distance_km", pd.Series(dtype=float)).sum()), 2) if not legs.empty else 0))

    st.caption(f"Output folder: `{output_dir}`")

    tab_map, tab_routes, tab_households, tab_cost, tab_files, tab_log = st.tabs(
        ["🗺️ Clear route map", "🚁 Routes", "🏠 Households", "💰 Cost", "📁 Output files", "🧾 Backend log"]
    )

    with tab_map:
        st.markdown("### Map Legend")

        st.markdown(
            """
            <div class="legend-row">

              <div class="legend-item">
                🔵 Train station and charging base
              </div>

              <div class="legend-item">
                🟣 Selected relay charger available to the routing model
              </div>

              <div class="legend-item">
                🟠 Active recharging stop actually visited by a drone route
              </div>

              <div class="legend-item">
                🟢 Route origin
              </div>

              <div class="legend-item">
                🔴 Final return station
              </div>

              <div class="legend-item">
                ⚪ Randomly selected household
              </div>

              <div class="legend-item">
                <span style="
                    display:inline-block;
                    width:18px;
                    height:18px;
                    border:2px solid #2563eb;
                    border-radius:50%;
                    margin-right:6px;
                    vertical-align:middle;
                "></span>
                Operating range around a train station
              </div>

              <div class="legend-item">
                <span style="
                    display:inline-block;
                    width:18px;
                    height:18px;
                    border:2px solid #a21caf;
                    border-radius:50%;
                    margin-right:6px;
                    vertical-align:middle;
                "></span>
                Operating range around a selected relay charger
              </div>

              <div class="legend-item">
                <span style="
                    display:inline-block;
                    width:26px;
                    border-top:3px solid #111827;
                    margin-right:6px;
                    vertical-align:middle;
                "></span>
                Railway corridor
              </div>

              <div class="legend-item">
                <span style="
                    display:inline-block;
                    width:18px;
                    height:18px;
                    border:2px solid #6b7280;
                    border-radius:50%;
                    margin-right:6px;
                    vertical-align:middle;
                "></span>
                300 km study-area buffer around the railway network
              </div>

              <div class="legend-item">
                <span style="
                    display:inline-block;
                    width:28px;
                    border-top:4px solid;
                    border-image:linear-gradient(
                        to right,
                        #dc2626,
                        #16a34a,
                        #2563eb,
                        #9333ea
                    ) 1;
                    margin-right:6px;
                    vertical-align:middle;
                "></span>
                Each line color represents a different drone delivery route
              </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "Open the layer control in the upper-right corner of the map to "
            "show or hide households, chargers, recharging stops, coverage "
            "circles, railway layers, and drone route legs. Click any route "
            "line to view its origin, visited households, recharging stops, "
            "return station, payload, and complete node sequence."
        )

        show_map(outputs["map_html"], height=780)
    with tab_routes:
        route_efficiency = read_csv_if_exists(output_dir / "route_efficiency_summary.csv")
        if not route_efficiency.empty:
            st.dataframe(route_efficiency, use_container_width=True, height=380)
        elif not routes.empty:
            st.dataframe(routes, use_container_width=True, height=380)
        else:
            st.warning("No route table was generated.")

    with tab_households:
        if not status.empty:
            cols = [c for c in [
                "delivery_household_id", "service_status", "food_kg", "demand_packages",
                "nearest_station_name", "nearest_station_distance_km", "nearest_base_name",
                "nearest_base_type", "nearest_base_distance_km", "unserved_reason"
            ] if c in status.columns]
            st.dataframe(status[cols], use_container_width=True, height=420)
        else:
            st.warning("No household status table was generated.")

    with tab_cost:
        selected_chargers = float(first_value(summary, "selected_relay_chargers", len(chargers)) or 0)
        unit_cost = float(first_value(summary, "charger_unit_cost_cad", charger_unit_cost) or 0)
        total_cost = float(first_value(summary, "estimated_total_charger_cost_cad", selected_chargers * unit_cost) or 0)
        c1, c2, c3 = st.columns(3)
        c1.metric("Relay chargers", f"{selected_chargers:,.0f}")
        c2.metric("Unit cost", f"${unit_cost:,.0f} CAD")
        c3.metric("Estimated charger cost", f"${total_cost:,.0f} CAD")
        st.markdown(f"Estimated charger cost = **{selected_chargers:,.0f} chargers × ${unit_cost:,.0f} CAD = ${total_cost:,.0f} CAD**")

    with tab_files:
        file_items = [
            ("Map HTML", outputs["map_html"], "text/html"),
            ("Summary CSV", outputs["summary_csv"], "text/csv"),
            ("Drone routes CSV", outputs["routes_csv"], "text/csv"),
            ("Route legs CSV", outputs["legs_csv"], "text/csv"),
            ("Household service status CSV", outputs["status_csv"], "text/csv"),
            ("Selected relay chargers CSV", outputs["chargers_csv"], "text/csv"),
            ("Route efficiency CSV", output_dir / "route_efficiency_summary.csv", "text/csv"),
            ("Coverage summary CSV", output_dir / "coverage_summary.csv", "text/csv"),
            ("Actual route recharging legs CSV", outputs["legs_csv"], "text/csv"),
        ]
        for label, path, mime in file_items:
            download_file_button(label, path, mime)

    with tab_log:
        st.code(st.session_state.get("last_log", ""), language="text")

if benchmark_outputs or permanent_outputs:
    st.markdown("---")
    st.markdown("## Model Evaluation Outputs")
    exp_tabs = []
    if benchmark_outputs:
        exp_tabs.append("⏱️ Runtime graph")
    if permanent_outputs:
        exp_tabs.append("📍 Permanent chargers")

    tabs = st.tabs(exp_tabs)
    tab_index = 0

    if benchmark_outputs:
        with tabs[tab_index]:
            tab_index += 1
            st.markdown("### Runtime graph: household count vs computation time")
            graph_path = Path(benchmark_outputs["benchmark_graph_png"])
            if graph_path.exists():
                st.image(str(graph_path), caption="Algorithm runtime as the number of randomly selected households increases")
            bench_summary = read_csv_if_exists(benchmark_outputs["benchmark_summary_csv"])
            bench_raw = read_csv_if_exists(benchmark_outputs["benchmark_csv"])
            if not bench_summary.empty:
                st.dataframe(bench_summary, use_container_width=True, height=260)
            with st.expander("Raw benchmark runs"):
                st.dataframe(bench_raw, use_container_width=True, height=260)
            download_file_button("Runtime benchmark CSV", benchmark_outputs["benchmark_csv"], "text/csv")
            download_file_button("Runtime benchmark summary CSV", benchmark_outputs["benchmark_summary_csv"], "text/csv")
            download_file_button("Runtime benchmark graph PNG", benchmark_outputs["benchmark_graph_png"], "image/png")
            with st.expander("Benchmark backend log"):
                st.code(st.session_state.get("benchmark_log", ""), language="text")

    if permanent_outputs:
        with tabs[tab_index]:
            st.markdown("### Permanent charger selection from multiple random scenarios")
            st.caption("This implements the methodology your supervisor suggested: daily households and routes change, so permanent chargers are selected by frequency and actual route use across several random scenarios.")
            rec = read_csv_if_exists(permanent_outputs.get("optimized_recommended_csv", permanent_outputs["recommended_csv"]))
            freq = read_csv_if_exists(permanent_outputs["frequency_csv"])
            if not rec.empty:
                st.markdown("#### Optimized recommended permanent chargers")
                st.dataframe(rec, use_container_width=True, height=330)
            map_to_show = permanent_outputs.get("optimized_frequency_map_html") or permanent_outputs.get("frequency_map_html")
            if map_to_show and Path(map_to_show).exists():
                show_map(map_to_show, height=680)
            with st.expander("All charger frequency results"):
                st.dataframe(freq, use_container_width=True, height=330)
            report_text = read_text_if_exists(permanent_outputs["methodology_report_md"])
            if report_text:
                with st.expander("Methodology report text"):
                    st.markdown(report_text)
            download_file_button("Optimized recommended permanent chargers CSV", permanent_outputs.get("optimized_recommended_csv", permanent_outputs["recommended_csv"]), "text/csv")
            download_file_button("Skipped too-close permanent chargers CSV", permanent_outputs.get("optimized_skipped_csv", ""), "text/csv")
            download_file_button("Permanent charger spacing summary CSV", permanent_outputs.get("optimized_spacing_summary_csv", ""), "text/csv")
            download_file_button("Original recommended permanent chargers CSV", permanent_outputs["recommended_csv"], "text/csv")
            download_file_button("Permanent charger frequency CSV", permanent_outputs["frequency_csv"], "text/csv")
            download_file_button("Permanent charger study runs CSV", permanent_outputs["runs_csv"], "text/csv")
            download_file_button("Optimized permanent charger map HTML", permanent_outputs.get("optimized_frequency_map_html", permanent_outputs["frequency_map_html"]), "text/html")
            download_file_button("Original permanent charger frequency map HTML", permanent_outputs["frequency_map_html"], "text/html")
            download_file_button("Methodology report Markdown", permanent_outputs["methodology_report_md"], "text/markdown")
            with st.expander("Permanent charger study backend log"):
                st.code(st.session_state.get("permanent_log", ""), language="text")



