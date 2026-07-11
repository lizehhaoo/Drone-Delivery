from __future__ import annotations

import math
import random
import json
import time
import urllib.parse
import urllib.request
from collections import deque
import heapq
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry import Point, LineString
except ImportError as exc:
    raise ImportError("Install geopandas and shapely: pip install geopandas shapely") from exc

try:
    from pyproj import Transformer
except ImportError as exc:
    raise ImportError("Install pyproj: pip install pyproj") from exc

try:
    from scipy.spatial import cKDTree
except ImportError as exc:
    raise ImportError("Install scipy: pip install scipy") from exc

try:
    import folium
    from folium.plugins import MarkerCluster, PolyLineTextPath
except ImportError as exc:
    raise ImportError("Install folium: pip install folium") from exc



# User settings
BASE_DIR = Path(".")
INPUT_ALL_POINTS_CSV = BASE_DIR / "manitoba_household_candidates_ALL_points.csv"
OUTDIR = BASE_DIR / "out_drone_delivery_model"

PROJECTED_CRS = "EPSG:3347"
WGS84_CRS = "EPSG:4326"

RNG_SEED = 42

RAIL_BUFFER_KM = 300.0

DRONE_RANGE_KM = 40.0
DRONE_RANGE_M = DRONE_RANGE_KM * 1000.0

# Train station and charger coverage radius.
CHARGER_RADIUS_KM = 40.0
CHARGER_RADIUS_M = CHARGER_RADIUS_KM * 1000.0

# Overlap allowed.
ALLOW_CHARGER_OVERLAP = True

# Use grid points to choose building chargers.
# Smaller number = more chargers and better relay coverage, but slower.
GRID_SPACING_KM = 30.0
GRID_SPACING_M = GRID_SPACING_KM * 1000.0


# Overlap is allowed, but chargers should not be too close to each other.
MIN_SELECTED_CHARGER_SPACING_KM = 25.0
MIN_SELECTED_CHARGER_SPACING_M = MIN_SELECTED_CHARGER_SPACING_KM * 1000.0

# Stop selecting chargers after this count.
MAX_SELECTED_CHARGERS = 90

TARGET_GRID_COVERAGE_RATIO = 0.70

# Show charger circles on map. If the map is still too busy, set this to False.
SHOW_CHARGER_COVERAGE_CIRCLES = True

# Map cleanup options for clearer presentation.
SHOW_ROUTE_DIRECTION_ARROWS = True
SHOW_ROUTE_POPUPS = True
SPLIT_CHARGER_LAYERS = False

GENERATE_ROUTE_MAP = True

# if some sampled households are still unserved, try adding local rescue
# chargers around those households and rerun routing. This helps cases where a
# household visually looks reachable, but the full charger/station -> household
# -> charger/station charge cycle is not <= 40 km yet.
ENABLE_UNSERVED_HOUSEHOLD_RESCUE = True
MAX_UNSERVED_RESCUE_CHARGERS = 20
MAX_RESCUE_CANDIDATES_PER_HOUSEHOLD = 1500
RESCUE_SEARCH_RADIUS_KM = 40.0
RESCUE_SEARCH_RADIUS_M = RESCUE_SEARCH_RADIUS_KM * 1000.0
MIN_RESCUE_CHARGER_DISTANCE_FROM_HOUSEHOLD_KM = 0.5
MIN_RESCUE_CHARGER_DISTANCE_FROM_HOUSEHOLD_M = MIN_RESCUE_CHARGER_DISTANCE_FROM_HOUSEHOLD_KM * 1000.0


# every selected charger must be reachable from the current station/charger network.
# Since the drone can fly 40 km per leg, the frontier range is also 40 km.
FRONTIER_RANGE_KM = DRONE_RANGE_KM
FRONTIER_RANGE_M = FRONTIER_RANGE_KM * 1000.0


# The first priority is still local Manitoba building candidates.
# Overpass is optional fallback when local building candidates cannot fill a connector gap.
ENABLE_GAP_REPAIR = True
MAX_CONNECTOR_CHARGERS = 80
# Gap repair should not stop just because one gap cannot be repaired.
# This controls total attempts, while MAX_CONNECTOR_CHARGERS controls successful additions.
MAX_GAP_REPAIR_ATTEMPTS = MAX_CONNECTOR_CHARGERS * 4
CONNECTOR_STEP_KM = 32.0
CONNECTOR_STEP_M = CONNECTOR_STEP_KM * 1000.0
LOCAL_CONNECTOR_SEARCH_RADIUS_KM = 12.0
LOCAL_CONNECTOR_SEARCH_RADIUS_M = LOCAL_CONNECTOR_SEARCH_RADIUS_KM * 1000.0
MIN_CONNECTOR_SPACING_KM = 8.0
MIN_CONNECTOR_SPACING_M = MIN_CONNECTOR_SPACING_KM * 1000.0

# If True, the script can query OpenStreetMap Overpass API when the local CSV
# cannot find a building/facility around a connector anchor.
# Keep the cap low to avoid slow runs or API timeouts.
USE_OVERPASS_IF_LOCAL_FAILS = True
MAX_OVERPASS_QUERIES = 20
OVERPASS_SEARCH_RADIUS_M = 5000
OVERPASS_TIMEOUT_SECONDS = 25
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Fixed charger mode: use a prepared CSV of permanent chargers.
# When this is enabled, the backend does NOT select extra relay chargers,
# does NOT run gap-repair connectors, and does NOT add rescue chargers.
USE_FIXED_PERMANENT_CHARGERS = False
FIXED_PERMANENT_CHARGERS_CSV = ""
FIXED_PERMANENT_CHARGERS_ONLY = False



# Household sampling.
N_RANDOM_HOUSEHOLDS = 40
MIN_HOUSEHOLD_DISTANCE_TO_STATION_KM = 60.0
MIN_HOUSEHOLD_TO_HOUSEHOLD_DISTANCE_KM = 12.0
RELAX_STEPS_KM = [60.0, 50.0, 40.0, 30.0, 20.0, 0.0]


# This prevents all random households from clustering near train stations.
# Format: (band_name, min_distance_to_nearest_station_km, max_distance_to_nearest_station_km, target_count)
HOUSEHOLD_DISTANCE_BANDS = [
    ("medium_40_80km", 40.0, 80.0, 10),
    ("far_80_120km", 80.0, 120.0, 10),
    ("very_far_120_180km", 120.0, 180.0, 10),
    ("extreme_far_180_260km", 180.0, 260.0, 10),
]

# If a far band cannot provide enough feasible households, keep what is available.
# The script will NOT secretly relax all the way back to station-near households.
STRICT_DISTANCE_BANDS = True


# Drone route.
# Current optimized mode only tests single routes and two-household merge routes.
MAX_HOUSEHOLDS_PER_DRONE_ROUTE = 2

# The route builder compares several possible household chunk sizes and chooses
# the lowest distance-per-household feasible route, instead of always forcing 4 households.
ENERGY_AWARE_ROUTING = True
MAX_EXTRA_DISTANCE_RATIO_TO_ADD_HOUSEHOLD = 1.10

# compared with serving those households separately. 0.95 means the combined route
# must be at least 5% shorter than separate routes; otherwise split the route.
COMBINE_ROUTE_ONLY_IF_SAVES_RATIO = 0.95

# Chargers are charging points only, not package origins. Routes start from a train station.
# One route serves one household by default. Two households are merged only if the
# merged route is shorter than serving them separately.
FORCE_SINGLE_HOUSEHOLD_ROUTES = False


# Map layers.
SHOW_ALL_CHARGER_CANDIDATES = False
SHOW_GRID = False
SHOW_UNSERVED_HOUSEHOLDS = True


# These models are used by the Streamlit interface. The values are editable
# so different model assumptions can be tested and compared.
DRONE_MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "DJI FlyCart 30": {
        "brand": "DJI",
        "model": "FlyCart 30",
        "range_km": 28.0,
        "payload_kg": 95.0,
        "speed_kmh": 72.0,
        "note": "Heavy cargo drone scenario: 28 km range, 95 kg payload, 72 km/h speed.",
    },
    "Wingcopter 198": {
        "brand": "Wingcopter",
        "model": "198",
        "range_km": 94.0,
        "payload_kg": 4.7,
        "speed_kmh": 90.0,
        "note": "Long-range logistics drone; useful for rural sensitivity analysis.",
    },
    "Drone Delivery Canada Sparrow": {
        "brand": "Drone Delivery Canada",
        "model": "Sparrow",
        "range_km": 20.0,
        "payload_kg": 4.0,
        "speed_kmh": 60.0,
        "note": "Canadian delivery drone reference: 20 km range, 4 kg payload, 60 km/h speed.",
    },
    "Drone Delivery Canada Canary": {
        "brand": "Drone Delivery Canada",
        "model": "Canary",
        "range_km": 20.0,
        "payload_kg": 4.5,
        "speed_kmh": 72.0,
        "note": "Canadian delivery drone reference: 20 km range, 4.5 kg payload, 72 km/h speed.",
    },
    "Zipline Platform 1": {
        "brand": "Zipline",
        "model": "Platform 1",
        "range_km": 193.0,
        "payload_kg": 1.8,
        "speed_kmh": 97.0,
        "note": "Very long-range scenario. The published value is 193+ km; this model uses 193 km as the numeric simulation value.",
    },
}

DRONE_MODEL_NAME = "DJI FlyCart 30"
DRONE_BRAND = DRONE_MODEL_CATALOG[DRONE_MODEL_NAME]["brand"]
DRONE_MODEL = DRONE_MODEL_CATALOG[DRONE_MODEL_NAME]["model"]
DRONE_PAYLOAD_KG = float(DRONE_MODEL_CATALOG[DRONE_MODEL_NAME]["payload_kg"])
DRONE_SPEED_KMH = float(DRONE_MODEL_CATALOG[DRONE_MODEL_NAME]["speed_kmh"])

# User-facing simulation controls.
NUMBER_OF_AGENTS = 15
FOOD_KG_MIN = 1.0
FOOD_KG_MAX = 5.0
CHARGER_UNIT_COST_CAD = 5000.0

# Visualization controls. Route-line offset makes overlapping routes easier to read.
ROUTE_LINE_SPACING_METERS = 650.0
USE_ROUTE_LINE_OFFSET = True
ROUTE_COLOR_BY_ROUTE = True


@dataclass
class SimulationConfig:
    """Configuration object used by the Streamlit app and by command-line runs."""
    drone_model_name: str = DRONE_MODEL_NAME
    drone_range_km: float = DRONE_RANGE_KM
    drone_payload_kg: float = DRONE_PAYLOAD_KG
    drone_speed_kmh: float = DRONE_SPEED_KMH
    number_of_agents: int = NUMBER_OF_AGENTS
    number_of_households: int = N_RANDOM_HOUSEHOLDS
    random_seed: int = RNG_SEED
    food_kg_min: float = FOOD_KG_MIN
    food_kg_max: float = FOOD_KG_MAX
    max_households_per_route: int = MAX_HOUSEHOLDS_PER_DRONE_ROUTE
    max_selected_chargers: int = MAX_SELECTED_CHARGERS
    charger_unit_cost_cad: float = CHARGER_UNIT_COST_CAD
    use_route_line_offset: bool = USE_ROUTE_LINE_OFFSET
    route_line_spacing_m: float = ROUTE_LINE_SPACING_METERS
    show_charger_coverage_circles: bool = SHOW_CHARGER_COVERAGE_CIRCLES
    use_overpass_if_local_fails: bool = False
    use_fixed_permanent_chargers: bool = False
    fixed_permanent_chargers_csv: str = ""
    fixed_permanent_chargers_only: bool = False
    generate_map: bool = True
    output_dir: str = str(OUTDIR)


"""
It applies the settings selected from the web app to the backend model, such as drone
model, range, payload, household number, random seed, and charger number.
"""
def apply_simulation_config(config: Optional[SimulationConfig] = None) -> SimulationConfig:
    """Apply interface/backend parameters to the model settings."""
    global DRONE_MODEL_NAME, DRONE_BRAND, DRONE_MODEL, DRONE_RANGE_KM, DRONE_RANGE_M
    global DRONE_PAYLOAD_KG, DRONE_SPEED_KMH, CHARGER_RADIUS_KM, CHARGER_RADIUS_M
    global FRONTIER_RANGE_KM, FRONTIER_RANGE_M, NUMBER_OF_AGENTS, N_RANDOM_HOUSEHOLDS
    global FOOD_KG_MIN, FOOD_KG_MAX, MAX_HOUSEHOLDS_PER_DRONE_ROUTE
    global MAX_SELECTED_CHARGERS, CHARGER_UNIT_COST_CAD, OUTDIR
    global RNG_SEED, HOUSEHOLD_DISTANCE_BANDS
    global USE_ROUTE_LINE_OFFSET, ROUTE_LINE_SPACING_METERS, SHOW_CHARGER_COVERAGE_CIRCLES
    global USE_OVERPASS_IF_LOCAL_FAILS, GENERATE_ROUTE_MAP
    global USE_FIXED_PERMANENT_CHARGERS, FIXED_PERMANENT_CHARGERS_CSV, FIXED_PERMANENT_CHARGERS_ONLY

    if config is None:
        config = SimulationConfig()

    if isinstance(config, dict):
        config = SimulationConfig(**config)

    DRONE_MODEL_NAME = str(config.drone_model_name)
    selected = DRONE_MODEL_CATALOG.get(DRONE_MODEL_NAME, {})
    DRONE_BRAND = str(selected.get("brand", DRONE_MODEL_NAME.split()[0] if DRONE_MODEL_NAME else "Custom"))
    DRONE_MODEL = str(selected.get("model", DRONE_MODEL_NAME))

    DRONE_RANGE_KM = float(config.drone_range_km)
    DRONE_RANGE_M = DRONE_RANGE_KM * 1000.0

    # In this project the charger service/relay radius follows the selected drone range.
    CHARGER_RADIUS_KM = DRONE_RANGE_KM
    CHARGER_RADIUS_M = CHARGER_RADIUS_KM * 1000.0
    FRONTIER_RANGE_KM = DRONE_RANGE_KM
    FRONTIER_RANGE_M = FRONTIER_RANGE_KM * 1000.0

    DRONE_PAYLOAD_KG = float(config.drone_payload_kg)
    DRONE_SPEED_KMH = float(config.drone_speed_kmh)
    NUMBER_OF_AGENTS = int(config.number_of_agents)
    N_RANDOM_HOUSEHOLDS = int(config.number_of_households)
    RNG_SEED = int(config.random_seed)

    # Make the random household distance bands follow the selected household count.
    # The four distance bands are scaled so larger simulations sample more households
    # without concentrating all demand near train stations.
    _band_template = [
        ("medium_40_80km", 40.0, 80.0),
        ("far_80_120km", 80.0, 120.0),
        ("very_far_120_180km", 120.0, 180.0),
        ("extreme_far_180_260km", 180.0, 260.0),
    ]
    _base_target = max(0, N_RANDOM_HOUSEHOLDS) // len(_band_template)
    _remainder = max(0, N_RANDOM_HOUSEHOLDS) % len(_band_template)
    HOUSEHOLD_DISTANCE_BANDS = [
        (name, min_km, max_km, _base_target + (1 if i < _remainder else 0))
        for i, (name, min_km, max_km) in enumerate(_band_template)
    ]

    FOOD_KG_MIN = float(config.food_kg_min)
    FOOD_KG_MAX = float(config.food_kg_max)
    MAX_HOUSEHOLDS_PER_DRONE_ROUTE = int(config.max_households_per_route)
    MAX_SELECTED_CHARGERS = int(config.max_selected_chargers)
    CHARGER_UNIT_COST_CAD = float(config.charger_unit_cost_cad)
    USE_ROUTE_LINE_OFFSET = bool(config.use_route_line_offset)
    ROUTE_LINE_SPACING_METERS = float(config.route_line_spacing_m)
    SHOW_CHARGER_COVERAGE_CIRCLES = bool(config.show_charger_coverage_circles)
    USE_OVERPASS_IF_LOCAL_FAILS = bool(config.use_overpass_if_local_fails)
    USE_FIXED_PERMANENT_CHARGERS = bool(getattr(config, "use_fixed_permanent_chargers", False))
    FIXED_PERMANENT_CHARGERS_CSV = str(getattr(config, "fixed_permanent_chargers_csv", "") or "")
    FIXED_PERMANENT_CHARGERS_ONLY = bool(getattr(config, "fixed_permanent_chargers_only", False))
    GENERATE_ROUTE_MAP = bool(getattr(config, "generate_map", True))
    OUTDIR = Path(config.output_dir)

    return config


"""
it converts the drone model catalog into a table, including models
such as DJI FlyCart 30, Wingcopter 198, and Zipline.
"""
def drone_model_catalog_dataframe() -> pd.DataFrame:
    rows = []
    for name, values in DRONE_MODEL_CATALOG.items():
        row = {"catalog_name": name}
        row.update(values)
        rows.append(row)
    return pd.DataFrame(rows)



"""
It runs the whole backend model and returns the output file paths,
such as the map, summary, routes, route legs, household status, and charger CSV files.
"""
def run_model(config: Optional[SimulationConfig] = None) -> Dict[str, str]:
    """Run the model from Streamlit or Python and return key output paths."""
    config = apply_simulation_config(config)
    main()
    return {
        "output_dir": str(OUTDIR),
        "map_html": str(OUTDIR / "final_drone_delivery_routes_map.html"),
        "summary_csv": str(OUTDIR / "summary.csv"),
        "routes_csv": str(OUTDIR / "drone_routes.csv"),
        "legs_csv": str(OUTDIR / "drone_route_legs.csv"),
        "status_csv": str(OUTDIR / "households_service_status.csv"),
        "chargers_csv": str(OUTDIR / "selected_connected_relay_chargers.csv"),
    }




# Static train stations
STATIC_STATION_COORDS = {
    "Cranberry Portage": (54.586111, -101.377220),
    "Flin Flon": (54.766735, -101.867054),
    "The Pas": (53.825000, -101.253330),
    "Cormorant": (54.254167, -100.533330),
    "Wabowden": (54.908889, -98.629722),
    "Thicket Portage": (55.318100, -97.686900),
    "Pikwitonei": (55.587778, -97.155556),
    "Thompson": (55.743333, -97.855278),
    "Gillam": (56.347222, -94.707778),
    "Churchill": (58.780833, -94.186944),
    "York Landing": (56.088889, -96.101667),
}

RAIL_ROUTE_ORDER = [
    "Flin Flon",
    "Cranberry Portage",
    "The Pas",
    "Cormorant",
    "Wabowden",
    "Thicket Portage",
    "Pikwitonei",
    "Thompson",
    "Gillam",
    "Churchill",
]



# Basic functions
"""
It checks whether the output folder exists. If not, it creates it.
"""
def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
"""
It calculates the straight-line distance between two points in meters.
"""
def euclidean_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))

"""
It reads manitoba_household_candidates_ALL_points.csv, checks for lat and lon,
and converts them into projected coordinates for distance calculation.
读取所有 Manitoba building / household 点。
"""
def load_all_points_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path}. Put manitoba_household_candidates_ALL_points.csv in the same folder."
        )

    df = pd.read_csv(path)
    if not {"lat", "lon"}.issubset(df.columns):
        raise ValueError("Input CSV must contain lat and lon columns.")

    df = df.copy().reset_index(drop=True)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).copy().reset_index(drop=True)

    if "household_id" not in df.columns:
        df["household_id"] = [f"MB_H{i:06d}" for i in range(len(df))]

    # Streamlit Cloud can fail on very large GeoPandas/Shapely point-array conversions.
    # Use pyproj directly for the 632k Manitoba building points; this is faster and more stable.
    transformer = Transformer.from_crs(WGS84_CRS, PROJECTED_CRS, always_xy=True)
    lon_arr = df["lon"].to_numpy(dtype="float64")
    lat_arr = df["lat"].to_numpy(dtype="float64")
    x_arr, y_arr = transformer.transform(lon_arr, lat_arr)
    df["x_m"] = np.asarray(x_arr, dtype="float64")
    df["y_m"] = np.asarray(y_arr, dtype="float64")
    return df.reset_index(drop=True)


"""
It converts the hard-coded Manitoba train station coordinates into a geographic table.
"""
def build_station_gdf() -> gpd.GeoDataFrame:
    rows = []
    for i, (name, (lat, lon)) in enumerate(STATIC_STATION_COORDS.items()):
        rows.append({
            "station_id": f"S{i:02d}",
            "base_id": f"ST{i:02d}",
            "base_type": "train_station",
            "name": name,
            "lat": lat,
            "lon": lon,
        })

    gdf = gpd.GeoDataFrame(
        rows,
        geometry=gpd.points_from_xy([r["lon"] for r in rows], [r["lat"] for r in rows]),
        crs=WGS84_CRS,
    ).to_crs(PROJECTED_CRS)

    gdf["x_m"] = gdf.geometry.x.astype(float)
    gdf["y_m"] = gdf.geometry.y.astype(float)
    return gdf


"""
It builds a railway line from train stations and creates a buffer area around the railway.
铁路范围
"""
def build_rail_line_and_buffer(station_gdf: gpd.GeoDataFrame):
    by_name = {str(r["name"]): r for _, r in station_gdf.iterrows()}
    coords = []
    for name in RAIL_ROUTE_ORDER:
        r = by_name[name]
        coords.append((float(r["x_m"]), float(r["y_m"])))

    rail_line = LineString(coords)
    rail_buffer = rail_line.buffer(RAIL_BUFFER_KM * 1000.0)
    return rail_line, rail_buffer


"""
It keeps only the building or household candidate points inside the railway buffer.
铁路附近的点
"""
def filter_points_within_rail_buffer(df: pd.DataFrame, rail_line, rail_buffer) -> pd.DataFrame:
    # Avoid creating 632k Shapely Point objects on Streamlit Cloud.
    # Instead, compute projected Euclidean distance from each candidate point to the railway polyline.
    # This is equivalent to filtering by a rail-line buffer for our projected CRS.
    if df.empty:
        out = df.copy()
        out["rail_distance_km"] = []
        return out

    pts = df[["x_m", "y_m"]].to_numpy(dtype="float64")
    line_coords = np.asarray(list(rail_line.coords), dtype="float64")
    if line_coords.ndim != 2 or line_coords.shape[0] < 2:
        raise ValueError("Rail line must contain at least two coordinate points.")

    min_dist = np.full(len(pts), np.inf, dtype="float64")
    for a, b in zip(line_coords[:-1], line_coords[1:]):
        vx = float(b[0] - a[0])
        vy = float(b[1] - a[1])
        denom = vx * vx + vy * vy
        if denom <= 0:
            continue
        wx = pts[:, 0] - float(a[0])
        wy = pts[:, 1] - float(a[1])
        t = np.clip((wx * vx + wy * vy) / denom, 0.0, 1.0)
        proj_x = float(a[0]) + t * vx
        proj_y = float(a[1]) + t * vy
        dist = np.hypot(pts[:, 0] - proj_x, pts[:, 1] - proj_y)
        min_dist = np.minimum(min_dist, dist)

    mask = min_dist <= (RAIL_BUFFER_KM * 1000.0)
    out = df.loc[mask].copy().reset_index(drop=True)
    out["rail_distance_km"] = (min_dist[mask] / 1000.0).round(6)
    return out

"""
It creates grid points inside the study area to check where charger coverage is needed.
判断哪里需要充电桩覆盖
"""
def make_grid_inside(poly, spacing_m: float) -> np.ndarray:
    minx, miny, maxx, maxy = poly.bounds
    xs = np.arange(minx, maxx + spacing_m, spacing_m)
    ys = np.arange(miny, maxy + spacing_m, spacing_m)

    pts = []
    for x in xs:
        for y in ys:
            p = Point(float(x), float(y))
            if poly.contains(p):
                pts.append((float(x), float(y)))

    if not pts:
        raise ValueError("No grid points generated. Try smaller GRID_SPACING_KM.")
    return np.array(pts, dtype=float)


"""
It calculates the distance from each point(household/charger) to the nearest train station in kilometers.
"""
def nearest_station_distance_km(points_xy: np.ndarray, station_xy: np.ndarray) -> np.ndarray:
    tree = cKDTree(station_xy)
    dist, _ = tree.query(points_xy, k=1)
    return dist / 1000.0




# Charger selection
"""
It selects relay chargers from building candidates. It grows the charger network outward from
train stations and ensures each new charger can connect back to the station network within
the drone range.
从 train station 网络开始往外扩展，确保新 charger 可以通过e.g.40 km 内的跳跃连接回 station
"""
def select_relay_chargers_from_grid(
    candidates_df: pd.DataFrame,
    station_gdf: gpd.GeoDataFrame,
    grid_xy: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    cand = candidates_df.copy().reset_index(drop=True)
    cand_xy = cand[["x_m", "y_m"]].to_numpy(dtype=float)
    grid_tree = cKDTree(grid_xy)

    station_xy = station_gdf[["x_m", "y_m"]].to_numpy(dtype=float)
    station_tree = cKDTree(station_xy)

    # Grid points already covered by train stations.
    d_station, _ = station_tree.query(grid_xy, k=1)
    covered = d_station <= DRONE_RANGE_M

    total_grid = len(grid_xy)
    target_covered = int(math.ceil(TARGET_GRID_COVERAGE_RATIO * total_grid))

    selected_idxs: List[int] = []
    selected_network_xy: List[Tuple[float, float]] = [
        (float(x), float(y)) for x, y in station_xy
    ]

    active = np.ones(len(cand), dtype=bool)

    # Candidates too close to train stations are not useful because station itself charges.
    cand_tree = cKDTree(cand_xy)
    near_station_candidate_idxs = cand_tree.query_ball_point(station_xy, r=MIN_SELECTED_CHARGER_SPACING_M)
    for idxs in near_station_candidate_idxs:
        if idxs:
            active[np.array(idxs, dtype=int)] = False

    selection_rows: List[dict] = []
    iteration = 0

    while int(covered.sum()) < target_covered and len(selected_idxs) < MAX_SELECTED_CHARGERS:
        iteration += 1

        # Build current network tree each round.
        network_xy = np.array(selected_network_xy, dtype=float)
        network_tree = cKDTree(network_xy)

        # Candidate must be within 40 km of the existing network.
        dist_to_network, nearest_network_idx = network_tree.query(cand_xy, k=1)
        frontier_mask = (dist_to_network <= FRONTIER_RANGE_M) & active

        frontier_idxs = np.where(frontier_mask)[0]

        if len(frontier_idxs) == 0:
            break

        best_ci = None
        best_gain = 0
        best_cover_idxs = None

        for ci in frontier_idxs:
            cover_idxs = grid_tree.query_ball_point(cand_xy[int(ci)], r=CHARGER_RADIUS_M)
            if not cover_idxs:
                continue

            cover_np = np.array(cover_idxs, dtype=int)
            gain = int(np.sum(~covered[cover_np]))

            # Tie-breaker: prefer candidates farther from the nearest network node,
            # because this helps extend the network outward instead of clustering.
            if gain > best_gain:
                best_gain = gain
                best_ci = int(ci)
                best_cover_idxs = cover_np
            elif gain == best_gain and gain > 0 and best_ci is not None:
                if dist_to_network[int(ci)] > dist_to_network[int(best_ci)]:
                    best_ci = int(ci)
                    best_cover_idxs = cover_np

        if best_ci is None or best_gain <= 0:
            # No frontier charger can add new coverage.
            break

        selected_idxs.append(best_ci)
        selected_network_xy.append((float(cand_xy[best_ci][0]), float(cand_xy[best_ci][1])))
        covered[best_cover_idxs] = True

        selection_rows.append({
            "iteration": iteration,
            "candidate_index": int(best_ci),
            "new_grid_points_covered": int(best_gain),
            "total_grid_points_covered_after_selection": int(covered.sum()),
            "coverage_ratio_after_selection": round(float(covered.sum()) / total_grid, 4),
            "distance_to_existing_network_km": round(float(dist_to_network[best_ci]) / 1000.0, 3),
            "frontier_range_km": FRONTIER_RANGE_KM,
        })

        # Density control: allow overlap, but avoid almost duplicate chargers.
        too_close = cand_tree.query_ball_point(cand_xy[best_ci], r=MIN_SELECTED_CHARGER_SPACING_M)
        if too_close:
            active[np.array(too_close, dtype=int)] = False

        active[best_ci] = False

    selected = cand.iloc[selected_idxs].copy().reset_index(drop=True)
    selected["charger_id"] = [f"C{i:04d}" for i in range(len(selected))]

    grid_status = pd.DataFrame(grid_xy, columns=["x_m", "y_m"])
    grid_status["covered_by_station_or_selected_charger"] = covered.astype(int)

    # Save selection iteration diagnostics immediately.
    if len(selection_rows) > 0:
        pd.DataFrame(selection_rows).to_csv(OUTDIR / "connected_charger_selection_iterations.csv", index=False)

    return selected, grid_status



"""
It combines train stations and selected chargers into one table because both can be
used as charging bases.
把车站和充电桩合并
"""
def build_charging_bases(station_gdf: gpd.GeoDataFrame, selected_chargers: pd.DataFrame) -> pd.DataFrame:
    stations_wgs = station_gdf.to_crs(WGS84_CRS)

    rows = []
    for _, r in stations_wgs.iterrows():
        rows.append({
            "base_id": str(r["base_id"]),
            "base_type": "train_station",
            "name": str(r["name"]),
            "source_id": str(r["station_id"]),
            "lat": float(r.geometry.y),
            "lon": float(r.geometry.x),
            "x_m": float(r["x_m"]),
            "y_m": float(r["y_m"]),
        })

    for _, r in selected_chargers.iterrows():
        rows.append({
            "base_id": str(r["charger_id"]),
            "base_type": "charger",
            "name": str(r["charger_id"]),
            "source_id": str(r["household_id"]),
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "x_m": float(r["x_m"]),
            "y_m": float(r["y_m"]),
        })

    return pd.DataFrame(rows).reset_index(drop=True)


"""
It automatically finds latitude and longitude columns in a CSV file,
even if the column names are different.
"""
def _find_lat_lon_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Find lat/lon columns in permanent-charger CSVs from different app versions."""
    if df.empty:
        return None, None
    lowered = {str(c).lower().strip(): c for c in df.columns}
    lat_candidates = [
        "lat", "latitude", "charger_lat", "selected_lat", "recommended_lat",
        "centroid_lat", "building_lat",
    ]
    lon_candidates = [
        "lon", "lng", "longitude", "charger_lon", "charger_lng",
        "selected_lon", "selected_lng", "recommended_lon", "recommended_lng",
        "centroid_lon", "centroid_lng", "building_lon", "building_lng",
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



"""
It loads the fixed permanent charger CSV and converts it into the backend charger format.
If fixed charger mode is enabled, these are the only chargers available.
读取绿色固定充电桩
"""
def load_fixed_permanent_chargers_csv(path: str | Path) -> pd.DataFrame:
    """Load fixed permanent chargers and convert them to the backend charger schema.

    This function prepares the permanent charger list for routing. When
    FIXED_PERMANENT_CHARGERS_ONLY is True, no extra relay, connector, or rescue
    chargers are added by the model.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Fixed permanent charger CSV not found: {p}")

    raw = pd.read_csv(p)
    if raw.empty:
        raise ValueError(f"Fixed permanent charger CSV is empty: {p}")

    lat_col, lon_col = _find_lat_lon_columns(raw)
    if not lat_col or not lon_col:
        raise ValueError("Fixed permanent charger CSV must contain latitude and longitude columns.")

    df = raw.copy().reset_index(drop=True)
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lon_col]).copy().reset_index(drop=True)
    if df.empty:
        raise ValueError("Fixed permanent charger CSV has no valid coordinates after cleaning.")

    if lat_col != "lat":
        df["lat"] = df[lat_col].astype(float)
    else:
        df["lat"] = df[lat_col].astype(float)
    if lon_col != "lon":
        df["lon"] = df[lon_col].astype(float)
    else:
        df["lon"] = df[lon_col].astype(float)

    # If projected coordinates are not already present, compute them from lat/lon.
    if "x_m" not in df.columns or "y_m" not in df.columns:
        gdf = gpd.GeoDataFrame(
            df.copy(),
            geometry=gpd.points_from_xy(df["lon"], df["lat"]),
            crs=WGS84_CRS,
        ).to_crs(PROJECTED_CRS)
        df["x_m"] = gdf.geometry.x.astype(float).values
        df["y_m"] = gdf.geometry.y.astype(float).values
    else:
        df["x_m"] = pd.to_numeric(df["x_m"], errors="coerce")
        df["y_m"] = pd.to_numeric(df["y_m"], errors="coerce")
        missing_xy = df["x_m"].isna() | df["y_m"].isna()
        if missing_xy.any():
            gdf = gpd.GeoDataFrame(
                df.loc[missing_xy].copy(),
                geometry=gpd.points_from_xy(df.loc[missing_xy, "lon"], df.loc[missing_xy, "lat"]),
                crs=WGS84_CRS,
            ).to_crs(PROJECTED_CRS)
            df.loc[missing_xy, "x_m"] = gdf.geometry.x.astype(float).values
            df.loc[missing_xy, "y_m"] = gdf.geometry.y.astype(float).values

    # Keep a stable source ID if the permanent-study output has one.
    if "household_id" not in df.columns:
        if "source_household_id" in df.columns:
            df["household_id"] = df["source_household_id"].astype(str)
        elif "location_key" in df.columns:
            df["household_id"] = df["location_key"].astype(str)
        elif "example_charger_id" in df.columns:
            df["household_id"] = df["example_charger_id"].astype(str)
        else:
            df["household_id"] = [f"FIXED_SOURCE_{i:04d}" for i in range(len(df))]

    # Use clear charger IDs so route legs show these fixed nodes.
    df["charger_id"] = [f"FP{i:04d}" for i in range(len(df))]
    df["charger_role"] = "fixed_permanent"
    df["source"] = "fixed_permanent_csv"
    df["fixed_permanent_charger"] = True

    # Remove exact duplicate coordinates, keeping the highest-ranked row order.
    df["_lat_round"] = df["lat"].round(7)
    df["_lon_round"] = df["lon"].round(7)
    df = df.drop_duplicates(subset=["_lat_round", "_lon_round"]).drop(columns=["_lat_round", "_lon_round"]).reset_index(drop=True)
    df["charger_id"] = [f"FP{i:04d}" for i in range(len(df))]

    return df



# Base graph and shortest relay path
"""
It builds a graph using all train stations and chargers. If two bases are within drone range,
an edge is created between them.
建立无人机可以从哪里飞到哪里的网络
"""
def build_base_graph(charging_bases: pd.DataFrame) -> Tuple[List[List[Tuple[int, float]]], cKDTree]:
    """
    Weighted graph nodes = train stations + selected chargers.
    Edge exists if distance <= 40 km.
    Edge weight = straight-line distance in meters.
    """
    base_xy = charging_bases[["x_m", "y_m"]].to_numpy(dtype=float)
    tree = cKDTree(base_xy)

    neighbor_lists = tree.query_ball_tree(tree, r=DRONE_RANGE_M)

    graph: List[List[Tuple[int, float]]] = []
    for i, neigh in enumerate(neighbor_lists):
        edges: List[Tuple[int, float]] = []
        a = (float(base_xy[i][0]), float(base_xy[i][1]))
        for j in neigh:
            j = int(j)
            if j == i:
                continue
            b = (float(base_xy[j][0]), float(base_xy[j][1]))
            edges.append((j, euclidean_m(a, b)))
        graph.append(edges)

    return graph, tree



"""
It finds the shortest path in the charging network using distance as the cost.
找最短的 charger/station 中转路线。
"""
def dijkstra_shortest_path(
    graph: List[List[Tuple[int, float]]],
    start_idxs: Sequence[int],
    target_idxs: Sequence[int],
) -> Optional[List[int]]:
    """
    Weighted shortest path from any start base to any target base.
    This is more energy-aware than BFS because distance is used as the cost.
    """
    targets = set(int(t) for t in target_idxs)
    pq: List[Tuple[float, int]] = []
    dist: Dict[int, float] = {}
    parent: Dict[int, Optional[int]] = {}

    for s in start_idxs:
        s = int(s)
        dist[s] = 0.0
        parent[s] = None
        heapq.heappush(pq, (0.0, s))

    found = None

    while pq:
        cur_dist, u = heapq.heappop(pq)

        if cur_dist > dist.get(u, float("inf")):
            continue

        if u in targets:
            found = u
            break

        for v, w in graph[u]:
            nd = cur_dist + float(w)
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))

    if found is None:
        return None

    path = []
    cur: Optional[int] = found
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path
def bfs_shortest_path(graph: List[List[Tuple[int, float]]], start_idxs: Sequence[int], target_idxs: Sequence[int]) -> Optional[List[int]]:
    return dijkstra_shortest_path(graph, start_idxs, target_idxs)





"""
It finds all stations or chargers within drone range of a given point.
找附近能充电的点
"""
def candidate_base_idxs_within_range(
    xy: Tuple[float, float],
    base_tree: cKDTree,
) -> List[int]:
    return [int(i) for i in base_tree.query_ball_point(np.array(xy, dtype=float), r=DRONE_RANGE_M)]

"""
It finds which charging bases are train stations.
"""
def get_station_base_indices(charging_bases: pd.DataFrame) -> List[int]:
    return [
        int(i)
        for i, r in charging_bases.reset_index(drop=True).iterrows()
        if str(r["base_type"]) == "train_station"
    ]



"""
It creates one flight leg record, such as station to charger or charger to household.
It records the start point, end point, distance, and whether the leg is feasible.
飞行记录检查是否超过航程
"""
def make_leg(
    route_id: str,
    leg_sequence: int,
    from_id: str,
    from_type: str,
    from_name: str,
    from_xy: Tuple[float, float],
    to_id: str,
    to_type: str,
    to_name: str,
    to_xy: Tuple[float, float],
) -> dict:
    dist_km = euclidean_m(from_xy, to_xy) / 1000.0
    return {
        "route_id": route_id,
        "leg_sequence": leg_sequence,
        "from_id": from_id,
        "from_type": from_type,
        "from_name": from_name,
        "from_x_m": from_xy[0],
        "from_y_m": from_xy[1],
        "to_id": to_id,
        "to_type": to_type,
        "to_name": to_name,
        "to_x_m": to_xy[0],
        "to_y_m": to_xy[1],
        "distance_km": round(dist_km, 3),
        "feasible_leg": dist_km <= DRONE_RANGE_KM,
    }



"""
It finds a feasible path between two points. If direct flight is possible, it flies directly;
otherwise, it uses chargers or stations as relay points.
从 A 点到 B 点，不够电就找充电桩中转
"""
def path_between_points_via_bases(
    route_id: str,
    start_id: str,
    start_type: str,
    start_name: str,
    start_xy: Tuple[float, float],
    end_id: str,
    end_type: str,
    end_name: str,
    end_xy: Tuple[float, float],
    charging_bases: pd.DataFrame,
    graph: List[List[Tuple[int, float]]],
    base_tree: cKDTree,
    first_leg_sequence: int,
) -> Tuple[Optional[List[dict]], int, Optional[str]]:
    """
    Find the lowest-distance feasible path from any point to any point.
    If direct <= 40 km, use direct leg.
    Otherwise insert charging bases as relay nodes using weighted Dijkstra.
    """
    direct_m = euclidean_m(start_xy, end_xy)
    if direct_m <= DRONE_RANGE_M:
        leg = make_leg(
            route_id, first_leg_sequence,
            start_id, start_type, start_name, start_xy,
            end_id, end_type, end_name, end_xy,
        )
        return [leg], first_leg_sequence + 1, None

    start_base_idxs = candidate_base_idxs_within_range(start_xy, base_tree)
    end_base_idxs = candidate_base_idxs_within_range(end_xy, base_tree)

    if not start_base_idxs:
        return None, first_leg_sequence, f"no charging base within 40 km of {start_id}"

    if not end_base_idxs:
        return None, first_leg_sequence, f"no charging base within 40 km of {end_id}"

    base_path = dijkstra_shortest_path(graph, start_base_idxs, end_base_idxs)
    if base_path is None:
        return None, first_leg_sequence, f"no relay path between {start_id} and {end_id}"

    legs: List[dict] = []
    seq = first_leg_sequence

    current_id = start_id
    current_type = start_type
    current_name = start_name
    current_xy = start_xy

    for bi in base_path:
        b = charging_bases.iloc[int(bi)]
        b_id = str(b["base_id"])
        b_type = str(b["base_type"])
        b_name = str(b["name"])
        b_xy = (float(b["x_m"]), float(b["y_m"]))

        if current_id == b_id:
            continue

        leg = make_leg(route_id, seq, current_id, current_type, current_name, current_xy, b_id, b_type, b_name, b_xy)
        if not leg["feasible_leg"]:
            return None, first_leg_sequence, f"relay leg {current_id}->{b_id} exceeds 40 km"
        legs.append(leg)
        seq += 1

        current_id, current_type, current_name, current_xy = b_id, b_type, b_name, b_xy

    if current_id != end_id:
        leg = make_leg(route_id, seq, current_id, current_type, current_name, current_xy, end_id, end_type, end_name, end_xy)
        if not leg["feasible_leg"]:
            return None, first_leg_sequence, f"final leg {current_id}->{end_id} exceeds 40 km"
        legs.append(leg)
        seq += 1

    return legs, seq, None



"""
It finds a route from the final household back to a train station. If direct return is not possible,
it uses chargers as relays.
送完以后回station
"""
def path_from_point_to_station_via_bases(
    route_id: str,
    start_id: str,
    start_type: str,
    start_name: str,
    start_xy: Tuple[float, float],
    charging_bases: pd.DataFrame,
    graph: List[List[Tuple[int, float]]],
    base_tree: cKDTree,
    station_base_idxs: List[int],
    first_leg_sequence: int,
) -> Tuple[Optional[List[dict]], int, Optional[str], Optional[str]]:
    """
    Return from the final household to the nearest/lowest-distance reachable train station.
    Intermediate chargers are allowed.
    """
    # Direct to nearest station if possible.
    best_direct = None
    best_direct_dist = float("inf")

    for bi in station_base_idxs:
        b = charging_bases.iloc[int(bi)]
        b_xy = (float(b["x_m"]), float(b["y_m"]))
        dist = euclidean_m(start_xy, b_xy)
        if dist <= DRONE_RANGE_M and dist < best_direct_dist:
            best_direct = int(bi)
            best_direct_dist = dist

    if best_direct is not None:
        b = charging_bases.iloc[best_direct]
        leg = make_leg(
            route_id, first_leg_sequence,
            start_id, start_type, start_name, start_xy,
            str(b["base_id"]), "train_station", str(b["name"]),
            (float(b["x_m"]), float(b["y_m"])),
        )
        return [leg], first_leg_sequence + 1, None, str(b["name"])

    start_base_idxs = candidate_base_idxs_within_range(start_xy, base_tree)
    if not start_base_idxs:
        return None, first_leg_sequence, f"no charging base within 40 km of {start_id} for return", None

    base_path = dijkstra_shortest_path(graph, start_base_idxs, station_base_idxs)
    if base_path is None:
        return None, first_leg_sequence, f"no relay path from {start_id} back to any train station", None

    legs: List[dict] = []
    seq = first_leg_sequence

    current_id = start_id
    current_type = start_type
    current_name = start_name
    current_xy = start_xy

    for bi in base_path:
        b = charging_bases.iloc[int(bi)]
        b_id = str(b["base_id"])
        b_type = str(b["base_type"])
        b_name = str(b["name"])
        b_xy = (float(b["x_m"]), float(b["y_m"]))

        if current_id == b_id:
            continue

        leg = make_leg(route_id, seq, current_id, current_type, current_name, current_xy, b_id, b_type, b_name, b_xy)
        if not leg["feasible_leg"]:
            return None, first_leg_sequence, f"return relay leg {current_id}->{b_id} exceeds 40 km", None

        legs.append(leg)
        seq += 1

        current_id, current_type, current_name, current_xy = b_id, b_type, b_name, b_xy

    end_station_name = str(charging_bases.iloc[int(base_path[-1])]["name"])
    return legs, seq, None, end_station_name





"""
It moves from one point toward another point by a certain distance and returns the
intermediate location.
"""
def point_along_line(
    start_xy: Tuple[float, float],
    target_xy: Tuple[float, float],
    step_m: float,
) -> Tuple[float, float]:
    dx = target_xy[0] - start_xy[0]
    dy = target_xy[1] - start_xy[1]
    dist = math.hypot(dx, dy)
    if dist == 0:
        return start_xy
    ratio = min(1.0, step_m / dist)
    return (start_xy[0] + dx * ratio, start_xy[1] + dy * ratio)


"""
It converts projected map coordinates back to latitude and longitude.
"""
def project_xy_to_latlon(x: float, y: float) -> Tuple[float, float]:
    gdf = gpd.GeoDataFrame(
        {"id": [0]},
        geometry=gpd.points_from_xy([float(x)], [float(y)]),
        crs=PROJECTED_CRS,
    ).to_crs(WGS84_CRS)
    return float(gdf.geometry.y.iloc[0]), float(gdf.geometry.x.iloc[0])


"""
It converts latitude and longitude into projected coordinates for distance calculation.
"""
def project_latlon_to_xy(lat: float, lon: float) -> Tuple[float, float]:
    gdf = gpd.GeoDataFrame(
        {"id": [0]},
        geometry=gpd.points_from_xy([float(lon)], [float(lat)]),
        crs=WGS84_CRS,
    ).to_crs(PROJECTED_CRS)
    return float(gdf.geometry.x.iloc[0]), float(gdf.geometry.y.iloc[0])


"""
If local building data cannot find a suitable charger location, it can query OpenStreetMap
Overpass for nearby buildings, shops, amenities, or power facilities.
"""
def query_overpass_facility_near(lat: float, lon: float, radius_m: int) -> Optional[dict]:
    query = f"""
    [out:json][timeout:{OVERPASS_TIMEOUT_SECONDS}];
    (
      node["building"](around:{radius_m},{lat},{lon});
      way["building"](around:{radius_m},{lat},{lon});
      relation["building"](around:{radius_m},{lat},{lon});
      node["amenity"](around:{radius_m},{lat},{lon});
      way["amenity"](around:{radius_m},{lat},{lon});
      node["shop"](around:{radius_m},{lat},{lon});
      way["shop"](around:{radius_m},{lat},{lon});
      node["industrial"](around:{radius_m},{lat},{lon});
      way["industrial"](around:{radius_m},{lat},{lon});
      node["power"](around:{radius_m},{lat},{lon});
      way["power"](around:{radius_m},{lat},{lon});
      node["railway"="station"](around:{radius_m},{lat},{lon});
      way["railway"="station"](around:{radius_m},{lat},{lon});
    );
    out center 20;
    """
    try:
        data = urllib.parse.urlencode({"data": query}).encode("utf-8")
        req = urllib.request.Request(
            OVERPASS_URL,
            data=data,
            headers={"User-Agent": "Manitoba-Drone-Delivery-Model/1.0"},
        )
        with urllib.request.urlopen(req, timeout=OVERPASS_TIMEOUT_SECONDS + 5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        elements = payload.get("elements", [])
        best = None
        best_dist = float("inf")

        for el in elements:
            if "lat" in el and "lon" in el:
                el_lat, el_lon = float(el["lat"]), float(el["lon"])
            elif "center" in el and "lat" in el["center"] and "lon" in el["center"]:
                el_lat, el_lon = float(el["center"]["lat"]), float(el["center"]["lon"])
            else:
                continue

            x, y = project_latlon_to_xy(el_lat, el_lon)
            ax, ay = project_latlon_to_xy(lat, lon)
            dist = euclidean_m((x, y), (ax, ay))

            if dist < best_dist:
                tags = el.get("tags", {})
                best_dist = dist
                best = {
                    "lat": el_lat,
                    "lon": el_lon,
                    "x_m": x,
                    "y_m": y,
                    "osm_id": str(el.get("id", "")),
                    "osm_type": str(el.get("type", "")),
                    "osm_tags": json.dumps(tags, ensure_ascii=False),
                    "distance_to_anchor_m": round(best_dist, 2),
                }
        return best
    except Exception as exc:
        return {
            "error": str(exc),
            "lat": None,
            "lon": None,
            "x_m": None,
            "y_m": None,
        }



"""
It recalculates which grid points are covered by stations or chargers.
重新计算哪些 grid points(地区) 已经被 station 或 charger 覆盖。
"""
def rebuild_connected_coverage(
    base_xy: np.ndarray,
    grid_xy: np.ndarray,
) -> np.ndarray:
    base_tree = cKDTree(base_xy)
    d, _ = base_tree.query(grid_xy, k=1)
    return d <= DRONE_RANGE_M

"""
It creates a coverage table showing whether each grid point is covered before and after gap repair.
"""
def build_grid_coverage_status(
    grid_xy: np.ndarray,
    charging_bases: pd.DataFrame,
    previous_grid_status: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build the grid coverage table after all connector chargers have been added.

    The output contains both:
    - covered_before_gap_repair
    - covered_after_gap_repair
    - newly_covered_by_gap_repair
    """
    if len(grid_xy) == 0:
        return pd.DataFrame()

    bases = charging_bases.reset_index(drop=True).copy()
    base_xy = bases[["x_m", "y_m"]].to_numpy(dtype=float)
    base_tree = cKDTree(base_xy)

    dist_to_base_m, nearest_base_idx = base_tree.query(grid_xy, k=1)

    out = pd.DataFrame(grid_xy, columns=["x_m", "y_m"])
    out["nearest_final_base_id"] = bases.iloc[nearest_base_idx]["base_id"].astype(str).values
    out["nearest_final_base_type"] = bases.iloc[nearest_base_idx]["base_type"].astype(str).values
    out["nearest_final_base_name"] = bases.iloc[nearest_base_idx]["name"].astype(str).values
    out["nearest_final_base_distance_km"] = dist_to_base_m / 1000.0
    out["covered_after_gap_repair"] = dist_to_base_m <= DRONE_RANGE_M

    if previous_grid_status is not None and len(previous_grid_status) == len(out):
        if "covered_by_station_or_selected_charger" in previous_grid_status.columns:
            out["covered_before_gap_repair"] = previous_grid_status[
                "covered_by_station_or_selected_charger"
            ].astype(bool).values
        elif "covered_after_gap_repair" in previous_grid_status.columns:
            out["covered_before_gap_repair"] = previous_grid_status[
                "covered_after_gap_repair"
            ].astype(bool).values
        else:
            out["covered_before_gap_repair"] = False
    else:
        out["covered_before_gap_repair"] = False

    out["newly_covered_by_gap_repair"] = (
        out["covered_after_gap_repair"] & ~out["covered_before_gap_repair"]
    )

    # Add lat/lon for easier map/debug inspection in CSV.
    grid_gdf = gpd.GeoDataFrame(
        out[["x_m", "y_m"]].copy(),
        geometry=gpd.points_from_xy(out["x_m"], out["y_m"]),
        crs=PROJECTED_CRS,
    ).to_crs(WGS84_CRS)
    out["lat"] = grid_gdf.geometry.y.astype(float).values
    out["lon"] = grid_gdf.geometry.x.astype(float).values

    return out


"""
It summarizes the grid coverage table into one row, including total grid points,
coverage before repair, coverage after repair, and uncovered points.
"""
def summarize_grid_coverage(grid_status: pd.DataFrame) -> pd.DataFrame:
    """
    One-row summary for report writing.
    """
    if len(grid_status) == 0:
        return pd.DataFrame([{
            "grid_points_total": 0,
            "covered_before_gap_repair": 0,
            "covered_after_gap_repair": 0,
            "newly_covered_by_gap_repair": 0,
            "coverage_ratio_before_gap_repair": 0.0,
            "coverage_ratio_after_gap_repair": 0.0,
            "coverage_ratio_improvement": 0.0,
            "uncovered_after_gap_repair": 0,
        }])

    total = int(len(grid_status))
    before = int(grid_status["covered_before_gap_repair"].sum())
    after = int(grid_status["covered_after_gap_repair"].sum())
    newly = int(grid_status["newly_covered_by_gap_repair"].sum())

    return pd.DataFrame([{
        "grid_points_total": total,
        "covered_before_gap_repair": before,
        "covered_after_gap_repair": after,
        "newly_covered_by_gap_repair": newly,
        "coverage_ratio_before_gap_repair": round(before / total, 4),
        "coverage_ratio_after_gap_repair": round(after / total, 4),
        "coverage_ratio_improvement": round((after - before) / total, 4),
        "uncovered_after_gap_repair": int(total - after),
    }])




"""
If the charger network has gaps, this function tries to add connector chargers.
It first uses local Manitoba building candidates and optionally uses Overpass facilities.
"""
def repair_gaps_with_connectors(
    selected_chargers: pd.DataFrame,
    candidates_df: pd.DataFrame,
    station_gdf: gpd.GeoDataFrame,
    grid_xy: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add connector chargers to reduce visual/network gaps.
    Search order for each connector anchor:
    1. local manitoba_household_candidates_ALL_points.csv building candidate;
    2. optional OSM Overpass search for building / amenity / shop / industrial / power / railway station.
    """
    if not ENABLE_GAP_REPAIR:
        selected = selected_chargers.copy()
        selected["charger_role"] = selected.get("charger_role", "service")
        return selected, pd.DataFrame([{
            "iteration": 0,
            "action": "gap_repair_disabled",
            "coverage_ratio_before": np.nan,
        }])

    selected = selected_chargers.copy().reset_index(drop=True)
    if "charger_role" not in selected.columns:
        selected["charger_role"] = "service"

    candidate_pool = candidates_df.copy().reset_index(drop=True)
    cand_xy = candidate_pool[["x_m", "y_m"]].to_numpy(dtype=float)
    cand_tree = cKDTree(cand_xy)

    station_xy = station_gdf[["x_m", "y_m"]].to_numpy(dtype=float)

    # Current network coordinates include train stations and selected chargers.
    current_base_xy_list: List[Tuple[float, float]] = [(float(x), float(y)) for x, y in station_xy]
    for _, r in selected.iterrows():
        current_base_xy_list.append((float(r["x_m"]), float(r["y_m"])))

    used_local_candidate_indices: set[int] = set()
    skipped_grid_idxs: set[int] = set()
    connector_rows: List[dict] = []
    repair_log: List[dict] = []

    overpass_queries = 0
    last_iteration = 0

    for iteration in range(1, MAX_GAP_REPAIR_ATTEMPTS + 1):
        last_iteration = iteration

        if len(connector_rows) >= MAX_CONNECTOR_CHARGERS:
            repair_log.append({
                "iteration": iteration,
                "action": "stop_max_connector_chargers_reached",
                "successful_connector_count": int(len(connector_rows)),
                "max_connector_chargers": int(MAX_CONNECTOR_CHARGERS),
            })
            break

        base_xy = np.array(current_base_xy_list, dtype=float)
        base_tree = cKDTree(base_xy)

        covered = rebuild_connected_coverage(base_xy, grid_xy)
        coverage_ratio = float(covered.sum()) / len(covered)

        if coverage_ratio >= TARGET_GRID_COVERAGE_RATIO:
            repair_log.append({
                "iteration": iteration,
                "action": "stop_target_coverage_reached",
                "coverage_ratio": round(coverage_ratio, 4),
                "successful_connector_count": int(len(connector_rows)),
                "failed_skipped_grid_count": int(len(skipped_grid_idxs)),
            })
            break

        uncovered_idxs = np.where(~covered)[0]
        if len(uncovered_idxs) == 0:
            repair_log.append({
                "iteration": iteration,
                "action": "stop_all_grid_points_covered",
                "coverage_ratio": round(coverage_ratio, 4),
            })
            break

        available_uncovered_idxs = [
            int(i) for i in uncovered_idxs if int(i) not in skipped_grid_idxs
        ]

        if len(available_uncovered_idxs) == 0:
            repair_log.append({
                "iteration": iteration,
                "action": "stop_all_remaining_uncovered_gaps_already_failed",
                "coverage_ratio": round(coverage_ratio, 4),
                "failed_skipped_grid_count": int(len(skipped_grid_idxs)),
            })
            break

        available_uncovered_xy = grid_xy[np.array(available_uncovered_idxs, dtype=int)]

        # Pick the available uncovered grid point closest to current network first.
        # If it fails, it will be skipped and the next iteration tries another gap.
        d_to_net, nearest_net = base_tree.query(available_uncovered_xy, k=1)
        order_pos = int(np.argmin(d_to_net))
        target_grid_idx = int(available_uncovered_idxs[order_pos])
        target_xy = (float(grid_xy[target_grid_idx][0]), float(grid_xy[target_grid_idx][1]))

        nearest_base_idx = int(nearest_net[order_pos])
        nearest_base_xy = (float(base_xy[nearest_base_idx][0]), float(base_xy[nearest_base_idx][1]))
        dist_to_network_m = float(d_to_net[order_pos])

        anchor_xy = point_along_line(nearest_base_xy, target_xy, CONNECTOR_STEP_M)

        failure_reason = "no_local_connector_found"

        # Local search first.
        local_idxs = cand_tree.query_ball_point(
            np.array(anchor_xy, dtype=float),
            r=LOCAL_CONNECTOR_SEARCH_RADIUS_M,
        )

        best_local_idx = None
        best_local_score = float("inf")

        for ci in local_idxs:
            ci = int(ci)
            if ci in used_local_candidate_indices:
                continue

            cxy = (float(cand_xy[ci][0]), float(cand_xy[ci][1]))

            # Must be within one hop of current network.
            d_net, _ = base_tree.query(np.array([cxy], dtype=float), k=1)
            if float(d_net[0]) > DRONE_RANGE_M:
                continue

            # Avoid almost duplicate connector points.
            if float(d_net[0]) < MIN_CONNECTOR_SPACING_M:
                continue

            score = euclidean_m(cxy, anchor_xy)
            if score < best_local_score:
                best_local_score = score
                best_local_idx = ci

        if best_local_idx is not None:
            r = candidate_pool.iloc[int(best_local_idx)]
            cid = f"GC_LOCAL_{len(connector_rows):04d}"
            connector = {
                "charger_id": cid,
                "household_id": str(r["household_id"]),
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
                "x_m": float(r["x_m"]),
                "y_m": float(r["y_m"]),
                "rail_distance_km": float(r.get("rail_distance_km", np.nan)),
                "charger_role": "connector_local_building",
                "source": "local_all_points_csv",
            }
            connector_rows.append(connector)
            selected = pd.concat([selected, pd.DataFrame([connector])], ignore_index=True)
            current_base_xy_list.append((connector["x_m"], connector["y_m"]))
            used_local_candidate_indices.add(int(best_local_idx))

            repair_log.append({
                "iteration": iteration,
                "action": "added_local_connector",
                "charger_id": cid,
                "target_grid_idx": target_grid_idx,
                "distance_target_to_network_before_km": round(dist_to_network_m / 1000.0, 3),
                "connector_source": "local_all_points_csv",
                "coverage_ratio_before": round(coverage_ratio, 4),
                "successful_connector_count": int(len(connector_rows)),
                "failed_skipped_grid_count": int(len(skipped_grid_idxs)),
            })
            continue

        # Optional Overpass fallback.
        if USE_OVERPASS_IF_LOCAL_FAILS and overpass_queries < MAX_OVERPASS_QUERIES:
            anchor_lat, anchor_lon = project_xy_to_latlon(anchor_xy[0], anchor_xy[1])
            result = query_overpass_facility_near(anchor_lat, anchor_lon, OVERPASS_SEARCH_RADIUS_M)
            overpass_queries += 1
            time.sleep(1.0)

            if result and result.get("x_m") is not None:
                oxy = (float(result["x_m"]), float(result["y_m"]))
                d_net, _ = base_tree.query(np.array([oxy], dtype=float), k=1)

                if float(d_net[0]) <= DRONE_RANGE_M and float(d_net[0]) >= MIN_CONNECTOR_SPACING_M:
                    cid = f"GC_OSM_{len(connector_rows):04d}"
                    connector = {
                        "charger_id": cid,
                        "household_id": str(result.get("osm_id", "")),
                        "lat": float(result["lat"]),
                        "lon": float(result["lon"]),
                        "x_m": float(result["x_m"]),
                        "y_m": float(result["y_m"]),
                        "rail_distance_km": np.nan,
                        "charger_role": "connector_osm_facility",
                        "source": "osm_overpass",
                        "osm_type": result.get("osm_type", ""),
                        "osm_tags": result.get("osm_tags", ""),
                    }
                    connector_rows.append(connector)
                    selected = pd.concat([selected, pd.DataFrame([connector])], ignore_index=True)
                    current_base_xy_list.append((connector["x_m"], connector["y_m"]))

                    repair_log.append({
                        "iteration": iteration,
                        "action": "added_osm_connector",
                        "charger_id": cid,
                        "target_grid_idx": target_grid_idx,
                        "distance_target_to_network_before_km": round(dist_to_network_m / 1000.0, 3),
                        "connector_source": "osm_overpass",
                        "coverage_ratio_before": round(coverage_ratio, 4),
                        "successful_connector_count": int(len(connector_rows)),
                        "failed_skipped_grid_count": int(len(skipped_grid_idxs)),
                    })
                    continue

                failure_reason = "osm_found_but_not_usable_too_far_or_duplicate"
                repair_log.append({
                    "iteration": iteration,
                    "action": "osm_found_but_not_usable",
                    "target_grid_idx": target_grid_idx,
                    "reason": failure_reason,
                    "coverage_ratio_before": round(coverage_ratio, 4),
                })
            else:
                failure_reason = "overpass_no_result_or_error"
                repair_log.append({
                    "iteration": iteration,
                    "action": "overpass_no_result_or_error",
                    "target_grid_idx": target_grid_idx,
                    "result": str(result),
                    "coverage_ratio_before": round(coverage_ratio, 4),
                })
        else:
            if USE_OVERPASS_IF_LOCAL_FAILS:
                failure_reason = "local_failed_and_overpass_query_limit_reached"
            else:
                failure_reason = "local_failed_and_overpass_disabled"

        # Record this gap as failed, skip it, and continue trying other uncovered grid points.
        skipped_grid_idxs.add(target_grid_idx)
        repair_log.append({
            "iteration": iteration,
            "action": "skipped_unrepairable_gap_and_continue",
            "target_grid_idx": target_grid_idx,
            "reason": failure_reason,
            "distance_target_to_network_before_km": round(dist_to_network_m / 1000.0, 3),
            "coverage_ratio_before": round(coverage_ratio, 4),
            "successful_connector_count": int(len(connector_rows)),
            "failed_skipped_grid_count": int(len(skipped_grid_idxs)),
        })
        continue

    # Final diagnostic row from inside the function.
    final_base_xy = np.array(current_base_xy_list, dtype=float)
    final_covered = rebuild_connected_coverage(final_base_xy, grid_xy)
    repair_log.append({
        "iteration": int(last_iteration) + 1,
        "action": "final_gap_repair_summary",
        "final_coverage_ratio": round(float(final_covered.sum()) / len(final_covered), 4),
        "successful_connector_count": int(len(connector_rows)),
        "failed_skipped_grid_count": int(len(skipped_grid_idxs)),
        "overpass_queries_used": int(overpass_queries),
    })

    return selected.reset_index(drop=True), pd.DataFrame(repair_log)




"""
It adds distance information to each household candidate, such as the nearest station,
nearest charging base, and distance.
给每个家庭点标注最近车站/充电点。
"""
def attach_distances_to_households(
    pool: pd.DataFrame,
    charging_bases: pd.DataFrame,
    station_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    df = pool.copy().reset_index(drop=True)
    xy = df[["x_m", "y_m"]].to_numpy(dtype=float)

    station_xy = station_gdf[["x_m", "y_m"]].to_numpy(dtype=float)
    station_tree = cKDTree(station_xy)
    station_dist, station_idx = station_tree.query(xy, k=1)
    df["nearest_station_id"] = station_gdf.iloc[station_idx]["station_id"].values
    df["nearest_station_name"] = station_gdf.iloc[station_idx]["name"].values
    df["nearest_station_distance_km"] = station_dist / 1000.0

    base_xy = charging_bases[["x_m", "y_m"]].to_numpy(dtype=float)
    base_tree = cKDTree(base_xy)
    base_dist, base_idx = base_tree.query(xy, k=1)
    df["nearest_base_id"] = charging_bases.iloc[base_idx]["base_id"].values
    df["nearest_base_type"] = charging_bases.iloc[base_idx]["base_type"].values
    df["nearest_base_name"] = charging_bases.iloc[base_idx]["name"].values
    df["nearest_base_distance_km"] = base_dist / 1000.0
    df["inside_base_coverage"] = df["nearest_base_distance_km"] <= CHARGER_RADIUS_KM
    return df

"""
It checks whether a drone can return from a household to a train station through the charger network.
检查送完这个家庭后能不能回车站
"""
def can_return_to_station(
    hh_row: pd.Series,
    charging_bases: pd.DataFrame,
    graph: List[List[int]],
    base_tree: cKDTree,
    station_base_idxs: List[int],
) -> bool:
    xy = (float(hh_row["x_m"]), float(hh_row["y_m"]))
    start_base_idxs = candidate_base_idxs_within_range(xy, base_tree)
    if not start_base_idxs:
        return False
    return bfs_shortest_path(graph, start_base_idxs, station_base_idxs) is not None



"""
It randomly samples delivery households from different distance bands,
so the households are not all clustered near train stations.
随机抽今天要送的家庭点。
"""
def sample_households(
    pool: pd.DataFrame,
    charging_bases: pd.DataFrame,
    graph: List[List[Tuple[int, float]]],
    base_tree: cKDTree,
    station_base_idxs: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified household sampling.

    Household eligibility rule:
    A household is NOT considered deliverable just because it is inside one
    charger/station circle. Because households cannot recharge the drone, the
    real single-household feasibility rule is:

        connected chargeable entry node -> household -> connected chargeable exit node
        must be <= DRONE_RANGE_KM in the same battery cycle.

    Therefore this sampler now uses the same charge-cycle feasibility test as
    the final routing diagnostics. This prevents confusing cases where a
    household looks inside a charger circle but later becomes unserved because
    the full charge cycle is longer than the drone range.
    """
    rng = random.Random(RNG_SEED)

    # First cheap screen: the household must be inside at least one station/charger circle.
    # The stronger charge-cycle test is applied below.
    eligible = pool[pool["inside_base_coverage"] == True].copy().reset_index(drop=True)

    selected_parts: List[pd.DataFrame] = []
    diagnostics_rows: List[dict] = []
    selected_xy_global: List[Tuple[float, float]] = []

    min_pair_m = MIN_HOUSEHOLD_TO_HOUSEHOLD_DISTANCE_KM * 1000.0

    for band_name, min_km, max_km, target_count in HOUSEHOLD_DISTANCE_BANDS:
        band_pool = eligible[
            (eligible["nearest_station_distance_km"] >= float(min_km)) &
            (eligible["nearest_station_distance_km"] < float(max_km))
        ].copy().reset_index(drop=True)

        idxs = list(range(len(band_pool)))
        rng.shuffle(idxs)

        band_selected_idxs: List[int] = []
        tested = 0
        feasible_charge_cycle_count = 0
        rejected_too_close_count = 0
        rejected_no_charge_cycle_count = 0
        rejected_reason_counts: Dict[str, int] = {}

        for idx in idxs:
            r = band_pool.iloc[idx]
            tested += 1

            # Strong test: this must match the route builder's battery-cycle logic.
            diag = best_charge_cycle_for_household(
                r,
                charging_bases=charging_bases,
                graph=graph,
                base_tree=base_tree,
                station_base_idxs=station_base_idxs,
            )
            if not bool(diag.get("charge_cycle_feasible", False)):
                rejected_no_charge_cycle_count += 1
                reason = str(diag.get("diagnosis", "charge_cycle_not_feasible"))
                rejected_reason_counts[reason] = rejected_reason_counts.get(reason, 0) + 1
                continue

            feasible_charge_cycle_count += 1
            xy = (float(r["x_m"]), float(r["y_m"]))

            # Keep households spatially spread out across all bands.
            if selected_xy_global and any(euclidean_m(xy, sxy) < min_pair_m for sxy in selected_xy_global):
                rejected_too_close_count += 1
                continue

            band_selected_idxs.append(idx)
            selected_xy_global.append(xy)

            if len(band_selected_idxs) >= int(target_count):
                break

        band_sample = band_pool.iloc[band_selected_idxs].copy().reset_index(drop=True)
        if len(band_sample) > 0:
            band_sample["distance_band"] = band_name
            band_sample["band_min_km"] = float(min_km)
            band_sample["band_max_km"] = float(max_km)
            band_sample["sampled_after_charge_cycle_feasibility_check"] = True
            selected_parts.append(band_sample)

        diagnostics_rows.append({
            "distance_band": band_name,
            "band_min_km": float(min_km),
            "band_max_km": float(max_km),
            "target_count": int(target_count),
            "candidate_count_inside_band_and_base_coverage": int(len(band_pool)),
            "tested_count": int(tested),
            "feasible_charge_cycle_count_during_test": int(feasible_charge_cycle_count),
            "rejected_no_charge_cycle_count": int(rejected_no_charge_cycle_count),
            "rejected_too_close_count": int(rejected_too_close_count),
            "selected_count": int(len(band_sample)),
            "charge_cycle_rule_used_for_sampling": True,
            "most_common_rejection_reasons": json.dumps(rejected_reason_counts, ensure_ascii=False),
        })

    if selected_parts:
        final = pd.concat(selected_parts, ignore_index=True)
    else:
        final = eligible.head(0).copy()

    final = final.head(N_RANDOM_HOUSEHOLDS).copy().reset_index(drop=True)

    rng2 = random.Random(RNG_SEED + 1000)
    final["delivery_household_id"] = [f"H{i:03d}" for i in range(len(final))]

    # Payload is checked later by the route builder, because demand is generated
    # after the spatial sample is selected.
    if len(final) > 0:
        food_min = min(float(FOOD_KG_MIN), float(FOOD_KG_MAX))
        food_max = max(float(FOOD_KG_MIN), float(FOOD_KG_MAX))
        final["food_kg"] = [round(rng2.uniform(food_min, food_max), 2) for _ in range(len(final))]
        final["demand_packages"] = [max(1, int(math.ceil(v / 1.5))) for v in final["food_kg"]]
    else:
        final["food_kg"] = []
        final["demand_packages"] = []

    diagnostics = pd.DataFrame(diagnostics_rows)

    return final, diagnostics



#Payload
"""
It calculates the total food weight of a group of households.
"""
def group_total_food_kg(group: pd.DataFrame) -> float:
    """Return total household food demand in kilograms for a route group."""
    if group is None or len(group) == 0 or "food_kg" not in group.columns:
        return 0.0
    return float(pd.to_numeric(group["food_kg"], errors="coerce").fillna(0.0).sum())

"""
It reads the food demand weight of one household.
"""
def household_food_kg(row: pd.Series) -> float:
    try:
        return float(row.get("food_kg", 0.0))
    except Exception:
        return 0.0


"""
It checks whether the total route payload exceeds the drone payload capacity.
"""
def route_payload_feasible(group: pd.DataFrame) -> bool:
    return group_total_food_kg(group) <= DRONE_PAYLOAD_KG + 1e-9


"""
If a route has multiple households, it uses a nearest-neighbor rule to decide the visiting order.
决定 household 的配送顺序by itself
"""
def order_households_nearest_neighbor(start_xy: Tuple[float, float], households: pd.DataFrame) -> pd.DataFrame:
    if len(households) <= 1:
        return households.copy().reset_index(drop=True)

    remaining = households.copy().reset_index(drop=True)
    ordered_rows = []
    cur_xy = start_xy

    while len(remaining) > 0:
        dists = [
            euclidean_m(cur_xy, (float(r["x_m"]), float(r["y_m"])))
            for _, r in remaining.iterrows()
        ]
        pos = int(np.argmin(dists))
        row = remaining.iloc[pos].copy()
        ordered_rows.append(row)
        cur_xy = (float(row["x_m"]), float(row["y_m"]))
        remaining = remaining.drop(index=remaining.index[pos]).reset_index(drop=True)

    return pd.DataFrame(ordered_rows).reset_index(drop=True)



"""
It adds up the distances of all flight legs in one route.
"""
def route_total_distance_km(legs: List[dict]) -> float:
    return float(sum(float(l["distance_km"]) for l in legs))


"""
It builds the full node sequence from route legs,
such as station → charger → household → charger → station.
整理路线顺序
"""
def route_node_sequence_from_legs(legs: List[dict]) -> str:
    if not legs:
        return ""
    return " -> ".join([str(legs[0]["from_id"])] + [str(l["to_id"]) for l in legs])


"""
It renumbers a route and updates the route ID for all its legs.
"""
def renumber_route_and_legs(route: dict, legs: List[dict], new_route_id: str) -> Tuple[dict, List[dict]]:
    """Copy a temporary route/legs pair and assign a final route id."""
    route2 = dict(route)
    route2["route_id"] = new_route_id

    legs2: List[dict] = []
    for leg in legs:
        leg2 = dict(leg)
        leg2["route_id"] = new_route_id
        legs2.append(leg2)

    return route2, legs2

"""
It extracts household IDs from a household group.
"""
def get_household_ids_from_group(group: pd.DataFrame) -> List[str]:
    return group["delivery_household_id"].astype(str).tolist()


"""
It calculates the total distance of a path through charging bases in meters.
"""
def base_idx_path_distance_m(base_idx_path: Sequence[int], charging_bases: pd.DataFrame) -> float:
    """Total straight-line distance of a base-to-base path."""
    if base_idx_path is None or len(base_idx_path) < 2:
        return 0.0

    total = 0.0
    for ai, bi in zip(base_idx_path[:-1], base_idx_path[1:]):
        a = charging_bases.iloc[int(ai)]
        b = charging_bases.iloc[int(bi)]
        total += euclidean_m((float(a["x_m"]), float(a["y_m"])), (float(b["x_m"]), float(b["y_m"])))
    return float(total)


"""
It converts a charger/station path into individual flight legs.
"""
def build_base_path_legs(
    route_id: str,
    first_leg_sequence: int,
    base_idx_path: Sequence[int],
    charging_bases: pd.DataFrame,
) -> Tuple[List[dict], int]:
    #Convert a base-index path into route legs.
    legs: List[dict] = []
    seq = first_leg_sequence

    if base_idx_path is None or len(base_idx_path) < 2:
        return legs, seq

    for ai, bi in zip(base_idx_path[:-1], base_idx_path[1:]):
        a = charging_bases.iloc[int(ai)]
        b = charging_bases.iloc[int(bi)]
        legs.append(
            make_leg(
                route_id=route_id,
                leg_sequence=seq,
                from_id=str(a["base_id"]),
                from_type=str(a["base_type"]),
                from_name=str(a["name"]),
                from_xy=(float(a["x_m"]), float(a["y_m"])),
                to_id=str(b["base_id"]),
                to_type=str(b["base_type"]),
                to_name=str(b["name"]),
                to_xy=(float(b["x_m"]), float(b["y_m"])),
            )
        )
        seq += 1

    return legs, seq


"""
This is one of the core routing functions. It builds a complete route that starts from a
train station, uses chargers if needed, serves households, and returns to a train station.
"""
def build_station_origin_route_for_order(
    route_id: str,
    ordered_group: pd.DataFrame,
    charging_bases: pd.DataFrame,
    graph: List[List[Tuple[int, float]]],
    base_tree: cKDTree,
    station_base_idxs: List[int],
) -> Tuple[Optional[dict], List[dict], Optional[str]]:
    """
    Build one candidate route under the corrected charge-cycle rule.
    train station -> charger relay -> [charge cycle containing household visit] ->
    charger relay -> train station
    """
    if len(ordered_group) == 0:
        return None, [], "empty ordered group"

    if len(ordered_group) > 2:
        return None, [], "this version only supports 1 or 2 households per route"

    total_food_kg = group_total_food_kg(ordered_group)
    if total_food_kg > DRONE_PAYLOAD_KG:
        hh_ids = ", ".join(get_household_ids_from_group(ordered_group))
        return None, [], (
            f"payload infeasible for household set [{hh_ids}]: "
            f"food demand {total_food_kg:.2f} kg exceeds drone payload {DRONE_PAYLOAD_KG:.2f} kg"
        )
    hh1 = ordered_group.iloc[0]
    hh1_id = str(hh1["delivery_household_id"])
    hh1_xy = (float(hh1["x_m"]), float(hh1["y_m"]))

    hh2 = None
    hh2_id = None
    hh2_xy = None
    if len(ordered_group) == 2:
        hh2 = ordered_group.iloc[1]
        hh2_id = str(hh2["delivery_household_id"])
        hh2_xy = (float(hh2["x_m"]), float(hh2["y_m"]))

    entry_base_idxs = candidate_base_idxs_within_range(hh1_xy, base_tree)
    exit_target_xy = hh2_xy if hh2_xy is not None else hh1_xy
    exit_base_idxs = candidate_base_idxs_within_range(exit_target_xy, base_tree)

    if not entry_base_idxs:
        return None, [], f"no chargeable base within 40 km of {hh1_id}"
    if not exit_base_idxs:
        last_hh_id = hh2_id if hh2_id is not None else hh1_id
        return None, [], f"no chargeable base within 40 km of {last_hh_id}"

    best_solution = None
    best_total_m = float("inf")

    for entry_idx in entry_base_idxs:
        pre_base_path = dijkstra_shortest_path(graph, station_base_idxs, [entry_idx])
        if pre_base_path is None:
            continue
        pre_m = base_idx_path_distance_m(pre_base_path, charging_bases)

        entry_base = charging_bases.iloc[int(entry_idx)]
        entry_xy = (float(entry_base["x_m"]), float(entry_base["y_m"]))

        for exit_idx in exit_base_idxs:
            exit_base = charging_bases.iloc[int(exit_idx)]
            exit_xy = (float(exit_base["x_m"]), float(exit_base["y_m"]))

            sortie_m = euclidean_m(entry_xy, hh1_xy)
            if hh2_xy is not None:
                sortie_m += euclidean_m(hh1_xy, hh2_xy)
                sortie_m += euclidean_m(hh2_xy, exit_xy)
            else:
                sortie_m += euclidean_m(hh1_xy, exit_xy)

            # Core corrected rule: the whole non-recharge sortie must fit in one battery cycle.
            if sortie_m > DRONE_RANGE_M:
                continue

            post_base_path = dijkstra_shortest_path(graph, [exit_idx], station_base_idxs)
            if post_base_path is None:
                continue
            post_m = base_idx_path_distance_m(post_base_path, charging_bases)

            total_m = pre_m + sortie_m + post_m
            if total_m < best_total_m:
                best_total_m = total_m
                best_solution = {
                    "pre_base_path": pre_base_path,
                    "entry_idx": int(entry_idx),
                    "exit_idx": int(exit_idx),
                    "post_base_path": post_base_path,
                    "sortie_m": sortie_m,
                }

    if best_solution is None:
        hh_ids = ", ".join(get_household_ids_from_group(ordered_group))
        return None, [], f"no feasible charge-cycle route for household set [{hh_ids}]"

    pre_base_path = best_solution["pre_base_path"]
    entry_idx = int(best_solution["entry_idx"])
    exit_idx = int(best_solution["exit_idx"])
    post_base_path = best_solution["post_base_path"]
    sortie_m = float(best_solution["sortie_m"])

    entry_base = charging_bases.iloc[entry_idx]
    exit_base = charging_bases.iloc[exit_idx]
    start_station = charging_bases.iloc[int(pre_base_path[0])]
    end_station = charging_bases.iloc[int(post_base_path[-1])]

    legs: List[dict] = []
    seq = 1

    pre_legs, seq = build_base_path_legs(route_id, seq, pre_base_path, charging_bases)
    legs.extend(pre_legs)

    # Delivery sortie: chargeable node -> household(s) -> chargeable node.
    legs.append(
        make_leg(
            route_id=route_id,
            leg_sequence=seq,
            from_id=str(entry_base["base_id"]),
            from_type=str(entry_base["base_type"]),
            from_name=str(entry_base["name"]),
            from_xy=(float(entry_base["x_m"]), float(entry_base["y_m"])),
            to_id=hh1_id,
            to_type="household",
            to_name=hh1_id,
            to_xy=hh1_xy,
        )
    )
    seq += 1

    if hh2_xy is not None:
        legs.append(
            make_leg(
                route_id=route_id,
                leg_sequence=seq,
                from_id=hh1_id,
                from_type="household",
                from_name=hh1_id,
                from_xy=hh1_xy,
                to_id=hh2_id,
                to_type="household",
                to_name=hh2_id,
                to_xy=hh2_xy,
            )
        )
        seq += 1
        last_hh_id = hh2_id
        last_hh_xy = hh2_xy
    else:
        last_hh_id = hh1_id
        last_hh_xy = hh1_xy

    legs.append(
        make_leg(
            route_id=route_id,
            leg_sequence=seq,
            from_id=last_hh_id,
            from_type="household",
            from_name=last_hh_id,
            from_xy=last_hh_xy,
            to_id=str(exit_base["base_id"]),
            to_type=str(exit_base["base_type"]),
            to_name=str(exit_base["name"]),
            to_xy=(float(exit_base["x_m"]), float(exit_base["y_m"])),
        )
    )
    seq += 1

    post_legs, seq = build_base_path_legs(route_id, seq, post_base_path, charging_bases)
    legs.extend(post_legs)

    total_distance_km = route_total_distance_km(legs)
    household_ids = get_household_ids_from_group(ordered_group)

    route = {
        "route_id": route_id,
        "start_id": str(start_station["base_id"]),
        "start_type": str(start_station["base_type"]),
        "start_station_name": str(start_station["name"]),
        "end_station_name": str(end_station["name"]),
        "num_households": int(len(ordered_group)),
        "total_food_kg": round(total_food_kg, 2),
        "drone_payload_kg": round(float(DRONE_PAYLOAD_KG), 2),
        "remaining_payload_kg": round(float(DRONE_PAYLOAD_KG) - total_food_kg, 2),
        "payload_feasible": bool(total_food_kg <= DRONE_PAYLOAD_KG),
        "drone_brand": DRONE_BRAND,
        "drone_model": DRONE_MODEL,
        "drone_model_name": DRONE_MODEL_NAME,
        "household_ids": ",".join(household_ids),
        "total_distance_km": round(total_distance_km, 3),
        "energy_aware_total_distance_km": round(total_distance_km, 3),
        "energy_aware_km_per_household": round(total_distance_km / max(1, len(ordered_group)), 3),
        "energy_aware_chunk_size_chosen": int(len(ordered_group)),
        "separate_routes_baseline_km": np.nan,
        "combined_vs_separate_ratio": np.nan,
        "route_selection_rule": "station-origin route; chargers are charging relays only; household visit(s) occur inside one <=40 km charge cycle between chargeable nodes",
        "charger_model": "charging_only_not_package_origin",
        "delivery_sortie_distance_km": round(sortie_m / 1000.0, 3),
        "route_node_sequence": route_node_sequence_from_legs(legs),
        "feasible_route": True,
    }

    return route, legs, None



"""
For two-household merging, evaluate both possible orders:
        H1 -> H2 and H2 -> H1.
For one household, there is only one order.
列出先送谁、后送谁的可能顺序
"""
def possible_household_orders_for_group(group: pd.DataFrame, start_xy: Tuple[float, float]) -> List[pd.DataFrame]:

    group = group.copy().reset_index(drop=True)

    if len(group) <= 1:
        return [group]

    if len(group) == 2:
        return [
            group.iloc[[0, 1]].copy().reset_index(drop=True),
            group.iloc[[1, 0]].copy().reset_index(drop=True),
        ]

    return [order_households_nearest_neighbor(start_xy, group)]



"""
It chooses the best feasible route among different household visiting orders.
"""
def best_station_origin_route_for_group(
    route_id: str,
    group: pd.DataFrame,
    charging_bases: pd.DataFrame,
    graph: List[List[Tuple[int, float]]],
    base_tree: cKDTree,
    station_base_idxs: List[int],
) -> Tuple[Optional[dict], List[dict], Optional[str]]:
    """
    Evaluate one-household and two-household ordered groups under the corrected
    charge-cycle rule and choose the lowest-distance feasible route.
    """
    if len(group) == 0:
        return None, [], "empty group"

    best_route = None
    best_legs: List[dict] = []
    best_total = float("inf")
    best_reason = None

    for ordered_group in possible_household_orders_for_group(group, (0.0, 0.0)):
        route, legs, reason = build_station_origin_route_for_order(
            route_id=route_id,
            ordered_group=ordered_group,
            charging_bases=charging_bases,
            graph=graph,
            base_tree=base_tree,
            station_base_idxs=station_base_idxs,
        )

        if route is None:
            best_reason = reason
            continue

        total_km = route_total_distance_km(legs)
        if total_km < best_total:
            best_total = total_km
            best_route = route
            best_legs = legs

    if best_route is None:
        return None, [], best_reason or "no feasible station-origin route"

    return best_route, best_legs, None



"""
It evaluates whether a small group of households can be served in one route and
calculates the result.
"""
def evaluate_route_chunk(
    route_id: str,
    chunk: pd.DataFrame,
    charging_bases: pd.DataFrame,
    graph: List[List[Tuple[int, float]]],
    base_tree: cKDTree,
    station_base_idxs: List[int],
) -> Tuple[Optional[dict], List[dict], Optional[str]]:
    return best_station_origin_route_for_group(
        route_id=route_id,
        group=chunk,
        charging_bases=charging_bases,
        graph=graph,
        base_tree=base_tree,
        station_base_idxs=station_base_idxs,
    )


"""
It finds the best route for serving one household.
"""
def best_single_household_route(
    route_id: str,
    hh: pd.Series,
    charging_bases: pd.DataFrame,
    graph: List[List[Tuple[int, float]]],
    base_tree: cKDTree,
    station_base_idxs: List[int],
) -> Tuple[Optional[dict], List[dict], Optional[str]]:

    #Single-household station-origin route.
    group = pd.DataFrame([hh]).reset_index(drop=True)
    route, legs, reason = best_station_origin_route_for_group(
        route_id=route_id,
        group=group,
        charging_bases=charging_bases,
        graph=graph,
        base_tree=base_tree,
        station_base_idxs=station_base_idxs,
    )

    if route is None:
        return None, [], reason or "no feasible station-origin single-household route"

    total_km = route_total_distance_km(legs)
    food_kg = household_food_kg(hh)
    route["total_food_kg"] = round(food_kg, 2)
    route["drone_payload_kg"] = round(float(DRONE_PAYLOAD_KG), 2)
    route["remaining_payload_kg"] = round(float(DRONE_PAYLOAD_KG) - food_kg, 2)
    route["payload_feasible"] = bool(food_kg <= DRONE_PAYLOAD_KG)
    route["energy_aware_total_distance_km"] = round(total_km, 3)
    route["energy_aware_km_per_household"] = round(total_km, 3)
    route["energy_aware_chunk_size_chosen"] = 1
    route["separate_routes_baseline_km"] = round(total_km, 3)
    route["combined_vs_separate_ratio"] = 1.0
    route["route_selection_rule"] = "single-household station-origin route; charger is relay only"
    route["charger_model"] = "charging_only_not_package_origin"
    route["route_node_sequence"] = route_node_sequence_from_legs(legs)

    return route, legs, None



"""
It checks whether two households can be merged into one route.
They are merged only if the combined route saves distance compared with serving them separately.
判断两个家庭一起送划不划算。
"""
def best_two_household_merge_route(
    route_id: str,
    hh_a: pd.Series,
    hh_b: pd.Series,
    separate_baseline_km: float,
    charging_bases: pd.DataFrame,
    graph: List[List[Tuple[int, float]]],
    base_tree: cKDTree,
    station_base_idxs: List[int],
) -> Tuple[Optional[dict], List[dict], Optional[str]]:

    #Try to merge exactly two households.
    group = pd.DataFrame([hh_a, hh_b]).reset_index(drop=True)
    route, legs, reason = best_station_origin_route_for_group(
        route_id=route_id,
        group=group,
        charging_bases=charging_bases,
        graph=graph,
        base_tree=base_tree,
        station_base_idxs=station_base_idxs,
    )

    if route is None:
        return None, [], reason or "no feasible two-household station-origin merge route"

    combined_km = route_total_distance_km(legs)
    ratio = combined_km / separate_baseline_km if separate_baseline_km > 0 else float("inf")

    route["energy_aware_total_distance_km"] = round(combined_km, 3)
    route["energy_aware_km_per_household"] = round(combined_km / 2.0, 3)
    route["energy_aware_chunk_size_chosen"] = 2
    route["separate_routes_baseline_km"] = round(separate_baseline_km, 3)
    route["combined_vs_separate_ratio"] = round(ratio, 4)
    route["distance_saved_vs_separate_km"] = round(separate_baseline_km - combined_km, 3)
    route["route_selection_rule"] = (
        "two-household merge candidate; accepted only if it saves distance vs separate station-origin routes"
    )
    route["charger_model"] = "charging_only_not_package_origin"
    route["route_node_sequence"] = route_node_sequence_from_legs(legs)

    return route, legs, None




"""
Station-origin routing with conditional two-household merging.
Main rules:
    1. Every route starts from a train station.
    2. Selected building chargers are only charging relays, not package origins.
    3. Each route must end at a train station.
    4. The default is one household per route.
    5. Two households are merged only when the combined route is shorter than
       serving those two households separately.
"""
def split_into_routes(
    households: pd.DataFrame,
    charging_bases: pd.DataFrame,
    graph: List[List[Tuple[int, float]]],
    base_tree: cKDTree,
    station_base_idxs: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    served_ids: List[str] = []
    routes: List[dict] = []
    legs: List[dict] = []
    unserved_reasons: Dict[str, str] = {}

    # Sort by distance band and station distance for stable output.
    sort_cols = []
    if "distance_band" in households.columns:
        sort_cols.append("distance_band")
    if "nearest_station_distance_km" in households.columns:
        sort_cols.append("nearest_station_distance_km")

    if sort_cols:
        ordered_households = households.sort_values(sort_cols).reset_index(drop=True)
    else:
        ordered_households = households.reset_index(drop=True)

    # Step A: build the best station-origin single route for every household.
    single_routes: Dict[str, Tuple[dict, List[dict]]] = {}
    feasible_hh_rows: Dict[str, pd.Series] = {}

    for _, hh in ordered_households.iterrows():
        hh_id = str(hh["delivery_household_id"])
        route, route_legs, reason = best_single_household_route(
            route_id=f"TMP_SINGLE_{hh_id}",
            hh=hh,
            charging_bases=charging_bases,
            graph=graph,
            base_tree=base_tree,
            station_base_idxs=station_base_idxs,
        )

        if route is None:
            unserved_reasons[hh_id] = reason or "cannot build station-origin route with final return to station"
            continue

        single_routes[hh_id] = (route, route_legs)
        feasible_hh_rows[hh_id] = hh

    # Step B: test all two-household combinations and keep only real distance-saving merges.
    merge_candidates: List[dict] = []
    feasible_ids = list(feasible_hh_rows.keys())

    if MAX_HOUSEHOLDS_PER_DRONE_ROUTE >= 2 and not FORCE_SINGLE_HOUSEHOLD_ROUTES:
        for i in range(len(feasible_ids)):
            for j in range(i + 1, len(feasible_ids)):
                id_a = feasible_ids[i]
                id_b = feasible_ids[j]

                baseline_km = float(single_routes[id_a][0]["total_distance_km"]) + float(single_routes[id_b][0]["total_distance_km"])

                pair_route, pair_legs, reason = best_two_household_merge_route(
                    route_id=f"TMP_PAIR_{id_a}_{id_b}",
                    hh_a=feasible_hh_rows[id_a],
                    hh_b=feasible_hh_rows[id_b],
                    separate_baseline_km=baseline_km,
                    charging_bases=charging_bases,
                    graph=graph,
                    base_tree=base_tree,
                    station_base_idxs=station_base_idxs,
                )

                if pair_route is None:
                    continue

                ratio = float(pair_route["combined_vs_separate_ratio"])
                distance_saved = float(pair_route["distance_saved_vs_separate_km"])

                if ratio <= COMBINE_ROUTE_ONLY_IF_SAVES_RATIO and distance_saved > 0:
                    merge_candidates.append({
                        "ids": (id_a, id_b),
                        "route": pair_route,
                        "legs": pair_legs,
                        "baseline_km": baseline_km,
                        "combined_km": float(pair_route["total_distance_km"]),
                        "ratio": ratio,
                        "distance_saved_km": distance_saved,
                    })

    # Choose non-overlapping pair merges greedily by largest absolute distance saving.
    merge_candidates.sort(key=lambda x: (x["distance_saved_km"], -x["ratio"]), reverse=True)
    used_ids: set[str] = set()
    selected_route_packages: List[Tuple[dict, List[dict]]] = []

    for cand in merge_candidates:
        id_a, id_b = cand["ids"]
        if id_a in used_ids or id_b in used_ids:
            continue

        route = dict(cand["route"])
        route["route_selection_rule"] = (
            f"accepted two-household merge because combined route is <= "
            f"{COMBINE_ROUTE_ONLY_IF_SAVES_RATIO:.2f} of separate route distance"
        )
        selected_route_packages.append((route, cand["legs"]))
        used_ids.add(id_a)
        used_ids.add(id_b)

    # Add remaining feasible households as single routes.
    for hh_id in feasible_ids:
        if hh_id in used_ids:
            continue
        selected_route_packages.append(single_routes[hh_id])
        used_ids.add(hh_id)

    # Step C: assign final route IDs and collect final outputs.
    for route_counter, (route, route_legs) in enumerate(selected_route_packages):
        final_route_id = f"R{route_counter:03d}"
        final_route, final_legs = renumber_route_and_legs(route, route_legs, final_route_id)

        routes.append(final_route)
        legs.extend(final_legs)

        for hh_id in str(final_route["household_ids"]).split(","):
            if hh_id:
                served_ids.append(hh_id)

    status = households.copy()
    status["service_status"] = np.where(
        status["delivery_household_id"].astype(str).isin(served_ids),
        "served",
        "unserved",
    )
    status["unserved_reason"] = status["delivery_household_id"].astype(str).map(unserved_reasons).fillna("")

    return pd.DataFrame(routes), pd.DataFrame(legs), status





"""
Unserved household diagnostics and rescue charger repair
It checks the best charging cycle for a household: from one charging base,
visit the household, and reach another charging base within the battery range.
"""
def best_charge_cycle_for_household(
    hh_row: pd.Series,
    charging_bases: pd.DataFrame,
    graph: List[List[Tuple[int, float]]],
    base_tree: cKDTree,
    station_base_idxs: List[int],
) -> dict:

    hh_id = str(hh_row.get("delivery_household_id", hh_row.get("household_id", "")))
    hh_xy = (float(hh_row["x_m"]), float(hh_row["y_m"]))

    base_idxs = candidate_base_idxs_within_range(hh_xy, base_tree)

    nearest_dist_km = np.nan
    nearest_base_id = ""
    if len(charging_bases) > 0:
        d, idx = base_tree.query(np.array([hh_xy], dtype=float), k=1)
        nearest_dist_km = float(d[0]) / 1000.0
        nearest_base_id = str(charging_bases.iloc[int(idx[0])]["base_id"])

    best_sortie_m = float("inf")
    best_entry_id = ""
    best_exit_id = ""
    best_pre_connected = False
    best_post_connected = False

    for entry_idx in base_idxs:
        pre_path = dijkstra_shortest_path(graph, station_base_idxs, [entry_idx])
        if pre_path is None:
            continue
        entry = charging_bases.iloc[int(entry_idx)]
        entry_xy = (float(entry["x_m"]), float(entry["y_m"]))

        for exit_idx in base_idxs:
            post_path = dijkstra_shortest_path(graph, [exit_idx], station_base_idxs)
            if post_path is None:
                continue
            exit_base = charging_bases.iloc[int(exit_idx)]
            exit_xy = (float(exit_base["x_m"]), float(exit_base["y_m"]))
            sortie_m = euclidean_m(entry_xy, hh_xy) + euclidean_m(hh_xy, exit_xy)
            if sortie_m < best_sortie_m:
                best_sortie_m = sortie_m
                best_entry_id = str(entry["base_id"])
                best_exit_id = str(exit_base["base_id"])
                best_pre_connected = True
                best_post_connected = True

    if math.isinf(best_sortie_m):
        best_sortie_km = np.nan
        charge_cycle_feasible = False
    else:
        best_sortie_km = best_sortie_m / 1000.0
        charge_cycle_feasible = best_sortie_m <= DRONE_RANGE_M

    if not base_idxs:
        reason = "No station/charger within 40 km of this household."
    elif math.isinf(best_sortie_m):
        reason = "There is a nearby station/charger, but it is not connected to a valid station-return network."
    elif not charge_cycle_feasible:
        reason = (
            "It looks reachable by one leg, but the full charge cycle "
            "station/charger -> household -> station/charger is more than 40 km."
        )
    else:
        reason = "Feasible under the charge-cycle rule. If unserved, check pairing/route construction."

    return {
        "delivery_household_id": hh_id,
        "service_status": str(hh_row.get("service_status", "")),
        "nearest_base_id": nearest_base_id,
        "nearest_base_distance_km": round(nearest_dist_km, 3) if not pd.isna(nearest_dist_km) else np.nan,
        "num_bases_within_40km": int(len(base_idxs)),
        "looks_reachable_by_single_leg": bool(len(base_idxs) > 0),
        "best_entry_base_id": best_entry_id,
        "best_exit_base_id": best_exit_id,
        "best_single_household_charge_cycle_km": round(best_sortie_km, 3) if not pd.isna(best_sortie_km) else np.nan,
        "charge_cycle_feasible": bool(charge_cycle_feasible),
        "pre_path_connected_to_station": bool(best_pre_connected),
        "post_path_connected_to_station": bool(best_post_connected),
        "diagnosis": reason,
    }



"""
It creates a diagnostic table for unserved households and explains why they cannot be served,
such as no nearby charger, no return path, or battery-cycle infeasibility.
"""
def build_unserved_household_diagnostics(
    households_status: pd.DataFrame,
    charging_bases: pd.DataFrame,
    graph: List[List[Tuple[int, float]]],
    base_tree: cKDTree,
    station_base_idxs: List[int],
) -> pd.DataFrame:
    rows = []
    for _, hh in households_status.iterrows():
        rows.append(best_charge_cycle_for_household(hh, charging_bases, graph, base_tree, station_base_idxs))
    return pd.DataFrame(rows)



"""
It checks whether one household can be served using the currently selected chargers.
"""
def route_feasible_for_single_household_with_selected_chargers(
    hh_row: pd.Series,
    station_gdf: gpd.GeoDataFrame,
    selected_chargers: pd.DataFrame,
) -> bool:
    temp_bases = build_charging_bases(station_gdf, selected_chargers)
    temp_graph, temp_tree = build_base_graph(temp_bases)
    temp_station_idxs = get_station_base_indices(temp_bases)
    route, legs, reason = best_single_household_route(
        route_id="TMP_RESCUE_TEST",
        hh=hh_row,
        charging_bases=temp_bases,
        graph=temp_graph,
        base_tree=temp_tree,
        station_base_idxs=temp_station_idxs,
    )
    return route is not None and len(legs) > 0



"""
If some households are unserved, it tries to add local rescue chargers nearby and checks again
whether they can be served.
"""
def rescue_unserved_households_with_local_chargers(
    selected_chargers: pd.DataFrame,
    candidates_df: pd.DataFrame,
    station_gdf: gpd.GeoDataFrame,
    households_status: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Try adding a small number of local rescue chargers for households that remain unserved.
    if not ENABLE_UNSERVED_HOUSEHOLD_RESCUE:
        return selected_chargers.copy().reset_index(drop=True), pd.DataFrame()

    selected = selected_chargers.copy().reset_index(drop=True)
    logs: List[dict] = []

    unserved = households_status[households_status["service_status"].astype(str) == "unserved"].copy().reset_index(drop=True)
    if len(unserved) == 0:
        return selected, pd.DataFrame([{"action": "no_unserved_households_to_rescue"}])

    cand = candidates_df.copy().reset_index(drop=True)
    cand_xy = cand[["x_m", "y_m"]].to_numpy(dtype=float)
    cand_tree = cKDTree(cand_xy)

    added_count = 0

    for _, hh in unserved.iterrows():
        if added_count >= MAX_UNSERVED_RESCUE_CHARGERS:
            logs.append({
                "action": "stop_max_rescue_chargers_reached",
                "max_rescue_chargers": int(MAX_UNSERVED_RESCUE_CHARGERS),
            })
            break

        hh_id = str(hh["delivery_household_id"])
        hh_source_id = str(hh.get("household_id", ""))
        hh_xy = (float(hh["x_m"]), float(hh["y_m"]))

        # If it became feasible because another household's rescue charger was added, skip.
        if route_feasible_for_single_household_with_selected_chargers(hh, station_gdf, selected):
            logs.append({
                "household_id": hh_id,
                "action": "already_feasible_after_previous_rescue",
            })
            continue

        existing_source_ids = set(selected.get("household_id", pd.Series(dtype=str)).astype(str).tolist())
        candidate_idxs = cand_tree.query_ball_point(np.array(hh_xy, dtype=float), r=RESCUE_SEARCH_RADIUS_M)

        if not candidate_idxs:
            logs.append({
                "household_id": hh_id,
                "action": "no_local_candidate_within_rescue_search_radius",
                "rescue_search_radius_km": RESCUE_SEARCH_RADIUS_KM,
            })
            continue

        # Sort candidates by distance to household, but test only a capped number.
        candidate_idxs = sorted(
            [int(i) for i in candidate_idxs],
            key=lambda i: euclidean_m(hh_xy, (float(cand_xy[i][0]), float(cand_xy[i][1]))),
        )[:MAX_RESCUE_CANDIDATES_PER_HOUSEHOLD]

        accepted = False
        tested_count = 0
        skipped_same_or_existing = 0
        skipped_too_close_to_household = 0
        skipped_not_connected = 0

        # Build current base tree once per household to reject isolated candidates fast.
        current_bases = build_charging_bases(station_gdf, selected)
        current_base_xy = current_bases[["x_m", "y_m"]].to_numpy(dtype=float)
        current_base_tree = cKDTree(current_base_xy)

        for ci in candidate_idxs:
            r = cand.iloc[int(ci)]
            c_source_id = str(r.get("household_id", ""))
            c_xy = (float(r["x_m"]), float(r["y_m"]))
            tested_count += 1

            # Do not place the charger directly on the same household point.
            if c_source_id == hh_source_id or c_source_id in existing_source_ids:
                skipped_same_or_existing += 1
                continue

            dist_hh_m = euclidean_m(hh_xy, c_xy)
            if dist_hh_m < MIN_RESCUE_CHARGER_DISTANCE_FROM_HOUSEHOLD_M:
                skipped_too_close_to_household += 1
                continue

            d_net, _ = current_base_tree.query(np.array([c_xy], dtype=float), k=1)
            if float(d_net[0]) > DRONE_RANGE_M:
                skipped_not_connected += 1
                continue

            rescue_id = f"RC_{added_count:04d}"
            rescue_row = {
                "charger_id": rescue_id,
                "household_id": c_source_id,
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
                "x_m": float(r["x_m"]),
                "y_m": float(r["y_m"]),
                "rail_distance_km": float(r.get("rail_distance_km", np.nan)),
                "charger_role": "rescue_for_unserved_household",
                "source": "local_all_points_csv_rescue",
                "rescued_household_id": hh_id,
            }

            trial_selected = pd.concat([selected, pd.DataFrame([rescue_row])], ignore_index=True)

            if route_feasible_for_single_household_with_selected_chargers(hh, station_gdf, trial_selected):
                selected = trial_selected.reset_index(drop=True)
                added_count += 1
                accepted = True
                logs.append({
                    "household_id": hh_id,
                    "action": "added_rescue_charger",
                    "charger_id": rescue_id,
                    "candidate_source_id": c_source_id,
                    "distance_household_to_rescue_charger_km": round(dist_hh_m / 1000.0, 3),
                    "distance_rescue_charger_to_existing_network_km": round(float(d_net[0]) / 1000.0, 3),
                    "tested_candidates": int(tested_count),
                })
                break

        if not accepted:
            logs.append({
                "household_id": hh_id,
                "action": "no_rescue_charger_found",
                "tested_candidates": int(tested_count),
                "skipped_same_or_existing": int(skipped_same_or_existing),
                "skipped_too_close_to_household": int(skipped_too_close_to_household),
                "skipped_not_connected_to_network": int(skipped_not_connected),
            })

    return selected.reset_index(drop=True), pd.DataFrame(logs)
ROUTE_COLOR_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]



"""
It gives each route a stable index so the map colors or order remain consistent.
"""
def stable_route_index(route_id: str) -> int:
    return sum(ord(ch) for ch in str(route_id))
"""
It chooses a color for a route based on the route ID.
"""
def route_line_color(route_id: str, is_relay: bool = False) -> str:
    if not ROUTE_COLOR_BY_ROUTE:
        return "darkblue" if not is_relay else "green"
    return ROUTE_COLOR_PALETTE[stable_route_index(str(route_id)) % len(ROUTE_COLOR_PALETTE)]
"""
If route lines overlap, it slightly offsets them so the map is easier to read
"""
def offset_segment_xy(
    from_xy: Tuple[float, float],
    to_xy: Tuple[float, float],
    route_id: str,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Offset a route segment sideways so overlapping lines become visible."""
    if not USE_ROUTE_LINE_OFFSET or ROUTE_LINE_SPACING_METERS <= 0:
        return from_xy, to_xy

    x1, y1 = float(from_xy[0]), float(from_xy[1])
    x2, y2 = float(to_xy[0]), float(to_xy[1])
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return from_xy, to_xy

    # Deterministic offset bucket: -3, -2, -1, 0, 1, 2, 3.
    bucket = (stable_route_index(route_id) % 7) - 3
    offset_m = bucket * float(ROUTE_LINE_SPACING_METERS)

    # Normal vector to the segment.
    nx, ny = -dy / length, dx / length
    return (x1 + nx * offset_m, y1 + ny * offset_m), (x2 + nx * offset_m, y2 + ny * offset_m)
"""
It converts x/y route segments back to latitude/longitude locations for map display.
"""
def xy_segment_to_latlon_locations(
    from_xy: Tuple[float, float],
    to_xy: Tuple[float, float],
    route_id: str,
) -> List[List[float]]:
    off_from_xy, off_to_xy = offset_segment_xy(from_xy, to_xy, route_id)
    seg_gdf = gpd.GeoDataFrame(
        {"id": [0, 1]},
        geometry=gpd.points_from_xy([float(off_from_xy[0]), float(off_to_xy[0])], [float(off_from_xy[1]), float(off_to_xy[1])]),
        crs=PROJECTED_CRS,
    ).to_crs(WGS84_CRS)
    return [
        [float(seg_gdf.geometry.y.iloc[0]), float(seg_gdf.geometry.x.iloc[0])],
        [float(seg_gdf.geometry.y.iloc[1]), float(seg_gdf.geometry.x.iloc[1])],
    ]
"""
It creates the final Folium map showing train stations, chargers, households, recharging stops,
route lines, and unserved households.
"""
def make_map(
    out_html: Path,
    station_gdf: gpd.GeoDataFrame,
    rail_line,
    rail_buffer,
    candidates_df: pd.DataFrame,
    selected_chargers: pd.DataFrame,
    households_status: pd.DataFrame,
    route_legs: pd.DataFrame,
) -> None:
    rail_wgs = gpd.GeoSeries([rail_line], crs=PROJECTED_CRS).to_crs(WGS84_CRS)
    buffer_wgs = gpd.GeoSeries([rail_buffer], crs=PROJECTED_CRS).to_crs(WGS84_CRS)
    stations_wgs = station_gdf.to_crs(WGS84_CRS)

    center_lat = float(stations_wgs.geometry.y.mean())
    center_lon = float(stations_wgs.geometry.x.mean())

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles="CartoDB positron",
        control_scale=True,
    )

    folium.GeoJson(
        data=buffer_wgs.iloc[0].__geo_interface__,
        name="300 km railway buffer",
        style_function=lambda _: {
            "color": "#777777",
            "weight": 1,
            "fillColor": "#cccccc",
            "fillOpacity": 0.10,
        },
    ).add_to(m)

    fg_rail = folium.FeatureGroup(name="Railway corridor line", show=True)
    rail_coords = [(lat, lon) for lon, lat in rail_wgs.iloc[0].coords]
    folium.PolyLine(rail_coords, color="black", weight=4, opacity=0.9).add_to(fg_rail)
    fg_rail.add_to(m)

    fg_st = folium.FeatureGroup(name="Train stations with charging radius", show=True)
    for _, r in stations_wgs.iterrows():
        folium.Circle(
            location=[float(r.geometry.y), float(r.geometry.x)],
            radius=DRONE_RANGE_M,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.04,
            weight=1,
            tooltip=f"{r['name']} charging radius = {DRONE_RANGE_KM:.0f} km",
        ).add_to(fg_st)

        folium.Marker(
            location=[float(r.geometry.y), float(r.geometry.x)],
            icon=folium.Icon(color="blue", icon="train", prefix="fa"),
            popup=f"<b>{r['name']}</b><br>Train station / charging base",
        ).add_to(fg_st)
    fg_st.add_to(m)

    if SHOW_ALL_CHARGER_CANDIDATES:
        fg_cand = folium.FeatureGroup(name="All charger candidates in 300 km buffer", show=False)
        cluster = MarkerCluster().add_to(fg_cand)
        for _, r in candidates_df.iterrows():
            folium.CircleMarker(
                location=[float(r["lat"]), float(r["lon"])],
                radius=1,
                color="gray",
                fill=True,
                fill_opacity=0.25,
                weight=0,
            ).add_to(cluster)
        fg_cand.add_to(m)

    has_only_fixed_chargers = (
        len(selected_chargers) > 0
        and "fixed_permanent_charger" in selected_chargers.columns
        and selected_chargers["fixed_permanent_charger"].astype(bool).all()
    )
    charger_layer_name = "Fixed permanent chargers only (green)" if has_only_fixed_chargers else "Selected charging-only relay chargers"
    fg_ch_service = folium.FeatureGroup(name=charger_layer_name, show=True)
    for _, r in selected_chargers.iterrows():
        charger_role = str(r.get("charger_role", ""))
        charger_source = str(r.get("source", "local"))
        is_fixed_permanent = bool(r.get("fixed_permanent_charger", False)) or charger_role == "fixed_permanent"
        layer = fg_ch_service
        marker_color = "green" if is_fixed_permanent else "purple"
        range_label_color = "green" if is_fixed_permanent else "purple"

        if SHOW_CHARGER_COVERAGE_CIRCLES:
            folium.Circle(
                location=[float(r["lat"]), float(r["lon"])],
                radius=DRONE_RANGE_M,
                color=marker_color,
                fill=True,
                fill_color=marker_color,
                fill_opacity=0.04,
                weight=1,
            ).add_to(layer)

        folium.Marker(
            location=[float(r["lat"]), float(r["lon"])],
            icon=folium.Icon(
                color=marker_color,
                icon="bolt",
                prefix="fa",
            ),
            popup=(
                f"<b>{r['charger_id']}</b><br>"
                f"Charging-only relay point<br>{DRONE_RANGE_KM:.0f} km charging range is shown as a {range_label_color} circle<br>"
                f"Source building/facility: {r['household_id']}<br>"
                f"Role: {charger_role}<br>"
                f"Source: {charger_source}<br>"
                f"Fixed permanent charger: {is_fixed_permanent}"
            ),
        ).add_to(layer)

    fg_ch_service.add_to(m)

    fg_hh = folium.FeatureGroup(name="Random households", show=True)
    for _, r in households_status.iterrows():
        status = str(r["service_status"])
        if status != "served":
            color = "black"
            radius = 6
            border_weight = 2
        else:
            band = str(r.get("distance_band", ""))
            if "medium" in band:
                color = "orange"
            elif "far_80" in band:
                color = "green"
            elif "very_far" in band:
                color = "cadetblue"
            elif "extreme_far" in band:
                color = "purple"
            else:
                color = "orange"
            radius = 5
            border_weight = 1

        if status == "served" or SHOW_UNSERVED_HOUSEHOLDS:
            popup = (
                f"<b>{r['delivery_household_id']}</b><br>"
                f"Status: {status}<br>"
                f"Demand: {r['demand_packages']}<br>"
                f"Distance band: {r.get('distance_band', '')}<br>"
                f"Nearest station: {r['nearest_station_name']} ({r['nearest_station_distance_km']:.1f} km)<br>"
                f"Nearest base: {r['nearest_base_id']} ({r['nearest_base_distance_km']:.1f} km)<br>"
                f"Reason: {r.get('unserved_reason', '')}"
            )
            folium.CircleMarker(
                location=[float(r["lat"]), float(r["lon"])],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                weight=border_weight,
                popup=popup,
            ).add_to(fg_hh)
    fg_hh.add_to(m)

    fg_routes = folium.FeatureGroup(name="Drone route legs with charger relays", show=True)
    if len(route_legs) > 0:
        from_gdf = gpd.GeoDataFrame(
            route_legs.copy(),
            geometry=gpd.points_from_xy(route_legs["from_x_m"], route_legs["from_y_m"]),
            crs=PROJECTED_CRS,
        ).to_crs(WGS84_CRS)
        to_gdf = gpd.GeoDataFrame(
            route_legs.copy(),
            geometry=gpd.points_from_xy(route_legs["to_x_m"], route_legs["to_y_m"]),
            crs=PROJECTED_CRS,
        ).to_crs(WGS84_CRS)

        # Build route-level statistics for clearer popups.
        route_stats: Dict[str, dict] = {}
        hh_food_lookup: Dict[str, float] = {}
        if len(households_status) > 0 and "food_kg" in households_status.columns:
            hh_food_lookup = {
                str(r["delivery_household_id"]): float(r.get("food_kg", 0.0))
                for _, r in households_status.iterrows()
            }
        for rid, grp in route_legs.groupby("route_id", sort=False):
            grp_sorted = grp.sort_values("leg_sequence")
            node_sequence = " -> ".join(
                [str(grp_sorted.iloc[0]["from_id"])] + grp_sorted["to_id"].astype(str).tolist()
            )
            household_ids_in_route: List[str] = []
            for _, leg_row in grp_sorted.iterrows():
                if str(leg_row.get("from_type", "")) == "household":
                    household_ids_in_route.append(str(leg_row.get("from_id", "")))
                if str(leg_row.get("to_type", "")) == "household":
                    household_ids_in_route.append(str(leg_row.get("to_id", "")))
            household_ids_in_route = list(dict.fromkeys([h for h in household_ids_in_route if h]))
            route_food_kg = sum(hh_food_lookup.get(h, 0.0) for h in household_ids_in_route)
            route_stats[str(rid)] = {
                "total_distance_km": round(float(grp_sorted["distance_km"].sum()), 3),
                "num_legs": int(len(grp_sorted)),
                "household_ids": ",".join(household_ids_in_route),
                "route_food_kg": round(float(route_food_kg), 2),
                "drone_payload_kg": round(float(DRONE_PAYLOAD_KG), 2),
                "remaining_payload_kg": round(float(DRONE_PAYLOAD_KG) - float(route_food_kg), 2),
                "num_relay_legs": int((
                    (grp_sorted["from_type"].astype(str) != "household")
                    & (grp_sorted["to_type"].astype(str) != "household")
                ).sum()),
                "start_node": str(grp_sorted.iloc[0]["from_id"]),
                "end_node": str(grp_sorted.iloc[-1]["to_id"]),
                "node_sequence": node_sequence,
            }

        for i, r in route_legs.reset_index(drop=True).iterrows():
            from_xy = (float(r["from_x_m"]), float(r["from_y_m"]))
            to_xy = (float(r["to_x_m"]), float(r["to_y_m"]))
            line_locations = xy_segment_to_latlon_locations(from_xy, to_xy, str(r["route_id"]))

            is_relay = ("household" not in [str(r["from_type"]), str(r["to_type"])])
            line_color = route_line_color(str(r["route_id"]), is_relay=is_relay)
            stats = route_stats.get(str(r["route_id"]), {})
            route_popup = (
                f"<b>Route {r['route_id']}</b><br>"
                f"Route total distance: {stats.get('total_distance_km', '')} km<br>"
                f"Route legs: {stats.get('num_legs', '')}<br>"
                f"Food demand: {stats.get('route_food_kg', '')} kg / Payload: {stats.get('drone_payload_kg', '')} kg<br>"
                f"Remaining payload: {stats.get('remaining_payload_kg', '')} kg<br>"
                f"Relay-only legs: {stats.get('num_relay_legs', '')}<br>"
                f"Start: {stats.get('start_node', '')}<br>"
                f"End: {stats.get('end_node', '')}<br>"
                f"<hr style='margin:4px 0;'>"
                f"Current leg {r['leg_sequence']}: {r['from_id']} → {r['to_id']}<br>"
                f"Current leg distance: {r['distance_km']} km<br>"
                f"<hr style='margin:4px 0;'>"
                f"Sequence:<br>{stats.get('node_sequence', '')}"
            )
            line = folium.PolyLine(
                locations=line_locations,
                color=line_color,
                weight=4 if not is_relay else 3,
                opacity=0.82 if not is_relay else 0.62,
                tooltip=f"{r['route_id']} leg {r['leg_sequence']}: {r['from_id']} -> {r['to_id']} ({r['distance_km']} km)",
                popup=(route_popup if SHOW_ROUTE_POPUPS else None),
            )
            line.add_to(fg_routes)
            if SHOW_ROUTE_DIRECTION_ARROWS:
                PolyLineTextPath(
                    line,
                    '➜',
                    repeat=True,
                    offset=8,
                    attributes={
                        'fill': line_color,
                        'font-weight': 'bold',
                        'font-size': '14'
                    },
                ).add_to(fg_routes)

    fg_routes.add_to(m)
    # show route origins, final returns, and actual recharging
    # locations in separate colors. These are derived from the final route legs,
    # not from all candidate chargers. A recharging stop is a train station or
    # selected charger that appears as an intermediate route node.
    fg_route_events = folium.FeatureGroup(name="Route origins, returns, and actual recharging stops", show=True)
    if len(route_legs) > 0:
        for rid, grp in route_legs.groupby("route_id", sort=False):
            grp_sorted = grp.sort_values("leg_sequence").reset_index(drop=True)
            if len(grp_sorted) == 0:
                continue

            nodes: List[dict] = []
            first = grp_sorted.iloc[0]
            nodes.append({
                "node_id": str(first["from_id"]),
                "node_type": str(first["from_type"]),
                "node_name": str(first["from_name"]),
                "x_m": float(first["from_x_m"]),
                "y_m": float(first["from_y_m"]),
            })
            for _, leg_row in grp_sorted.iterrows():
                nodes.append({
                    "node_id": str(leg_row["to_id"]),
                    "node_type": str(leg_row["to_type"]),
                    "node_name": str(leg_row["to_name"]),
                    "x_m": float(leg_row["to_x_m"]),
                    "y_m": float(leg_row["to_y_m"]),
                })

            if not nodes:
                continue

            # Convert all route event nodes in one projection call.
            event_gdf = gpd.GeoDataFrame(
                nodes,
                geometry=gpd.points_from_xy([float(n["x_m"]) for n in nodes], [float(n["y_m"]) for n in nodes]),
                crs=PROJECTED_CRS,
            ).to_crs(WGS84_CRS)

            # Route origin marker.
            origin = event_gdf.iloc[0]
            folium.Marker(
                location=[float(origin.geometry.y), float(origin.geometry.x)],
                icon=folium.Icon(color="green", icon="play", prefix="fa"),
                tooltip=f"{rid} origin: {origin['node_id']}",
                popup=(
                    f"<b>{rid} origin</b><br>"
                    f"Start node: {origin['node_id']}<br>"
                    f"Type: {origin['node_type']}"
                ),
            ).add_to(fg_route_events)

            # Final return marker.
            final = event_gdf.iloc[-1]
            folium.Marker(
                location=[float(final.geometry.y), float(final.geometry.x)],
                icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa"),
                tooltip=f"{rid} return: {final['node_id']}",
                popup=(
                    f"<b>{rid} final return</b><br>"
                    f"Return node: {final['node_id']}<br>"
                    f"Type: {final['node_type']}"
                ),
            ).add_to(fg_route_events)

            # Intermediate chargeable nodes = actual recharging stops.
            for step_idx in range(1, max(1, len(event_gdf) - 1)):
                node = event_gdf.iloc[step_idx]
                if str(node["node_type"]) not in {"charger", "train_station"}:
                    continue
                folium.Marker(
                    location=[float(node.geometry.y), float(node.geometry.x)],
                    icon=folium.Icon(color="orange", icon="bolt", prefix="fa"),
                    tooltip=f"{rid} recharging stop: {node['node_id']}",
                    popup=(
                        f"<b>Actual recharging stop</b><br>"
                        f"Route: {rid}<br>"
                        f"Step in route: {step_idx}<br>"
                        f"Node: {node['node_id']}<br>"
                        f"Type: {node['node_type']}<br>"
                        f"This marker shows where the drone recharges during the selected route."
                    ),
                ).add_to(fg_route_events)

    fg_route_events.add_to(m)

    served = int((households_status["service_status"] == "served").sum()) if len(households_status) else 0
    unserved = int((households_status["service_status"] == "unserved").sum()) if len(households_status) else 0

    # No fixed bottom-left legend is added. Use LayerControl and marker/line popups instead.
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(out_html)



# Main
def main(config: Optional[SimulationConfig] = None) -> None:
    if config is not None:
        apply_simulation_config(config)
    ensure_outdir(OUTDIR)

    print("Loading all Manitoba building centroid points ...")
    df_all = load_all_points_csv(INPUT_ALL_POINTS_CSV)
    print(f"  Loaded: {len(df_all):,}")

    print("Building stations, railway line, and 300 km railway buffer ...")
    station_gdf = build_station_gdf()
    rail_line, rail_buffer = build_rail_line_and_buffer(station_gdf)

    print("Filtering candidate points inside 300 km rail buffer ...")
    candidates_df = filter_points_within_rail_buffer(df_all, rail_line, rail_buffer)
    station_xy = station_gdf[["x_m", "y_m"]].to_numpy(dtype=float)
    candidates_df["nearest_station_distance_km"] = nearest_station_distance_km(
        candidates_df[["x_m", "y_m"]].to_numpy(dtype=float),
        station_xy,
    )
    candidates_df.to_csv(OUTDIR / "charger_candidates_within_300km_rail.csv", index=False)
    print(f"  Candidates in rail buffer: {len(candidates_df):,}")

    print("Building grid and selecting/loading chargers ...")
    grid_xy = make_grid_inside(rail_buffer, GRID_SPACING_M)

    if USE_FIXED_PERMANENT_CHARGERS:
        print("  Loading fixed permanent chargers from CSV ...")
        selected_chargers = load_fixed_permanent_chargers_csv(FIXED_PERMANENT_CHARGERS_CSV)
        grid_status = pd.DataFrame(grid_xy, columns=["x_m", "y_m"])
        grid_status["covered_by_station_or_selected_charger"] = False
        gap_repair_log = pd.DataFrame([{
            "action": "fixed_permanent_chargers_only_no_gap_repair",
            "fixed_permanent_chargers_csv": str(FIXED_PERMANENT_CHARGERS_CSV),
            "fixed_permanent_chargers_loaded": int(len(selected_chargers)),
            "additional_temporary_chargers_added": 0,
        }])
        print(f"  Fixed permanent chargers loaded: {len(selected_chargers):,}")
        print("  No model-selected temporary/connector chargers were added.")
    else:
        print("  Normal mode: selecting CONNECTED relay chargers from all candidate points ...")
        selected_chargers, grid_status = select_relay_chargers_from_grid(candidates_df, station_gdf, grid_xy)

        print("Repairing gaps using local building points and optional OSM Overpass facilities ...")
        selected_chargers, gap_repair_log = repair_gaps_with_connectors(
            selected_chargers=selected_chargers,
            candidates_df=candidates_df,
            station_gdf=station_gdf,
            grid_xy=grid_xy,
        )
        print(f"  Selected chargers after gap repair: {len(selected_chargers):,}")
        print(f"  Gap repair actions: {len(gap_repair_log):,}")

    selected_chargers.to_csv(OUTDIR / "selected_connected_relay_chargers.csv", index=False)
    gap_repair_log.to_csv(OUTDIR / "gap_repair_log.csv", index=False)

    print("Building charging bases and FINAL grid coverage diagnostics ...")
    charging_bases = build_charging_bases(station_gdf, selected_chargers)
    charging_bases.to_csv(OUTDIR / "charging_bases_station_plus_chargers.csv", index=False)
    # Recalculate grid coverage AFTER gap repair, so coverage_grid_status.csv
    # represents the final charger network, not the pre-repair network.
    final_grid_status = build_grid_coverage_status(
        grid_xy=grid_xy,
        charging_bases=charging_bases,
        previous_grid_status=grid_status,
    )
    coverage_summary = summarize_grid_coverage(final_grid_status)
    final_grid_status.to_csv(OUTDIR / "coverage_grid_status.csv", index=False)
    coverage_summary.to_csv(OUTDIR / "coverage_summary.csv", index=False)

    if len(coverage_summary) > 0:
        cov_row = coverage_summary.iloc[0]
        print(
            "  Grid coverage before gap repair: "
            f"{cov_row['coverage_ratio_before_gap_repair']:.4f}"
        )
        print(
            "  Grid coverage after gap repair:  "
            f"{cov_row['coverage_ratio_after_gap_repair']:.4f}"
        )
        print(
            "  Newly covered grid points: "
            f"{int(cov_row['newly_covered_by_gap_repair'])}"
        )

    print("Building charging base graph: stations + chargers ...")
    graph, base_tree = build_base_graph(charging_bases)
    station_base_idxs = get_station_base_indices(charging_bases)

    # Diagnostics: connected components roughly not needed, but degree is useful.
    degrees = [len(n) for n in graph]
    pd.DataFrame({
        "base_id": charging_bases["base_id"],
        "base_type": charging_bases["base_type"],
        "degree_within_40km": degrees,
    }).to_csv(OUTDIR / "charging_base_graph_diagnostics.csv", index=False)

    print("Preparing household candidate pool with station/base distances ...")
    household_pool = attach_distances_to_households(candidates_df, charging_bases, station_gdf)
    household_pool.to_csv(OUTDIR / "household_candidate_pool_with_distances.csv", index=False)

    print("Selecting STRATIFIED far households feasible under the full charge-cycle rule ...")
    random_households, sampling_diagnostics = sample_households(
        household_pool, charging_bases, graph, base_tree, station_base_idxs
    )
    random_households.to_csv(OUTDIR / "random_40_households.csv", index=False)
    sampling_diagnostics.to_csv(OUTDIR / "sampling_diagnostics.csv", index=False)
    print(f"  Random households selected: {len(random_households):,}")

    print("Building STATION-ORIGIN routes with conditional two-household merge ...")
    routes_df, legs_df, status_df = split_into_routes(
        random_households, charging_bases, graph, base_tree, station_base_idxs
    )

    diagnostics_before_rescue = build_unserved_household_diagnostics(
        status_df, charging_bases, graph, base_tree, station_base_idxs
    )
    diagnostics_before_rescue.to_csv(OUTDIR / "unserved_household_diagnostics_before_rescue.csv", index=False)

    rescue_log = pd.DataFrame()
    unserved_before_rescue = int((status_df["service_status"] == "unserved").sum()) if len(status_df) else 0

    if ENABLE_UNSERVED_HOUSEHOLD_RESCUE and unserved_before_rescue > 0 and not FIXED_PERMANENT_CHARGERS_ONLY:
        print(f"Trying rescue chargers for {unserved_before_rescue} unserved households ...")
        selected_chargers, rescue_log = rescue_unserved_households_with_local_chargers(
            selected_chargers=selected_chargers,
            candidates_df=candidates_df,
            station_gdf=station_gdf,
            households_status=status_df,
        )
        rescue_log.to_csv(OUTDIR / "unserved_household_rescue_log.csv", index=False)

        rescue_added = int((rescue_log.get("action", pd.Series(dtype=str)) == "added_rescue_charger").sum()) if len(rescue_log) else 0
        if rescue_added > 0:
            print(f"  Rescue chargers added: {rescue_added}. Rebuilding graph and rerouting ...")
            selected_chargers.to_csv(OUTDIR / "selected_connected_relay_chargers.csv", index=False)
            charging_bases = build_charging_bases(station_gdf, selected_chargers)
            charging_bases.to_csv(OUTDIR / "charging_bases_station_plus_chargers.csv", index=False)
            graph, base_tree = build_base_graph(charging_bases)
            station_base_idxs = get_station_base_indices(charging_bases)

            degrees = [len(n) for n in graph]
            pd.DataFrame({
                "base_id": charging_bases["base_id"],
                "base_type": charging_bases["base_type"],
                "degree_within_40km": degrees,
            }).to_csv(OUTDIR / "charging_base_graph_diagnostics.csv", index=False)

            # Recompute nearest base values for household popups after rescue chargers.
            random_households = attach_distances_to_households(random_households, charging_bases, station_gdf)

            routes_df, legs_df, status_df = split_into_routes(
                random_households, charging_bases, graph, base_tree, station_base_idxs
            )

            # Recompute final grid coverage after rescue chargers too.
            final_grid_status = build_grid_coverage_status(
                grid_xy=grid_xy,
                charging_bases=charging_bases,
                previous_grid_status=grid_status,
            )
            coverage_summary = summarize_grid_coverage(final_grid_status)
            final_grid_status.to_csv(OUTDIR / "coverage_grid_status.csv", index=False)
            coverage_summary.to_csv(OUTDIR / "coverage_summary.csv", index=False)
    else:
        if FIXED_PERMANENT_CHARGERS_ONLY and unserved_before_rescue > 0:
            rescue_log = pd.DataFrame([{
                "action": "skipped_rescue_because_fixed_permanent_chargers_only",
                "unserved_households_before_rescue": int(unserved_before_rescue),
                "additional_temporary_chargers_added": 0,
            }])
        rescue_log.to_csv(OUTDIR / "unserved_household_rescue_log.csv", index=False)

    diagnostics_after_rescue = build_unserved_household_diagnostics(
        status_df, charging_bases, graph, base_tree, station_base_idxs
    )
    diagnostics_after_rescue.to_csv(OUTDIR / "unserved_household_diagnostics_after_rescue.csv", index=False)

    routes_df.to_csv(OUTDIR / "drone_routes.csv", index=False)
    legs_df.to_csv(OUTDIR / "drone_route_legs.csv", index=False)
    status_df.to_csv(OUTDIR / "households_service_status.csv", index=False)
    selected_chargers.to_csv(OUTDIR / "selected_connected_relay_chargers.csv", index=False)

    if len(routes_df) > 0:
        cols = [
            "route_id",
            "num_households",
            "total_food_kg",
            "drone_payload_kg",
            "remaining_payload_kg",
            "payload_feasible",
            "total_distance_km",
            "energy_aware_km_per_household",
            "energy_aware_chunk_size_chosen",
            "separate_routes_baseline_km",
            "combined_vs_separate_ratio",
            "delivery_sortie_distance_km",
            "end_station_name",
            "household_ids",
            "route_node_sequence",
        ]
        route_efficiency = routes_df[[c for c in cols if c in routes_df.columns]].copy()
    else:
        route_efficiency = pd.DataFrame()
    route_efficiency.to_csv(OUTDIR / "route_efficiency_summary.csv", index=False)

    served = int((status_df["service_status"] == "served").sum()) if len(status_df) else 0
    unserved = int((status_df["service_status"] == "unserved").sum()) if len(status_df) else 0

    print(f"  Served households: {served}")
    print(f"  Unserved households: {unserved}")
    print(f"  Drone routes: {len(routes_df)}")

    if GENERATE_ROUTE_MAP:
        print("Drawing final map ...")
        make_map(
            OUTDIR / "final_drone_delivery_routes_map.html",
            station_gdf,
            rail_line,
            rail_buffer,
            candidates_df,
            selected_chargers,
            status_df,
            legs_df,
        )
    else:
        print("Skipping map drawing for runtime/study experiment ...")

    total_distance = float(legs_df["distance_km"].sum()) if len(legs_df) else 0.0

    summary = pd.DataFrame([{
        "input_all_points_count": int(len(df_all)),
        "candidate_points_within_300km_rail": int(len(candidates_df)),
        "selected_relay_chargers": int(len(selected_chargers)),
        "gap_repair_enabled": ENABLE_GAP_REPAIR,
        "gap_repair_actions": int(len(gap_repair_log)),
        "gap_repair_successful_connectors": int((gap_repair_log.get("action", pd.Series(dtype=str)) == "added_local_connector").sum() + (gap_repair_log.get("action", pd.Series(dtype=str)) == "added_osm_connector").sum()) if len(gap_repair_log) else 0,
        "gap_repair_skipped_unrepairable_gaps": int((gap_repair_log.get("action", pd.Series(dtype=str)) == "skipped_unrepairable_gap_and_continue").sum()) if len(gap_repair_log) else 0,
        "unserved_household_rescue_enabled": bool(ENABLE_UNSERVED_HOUSEHOLD_RESCUE and not FIXED_PERMANENT_CHARGERS_ONLY),
        "unserved_household_rescue_added_chargers": int((rescue_log.get("action", pd.Series(dtype=str)) == "added_rescue_charger").sum()) if len(rescue_log) else 0,
        "unserved_household_rescue_max_chargers": int(MAX_UNSERVED_RESCUE_CHARGERS),
        "max_gap_repair_attempts": int(MAX_GAP_REPAIR_ATTEMPTS),
        "use_overpass_if_local_fails": USE_OVERPASS_IF_LOCAL_FAILS,
        "generate_route_map": GENERATE_ROUTE_MAP,
        "use_fixed_permanent_chargers": bool(USE_FIXED_PERMANENT_CHARGERS),
        "fixed_permanent_chargers_only": bool(FIXED_PERMANENT_CHARGERS_ONLY),
        "fixed_permanent_chargers_csv": str(FIXED_PERMANENT_CHARGERS_CSV),
        "blue_temporary_chargers_added": 0 if FIXED_PERMANENT_CHARGERS_ONLY else None,
        "max_selected_chargers": MAX_SELECTED_CHARGERS,
        "min_selected_charger_spacing_km": MIN_SELECTED_CHARGER_SPACING_KM,
        "target_grid_coverage_ratio": TARGET_GRID_COVERAGE_RATIO,
        "grid_points_total": int(coverage_summary.iloc[0]["grid_points_total"]) if len(coverage_summary) else 0,
        "coverage_ratio_before_gap_repair": float(coverage_summary.iloc[0]["coverage_ratio_before_gap_repair"]) if len(coverage_summary) else 0.0,
        "coverage_ratio_after_gap_repair": float(coverage_summary.iloc[0]["coverage_ratio_after_gap_repair"]) if len(coverage_summary) else 0.0,
        "coverage_ratio_improvement": float(coverage_summary.iloc[0]["coverage_ratio_improvement"]) if len(coverage_summary) else 0.0,
        "newly_covered_by_gap_repair": int(coverage_summary.iloc[0]["newly_covered_by_gap_repair"]) if len(coverage_summary) else 0,
        "uncovered_after_gap_repair": int(coverage_summary.iloc[0]["uncovered_after_gap_repair"]) if len(coverage_summary) else 0,
        "connectivity_required_for_selected_chargers": True,
        "frontier_range_km": FRONTIER_RANGE_KM,
        "train_stations_as_charging_bases": int(len(station_gdf)),
        "total_charging_bases": int(len(charging_bases)),
        "drone_brand": DRONE_BRAND,
        "drone_model": DRONE_MODEL,
        "drone_model_name": DRONE_MODEL_NAME,
        "drone_range_km": DRONE_RANGE_KM,
        "drone_payload_kg": DRONE_PAYLOAD_KG,
        "drone_speed_kmh": DRONE_SPEED_KMH,
        "number_of_agents": NUMBER_OF_AGENTS,
        "agents_sufficient_for_routes": bool(len(routes_df) <= NUMBER_OF_AGENTS),
        "food_kg_min": FOOD_KG_MIN,
        "food_kg_max": FOOD_KG_MAX,
        "charger_unit_cost_cad": CHARGER_UNIT_COST_CAD,
        "estimated_total_charger_cost_cad": round(float(len(selected_chargers)) * float(CHARGER_UNIT_COST_CAD), 2),
        "max_distance_between_chargeable_nodes_km": DRONE_RANGE_KM,
        "households_cannot_recharge": True,
        "delivery_sortie_from_chargeable_node_to_chargeable_node_with_households_must_be_within_km": DRONE_RANGE_KM,
        "charger_overlap_allowed": True,
        "package_origin_must_be_train_station": True,
        "chargers_are_package_origins": False,
        "chargers_are_charging_relays_only": True,
        "final_return_must_be_train_station": True,
        "return_can_use_intermediate_chargers": True,
        "energy_aware_routing": ENERGY_AWARE_ROUTING,
        "max_extra_distance_ratio_to_add_household": MAX_EXTRA_DISTANCE_RATIO_TO_ADD_HOUSEHOLD,
        "combine_route_only_if_saves_ratio": COMBINE_ROUTE_ONLY_IF_SAVES_RATIO,
        "force_single_household_routes": FORCE_SINGLE_HOUSEHOLD_ROUTES,
        "requested_households": N_RANDOM_HOUSEHOLDS,
        "selected_households": int(len(random_households)),
        "served_households": served,
        "unserved_households": unserved,
        "drone_routes": int(len(routes_df)),
        "route_legs": int(len(legs_df)),
        "total_route_distance_km": round(total_distance, 3),
        "grid_spacing_km_for_charger_selection": GRID_SPACING_KM,
        "min_household_distance_to_station_preferred_km": MIN_HOUSEHOLD_DISTANCE_TO_STATION_KM,
        "max_households_per_drone_route": MAX_HOUSEHOLDS_PER_DRONE_ROUTE,
        "stratified_household_sampling": True,
        "household_distance_bands": "; ".join([f"{b[0]}:{b[1]}-{b[2]}km target={b[3]}" for b in HOUSEHOLD_DISTANCE_BANDS]),
    }])
    summary.to_csv(OUTDIR / "summary.csv", index=False)

    print("DONE")
    print(f"  Output folder: {OUTDIR}")
    print("  Main outputs:")
    print(f"    - {OUTDIR / 'final_drone_delivery_routes_map.html'}")
    print(f"    - {OUTDIR / 'summary.csv'}")
    print(f"    - {OUTDIR / 'coverage_summary.csv'}")
    print(f"    - {OUTDIR / 'coverage_grid_status.csv'}")
    print(f"    - {OUTDIR / 'selected_connected_relay_chargers.csv'}")
    print(f"    - {OUTDIR / 'gap_repair_log.csv'}")
    print(f"    - {OUTDIR / 'charging_bases_station_plus_chargers.csv'}")
    print(f"    - {OUTDIR / 'random_40_households.csv'}")
    print(f"    - {OUTDIR / 'drone_routes.csv'}")
    print(f"    - {OUTDIR / 'route_efficiency_summary.csv'}")
    print(f"    - {OUTDIR / 'drone_route_legs.csv'}")
    print(f"    - {OUTDIR / 'households_service_status.csv'}")

    print("Check map and coverage_summary.csv first. If network still has gaps, increase MAX_SELECTED_CHARGERS or reduce MIN_SELECTED_CHARGER_SPACING_KM.")

def _copy_config(config: Optional[SimulationConfig]) -> SimulationConfig:
    """Return an independent SimulationConfig object."""
    if config is None:
        return SimulationConfig()
    if isinstance(config, dict):
        return SimulationConfig(**config)
    return replace(config)



"""
It measures model runtime using different household numbers, such as 10, 20, 40, 60, 80, and 100, to show how computation time changes.
"""
def run_runtime_benchmark(
    config: Optional[SimulationConfig] = None,
    household_counts: Sequence[int] = (10, 20, 40, 60, 80, 100),
    seeds_per_count: int = 1,
    seed_start: int = 42,
    output_dir: Optional[str | Path] = None,
    include_map_drawing: bool = False,
) -> Dict[str, str]:

    #Measure how long the algorithm takes as the number of random households grows.
    base_config = _copy_config(config)
    study_dir = Path(output_dir) if output_dir is not None else Path(base_config.output_dir) / "runtime_benchmark"
    study_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    counts = [int(c) for c in household_counts if int(c) > 0]
    seeds_per_count = max(1, int(seeds_per_count))

    for count in counts:
        for rep in range(seeds_per_count):
            seed = int(seed_start) + rep
            run_dir = study_dir / f"H{count}_seed{seed}"
            run_config = replace(
                base_config,
                number_of_households=int(count),
                random_seed=int(seed),
                generate_map=bool(include_map_drawing),
                output_dir=str(run_dir),
            )

            t0 = time.perf_counter()
            outputs = run_model(run_config)
            elapsed = time.perf_counter() - t0

            summary_path = Path(outputs["summary_csv"])
            summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
            row0 = summary.iloc[0].to_dict() if len(summary) else {}
            rows.append({
                "number_of_households": int(count),
                "seed": int(seed),
                "runtime_seconds": round(float(elapsed), 3),
                "include_map_drawing": bool(include_map_drawing),
                "served_households": int(row0.get("served_households", 0) or 0),
                "unserved_households": int(row0.get("unserved_households", 0) or 0),
                "drone_routes": int(row0.get("drone_routes", 0) or 0),
                "selected_relay_chargers": int(row0.get("selected_relay_chargers", 0) or 0),
                "total_route_distance_km": float(row0.get("total_route_distance_km", 0.0) or 0.0),
                "output_dir": str(run_dir),
            })

    results = pd.DataFrame(rows)
    csv_path = study_dir / "runtime_benchmark.csv"
    results.to_csv(csv_path, index=False)

    mean_df = pd.DataFrame()
    if len(results) > 0:
        mean_df = (
            results.groupby("number_of_households", as_index=False)
            .agg(
                mean_runtime_seconds=("runtime_seconds", "mean"),
                min_runtime_seconds=("runtime_seconds", "min"),
                max_runtime_seconds=("runtime_seconds", "max"),
                mean_routes=("drone_routes", "mean"),
                mean_served=("served_households", "mean"),
            )
        )
    mean_csv = study_dir / "runtime_benchmark_summary.csv"
    mean_df.to_csv(mean_csv, index=False)

    png_path = study_dir / "runtime_benchmark_graph.png"
    try:
        import matplotlib.pyplot as plt
        if len(mean_df) > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(mean_df["number_of_households"], mean_df["mean_runtime_seconds"], marker="o")
            ax.set_xlabel("Number of randomly selected households")
            ax.set_ylabel("Computation time (seconds)")
            ax.set_title("Algorithm runtime as household count increases")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(png_path, dpi=180)
            plt.close(fig)
    except Exception as exc:
        (study_dir / "runtime_graph_error.txt").write_text(str(exc), encoding="utf-8")

    return {
        "output_dir": str(study_dir),
        "benchmark_csv": str(csv_path),
        "benchmark_summary_csv": str(mean_csv),
        "benchmark_graph_png": str(png_path),
    }


"""
It creates a unique location key for each charger so the model can recognize the same charger
location across different scenarios.
"""
def _charger_location_key(row: pd.Series) -> str:
    """Stable key across scenarios for a charger location."""
    for col in ["household_id", "source_id", "osm_id", "charger_id"]:
        if col in row and str(row.get(col, "")).strip() not in {"", "nan", "None"}:
            return str(row.get(col))
    return f"{float(row.get('lat', 0.0)):.6f},{float(row.get('lon', 0.0)):.6f}"


"""
It creates a map for the permanent charger study, showing which chargers are frequently
selected and used.
"""
def _make_permanent_charger_frequency_map(
    out_html: Path,
    frequency_df: pd.DataFrame,
    recommended_df: pd.DataFrame,
) -> None:
    station_gdf = build_station_gdf()
    rail_line, _rail_buffer = build_rail_line_and_buffer(station_gdf)
    rail_wgs = gpd.GeoSeries([rail_line], crs=PROJECTED_CRS).to_crs(WGS84_CRS)
    stations_wgs = station_gdf.to_crs(WGS84_CRS)
    center_lat = float(stations_wgs.geometry.y.mean())
    center_lon = float(stations_wgs.geometry.x.mean())

    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="CartoDB positron", control_scale=True)
    rail_coords = [(lat, lon) for lon, lat in rail_wgs.iloc[0].coords]
    folium.PolyLine(rail_coords, color="black", weight=4, opacity=0.8, tooltip="Railway corridor").add_to(m)

    fg_st = folium.FeatureGroup(name="Train stations", show=True)
    for _, r in stations_wgs.iterrows():
        folium.Marker(
            location=[float(r.geometry.y), float(r.geometry.x)],
            icon=folium.Icon(color="blue", icon="train", prefix="fa"),
            popup=f"<b>{r['name']}</b><br>Train station",
        ).add_to(fg_st)
    fg_st.add_to(m)

    fg_all = folium.FeatureGroup(name="All candidate chargers observed across scenarios", show=False)
    for _, r in frequency_df.iterrows():
        folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=3,
            color="gray",
            fill=True,
            fill_opacity=0.45,
            weight=0,
            popup=(
                f"<b>{r['location_key']}</b><br>"
                f"Selected in scenarios: {r['selected_scenario_count']}<br>"
                f"Used as recharge in scenarios: {r['used_recharge_scenario_count']}<br>"
                f"Route-use count: {r['route_use_count']}<br>"
                f"Score: {r['permanent_priority_score']}"
            ),
        ).add_to(fg_all)
    fg_all.add_to(m)

    fg_rec = folium.FeatureGroup(name="Recommended permanent chargers", show=True)
    for rank, (_, r) in enumerate(recommended_df.iterrows(), start=1):
        folium.Marker(
            location=[float(r["lat"]), float(r["lon"])],
            icon=folium.Icon(color="darkgreen", icon="star", prefix="fa"),
            tooltip=f"Permanent charger #{rank}: score {r['permanent_priority_score']}",
            popup=(
                f"<b>Recommended permanent charger #{rank}</b><br>"
                f"Location key: {r['location_key']}<br>"
                f"Selected frequency: {r['selected_frequency_percent']}%<br>"
                f"Recharge-use frequency: {r['used_recharge_frequency_percent']}%<br>"
                f"Route-use count: {r['route_use_count']}<br>"
                f"Priority score: {r['permanent_priority_score']}"
            ),
        ).add_to(fg_rec)
    fg_rec.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(out_html)



"""
It runs multiple random delivery scenarios, counts how often each charger is selected
and actually used, and recommends the best locations for permanent chargers.

"""
def run_permanent_charger_study(
    config: Optional[SimulationConfig] = None,
    num_scenarios: int = 5,
    seed_start: int = 1000,
    top_n_chargers: int = 30,
    output_dir: Optional[str | Path] = None,
) -> Dict[str, str]:

    #Run multiple random demand scenarios and rank charger locations for permanent infrastructure.
    base_config = _copy_config(config)
    study_dir = Path(output_dir) if output_dir is not None else Path(base_config.output_dir) / "permanent_charger_study"
    study_dir.mkdir(parents=True, exist_ok=True)

    num_scenarios = max(1, int(num_scenarios))
    top_n_chargers = max(1, int(top_n_chargers))

    locations: Dict[str, dict] = {}
    run_rows: List[dict] = []

    for scenario_idx in range(num_scenarios):
        seed = int(seed_start) + int(scenario_idx)
        run_dir = study_dir / f"scenario_{scenario_idx + 1:03d}_seed{seed}"
        run_config = replace(
            base_config,
            random_seed=seed,
            generate_map=False,
            output_dir=str(run_dir),
        )

        t0 = time.perf_counter()
        outputs = run_model(run_config)
        elapsed = time.perf_counter() - t0

        chargers_path = Path(outputs["chargers_csv"])
        legs_path = Path(outputs["legs_csv"])
        summary_path = Path(outputs["summary_csv"])

        chargers = pd.read_csv(chargers_path) if chargers_path.exists() else pd.DataFrame()
        legs = pd.read_csv(legs_path) if legs_path.exists() else pd.DataFrame()
        summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
        row0 = summary.iloc[0].to_dict() if len(summary) else {}

        charger_id_to_key: Dict[str, str] = {}
        scenario_selected_keys: set[str] = set()
        scenario_used_keys: set[str] = set()

        for _, ch in chargers.iterrows():
            key = _charger_location_key(ch)
            charger_id_to_key[str(ch.get("charger_id", ""))] = key
            scenario_selected_keys.add(key)
            loc = locations.setdefault(key, {
                "location_key": key,
                "example_charger_id": str(ch.get("charger_id", "")),
                "source_household_id": str(ch.get("household_id", "")),
                "lat": float(ch.get("lat", np.nan)),
                "lon": float(ch.get("lon", np.nan)),
                "x_m": float(ch.get("x_m", np.nan)),
                "y_m": float(ch.get("y_m", np.nan)),
                "charger_role_examples": set(),
                "selected_scenarios": set(),
                "used_recharge_scenarios": set(),
                "route_use_count": 0,
            })
            loc["selected_scenarios"].add(scenario_idx + 1)
            role = str(ch.get("charger_role", ""))
            if role:
                loc["charger_role_examples"].add(role)

        if len(legs) > 0:
            for _, leg in legs.iterrows():
                for id_col, type_col in [("from_id", "from_type"), ("to_id", "to_type")]:
                    if str(leg.get(type_col, "")) != "charger":
                        continue
                    charger_id = str(leg.get(id_col, ""))
                    key = charger_id_to_key.get(charger_id)
                    if key is None:
                        continue
                    loc = locations.get(key)
                    if loc is None:
                        continue
                    loc["route_use_count"] += 1
                    loc["used_recharge_scenarios"].add(scenario_idx + 1)
                    scenario_used_keys.add(key)

        run_rows.append({
            "scenario": scenario_idx + 1,
            "seed": seed,
            "runtime_seconds": round(float(elapsed), 3),
            "selected_households": int(row0.get("selected_households", 0) or 0),
            "served_households": int(row0.get("served_households", 0) or 0),
            "unserved_households": int(row0.get("unserved_households", 0) or 0),
            "drone_routes": int(row0.get("drone_routes", 0) or 0),
            "selected_relay_chargers": int(row0.get("selected_relay_chargers", len(chargers)) or len(chargers)),
            "unique_selected_charger_locations": int(len(scenario_selected_keys)),
            "unique_chargers_used_as_recharge": int(len(scenario_used_keys)),
            "output_dir": str(run_dir),
        })

    frequency_rows: List[dict] = []
    for key, loc in locations.items():
        selected_count = len(loc["selected_scenarios"])
        used_count = len(loc["used_recharge_scenarios"])
        route_use_count = int(loc["route_use_count"])
        # Route-use frequency is weighted most heavily because it shows where
        # recharging actually occurs in changing daily delivery routes.
        score = (2.0 * used_count) + (0.25 * route_use_count) + (0.5 * selected_count)
        frequency_rows.append({
            "location_key": key,
            "example_charger_id": loc["example_charger_id"],
            "source_household_id": loc["source_household_id"],
            "lat": loc["lat"],
            "lon": loc["lon"],
            "x_m": loc["x_m"],
            "y_m": loc["y_m"],
            "charger_role_examples": ";".join(sorted(loc["charger_role_examples"])),
            "selected_scenario_count": int(selected_count),
            "selected_frequency_percent": round(100.0 * selected_count / num_scenarios, 1),
            "used_recharge_scenario_count": int(used_count),
            "used_recharge_frequency_percent": round(100.0 * used_count / num_scenarios, 1),
            "route_use_count": int(route_use_count),
            "permanent_priority_score": round(float(score), 3),
        })

    frequency_df = pd.DataFrame(frequency_rows)
    if len(frequency_df) > 0:
        frequency_df = frequency_df.sort_values(
            ["permanent_priority_score", "used_recharge_scenario_count", "route_use_count", "selected_scenario_count"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
    recommended_df = frequency_df.head(top_n_chargers).copy()
    if len(recommended_df) > 0:
        recommended_df.insert(0, "recommended_rank", range(1, len(recommended_df) + 1))

    runs_df = pd.DataFrame(run_rows)
    runs_csv = study_dir / "permanent_charger_study_runs.csv"
    frequency_csv = study_dir / "permanent_charger_frequency.csv"
    recommended_csv = study_dir / "recommended_permanent_chargers.csv"
    map_html = study_dir / "permanent_charger_frequency_map.html"
    report_md = study_dir / "permanent_charger_methodology_report.md"

    runs_df.to_csv(runs_csv, index=False)
    frequency_df.to_csv(frequency_csv, index=False)
    recommended_df.to_csv(recommended_csv, index=False)

    if len(frequency_df) > 0:
        _make_permanent_charger_frequency_map(map_html, frequency_df, recommended_df)

    report_md.write_text(
        "# Permanent charger selection methodology\n\n"
        "Charging stations are treated as long-term infrastructure, while daily household demand changes. "
        "Therefore, this study runs multiple random household delivery scenarios and records which charger locations are repeatedly useful.\n\n"
        "## Ranking logic\n\n"
        "Each charger location is identified by a stable source building ID where possible. For every scenario, the model records whether the location was selected as a charger and whether it was actually used as a recharging stop in the final drone route legs. "
        "The priority score weights actual route-use frequency most strongly, because this shows that the charger is important for real delivery operations instead of only appearing in the coverage network.\n\n"
        f"Scenarios run: {num_scenarios}\n\n"
        f"Recommended permanent chargers: {len(recommended_df)}\n\n"
        "Main outputs:\n"
        "- permanent_charger_study_runs.csv\n"
        "- permanent_charger_frequency.csv\n"
        "- recommended_permanent_chargers.csv\n"
        "- permanent_charger_frequency_map.html\n",
        encoding="utf-8",
    )

    return {
        "output_dir": str(study_dir),
        "runs_csv": str(runs_csv),
        "frequency_csv": str(frequency_csv),
        "recommended_csv": str(recommended_csv),
        "frequency_map_html": str(map_html),
        "methodology_report_md": str(report_md),
    }


if __name__ == "__main__":
    main()
