from pathlib import Path
import time

import geopandas as gpd
import pandas as pd
import folium
from folium.plugins import FastMarkerCluster, MarkerCluster


# File paths

BASE_DIR = Path(".")  
BUILDING_GEOJSON = BASE_DIR / "Manitoba Microsoft Canadian Building Footprints GitHub.geojson"

OUTPUT_ALL_POINTS_CSV = BASE_DIR / "manitoba_household_candidates_ALL_points.csv"
OUTPUT_ALL_POINTS_MAP = BASE_DIR / "manitoba_household_candidates_ALL_points_map.html"
OUTPUT_SAMPLE_PREVIEW_MAP = BASE_DIR / "manitoba_household_candidates_sample_preview_map.html"



# Read building footprint GeoJSON

t0 = time.time()

print("[1/5] Reading Manitoba building footprint GeoJSON...")

buildings = gpd.read_file(BUILDING_GEOJSON)

print(f"  Loaded buildings: {len(buildings):,}")
print(f"  Original CRS: {buildings.crs}")



# Convert building polygons to centroid household points

print("[2/5] Converting building polygons to centroid points...")

# 如果原文件没有 CRS，就假设它是 WGS84 经纬度坐标
if buildings.crs is None:
    buildings = buildings.set_crs(epsg=4326)

# 注意：
# 不建议直接在 EPSG:4326 经纬度坐标下计算 centroid。
# 所以这里先转成加拿大常用投影 EPSG:3347，再计算 centroid。
buildings_projected = buildings.to_crs(epsg=3347)

centroids_projected = buildings_projected.geometry.centroid

# 算完 centroid 后，再转回 EPSG:4326，这样可以得到 lat/lon
centroids = gpd.GeoDataFrame(
    geometry=centroids_projected,
    crs="EPSG:3347"
).to_crs(epsg=4326)

# 生成 household candidate dataframe
households_df = pd.DataFrame({
    "household_id": [f"MB_H{i:06d}" for i in range(len(centroids))],
    "lat": centroids.geometry.y.astype(float),
    "lon": centroids.geometry.x.astype(float),
})

# 删除无效坐标，防止地图报错
households_df = households_df[
    households_df["lat"].between(-90, 90)
    & households_df["lon"].between(-180, 180)
].reset_index(drop=True)

print(f"  Valid centroid points: {len(households_df):,}")



# 4. Save all household candidate points to CSV

print("[3/5] Saving all centroid household candidate points CSV...")

households_df.to_csv(OUTPUT_ALL_POINTS_CSV, index=False)

print(f"  Saved CSV: {OUTPUT_ALL_POINTS_CSV}")


# ============================================================
# 5. Create all-points interactive map
# ============================================================

print("[4/5] Creating all-points interactive map with FastMarkerCluster...")

center_lat = float(households_df["lat"].mean())
center_lon = float(households_df["lon"].mean())

all_map = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=6,
    tiles="CartoDB positron",
    control_scale=True,
)

# 注意：
# 这里有 632,982 个点，不能用普通 Marker。
# FastMarkerCluster 比普通 MarkerCluster 更适合大量点。
coords = households_df[["lat", "lon"]].values.tolist()

FastMarkerCluster(
    coords,
    name=f"All building centroid household candidates ({len(households_df):,})"
).add_to(all_map)

# 加一个中心说明点
folium.Marker(
    location=[center_lat, center_lon],
    popup=(
        f"<b>Manitoba building centroid household candidates</b><br>"
        f"Total points: {len(households_df):,}<br>"
        f"Generated from Microsoft Canadian Building Footprints GeoJSON."
    ),
    tooltip="Map summary"
).add_to(all_map)

folium.LayerControl().add_to(all_map)

all_map.save(OUTPUT_ALL_POINTS_MAP)

print(f"  Saved all-points map: {OUTPUT_ALL_POINTS_MAP}")


# ============================================================
# 6. Create lighter sample preview map
# ============================================================

print("[5/5] Creating lighter preview map with 2,000 sampled points...")

sample_n = min(2000, len(households_df))

sample_df = households_df.sample(
    n=sample_n,
    random_state=42
)

preview_map = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=6,
    tiles="CartoDB positron",
    control_scale=True,
)

sample_cluster = MarkerCluster(
    name=f"Sample preview household candidates ({sample_n:,})"
).add_to(preview_map)

for _, row in sample_df.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=2,
        popup=(
            f"{row['household_id']}<br>"
            f"Lat: {row['lat']:.6f}<br>"
            f"Lon: {row['lon']:.6f}"
        ),
        fill=True,
        fill_opacity=0.6,
        weight=0
    ).add_to(sample_cluster)

folium.LayerControl().add_to(preview_map)

preview_map.save(OUTPUT_SAMPLE_PREVIEW_MAP)

print(f"  Saved preview map: {OUTPUT_SAMPLE_PREVIEW_MAP}")
print(f"DONE in {(time.time() - t0):.1f} seconds")
