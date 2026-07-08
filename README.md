# 🚁 Multi-Agent Drone Delivery Routing & Charging Optimization

**An Interactive Simulation Framework for Manitoba Train-Drone Logistics**

> **Live Demo:** [Explore the Interactive Dashboard here!](https://YOUR-STREAMLIT-APP-NAME.streamlit.app/)

---

## Overview

This project develops an interactive train-drone delivery simulation model for rural and remote household service in **Manitoba, Canada**. The system combines existing railway stations, building-based household candidate points, relay charging stations, and drone routing constraints.

The objective is to evaluate whether randomly selected households can be served by drones while minimizing the number of charging facilities, reducing drone routes, and maintaining range and payload feasibility. All drone routes start from a train station, may use selected building-based chargers as relay points, serve households, and finally return to a train station.

---

## Key Features

- **Real Manitoba building data:** household candidates are generated from building centroid points.
- **Train-station depots:** existing Manitoba railway stations act as package origin points and charging bases.
- **Drone model comparison:** supports DJI FlyCart 30, Wingcopter 198, Drone Delivery Canada Sparrow, Drone Delivery Canada Canary, and Zipline Platform 1.
- **Energy-aware routing:** every flight leg must satisfy the selected drone range limit.
- **Charger relay planning:** selects building-based chargers to connect households back to the station network.
- **Permanent charger study:** runs multiple random scenarios and recommends frequently used permanent chargers.
- **Interactive Streamlit dashboard:** users can change household count, drone range, payload, speed, charger cost, and random seed.
- **Map visualization:** Folium maps display stations, households, chargers, drone route legs, and unserved households.

---

## Technical Stack

- **Language:** Python
- **Web App:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Spatial Analysis:** GeoPandas, Shapely, SciPy KDTree
- **Map Rendering:** Folium / Leaflet
- **Optimization Logic:** Greedy charger selection, connected charger network repair, shortest relay path search, and Clarke-Wright-inspired household route merging

---

## Methodology

### 1. Study Network

The system is represented as a graph:

\[
G=(B,E)
\]

where \(B\) includes train stations and selected charging bases, and \(E\) includes feasible drone flight legs. An edge exists only when the distance between two bases is within the drone range:

\[
d(i,j) \leq R
\]

where \(R\) is the selected drone range.

### 2. Household Service Feasibility

Each household must be reachable through a feasible route that starts from a train station, visits one or more households, uses chargers if needed, and returns to a train station:

\[
S \rightarrow C_1 \rightarrow \cdots \rightarrow H_i \rightarrow \cdots \rightarrow C_k \rightarrow S
\]

Every individual flight leg must satisfy:

\[
d(v_a,v_b) \leq R
\]

### 3. Payload Constraint

For each drone route \(r\), the total household demand must not exceed the drone payload capacity:

\[
\sum_{i \in H_r} q_i \leq Q
\]

where \(q_i\) is household demand and \(Q\) is drone payload capacity.

### 4. Route Merging Logic

The model compares separate household service routes with combined routes. Two households are merged only when the combined route is feasible and shorter than serving them separately.

---

## Installation & Quick Start

1. Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME.git
cd YOUR-REPOSITORY-NAME
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit dashboard:

```bash
streamlit run interactive_model_app.py
```

---

## Project Structure

```text
.
├── interactive_model_app.py                         # Main Streamlit dashboard
├── payload_drone_backend_interactive_fixed_only.py  # Routing, charger selection, and simulation backend
├── plot_map.py                                      # According to Statistics Canada, regenerate the CSV file containing the list of candidate households
├── manitoba_household_candidates_ALL_points.csv     # Manitoba building centroid household candidates
├── requirements.txt                                 # Python dependencies
├── STREAMLIT_DEPLOYMENT_GUIDE.md                    # Live demo deployment instructions
└── README.md                                        # Project documentation
```

---

## Main Outputs

After running the model, output files are saved under `interactive_runs/` or `out_drone_delivery_model/`, including:

- `final_drone_delivery_routes_map.html`
- `summary.csv`
- `drone_routes.csv`
- `drone_route_legs.csv`
- `households_service_status.csv`
- `selected_connected_relay_chargers.csv`
- `recommended_permanent_chargers_optimized.csv`

These generated output folders are ignored by Git because they can be recreated by running the app.

---

## Live Demo Deployment

This project can be deployed on Streamlit Community Cloud. The main file path should be:

```text
interactive_model_app.py
```

After deployment, replace the Live Demo placeholder at the top of this README with your Streamlit app URL.

---

## Academic Context

**Institution:** Western University  
**Project:** MEng On-Demand Train-Drone Delivery Simulation  
**Study Area:** Manitoba, Canada  
**Author:** Zehao Li
