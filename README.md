# 🚁 Multi-Agent Drone Delivery Routing & Charging Optimization

**An Interactive Simulation Framework for Manitoba Train-Drone Logistics**

> **Live Demo:** [Drone Delivery Live Demo](https://drone-delivery-us9sqrexpcedig9m3efhiy.streamlit.app/)
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



## 1. Study Network

The charging network is represented as a graph:

$$
G^{B}=(B,E^{B})
$$

where $B$ includes the train stations and selected charging bases:

$$
B=S\cup C^{*}
$$

The edge set $E^{B}$ contains all feasible direct flight legs between charging nodes:

$$
E^{B}=\{(i,j)\in B\times B\mid i\neq j,\ d(i,j)\leq R\}
$$

where $d(i,j)$ is the distance between nodes $i$ and $j$, and $R$ is the selected drone range.

## 2. Household Service Feasibility

Each household must be served by a feasible route. The route starts from a train station, may visit charging bases, serves one or more households, and finally returns to a train station:

$$
s_{0}
\rightarrow
c_{1}
\rightarrow
\cdots
\rightarrow
h_{1}
\rightarrow
\cdots
\rightarrow
h_{m}
\rightarrow
c_{k}
\rightarrow
s_{1}
$$

where $s_{0},s_{1}\in S$, $c_{1},\ldots,c_{k}\in C^{*}$, and $h_{1},\ldots,h_{m}\in H$.

Every individual flight leg between two consecutive route nodes must satisfy:

$$
d(v_{a},v_{a+1})\leq R
$$

Households are not charging nodes. Therefore, the complete flight segment between two consecutive charging-capable nodes must also satisfy the drone battery constraint.

For example, if a drone serves one household between charging nodes $b_{1}$ and $b_{2}$, then:

$$
d(b_{1},h_{i})+d(h_{i},b_{2})\leq R
$$

For two households, the condition becomes:

$$
d(b_{1},h_{i})
+
d(h_{i},h_{j})
+
d(h_{j},b_{2})
\leq R
$$

## 3. Payload Constraint

For each drone route $r$, the total household demand must not exceed the drone payload capacity:

$$
\sum_{i\in H_{r}}q_{i}\leq Q
$$

where $H_{r}$ is the set of households served by route $r$, $q_{i}$ is the demand of household $i$, and $Q$ is the drone payload capacity.

## 4. Route Merging Logic

The model first constructs the shortest feasible route for each household. It then evaluates whether two individual household routes can be combined into one route.

For households $i$ and $j$, the total distance of serving them separately is:

$$L_{ij}^{\mathrm{sep}}=L_i^{\min}+L_j^{\min}$$

where $L_i^{\min}$ and $L_j^{\min}$ are the shortest feasible route distances for serving households $i$ and $j$ individually.

The model evaluates both possible service orders for the combined route:

$$L_{ij}^{\mathrm{com}}=\min(L_{i\rightarrow j},L_{j\rightarrow i})$$

where $L_{i\rightarrow j}$ is the shortest feasible route that serves household $i$ before household $j$. Similarly, $L_{j\rightarrow i}$ serves household $j$ before household $i$.

The distance saving obtained by merging the two routes is:

$$S_{ij}=L_{ij}^{\mathrm{sep}}-L_{ij}^{\mathrm{com}}$$

A combined route is accepted only when it reduces the total travel distance:

$$S_{ij}>0$$

The combined household demand must not exceed the drone payload capacity:

$$q_i+q_j\leq Q$$

where $q_i$ and $q_j$ are household demands, and $Q$ is the drone payload capacity.

The merged route must also satisfy the drone-range constraint. For the service order $i\rightarrow j$, the flight segment between two consecutive charging-capable nodes must satisfy:

$$d(e,h_i)+d(h_i,h_j)+d(h_j,x)\leq R$$

For the reverse service order $j\rightarrow i$, the condition is:

$$d(e,h_j)+d(h_j,h_i)+d(h_i,x)\leq R$$

where $e$ is the charging-capable entry node, $x$ is the charging-capable exit node, and $R$ is the selected drone range. Both $e$ and $x$ belong to the charging network:

$$e,x\in B$$

A merged route is feasible only when it satisfies the distance-saving, payload, battery-range, charging-network connectivity, and train-station return requirements. Feasible household pairs are considered from the largest to the smallest value of $S_{ij}$.


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

## 🎓 Academic Context

**Institution:** Western University (UWO) 

**Project Team:** Zehao Li

**Supervisors:** Prof. Roorda, Prof. Tang 
