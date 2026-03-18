import streamlit as st
import pandas as pd
import pydeck as pdk
import geopandas as gpd
import json
import numpy as np
import heapq
import requests
from shapely.geometry import Point
from sklearn.neighbors import BallTree

# ===== PAGINA INSTELLINGEN =====
st.set_page_config(page_title="Laadpalen Nederland", page_icon="⚡", layout="wide")

# ===== CUSTOM CSS =====
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Space Mono', monospace !important;
        letter-spacing: -0.5px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        border-bottom: 2px solid #22c55e;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Space Mono', monospace;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        padding: 12px 28px;
        background: transparent;
        border: none;
        color: #6b7280;
        transition: all 0.2s;
    }

    .stTabs [aria-selected="true"] {
        color: #22c55e !important;
        background: rgba(34,197,94,0.08) !important;
        border-bottom: 2px solid #22c55e !important;
    }

    .result-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #22c55e;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 12px 0;
        color: white;
    }

    .result-card h4 {
        font-family: 'Space Mono', monospace;
        color: #22c55e;
        margin: 0 0 8px 0;
        font-size: 14px;
    }

    .result-card p {
        margin: 4px 0;
        color: #94a3b8;
        font-size: 13px;
    }

    .result-card .distance {
        font-family: 'Space Mono', monospace;
        font-size: 22px;
        color: white;
        font-weight: 700;
    }

    .algo-badge {
        display: inline-block;
        background: rgba(34,197,94,0.15);
        border: 1px solid #22c55e;
        color: #22c55e;
        font-family: 'Space Mono', monospace;
        font-size: 11px;
        padding: 3px 10px;
        border-radius: 20px;
        margin-bottom: 16px;
        letter-spacing: 1px;
    }

    .stTextInput > div > div > input {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        color: white;
        font-family: 'DM Sans', sans-serif;
        padding: 12px 16px;
    }

    .stTextInput > div > div > input:focus {
        border-color: #22c55e;
        box-shadow: 0 0 0 2px rgba(34,197,94,0.2);
    }

    .stButton > button {
        background: #22c55e;
        color: #0f172a;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        font-size: 13px;
        letter-spacing: 1px;
        border: none;
        border-radius: 8px;
        padding: 12px 32px;
        width: 100%;
        transition: all 0.2s;
        text-transform: uppercase;
    }

    .stButton > button:hover {
        background: #16a34a;
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(34,197,94,0.4);
    }

    .metric-row {
        display: flex;
        gap: 16px;
        margin: 16px 0;
    }

    .metric-box {
        flex: 1;
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }

    .metric-box .label {
        font-size: 11px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-family: 'Space Mono', monospace;
    }

    .metric-box .value {
        font-size: 24px;
        font-weight: 700;
        color: #22c55e;
        font-family: 'Space Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# ⚡ Laadpalen Nederland")

# ===== DATA LADEN =====
@st.cache_data
def load_data():
    df = pd.read_csv('open_charge_map_NL.csv')
    df.columns = df.columns.str.strip()
    df['AddressInfo_Latitude']  = pd.to_numeric(df['AddressInfo_Latitude'],  errors='coerce')
    df['AddressInfo_Longitude'] = pd.to_numeric(df['AddressInfo_Longitude'], errors='coerce')
    df = df.dropna(subset=['AddressInfo_Latitude', 'AddressInfo_Longitude'])

    df['title_norm']   = df['AddressInfo_Title'].astype(str).str.strip().str.lower()
    df['address_norm'] = df['AddressInfo_AddressLine1'].astype(str).str.strip().str.lower()
    df['town_norm']    = df['AddressInfo_Town'].astype(str).str.strip().str.lower()
    df = df.drop_duplicates(subset=['title_norm', 'address_norm', 'town_norm',
                                     'AddressInfo_Latitude', 'AddressInfo_Longitude'])
    df = df.reset_index(drop=True)

    try:
        provincies_gdf = gpd.read_file('provincies.geojson')
        provincies_gdf = provincies_gdf[['statnaam', 'geometry']].rename(columns={'statnaam': 'Provincie'})
        provincies_gdf = provincies_gdf.to_crs("EPSG:4326")

        gdf = gpd.GeoDataFrame(
            df,
            geometry=[Point(lon, lat) for lon, lat in zip(df['AddressInfo_Longitude'], df['AddressInfo_Latitude'])],
            crs="EPSG:4326"
        )
        gdf = gdf.sjoin(provincies_gdf, how='left', predicate='within')
        gdf['Provincie'] = gdf['Provincie'].fillna('Onbekend')
        df = pd.DataFrame(gdf.drop(columns=['geometry', 'index_right'], errors='ignore'))
        df = df.reset_index(drop=True)
        return df, provincies_gdf
    except Exception:
        df['Provincie'] = 'Onbekend'
        return df, None

@st.cache_resource
def build_graph(_df, k=8):
    """Build a k-NN graph of charging stations for Dijkstra."""
    coords_rad = np.radians(_df[['AddressInfo_Latitude', 'AddressInfo_Longitude']].values)
    tree = BallTree(coords_rad, metric='haversine')

    # k+1 because the point itself is included
    distances, indices = tree.query(coords_rad, k=k + 1)

    R = 6371.0  # Earth radius km
    graph = {i: [] for i in range(len(_df))}
    for i in range(len(_df)):
        for dist_rad, j in zip(distances[i][1:], indices[i][1:]):
            dist_km = dist_rad * R
            graph[i].append((j, dist_km))
            graph[j].append((i, dist_km))

    return graph, tree

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def dijkstra(graph, start_node):
    """Standard Dijkstra from start_node. Returns (dist_dict, prev_dict)."""
    dist = {node: float('inf') for node in graph}
    prev = {node: None for node in graph}
    dist[start_node] = 0.0
    pq = [(0.0, start_node)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    return dist, prev

def reconstruct_path(prev, target):
    path = []
    node = target
    while node is not None:
        path.append(node)
        node = prev[node]
    return list(reversed(path))

def geocode_address(address: str):
    """Geocode address using Nominatim (free, no key needed)."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": address + ", Netherlands", "format": "json", "limit": 1}
    headers = {"User-Agent": "LaadpalenNL-StreamlitApp/1.0"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=5)
        results = r.json()
        if results:
            return float(results[0]['lat']), float(results[0]['lon']), results[0]['display_name']
    except Exception:
        pass
    return None, None, None

def find_nearest_charger_dijkstra(user_lat, user_lon, df, graph, tree):
    """
    1. Find nearest node to user via BallTree (O(log n)).
    2. Add virtual 'user' node with edges to K nearest chargers.
    3. Run Dijkstra from that virtual node.
    4. Return nearest charger index + path indices + total distance.
    """
    coords_rad = np.radians([[user_lat, user_lon]])
    K_connect = 5
    distances, indices = tree.query(coords_rad, k=K_connect)
    R = 6371.0

    # Build augmented graph with virtual start node
    virtual_id = len(df)
    aug_graph = {k: list(v) for k, v in graph.items()}
    aug_graph[virtual_id] = []

    for dist_rad, j in zip(distances[0], indices[0]):
        dist_km = float(dist_rad) * R
        aug_graph[virtual_id].append((int(j), dist_km))
        aug_graph[int(j)].append((virtual_id, dist_km))

    dist_map, prev_map = dijkstra(aug_graph, virtual_id)

    # Find nearest reachable charger
    best_idx = min(
        (i for i in range(len(df)) if dist_map[i] < float('inf')),
        key=lambda i: dist_map[i]
    )
    path_nodes = reconstruct_path(prev_map, best_idx)
    return best_idx, path_nodes, dist_map[best_idx]


df, provincies_gdf = load_data()
graph, ball_tree = build_graph(df)

# ===== TABS =====
tab1, tab2 = st.tabs(["🗺️  Laadpalen Kaart", "🔍  Kortste Route (Dijkstra)"])

# ==================== TAB 1: ORIGINAL MAP ====================
with tab1:
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        filter_type = st.radio("Bekijken per:", ["Alle locaties", "Provincie", "Gemeente"])

    with col2:
        filtered_df = df.copy()
        gekozen = None

        if filter_type == "Provincie":
            gekozen = st.selectbox("Kies een provincie:", sorted(df['Provincie'].unique()))
            filtered_df = df[df['Provincie'] == gekozen]
        elif filter_type == "Gemeente":
            gekozen = st.selectbox("Kies een gemeente:", sorted(df['AddressInfo_Town'].dropna().unique()))
            filtered_df = df[df['AddressInfo_Town'] == gekozen]
        else:
            st.write("")

    with col3:
        st.metric("Laadpalen zichtbaar", len(filtered_df))

    st.divider()

    if len(filtered_df) == 0:
        st.warning("Geen laadpalen gevonden voor deze selectie.")
    else:
        center_lat = filtered_df['AddressInfo_Latitude'].mean()
        center_lng = filtered_df['AddressInfo_Longitude'].mean()
        zoom = 6 if filter_type == "Alle locaties" else (8 if filter_type == "Provincie" else 11)

        map_df = filtered_df[['AddressInfo_Latitude', 'AddressInfo_Longitude',
                               'AddressInfo_Title', 'AddressInfo_AddressLine1',
                               'AddressInfo_Town', 'Provincie',
                               'Connections_0_PowerKW']].copy()
        map_df.columns = ['lat', 'lon', 'title', 'address', 'town', 'provincie', 'power']
        map_df = map_df.fillna('N/A')

        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position='[lon, lat]',
            get_fill_color='[34, 197, 94, 200]',
            get_radius=5,
            radius_min_pixels=4,
            radius_max_pixels=12,
            pickable=True,
        )

        layers = [scatter_layer]

        if filter_type == "Provincie" and gekozen and provincies_gdf is not None:
            provincie_shape = provincies_gdf[provincies_gdf['Provincie'] == gekozen]
            if not provincie_shape.empty:
                geojson_data = json.loads(provincie_shape.to_json())
                grens_layer = pdk.Layer(
                    "GeoJsonLayer",
                    data=geojson_data,
                    stroked=True,
                    filled=True,
                    get_line_color=[220, 38, 38, 255],
                    get_fill_color=[220, 38, 38, 20],
                    line_width_min_pixels=3,
                )
                layers.append(grens_layer)

        view = pdk.ViewState(latitude=center_lat, longitude=center_lng, zoom=zoom, pitch=0)

        tooltip = {
            "html": "<b>{title}</b><br>{address}<br>{town}<br>Vermogen: {power} kW<br>Provincie: {provincie}",
            "style": {"backgroundColor": "#1e293b", "color": "white", "fontSize": "13px", "padding": "8px"}
        }

        st.pydeck_chart(
            pdk.Deck(layers=layers, initial_view_state=view, tooltip=tooltip,
                     map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"),
            use_container_width=True,
            height=1000
        )

# ==================== TAB 2: DIJKSTRA ====================
with tab2:
    st.markdown('<div class="algo-badge">DIJKSTRA\'S ALGORITHM</div>', unsafe_allow_html=True)
    st.markdown("### Vind de dichtstbijzijnde laadpaal")
    st.markdown("Voer een adres, plaatsnaam of postcode in. Het algoritme berekent de kortste route door het laadpalennetwerk.")

    col_search, col_btn = st.columns([3, 1])

    with col_search:
        address_input = st.text_input(
            "Startlocatie",
            placeholder="bijv. Kalverstraat 1, Amsterdam  •  3012 Rotterdam  •  Eindhoven Centrum",
            label_visibility="collapsed"
        )

    with col_btn:
        search_clicked = st.button("⚡ Zoek Route", use_container_width=True)

    # ---- Results ----
    if search_clicked and address_input.strip():
        with st.spinner("Locatie zoeken en route berekenen..."):
            user_lat, user_lon, display_name = geocode_address(address_input.strip())

        if user_lat is None:
            st.error("❌ Adres niet gevonden. Probeer een andere zoekopdracht.")
        else:
            with st.spinner("Dijkstra's algoritme berekent de optimale route..."):
                best_idx, path_nodes, total_dist = find_nearest_charger_dijkstra(
                    user_lat, user_lon, df, graph, ball_tree
                )

            charger = df.iloc[best_idx]

            # --- Metrics ---
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-box">
                    <div class="label">Afstand</div>
                    <div class="value">{total_dist:.2f} km</div>
                </div>
                <div class="metric-box">
                    <div class="label">Knopen in pad</div>
                    <div class="value">{len(path_nodes)}</div>
                </div>
                <div class="metric-box">
                    <div class="label">Laadpalen netwerk</div>
                    <div class="value">{len(df):,}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            power_val = charger.get('Connections_0_PowerKW', 'N/A')

            st.markdown(f"""
            <div class="result-card">
                <h4>📍 Startpunt</h4>
                <p>{display_name}</p>
                <p style="color:#64748b; font-size:11px; font-family:'Space Mono',monospace;">
                    {user_lat:.5f}° N, {user_lon:.5f}° E
                </p>
            </div>
            <div class="result-card">
                <h4>⚡ Dichtstbijzijnde Laadpaal</h4>
                <p style="font-size:16px; color:white; font-weight:600;">{charger['AddressInfo_Title']}</p>
                <p>{charger.get('AddressInfo_AddressLine1', '')} — {charger.get('AddressInfo_Town', '')}</p>
                <p>Vermogen: <span style="color:#22c55e; font-weight:600;">{power_val} kW</span></p>
                <p>Postcode: {charger.get('AddressInfo_Postcode', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)

            # --- Build map data ---
            # All chargers (dimmed background)
            all_chargers_map = df[['AddressInfo_Latitude', 'AddressInfo_Longitude',
                                   'AddressInfo_Title']].copy()
            all_chargers_map.columns = ['lat', 'lon', 'title']

            # Path edges as line segments
            path_coords = []
            for i in range(len(path_nodes) - 1):
                n1 = path_nodes[i]
                n2 = path_nodes[i + 1]

                if n1 == len(df):
                    lat1, lon1 = user_lat, user_lon
                else:
                    lat1 = float(df.iloc[n1]['AddressInfo_Latitude'])
                    lon1 = float(df.iloc[n1]['AddressInfo_Longitude'])

                if n2 == len(df):
                    lat2, lon2 = user_lat, user_lon
                else:
                    lat2 = float(df.iloc[n2]['AddressInfo_Latitude'])
                    lon2 = float(df.iloc[n2]['AddressInfo_Longitude'])

                path_coords.append({
                    "start": [lon1, lat1],
                    "end":   [lon2, lat2],
                })

            # Path node markers
            path_node_markers = []
            for n in path_nodes:
                if n == len(df):
                    path_node_markers.append({"lat": user_lat, "lon": user_lon, "type": "start"})
                elif n == best_idx:
                    path_node_markers.append({
                        "lat": float(df.iloc[n]['AddressInfo_Latitude']),
                        "lon": float(df.iloc[n]['AddressInfo_Longitude']),
                        "type": "end"
                    })
                else:
                    path_node_markers.append({
                        "lat": float(df.iloc[n]['AddressInfo_Latitude']),
                        "lon": float(df.iloc[n]['AddressInfo_Longitude']),
                        "type": "mid"
                    })

            # Special markers: user pin + destination
            special_points = [
                {"lat": user_lat, "lon": user_lon,
                 "label": "📍 Start", "color": [59, 130, 246]},
                {"lat": float(charger['AddressInfo_Latitude']),
                 "lon": float(charger['AddressInfo_Longitude']),
                 "label": f"⚡ {charger['AddressInfo_Title']}", "color": [34, 197, 94]},
            ]

            # --- Layers ---
            bg_layer = pdk.Layer(
                "ScatterplotLayer",
                data=all_chargers_map,
                get_position='[lon, lat]',
                get_fill_color='[100, 116, 139, 80]',
                get_radius=4,
                radius_min_pixels=2,
                radius_max_pixels=6,
                pickable=False,
            )

            path_line_layer = pdk.Layer(
                "LineLayer",
                data=path_coords,
                get_source_position="start",
                get_target_position="end",
                get_color=[251, 191, 36, 230],   # amber path line
                get_width=4,
                width_min_pixels=3,
            )

            # Waypoint dots
            mid_points = [p for p in path_node_markers if p['type'] == 'mid']
            if mid_points:
                mid_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=mid_points,
                    get_position='[lon, lat]',
                    get_fill_color='[251, 191, 36, 200]',
                    get_radius=6,
                    radius_min_pixels=4,
                    radius_max_pixels=10,
                    pickable=False,
                )
            else:
                mid_layer = None

            start_layer = pdk.Layer(
                "ScatterplotLayer",
                data=[special_points[0]],
                get_position='[lon, lat]',
                get_fill_color='[59, 130, 246, 255]',
                get_radius=12,
                radius_min_pixels=10,
                radius_max_pixels=22,
                pickable=True,
            )

            end_layer = pdk.Layer(
                "ScatterplotLayer",
                data=[special_points[1]],
                get_position='[lon, lat]',
                get_fill_color='[34, 197, 94, 255]',
                get_radius=14,
                radius_min_pixels=12,
                radius_max_pixels=24,
                pickable=True,
            )

            layers_map = [bg_layer, path_line_layer]
            if mid_layer:
                layers_map.append(mid_layer)
            layers_map += [start_layer, end_layer]

            # View centred between start and destination
            mid_lat = (user_lat + float(charger['AddressInfo_Latitude'])) / 2
            mid_lon = (user_lon + float(charger['AddressInfo_Longitude'])) / 2

            # Estimate zoom from distance
            if total_dist < 1:
                zoom_level = 14
            elif total_dist < 5:
                zoom_level = 12
            elif total_dist < 20:
                zoom_level = 10
            else:
                zoom_level = 8

            view = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=zoom_level, pitch=0)

            tooltip_map = {
                "html": "<b>{label}</b>",
                "style": {"backgroundColor": "#0f172a", "color": "white",
                          "fontSize": "13px", "padding": "8px",
                          "borderRadius": "6px", "border": "1px solid #22c55e"}
            }

            st.pydeck_chart(
                pdk.Deck(
                    layers=layers_map,
                    initial_view_state=view,
                    tooltip=tooltip_map,
                    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
                ),
                use_container_width=True,
                height=620,
            )

            st.markdown("""
            <div style="margin-top:12px; padding:12px 16px; background:#0f172a; border-radius:8px;
                        border-left:3px solid #fbbf24; font-size:12px; color:#94a3b8;">
                <span style="color:#fbbf24; font-family:'Space Mono',monospace; font-weight:700;">
                    HOE WERKT HET
                </span><br><br>
                Het netwerk modelleert alle laadpalen als knopen. Elke laadpaal is verbonden met zijn 
                <strong style="color:white;">8 dichtstbijzijnde buren</strong> via gewogen kanten 
                (gewicht = Haversine-afstand in km). Dijkstra's algoritme zoekt vanuit uw startpunt 
                het <strong style="color:white;">kortste pad</strong> door dit netwerk naar de 
                dichtstbijzijnde bereikbare laadpaal. De gele lijn toont de optimale route door het netwerk.
            </div>
            """, unsafe_allow_html=True)

    elif search_clicked:
        st.warning("Voer een adres in om te zoeken.")
    else:
        # Placeholder state
        st.markdown("""
        <div style="margin-top:40px; text-align:center; color:#334155;">
            <div style="font-size:48px; margin-bottom:16px;">🔍</div>
            <p style="font-family:'Space Mono',monospace; font-size:14px; color:#64748b;">
                Voer een adres in en klik op "Zoek Route"<br>om de dichtstbijzijnde laadpaal te vinden.
            </p>
        </div>
        """, unsafe_allow_html=True)