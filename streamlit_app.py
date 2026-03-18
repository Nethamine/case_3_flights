import streamlit as st
import pandas as pd
import pydeck as pdk
import geopandas as gpd
import json
import numpy as np
import heapq
import requests
import plotly.express as px
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

@st.cache_data
def load_voertuigen():
    df = pd.read_csv('rdw_voertuigen_clean.csv')
    df["datum_tenaamstelling"] = pd.to_datetime(df["datum_tenaamstelling"], errors="coerce")
    df["jaar_maand"] = df["datum_tenaamstelling"].dt.to_period("M").dt.to_timestamp()
    return df

@st.cache_resource
def build_balltree(_df):
    """Build a BallTree spatial index over all charging station coordinates."""
    coords_rad = np.radians(_df[['AddressInfo_Latitude', 'AddressInfo_Longitude']].values)
    tree = BallTree(coords_rad, metric='haversine')
    return tree

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

def _osrm_route(origin_lon, origin_lat, dest_lon, dest_lat):
    """
    Fetch a driving route from the public OSRM demo server.
    Returns (road_distance_km, road_duration_min, route_coords) or (None, None, None) on failure.
    route_coords is a list of {"lat": ..., "lon": ...} dicts decoded from the polyline geometry.
    """
    url = (
        f"http://router.project-osrm.org/route/v1/driving/"
        f"{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
        f"?overview=full&geometries=geojson&steps=false"
    )
    headers = {"User-Agent": "LaadpalenNL-StreamlitApp/1.0"}
    try:
        r = requests.get(url, headers=headers, timeout=8)
        data = r.json()
        if data.get("code") != "Ok" or not data.get("routes"):
            return None, None, None
        route = data["routes"][0]
        dist_km  = route["distance"] / 1000.0
        dur_min  = route["duration"] / 60.0
        coords   = [
            {"lat": c[1], "lon": c[0]}
            for c in route["geometry"]["coordinates"]
        ]
        return dist_km, dur_min, coords
    except Exception:
        return None, None, None

def find_nearest_charger_dijkstra(user_lat, user_lon, df, tree, n_candidates=10):
    """
    Geo-based Dijkstra using real road distances via OSRM:
    1. BallTree selects the N nearest charger candidates by straight-line distance.
    2. For each candidate, fetch the actual road distance from OSRM.
    3. Build a single-source graph: user → each candidate, weight = road distance (km).
    4. Dijkstra selects the candidate with the minimum road distance.
    5. Fetch the final road geometry for the winning charger to draw on the map.
    Returns: (best_charger_idx, road_dist_km, road_duration_min, route_coords)
    """
    R = 6371.0

    # Step 1: get N nearest by straight-line as candidates
    user_rad = np.radians([[user_lat, user_lon]])
    distances_rad, indices = tree.query(user_rad, k=n_candidates)

    # Step 2 & 3: query OSRM for each candidate, build Dijkstra graph
    USER_NODE = 0
    graph        = {USER_NODE: []}
    route_cache  = {}   # node_id → (dist_km, dur_min, coords)

    for rank, charger_idx in enumerate(indices[0]):
        node_id     = rank + 1
        c_lat = float(df.iloc[charger_idx]['AddressInfo_Latitude'])
        c_lon = float(df.iloc[charger_idx]['AddressInfo_Longitude'])

        dist_km, dur_min, coords = _osrm_route(user_lon, user_lat, c_lon, c_lat)

        if dist_km is None:
            # Fall back to Haversine if OSRM fails for this candidate
            dist_km  = float(distances_rad[0][rank]) * R
            dur_min  = None
            coords   = None

        graph[USER_NODE].append((node_id, dist_km))
        graph[node_id]  = []
        route_cache[node_id] = (int(charger_idx), dist_km, dur_min, coords)

    # Step 4: Dijkstra (single-source star graph → trivially picks min edge)
    dist_map = {node: float('inf') for node in graph}
    prev_map = {node: None         for node in graph}
    dist_map[USER_NODE] = 0.0
    pq = [(0.0, USER_NODE)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist_map[u]:
            continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist_map[v]:
                dist_map[v] = nd
                prev_map[v] = u
                heapq.heappush(pq, (nd, v))

    best_node = min(
        (n for n in graph if n != USER_NODE),
        key=lambda n: dist_map[n]
    )

    best_charger_idx, road_dist_km, road_dur_min, road_coords = route_cache[best_node]

    # If we only have a fallback distance but no geometry, fetch geometry now
    if road_coords is None:
        c_lat = float(df.iloc[best_charger_idx]['AddressInfo_Latitude'])
        c_lon = float(df.iloc[best_charger_idx]['AddressInfo_Longitude'])
        _, _, road_coords = _osrm_route(user_lon, user_lat, c_lon, c_lat)

    return best_charger_idx, road_dist_km, road_dur_min, road_coords


df, provincies_gdf = load_data()
df_voer = load_voertuigen()
ball_tree = build_balltree(df)

# ===== TABS =====
tab3, tab1, tab2 = st.tabs(["📈  Voertuigregistraties", "🗺️  Laadpalen Kaart", "🔍  Kortste Route (Dijkstra)"])

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
            with st.spinner("Rijroute berekenen via OSRM & Dijkstra..."):
                best_idx, road_dist, road_dur, road_coords = find_nearest_charger_dijkstra(
                    user_lat, user_lon, df, ball_tree
                )

            charger = df.iloc[best_idx]
            charger_lat = float(charger['AddressInfo_Latitude'])
            charger_lon = float(charger['AddressInfo_Longitude'])

            dur_str = f"{road_dur:.0f} min" if road_dur is not None else "N/A"

            # --- Metrics ---
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-box">
                    <div class="label">Rijafstand</div>
                    <div class="value">{road_dist:.2f} km</div>
                </div>
                <div class="metric-box">
                    <div class="label">Reistijd</div>
                    <div class="value">{dur_str}</div>
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

            # --- Build map layers ---

            # Background: all chargers (dimmed)
            all_chargers_map = df[['AddressInfo_Latitude', 'AddressInfo_Longitude',
                                   'AddressInfo_Title']].copy()
            all_chargers_map.columns = ['lat', 'lon', 'title']

            bg_layer = pdk.Layer(
                "ScatterplotLayer",
                data=all_chargers_map,
                get_position='[lon, lat]',
                get_fill_color='[100, 116, 139, 70]',
                get_radius=4,
                radius_min_pixels=2,
                radius_max_pixels=7,
                pickable=False,
            )

            # Road route line — real OSRM geometry if available, else straight fallback
            if road_coords and len(road_coords) >= 2:
                route_segments = [
                    {"start": [road_coords[i]["lon"], road_coords[i]["lat"]],
                     "end":   [road_coords[i+1]["lon"], road_coords[i+1]["lat"]]}
                    for i in range(len(road_coords) - 1)
                ]
            else:
                route_segments = [{
                    "start": [user_lon, user_lat],
                    "end":   [charger_lon, charger_lat]
                }]

            route_layer = pdk.Layer(
                "LineLayer",
                data=route_segments,
                get_source_position="start",
                get_target_position="end",
                get_color=[251, 191, 36, 240],
                get_width=5,
                width_min_pixels=3,
            )

            # User location (blue)
            start_layer = pdk.Layer(
                "ScatterplotLayer",
                data=[{"lat": user_lat, "lon": user_lon, "label": "📍 Startpunt"}],
                get_position='[lon, lat]',
                get_fill_color='[59, 130, 246, 255]',
                get_radius=14,
                radius_min_pixels=10,
                radius_max_pixels=22,
                pickable=True,
            )

            # Destination charger (bright green)
            end_layer = pdk.Layer(
                "ScatterplotLayer",
                data=[{"lat": charger_lat, "lon": charger_lon,
                       "label": f"⚡ {charger['AddressInfo_Title']}"}],
                get_position='[lon, lat]',
                get_fill_color='[34, 197, 94, 255]',
                get_radius=16,
                radius_min_pixels=12,
                radius_max_pixels=26,
                pickable=True,
            )

            # View: fit around the route
            mid_lat = (user_lat + charger_lat) / 2
            mid_lon = (user_lon + charger_lon) / 2

            if road_dist < 1:
                zoom_level = 14
            elif road_dist < 3:
                zoom_level = 13
            elif road_dist < 8:
                zoom_level = 12
            elif road_dist < 20:
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
                    layers=[bg_layer, route_layer, start_layer, end_layer],
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
                De BallTree selecteert de <strong style="color:white;">10 geografisch dichtstbijzijnde 
                laadpalen</strong> als kandidaten. Voor elke kandidaat wordt de echte 
                <strong style="color:white;">rijafstand via OSRM</strong> (OpenStreetMap routing) 
                opgehaald. Dijkstra's algoritme kiest de kandidaat met de 
                <strong style="color:white;">kortste rijafstand over de weg</strong> — niet de 
                hemelsbreed dichtstbijzijnde. De gele lijn volgt de exacte rijroute.
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

# ==================== TAB 3: VOERTUIGREGISTRATIES ====================
with tab3:

    st.markdown('<div class="algo-badge">RDW OPEN DATA</div>', unsafe_allow_html=True)
    st.markdown("### Aandeel voertuigen per aandrijflijn")
    st.markdown("Cumulatief aandeel van het Nederlandse wagenpark per brandstofcategorie over de tijd.")

    # --- Categorisering ---
    def categoriseer_brandstof(brandstof):
        if pd.isna(brandstof):
            return None
        b = brandstof.strip().lower()
        if b == "elektriciteit":
            return "🔋 Volledig elektrisch"
        elif b in ["benzine/elektriciteit", "diesel/elektriciteit"]:
            return "⚡ Plug-in hybride"
        elif b in ["benzine", "diesel", "lpg", "cng", "waterstof", "overig"]:
            return "⛽ Fossiel"
        else:
            return None

    df_cat = df_voer.copy()
    df_cat["categorie"] = df_cat["brandstof_omschrijving"].apply(categoriseer_brandstof)
    df_cat = df_cat[df_cat["categorie"].notna()].copy()

    # --- Groeperen en cumuleren ---
    df_groep = (
        df_cat
        .groupby(["jaar_maand", "categorie"])
        .size()
        .reset_index(name="aantal")
        .sort_values("jaar_maand")
    )
    df_groep["cumulatief"] = df_groep.groupby("categorie")["aantal"].cumsum()

    # --- Percentage t.o.v. totaal lopend bestand per maand ---
    totaal_per_maand = df_groep.groupby("jaar_maand")["cumulatief"].transform("sum")
    df_groep["percentage"] = (df_groep["cumulatief"] / totaal_per_maand * 100).round(2)

    # --- Kleurmap ---
    kleur_map = {
        "🔋 Volledig elektrisch": "#22c55e",
        "⚡ Plug-in hybride":     "#f59e0b",
        "⛽ Fossiel":              "#64748b",
    }

    # --- Snelle statistieken bovenaan ---
    laatste_maand = df_groep["jaar_maand"].max()
    laatste = df_groep[df_groep["jaar_maand"] == laatste_maand].set_index("categorie")

    def pct(cat):
        return f"{laatste.loc[cat, 'percentage']:.1f}%" if cat in laatste.index else "N/A"

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(f"""
        <div class="metric-box" style="background:#0f172a; border:1px solid #22c55e; border-radius:10px;
             padding:16px; text-align:center;">
            <div class="label" style="font-size:11px; color:#64748b; text-transform:uppercase;
                 letter-spacing:1px; font-family:'Space Mono',monospace;">Volledig elektrisch</div>
            <div class="value" style="font-size:28px; font-weight:700; color:#22c55e;
                 font-family:'Space Mono',monospace;">{pct("🔋 Volledig elektrisch")}</div>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div class="metric-box" style="background:#0f172a; border:1px solid #f59e0b; border-radius:10px;
             padding:16px; text-align:center;">
            <div class="label" style="font-size:11px; color:#64748b; text-transform:uppercase;
                 letter-spacing:1px; font-family:'Space Mono',monospace;">Plug-in hybride</div>
            <div class="value" style="font-size:28px; font-weight:700; color:#f59e0b;
                 font-family:'Space Mono',monospace;">{pct("⚡ Plug-in hybride")}</div>
        </div>""", unsafe_allow_html=True)
    with col_c:
        st.markdown(f"""
        <div class="metric-box" style="background:#0f172a; border:1px solid #64748b; border-radius:10px;
             padding:16px; text-align:center;">
            <div class="label" style="font-size:11px; color:#64748b; text-transform:uppercase;
                 letter-spacing:1px; font-family:'Space Mono',monospace;">Fossiel</div>
            <div class="value" style="font-size:28px; font-weight:700; color:#94a3b8;
                 font-family:'Space Mono',monospace;">{pct("⛽ Fossiel")}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)

    # --- Lijndiagram aandeel ---
    if df_groep.empty:
        st.warning("Geen data beschikbaar.")
    else:
        fig_aandeel = px.line(
            df_groep,
            x="jaar_maand",
            y="percentage",
            color="categorie",
            color_discrete_map=kleur_map,
            labels={
                "jaar_maand": "Datum",
                "percentage": "Aandeel (%)",
                "categorie": "Aandrijflijn"
            },
            template="plotly_dark",
        )
        fig_aandeel.update_traces(mode="lines", line_shape="spline", line=dict(width=2.5))
        fig_aandeel.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,1)",
            font=dict(family="DM Sans", color="#94a3b8"),
            legend_title="Aandrijflijn",
            xaxis_title="Datum",
            yaxis_title="Aandeel van lopend bestand (%)",
            hovermode="x unified",
            yaxis=dict(ticksuffix="%", gridcolor="#1e293b"),
            xaxis=dict(gridcolor="#1e293b"),
            height=460,
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_aandeel, use_container_width=True)

    st.divider()

    # --- Absolute aantallen per categorie ---
    st.markdown("### Absolute registraties per aandrijflijn")

    fig_abs = px.line(
        df_groep,
        x="jaar_maand",
        y="cumulatief",
        color="categorie",
        color_discrete_map=kleur_map,
        labels={
            "jaar_maand": "Datum",
            "cumulatief": "Cumulatief aantal voertuigen",
            "categorie": "Aandrijflijn"
        },
        template="plotly_dark",
    )
    fig_abs.update_traces(mode="lines", line_shape="spline", line=dict(width=2.5))
    fig_abs.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,1)",
        font=dict(family="DM Sans", color="#94a3b8"),
        legend_title="Aandrijflijn",
        xaxis_title="Datum",
        yaxis_title="Cumulatief aantal",
        hovermode="x unified",
        yaxis=dict(gridcolor="#1e293b", tickformat=","),
        xaxis=dict(gridcolor="#1e293b"),
        height=420,
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_abs, use_container_width=True)

    # --- Voertuigsoort filter ---
    st.divider()
    st.markdown("### Uitsplitsing per voertuigsoort")

    voertuigsoorten = sorted(df_voer["voertuigsoort"].dropna().unique())
    gekozen_soort = st.selectbox("Filter op voertuigsoort:", ["Alle"] + voertuigsoorten)

    df_soort = df_cat.copy()
    if gekozen_soort != "Alle":
        df_soort = df_soort[df_soort["voertuigsoort"] == gekozen_soort]

    df_soort_groep = (
        df_soort
        .groupby(["jaar_maand", "categorie"])
        .size()
        .reset_index(name="aantal")
        .sort_values("jaar_maand")
    )
    df_soort_groep["cumulatief"] = df_soort_groep.groupby("categorie")["aantal"].cumsum()

    fig_soort = px.area(
        df_soort_groep,
        x="jaar_maand",
        y="cumulatief",
        color="categorie",
        color_discrete_map=kleur_map,
        labels={
            "jaar_maand": "Datum",
            "cumulatief": "Cumulatief aantal",
            "categorie": "Aandrijflijn"
        },
        template="plotly_dark",
    )
    fig_soort.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,1)",
        font=dict(family="DM Sans", color="#94a3b8"),
        legend_title="Aandrijflijn",
        hovermode="x unified",
        yaxis=dict(gridcolor="#1e293b", tickformat=","),
        xaxis=dict(gridcolor="#1e293b"),
        height=400,
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_soort, use_container_width=True)