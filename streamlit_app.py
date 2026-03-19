# ==================== IMPORTS =============================================
import streamlit as st
import pandas as pd
import pydeck as pdk
import geopandas as gpd
import json
import numpy as np
import heapq
import requests
from collections import Counter
import plotly.express as px
from shapely.geometry import Point
from sklearn.neighbors import BallTree
from thefuzz import process as fuzz_process

# ==================== PAGINA INSTELLINGEN ==================================
st.set_page_config(page_title="Groene Mobiliteit Nederland", page_icon="🌱", layout="wide")

# ==================== CUSTOM CSS ===========================================
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

    div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button {
        background: #1e293b !important;
        color: #94a3b8 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 12px !important;
        font-weight: 400 !important;
        letter-spacing: 0 !important;
        text-transform: none !important;
        border: 1px solid #334155 !important;
        border-radius: 6px !important;
        padding: 6px 10px !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button:hover {
        background: #22c55e !important;
        color: #0f172a !important;
        border-color: #22c55e !important;
        transform: none !important;
        box-shadow: none !important;
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

st.markdown("# 🌱 Groene Mobiliteit Nederland")

# ==================== DATA LADEN ============================================
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
    required_columns = ["datum_eerste_toelating", "brandstof_omschrijving"]
    fossiel_brandstoffen = {"benzine", "diesel", "lpg", "cng", "waterstof", "alcohol", "overig"}

    def _prepare_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
        raw = chunk["datum_eerste_toelating"]

        if pd.api.types.is_datetime64_any_dtype(raw):
            dt = raw
        else:
            raw_str = raw.astype("string").str.strip()
            compact_mask = raw_str.str.match(r"^\d{8}", na=False)
            dt_default = pd.to_datetime(raw_str, errors="coerce")
            dt_compact = pd.to_datetime(raw_str.str[:8], format="%Y%m%d", errors="coerce")
            dt = dt_default.where(~compact_mask, dt_compact)

        jaar_maand = dt.dt.to_period("M").dt.to_timestamp()
        brandstof = chunk["brandstof_omschrijving"].astype("string").str.strip().str.lower()

        categorie = pd.Series(pd.NA, index=chunk.index, dtype="string")
        categorie.loc[brandstof.eq("elektriciteit")] = "🔋 Volledig elektrisch"
        categorie.loc[brandstof.isin(fossiel_brandstoffen)] = "⛽ Fossiel"

        out = pd.DataFrame({"jaar_maand": jaar_maand, "categorie": categorie})
        out = out.dropna(subset=["jaar_maand", "categorie"])
        out = out[out["jaar_maand"] >= pd.Timestamp("2005-01-01")]
        return out

    maand_tellingen: Counter = Counter()

    try:
        import pyarrow.dataset as ds

        dataset = ds.dataset("rdw_voertuigen.parquet", format="parquet")
        scanner = dataset.scanner(columns=required_columns, batch_size=250_000)

        for batch in scanner.to_batches():
            chunk = batch.to_pandas()
            if chunk.empty:
                continue
            prepared = _prepare_chunk(chunk)
            if prepared.empty:
                continue
            grouped = prepared.groupby(["jaar_maand", "categorie"]).size()
            for (jaar_maand, categorie), aantal in grouped.items():
                maand_tellingen[(pd.Timestamp(jaar_maand), str(categorie))] += int(aantal)

    except Exception:
        chunk = pd.read_parquet("rdw_voertuigen.parquet", columns=required_columns)
        prepared = _prepare_chunk(chunk)
        grouped = prepared.groupby(["jaar_maand", "categorie"]).size()
        for (jaar_maand, categorie), aantal in grouped.items():
            maand_tellingen[(pd.Timestamp(jaar_maand), str(categorie))] += int(aantal)

    if not maand_tellingen:
        return pd.DataFrame(columns=["jaar_maand", "categorie", "aantal"])

    df_groep = pd.DataFrame(
        [
            {"jaar_maand": ym, "categorie": cat, "aantal": count}
            for (ym, cat), count in maand_tellingen.items()
        ]
    )
    df_groep = df_groep.sort_values("jaar_maand").reset_index(drop=True)
    return df_groep


@st.cache_resource
def build_balltree(_df):
    coords_rad = np.radians(_df[['AddressInfo_Latitude', 'AddressInfo_Longitude']].values)
    tree = BallTree(coords_rad, metric='haversine')
    return tree


# ==================== HULPFUNCTIES: GEOCODING & AUTOCOMPLETE ===============
def geocode_address(address: str):
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


_NL_PLAATSNAMEN_FALLBACK = [
    "Amsterdam", "Rotterdam", "Den Haag", "Utrecht", "Eindhoven", "Tilburg",
    "Groningen", "Almere", "Breda", "Nijmegen", "Enschede", "Haarlem",
    "Arnhem", "Zaanstad", "Amersfoort", "Apeldoorn", "Den Bosch", "Hoofddorp",
    "Maastricht", "Leiden", "Dordrecht", "Zoetermeer", "Zwolle", "Deventer",
    "Delft", "Alkmaar", "Venlo", "Leeuwarden", "Sittard", "Emmen",
    "Helmond", "Hilversum", "Hengelo", "Purmerend", "Oss", "Roosendaal",
    "Vlaardingen", "Schiedam", "Spijkenisse", "Lelystad", "Bergen op Zoom",
    "Hoorn", "Velsen", "Ede", "Gouda", "Westland", "Meierijstad",
]


@st.cache_data(ttl=86400, show_spinner=False)
def load_nl_plaatsnamen() -> list[str]:
    url = "https://opendata.cbs.nl/ODataApi/OData/86097NED/WoonplaatsenPerGemeente"
    headers = {"User-Agent": "LaadpalenNL-StreamlitApp/1.0", "Accept": "application/json"}
    plaatsnamen = []
    try:
        skip = 0
        page_size = 1000
        while True:
            params = {
                "$format": "json",
                "$select": "WoonplaatsNaam",
                "$top": page_size,
                "$skip": skip,
            }
            r = requests.get(url, params=params, headers=headers, timeout=10)
            r.raise_for_status()
            data = r.json().get("value", [])
            if not data:
                break
            for item in data:
                naam = item.get("WoonplaatsNaam", "").strip()
                if naam:
                    plaatsnamen.append(naam)
            if len(data) < page_size:
                break
            skip += page_size

        plaatsnamen = sorted(set(plaatsnamen))
        return plaatsnamen if plaatsnamen else _NL_PLAATSNAMEN_FALLBACK

    except Exception:
        return _NL_PLAATSNAMEN_FALLBACK


NL_PLAATSNAMEN = load_nl_plaatsnamen()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_suggestions(query: str):
    q = query.strip()
    if len(q) < 3:
        return []

    words = q.split()
    corrected_words = []
    for word in words:
        if len(word) >= 4:
            match, score = fuzz_process.extractOne(word, NL_PLAATSNAMEN)
            if score >= 75:
                corrected_words.append(match)
            else:
                corrected_words.append(word)
        else:
            corrected_words.append(word)
    corrected_q = " ".join(corrected_words)

    def _photon(search_q: str):
        url = "https://photon.komoot.io/api/"
        params = {"q": search_q, "limit": 6, "lang": "nl", "bbox": "3.2,50.7,7.3,53.6"}
        headers = {"User-Agent": "LaadpalenNL-StreamlitApp/1.0"}
        try:
            r = requests.get(url, params=params, headers=headers, timeout=5)
            features = r.json().get("features", [])
            results = []
            for f in features:
                props = f.get("properties", {})
                if props.get("countrycode", "").upper() != "NL":
                    continue
                parts = [
                    props.get("name", ""),
                    props.get("street", ""),
                    props.get("housenumber", ""),
                    props.get("city", "") or props.get("town", "") or props.get("village", ""),
                    props.get("state", ""),
                    "Nederland",
                ]
                label = ", ".join(p for p in parts if p)
                if label:
                    results.append(label)
            return results
        except Exception:
            return []

    def _nominatim(search_q: str):
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": search_q + ", Netherlands",
            "format": "json",
            "limit": 6,
            "addressdetails": 0,
            "countrycodes": "nl",
        }
        headers = {"User-Agent": "LaadpalenNL-StreamlitApp/1.0"}
        try:
            r = requests.get(url, params=params, headers=headers, timeout=5)
            return [res["display_name"] for res in r.json()]
        except Exception:
            return []

    results = _photon(corrected_q)
    if not results and corrected_q != q:
        results = _photon(q)
    if not results:
        results = _nominatim(corrected_q)
    if not results and corrected_q != q:
        results = _nominatim(q)

    seen = set()
    unique = []
    for r in results:
        key = r.lower()
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique[:6]


# ==================== HULPFUNCTIES: ROUTING & DIJKSTRA =====================
def _osrm_route(origin_lon, origin_lat, dest_lon, dest_lat):
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
    R = 6371.0

    user_rad = np.radians([[user_lat, user_lon]])
    distances_rad, indices = tree.query(user_rad, k=n_candidates)

    USER_NODE = 0
    graph        = {USER_NODE: []}
    route_cache  = {}

    for rank, charger_idx in enumerate(indices[0]):
        node_id = rank + 1
        c_lat = float(df.iloc[charger_idx]['AddressInfo_Latitude'])
        c_lon = float(df.iloc[charger_idx]['AddressInfo_Longitude'])

        dist_km, dur_min, coords = _osrm_route(user_lon, user_lat, c_lon, c_lat)

        if dist_km is None:
            dist_km  = float(distances_rad[0][rank]) * R
            dur_min  = None
            coords   = None

        graph[USER_NODE].append((node_id, dist_km))
        graph[node_id]  = []
        route_cache[node_id] = (int(charger_idx), dist_km, dur_min, coords)

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

    if road_coords is None:
        c_lat = float(df.iloc[best_charger_idx]['AddressInfo_Latitude'])
        c_lon = float(df.iloc[best_charger_idx]['AddressInfo_Longitude'])
        _, _, road_coords = _osrm_route(user_lon, user_lat, c_lon, c_lat)

    return best_charger_idx, road_dist_km, road_dur_min, road_coords


# ==================== APP OPSTARTEN ========================================
df, provincies_gdf = load_data()
df_voer_groep = load_voertuigen()
ball_tree = build_balltree(df)

# ==================== TABS =================================================
tab3, tab1, tab2 = st.tabs(["📈  Voertuigregistraties", "🗺️  Laadpalen Kaart", "🔍  Laadpalen dichtbij mij"])

# ==================== TAB 1: LAADPALEN KAART ===============================
with tab1:
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        filter_type = st.radio("Bekijken per:", ["Alle locaties", "Provincie", "Plaats"])

    with col2:
        filtered_df = df.copy()
        gekozen = None

        if filter_type == "Provincie":
            gekozen = st.selectbox("Kies een provincie:", sorted(df['Provincie'].unique()))
            filtered_df = df[df['Provincie'] == gekozen]
        elif filter_type == "Plaats":
            gekozen = st.selectbox("Kies een plaats:", sorted(df['AddressInfo_Town'].dropna().unique()))
            filtered_df = df[df['AddressInfo_Town'] == gekozen]
        else:
            st.write("")

    with col3:
        st.metric("Laadpalen zichtbaar", len(filtered_df))

    st.divider()

    # Legenda uitleg
    st.markdown("""
    <div style="display:flex; gap:24px; margin-bottom:16px; flex-wrap:wrap;">
        <div style="display:flex; align-items:center; gap:8px;">
            <div style="width:14px; height:14px; border-radius:50%; background:#94a3b8;"></div>
            <span style="font-size:12px; color:#94a3b8; font-family:'Space Mono',monospace;">Onbekend</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
            <div style="width:14px; height:14px; border-radius:50%; background:#60a5fa;"></div>
            <span style="font-size:12px; color:#94a3b8; font-family:'Space Mono',monospace;">Langzaam &lt; 22 kW</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
            <div style="width:14px; height:14px; border-radius:50%; background:#facc15;"></div>
            <span style="font-size:12px; color:#94a3b8; font-family:'Space Mono',monospace;">Snel 22–100 kW</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
            <div style="width:14px; height:14px; border-radius:50%; background:#f97316;"></div>
            <span style="font-size:12px; color:#94a3b8; font-family:'Space Mono',monospace;">Supersnel 100–150 kW</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
            <div style="width:14px; height:14px; border-radius:50%; background:#ef4444;"></div>
            <span style="font-size:12px; color:#94a3b8; font-family:'Space Mono',monospace;">Ultrasnel &gt; 150 kW</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
        map_df['power_num'] = pd.to_numeric(map_df['power'], errors='coerce')

        def power_color(kw):
            if pd.isna(kw):
                return [148, 163, 184, 200]   # grijs — onbekend
            elif kw < 22:
                return [96, 165, 250, 220]    # blauw — langzaam
            elif kw < 100:
                return [250, 204, 21, 220]    # geel — snel
            elif kw < 150:
                return [249, 115, 22, 220]    # oranje — supersnel
            else:
                return [239, 68, 68, 220]     # rood — ultrasnel

        map_df['color'] = map_df['power_num'].apply(power_color)
        map_df['power'] = map_df['power'].fillna('N/A')
        map_df = map_df.fillna('N/A')

        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position='[lon, lat]',
            get_fill_color='color',
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
            width="stretch",
            height=1000
        )

# ==================== TAB 2: DIJKSTRA =====================================
with tab2:
    st.markdown('<div class="algo-badge">DIJKSTRA\'S ALGORITHM</div>', unsafe_allow_html=True)
    st.markdown("### Vind de dichtstbijzijnde laadpaal")
    st.markdown("Voer een adres, plaatsnaam of postcode in. Het algoritme berekent de kortste route door het laadpalennetwerk.")

    for _k, _v in [
        ("address_input",    ""),
        ("suggestions",      []),
        ("selected_address", ""),
        ("input_version",    0),
    ]:
        if _k not in st.session_state:
            st.session_state[_k] = _v

    col_search, col_btn = st.columns([3, 1])

    input_key = f"address_input_v{st.session_state['input_version']}"

    with col_search:
        typed = st.text_input(
            "Startlocatie",
            value=st.session_state["address_input"],
            placeholder="bijv. Kalverstraat 1, Amsterdam  •  3012 Rotterdam  •  Eindhoven Centrum",
            label_visibility="collapsed",
            key=input_key,
        )

    with col_btn:
        search_clicked = st.button("⚡ Zoek Route", width="stretch")

    if typed != st.session_state["address_input"]:
        st.session_state["address_input"]    = typed
        st.session_state["selected_address"] = ""
        if len(typed.strip()) >= 3:
            st.session_state["suggestions"] = fetch_suggestions(typed.strip())
        else:
            st.session_state["suggestions"] = []

    suggestions = st.session_state["suggestions"]
    if suggestions and not st.session_state["selected_address"]:
        st.markdown(
            "<div style='margin: 6px 0 4px 0; font-size:11px; color:#64748b; "
            "font-family:Space Mono,monospace; letter-spacing:1px;'>SUGGESTIES</div>",
            unsafe_allow_html=True,
        )
        sug_cols = st.columns(len(suggestions))
        for i, suggestion in enumerate(suggestions):
            short_label = ", ".join(suggestion.split(", ")[:2])
            with sug_cols[i]:
                if st.button(short_label, key=f"sug_{i}"):
                    st.session_state["selected_address"] = suggestion
                    st.session_state["address_input"]    = suggestion
                    st.session_state["suggestions"]      = []
                    st.session_state["input_version"]   += 1
                    st.rerun()

    final_address = st.session_state["selected_address"] or st.session_state["address_input"]

    if search_clicked and final_address.strip():
        with st.spinner("Locatie zoeken en route berekenen..."):
            user_lat, user_lon, display_name = geocode_address(final_address.strip())

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
                width="stretch",
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
        st.markdown("""
        <div style="margin-top:40px; text-align:center; color:#334155;">
            <div style="font-size:48px; margin-bottom:16px;">🔍</div>
            <p style="font-family:'Space Mono',monospace; font-size:14px; color:#64748b;">
                Voer een adres in en klik op "Zoek Route"<br>om de dichtstbijzijnde laadpaal te vinden.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ==================== TAB 3: VOERTUIGREGISTRATIES ==========================
with tab3:
    st.markdown('<div class="algo-badge">RDW OPEN DATA</div>', unsafe_allow_html=True)
    st.markdown("### Aandeel voertuigen per aandrijflijn")
    st.markdown("Cumulatief aandeel van het Nederlandse wagenpark per brandstofcategorie over de tijd.")

    df_groep = df_voer_groep.copy()
    df_groep = df_groep.sort_values("jaar_maand")
    df_groep["cumulatief"] = df_groep.groupby("categorie")["aantal"].cumsum()

    totaal_per_maand = df_groep.groupby("jaar_maand")["cumulatief"].transform("sum")
    df_groep["percentage"] = (df_groep["cumulatief"] / totaal_per_maand * 100).round(2)

    kleur_map = {
        "🔋 Volledig elektrisch": "#22c55e",
        "⛽ Fossiel":              "#64748b",
    }

    if df_groep.empty:
        st.warning("Geen voertuigdata beschikbaar.")
    else:
        min_datum = df_groep["jaar_maand"].min().year
        max_datum = df_groep["jaar_maand"].max().year

        jaar_range = st.slider(
            "Selecteer tijdsbereik (jaren):",
            min_value=min_datum,
            max_value=max_datum,
            value=(min_datum, max_datum)
        )

        df_groep_gefilterd = df_groep[
            (df_groep["jaar_maand"].dt.year >= jaar_range[0]) &
            (df_groep["jaar_maand"].dt.year <= jaar_range[1])
        ]

        if df_groep_gefilterd.empty:
            st.warning("Geen data beschikbaar voor de geselecteerde periode.")
        else:
            laatste_maand_gefilterd = df_groep_gefilterd["jaar_maand"].max()
            laatste = df_groep_gefilterd[
                df_groep_gefilterd["jaar_maand"] == laatste_maand_gefilterd
            ].set_index("categorie")

            def pct(cat):
                return f"{laatste.loc[cat, 'percentage']:.1f}%" if cat in laatste.index else "N/A"

            col_a, col_c = st.columns(2)
            with col_a:
                st.markdown(f"""
                <div class="metric-box" style="background:#0f172a; border:1px solid #22c55e; border-radius:10px;
                     padding:16px; text-align:center;">
                    <div class="label" style="font-size:11px; color:#64748b; text-transform:uppercase;
                         letter-spacing:1px; font-family:'Space Mono',monospace;">Volledig elektrisch</div>
                    <div class="value" style="font-size:28px; font-weight:700; color:#22c55e;
                         font-family:'Space Mono',monospace;">{pct("🔋 Volledig elektrisch")}</div>
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

            fig_aandeel = px.line(
                df_groep_gefilterd,
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
            fig_aandeel.update_traces(line=dict(width=2.5))
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

            st.markdown("### Registraties per aandrijflijn")

            fig_abs = px.line(
                df_groep_gefilterd,
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
            fig_abs.update_traces(line=dict(width=2.5))
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
            # ── na de bestaande fig_abs plotly_chart ──────────────────────

            st.divider()
            st.markdown("### 🔮 Voorspelling tot 2050")
            st.markdown("Lineaire regressie op basis van historische maanddata.")

            from sklearn.linear_model import LinearRegression
            import numpy as np

            # ── helper: regressie + voorspelling ────────────────────────────
            def voorspel(df_cat: pd.DataFrame, categorie: str) -> pd.DataFrame:
                sub = df_cat[(df_cat["categorie"] == categorie) &
                (df_cat["jaar_maand"] >= pd.Timestamp("2024-01-01")) ].copy()
                if sub.empty:
                    return pd.DataFrame()

                # X = maanden sinds eerste datapunt (integer)
                t0 = sub["jaar_maand"].min()
                sub["t"] = ((sub["jaar_maand"].dt.year - t0.year) * 12
                            + (sub["jaar_maand"].dt.month - t0.month))

                model = LinearRegression()
                model.fit(sub[["t"]], sub["cumulatief"])

                # Voorspel tot december 2050
                t_max = (2050 - t0.year) * 12 + (12 - t0.month)
                t_toekomst = np.arange(sub["t"].max() + 1, t_max + 1)
                datums_toekomst = [
                    t0 + pd.DateOffset(months=int(t)) for t in t_toekomst
                ]
                voorspeld = model.predict(t_toekomst.reshape(-1, 1))
                voorspeld = np.maximum(voorspeld, 0)  # geen negatieve aantallen

                return pd.DataFrame({
                    "jaar_maand": datums_toekomst,
                    "cumulatief": voorspeld,
                    "categorie":  categorie,
                    "type":       "Voorspelling",
                })

            # ── combineer historisch + voorspelling ─────────────────────────
            historisch = df_groep_gefilterd.copy()
            historisch["type"] = "Historisch"

            frames = [historisch]
            for cat in ["🔋 Volledig elektrisch", "⛽ Fossiel"]:
                pred = voorspel(df_groep, cat)   # regressie op volledige dataset
                if not pred.empty:
                    frames.append(pred)

            df_pred = pd.concat(frames, ignore_index=True)

            # ── cumulatief grafiek met voorspelling ──────────────────────────
            lijn_stijl = {"Historisch": "solid", "Voorspelling": "dot"}

            fig_pred = px.line(
                df_pred,
                x="jaar_maand",
                y="cumulatief",
                color="categorie",
                line_dash="type",
                line_dash_map=lijn_stijl,
                color_discrete_map=kleur_map,
                labels={
                    "jaar_maand": "Datum",
                    "cumulatief": "Cumulatief aantal voertuigen",
                    "categorie":  "Aandrijflijn",
                    "type":       "Type",
                },
                template="plotly_dark",
            )
            fig_pred.update_traces(line=dict(width=2.5))

            # Verticale lijn op vandaag
            fig_pred.add_vline(
                x=pd.Timestamp("today").timestamp() * 1000,
                line_dash="dash",
                line_color="#fbbf24",
                annotation_text="Vandaag",
                annotation_font_color="#fbbf24",
                annotation_position="top right",
            )

            fig_pred.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,1)",
                font=dict(family="DM Sans", color="#94a3b8"),
                legend_title="Aandrijflijn",
                xaxis_title="Datum",
                yaxis_title="Cumulatief aantal",
                hovermode="x unified",
                yaxis=dict(gridcolor="#1e293b", tickformat=","),
                xaxis=dict(gridcolor="#1e293b"),
                height=460,
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_pred, use_container_width=True)

            st.divider()

            # ── verhouding elektrisch / fossiel voorspelling ─────────────────
            st.markdown("### ⚖️ Voorspelde verhouding elektrisch vs fossiel")

            # Bereken verhouding op gecombineerde data (historisch + voorspelling)
            df_ratio = (
                df_pred
                .groupby(["jaar_maand", "categorie"])["cumulatief"]
                .sum()
                .reset_index()
            )
            totaal = df_ratio.groupby("jaar_maand")["cumulatief"].transform("sum")
            df_ratio["percentage"] = (df_ratio["cumulatief"] / totaal * 100).round(2)

            # Voeg type toe voor stippellijn na vandaag
            vandaag = pd.Timestamp("today")
            df_ratio["type"] = df_ratio["jaar_maand"].apply(
                lambda d: "Historisch" if d <= vandaag else "Voorspelling"
            )

            fig_ratio = px.line(
                df_ratio,
                x="jaar_maand",
                y="percentage",
                color="categorie",
                line_dash="type",
                line_dash_map=lijn_stijl,
                color_discrete_map=kleur_map,
                labels={
                    "jaar_maand": "Datum",
                    "percentage": "Aandeel (%)",
                    "categorie":  "Aandrijflijn",
                    "type":       "Type",
                },
                template="plotly_dark",
            )
            fig_ratio.update_traces(line=dict(width=2.5))
            fig_ratio.add_vline(
                x=pd.Timestamp("today").timestamp() * 1000,
                line_dash="dash",
                line_color="#fbbf24",
                annotation_text="Vandaag",
                annotation_font_color="#fbbf24",
                annotation_position="top right",
            )
            fig_ratio.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,1)",
                font=dict(family="DM Sans", color="#94a3b8"),
                legend_title="Aandrijflijn",
                xaxis_title="Datum",
                yaxis_title="Aandeel van wagenpark (%)",
                hovermode="x unified",
                yaxis=dict(ticksuffix="%", gridcolor="#1e293b"),
                xaxis=dict(gridcolor="#1e293b"),
                height=420,
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_ratio, use_container_width=True)

            # ── mijlpalen tabel ──────────────────────────────────────────────
            st.markdown("### 📅 Voorspelde mijlpalen")

            mijlpalen = []
            for drempel in [10, 25, 50, 75]:
                elek = df_ratio[
                    (df_ratio["categorie"] == "🔋 Volledig elektrisch")
                    & (df_ratio["percentage"] >= drempel)
                    & (df_ratio["jaar_maand"] > vandaag)
                ]
                if not elek.empty:
                    datum = elek["jaar_maand"].min()
                    mijlpalen.append({
                        "Mijlpaal": f"{drempel}% elektrisch",
                        "Voorspeld jaar": datum.year,
                        "Voorspelde maand": datum.strftime("%B %Y"),
                    })

            if mijlpalen:
                df_mijl = pd.DataFrame(mijlpalen)
                st.dataframe(
                    df_mijl,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Mijlpaal":          st.column_config.TextColumn("Mijlpaal"),
                        "Voorspeld jaar":    st.column_config.NumberColumn("Jaar", format="%d"),
                        "Voorspelde maand":  st.column_config.TextColumn("Datum"),
                    }
                )
            else:
                st.info("Geen mijlpalen bereikt binnen de voorspellingsperiode op basis van lineaire regressie.")

            st.markdown("""
            <div style="margin-top:12px; padding:12px 16px; background:#0f172a; border-radius:8px;
                        border-left:3px solid #fbbf24; font-size:12px; color:#94a3b8;">
                <span style="color:#fbbf24; font-family:'Space Mono',monospace; font-weight:700;">
                    METHODOLOGIE
                </span><br><br>
                Lineaire regressie (OLS) op cumulatieve maandtotalen per categorie vanaf 2005.
                De stippellijn toont de voorspelling — de doorgaande lijn de historische data.
                Lineaire regressie houdt geen rekening met beleidsveranderingen, economische schokken
                of verzadigingseffecten. Gebruik als indicatie, niet als prognose.
            </div>
            """, unsafe_allow_html=True)