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

# ==================== INTRO PAGE ============================================
st.markdown("""
<style>
.intro-wrap {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
}
}
.intro-wrap::before {
    content: '';
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(rgba(34,197,94,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(34,197,94,0.04) 1px, transparent 1px);
    background-size: 40px 40px;
    animation: gridPan 20s linear infinite;
    pointer-events: none;
}
.intro-wrap::after {
    content: '';
    position: absolute;
    top: -120px;
    right: -80px;
    width: 500px;
    height: 500px;
    background: radial-gradient(ellipse at center, rgba(34,197,94,0.12) 0%, transparent 70%);
    pointer-events: none;
    animation: pulse 6s ease-in-out infinite;
}
@keyframes gridPan {
    0%   { background-position: 0 0; }
    100% { background-position: 40px 40px; }
}
@keyframes pulse {
    0%, 100% { opacity: 0.6; transform: scale(1); }
    50%       { opacity: 1;   transform: scale(1.1); }
}
.intro-inner {
    position: relative;
    z-index: 2;
}
.intro-label {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.3);
    color: #22c55e;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 5px 14px;
    border-radius: 999px;
    margin-bottom: 28px;
    animation: fadeUp 0.6s ease both;
}
.intro-dot {
    width: 7px;
    height: 7px;
    background: #22c55e;
    border-radius: 50%;
    display: inline-block;
    animation: blink 1.4s ease-in-out infinite;
}
@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.2; }
}
.intro-title {
    font-family: 'Space Mono', monospace;
    font-size: 52px;
    font-weight: 700;
    line-height: 1.1;
    letter-spacing: -1px;
    color: #f1f5f9;
    margin: 0 0 6px;
    animation: fadeUp 0.6s 0.1s ease both;
}
.intro-title span {
    color: #22c55e;
}
.intro-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 18px;
    font-weight: 300;
    color: #64748b;
    margin: 0 0 40px;
    max-width: 560px;
    line-height: 1.6;
    animation: fadeUp 0.6s 0.2s ease both;
}
.intro-stats {
    display: flex;
    gap: 0;
    margin-bottom: 44px;
    background: transparent !important;
    border: 1px solid #1e293b !important;
    border-radius: 12px;
    overflow: hidden;
    animation: fadeUp 0.6s 0.3s ease both;
}
.stat-item {
    flex: 1;
    padding: 22px 24px;
    background: #0f172a !important;
    border-right: 1px solid #1e293b;
    transition: background 0.2s;
}
.stat-item:last-child { border-right: none; }
.stat-item:hover { background: rgba(34,197,94,0.05); }
.stat-number {
    font-family: 'Space Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    color: #22c55e;
    line-height: 1;
    margin-bottom: 4px;
}
.stat-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 18px;
    color: #475569;
    letter-spacing: 0.5px;
}
.intro-features {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    animation: fadeUp 0.6s 0.4s ease both;
}
.feature-card {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px;
    padding: 20px;
    transition: all 0.25s;
}
.feature-card:hover {
    border-color: rgba(34,197,94,0.4);
    background: rgba(34,197,94,0.05);
    transform: translateY(-2px);
}
.feature-icon {
    font-size: 26px;
    margin-bottom: 14px;
    display: block;
}
.feature-title {
    font-family: 'Space Mono', monospace;
    font-size: 20px;
    font-weight: 700;
    color: #22c55e !important;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
    text-transform: uppercase;
}
.feature-desc {
    font-family: 'DM Sans', sans-serif;
    font-size: 18px;
    color: #94a3b8 !important;
    line-height: 1.5;
}
.scroll-hint {
    text-align: center;
    margin-top: 16px;
    margin-bottom: 40px;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    color: #334155;
    text-transform: uppercase;
    animation: bounce 2s 2s ease-in-out infinite;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes bounce {
    0%, 100% { transform: translateY(0); }
    50%       { transform: translateY(5px); }
}
            
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="intro-wrap">
    <div class="intro-inner">
        <div class="intro-label">
            <span class="intro-dot"></span>
            Live &nbsp;·&nbsp; Nederland &nbsp;·&nbsp; 2025
        </div>
        <h1 class="intro-title">Groene<br><span>Mobiliteit</span><br>Nederland</h1>
        <p class="intro-sub">
            Interactief dashboard voor laadpalen, elektrische voertuigen
            en de energietransitie op de Nederlandse weg.
        </p>
        <div class="intro-stats">
            <div class="stat-item">
                <div class="stat-number">8.134</div>
                <div class="stat-label">Laadpunten in NL</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">700K+</div>
                <div class="stat-label">Elektrische voertuigen</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">12</div>
                <div class="stat-label">Provincies in kaart</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">2050</div>
                <div class="stat-label">Voorspelling horizon</div>
            </div>
        </div>
        <div class="intro-features">
            <div class="feature-card">
                <span class="feature-icon">📈</span>
                <div class="feature-title">Voertuigregistraties</div>
                <div class="feature-desc">
                    Historische groei van elektrisch vs. fossiel rijden,
                    inclusief lineaire regressie tot 2050.
                </div>
            </div>
            <div class="feature-card">
                <span class="feature-icon">🗺️</span>
                <div class="feature-title">Laadpalen Kaart</div>
                <div class="feature-desc">
                    Interactieve kaart van alle publieke laadpunten,
                    gefilterd per provincie en aansluittype.
                </div>
            </div>
            <div class="feature-card">
                <span class="feature-icon">🔍</span>
                <div class="feature-title">Laadpalen Dichtbij</div>
                <div class="feature-desc">
                    Vind de dichtstbijzijnde laadpalen op basis van
                    jouw locatie via slim zoeken met Dijkstra &amp; KNN.
                </div>
            </div>
        </div>
    </div>
</div>
<div class="scroll-hint">&#8595; &nbsp; Kies een tab om te beginnen &nbsp; &#8595;</div>
""", unsafe_allow_html=True)

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
    st.markdown("### Aandeel voertuigen per brandstoftype")
    st.markdown("Cumulatief aandeel van de Nederlandse automarkt per brandstofcategorie over de tijd.")

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    import numpy as np

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
        alle_maanden = sorted(df_groep["jaar_maand"].unique())

        geselecteerde_maanden = st.select_slider(
            "Selecteer tijdsbereik:",
            options=alle_maanden,
            value=(alle_maanden[0], alle_maanden[-1]),
            format_func=lambda x: pd.Timestamp(x).strftime("%b %Y"),
        )

        df_groep_gefilterd = df_groep[
            (df_groep["jaar_maand"] >= geselecteerde_maanden[0]) &
            (df_groep["jaar_maand"] <= geselecteerde_maanden[1])
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
            # ── 1. CUMULATIEF ────────────────────────────────────────────────
            st.markdown("### Registraties per brandstoftype")
            st.info(" De grafieken hieronder zijn gebaseerd op 7,1 miljoen voertuigregistraties uit de RDW dataset. Dit is een subset van het volledige Nederlandse aantal.")
            fig_abs = px.line(
                df_groep_gefilterd,
                x="jaar_maand",
                y="cumulatief",
                color="categorie",
                color_discrete_map=kleur_map,
                labels={"jaar_maand": "Datum", "cumulatief": "Cumulatief aantal voertuigen", "categorie": "Brandstoftype"},
                template="plotly_dark",
            )
            fig_abs.update_traces(line=dict(width=2.5))
            fig_abs.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,1)",
                font=dict(family="DM Sans", color="#94a3b8"),
                legend_title="Brandstoftype",
                xaxis_title="Datum",
                yaxis_title="Cumulatief aantal auto's",
                hovermode="x unified",
                yaxis=dict(gridcolor="#1e293b", tickformat=","),
                xaxis=dict(gridcolor="#1e293b"),
                height=420,
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_abs, use_container_width=True)

            st.divider()

            # ── 2. VERHOUDING ────────────────────────────────────────────────
            st.markdown("### Aandeel per Brandstoftype")

            fig_aandeel = px.line(
                df_groep_gefilterd,
                x="jaar_maand",
                y="percentage",
                color="categorie",
                color_discrete_map=kleur_map,
                labels={"jaar_maand": "Datum", "percentage": "Aandeel (%)", "categorie": "Brandstoftype"},
                template="plotly_dark",
            )
            fig_aandeel.update_traces(line=dict(width=2.5))
            fig_aandeel.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,1)",
                font=dict(family="DM Sans", color="#94a3b8"),
                legend_title="Brandstoftype",
                xaxis_title="Datum",
                yaxis_title="Aandeel van brandstoftype (%)",
                hovermode="x unified",
                yaxis=dict(ticksuffix="%", gridcolor="#1e293b"),
                xaxis=dict(gridcolor="#1e293b"),
                height=460,
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_aandeel, use_container_width=True)

            st.divider()

            # ── 3. VOORSPELLING ──────────────────────────────────────────────
            st.markdown("###  Voorspelling tot 2050 voor de brandstofsoort")

            elek = df_groep[df_groep["categorie"] == "🔋 Volledig elektrisch"].copy()
            elek = elek[elek["jaar_maand"] >= pd.Timestamp("2018-01-01")]

            MIN_PUNTEN = 12

            beste_r2         = -np.inf
            beste_startdatum = None
            beste_model      = None
            beste_sub        = None

            kandidaat_data = elek.sort_values("jaar_maand").reset_index(drop=True)

            for i in range(len(kandidaat_data) - MIN_PUNTEN):
                sub = kandidaat_data.iloc[i:].copy()
                t0  = sub["jaar_maand"].iloc[0]
                sub["t"] = (
                    (sub["jaar_maand"].dt.year  - t0.year)  * 12
                    + (sub["jaar_maand"].dt.month - t0.month)
                )
                model = LinearRegression()
                model.fit(sub[["t"]], sub["percentage"])
                r2 = r2_score(sub["percentage"], model.predict(sub[["t"]]))

                if r2 > beste_r2:
                    beste_r2         = r2
                    beste_startdatum = t0
                    beste_model      = model
                    beste_sub        = sub

            st.markdown(
                f"<div style='margin-bottom:16px;'>"
                f"<span style='font-family:Space Mono,monospace; font-size:12px; color:#64748b;'>"
                f"BESTE STARTPUNT: </span>"
                f"<span style='font-family:Space Mono,monospace; font-size:12px; color:#22c55e;'>"
                f"{beste_startdatum.strftime('%B %Y')} &nbsp;|&nbsp; R² = {beste_r2:.4f}"
                f"</span></div>",
                unsafe_allow_html=True,
            )

            def voorspel(categorie: str, startdatum: pd.Timestamp) -> pd.DataFrame:
                sub = df_groep[
                    (df_groep["categorie"] == categorie) &
                    (df_groep["jaar_maand"] >= startdatum)
                ].copy()
                if sub.empty:
                    return pd.DataFrame()

                t0 = sub["jaar_maand"].min()
                sub["t"] = (
                    (sub["jaar_maand"].dt.year  - t0.year)  * 12
                    + (sub["jaar_maand"].dt.month - t0.month)
                )
                model = LinearRegression()
                model.fit(sub[["t"]], sub["percentage"])

                t_max      = (2050 - t0.year) * 12 + (12 - t0.month)
                t_toekomst = np.arange(sub["t"].max() + 1, t_max + 1)
                datums     = [t0 + pd.DateOffset(months=int(t)) for t in t_toekomst]
                voorspeld  = np.clip(model.predict(t_toekomst.reshape(-1, 1)), 0, 100)

                return pd.DataFrame({
                    "jaar_maand": datums,
                    "percentage": voorspeld,
                    "categorie":  categorie,
                    "type":       "Voorspelling",
                })

            df_hist = df_groep[["jaar_maand", "percentage", "categorie"]].copy()
            df_hist["type"] = "Historisch"

            frames = [df_hist]
            for cat in ["🔋 Volledig elektrisch", "⛽ Fossiel"]:
                pred = voorspel(cat, beste_startdatum)
                if not pred.empty:
                    frames.append(pred)

            df_ratio = pd.concat(frames, ignore_index=True)

            fig_ratio = px.line(
                df_ratio,
                x="jaar_maand",
                y="percentage",
                color="categorie",
                line_dash="type",
                line_dash_map={"Historisch": "solid", "Voorspelling": "dot"},
                color_discrete_map=kleur_map,
                labels={
                    "jaar_maand": "Datum",
                    "percentage": "Aandeel (%)",
                    "categorie":  "Brandstoftype",
                    "type":       "Type",
                },
                template="plotly_dark",
            )
            fig_ratio.update_traces(line=dict(width=2.5))
            # ── kruispunt elektrisch > fossiel ───────────────────────────────
            elek_ratio = df_ratio[df_ratio["categorie"] == "🔋 Volledig elektrisch"].copy()
            foss_ratio = df_ratio[df_ratio["categorie"] == "⛽ Fossiel"].copy()

            kruispunt = elek_ratio.merge(foss_ratio, on="jaar_maand", suffixes=("_elek", "_foss"))
            kruispunt = kruispunt[
                (kruispunt["type_elek"] == "Voorspelling") &
                (kruispunt["percentage_elek"] >= kruispunt["percentage_foss"])
            ]

            if not kruispunt.empty:
                kruis_datum  = kruispunt["jaar_maand"].iloc[0]
                kruis_pct    = kruispunt["percentage_elek"].iloc[0]

                fig_ratio.add_annotation(
                    x=kruis_datum,
                    y=kruis_pct,
                    text=f" {kruis_datum.strftime('%B %Y')}",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="#22c55e",
                    arrowwidth=1.5,
                    ax=-80,
                    ay=0,
                    font=dict(
                        family="Space Mono, monospace",
                        size=11,
                        color="#22c55e",
                    ),
                    bgcolor="#0f172a",
                    bordercolor="#22c55e",
                    borderwidth=1,
                    borderpad=6,
                )
            fig_ratio.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,1)",
                font=dict(family="DM Sans", color="#94a3b8"),
                legend_title="Brandstoftype",
                xaxis_title="Datum",
                yaxis_title="Aandeel van automarkt (%)",
                hovermode="x unified",
                yaxis=dict(ticksuffix="%", gridcolor="#1e293b", range=[0, 100]),
                xaxis=dict(gridcolor="#1e293b"),
                height=480,
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_ratio, use_container_width=True)

            # ── mijlpalen tabel ──────────────────────────────────────────────
            st.markdown("###  Voorspelde mijlpalen elektrisch")

            mijlpalen = []
            elek_pred = df_ratio[
                (df_ratio["categorie"] == "🔋 Volledig elektrisch") &
                (df_ratio["type"] == "Voorspelling")
            ]
            for drempel in [25, 50, 75, 100]:
                bereikt = elek_pred[elek_pred["percentage"] >= drempel]
                if not bereikt.empty:
                    datum = bereikt["jaar_maand"].min()
                    mijlpalen.append({
                        "Mijlpaal":         f"{drempel}% elektrisch",
                        "Voorspelde maand": datum.strftime("%B %Y"),
                    })
                else:
                    mijlpalen.append({
                        "Mijlpaal":         f"{drempel}% elektrisch",
                        "Voorspelde maand": "Na 2050",
                    })

            st.dataframe(
                pd.DataFrame(mijlpalen),
                hide_index=True,
                use_container_width=True,
            )

