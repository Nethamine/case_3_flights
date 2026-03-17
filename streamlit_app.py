import streamlit as st
import pandas as pd
import pydeck as pdk
import geopandas as gpd
from shapely.geometry import Point

# ===== PAGINA INSTELLINGEN =====
st.set_page_config(page_title="Laadpalen Nederland", page_icon="⚡", layout="wide")
st.title("⚡ Laadpalen Nederland")

# ===== DATA LADEN =====
@st.cache_data
def load_data():
    # Laadpalen
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

    # Provinciegrenzen laden vanuit lokaal bestand
    provincies_gdf = gpd.read_file('provincies.geojson')
    provincies_gdf = provincies_gdf[['statnaam', 'geometry']].rename(columns={'statnaam': 'Provincie'})

    # Coördinaten omzetten naar GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(lon, lat) for lon, lat in zip(df['AddressInfo_Longitude'], df['AddressInfo_Latitude'])],
        crs="EPSG:4326"
    )

    # Zorg dat beide hetzelfde CRS hebben
    provincies_gdf = provincies_gdf.to_crs("EPSG:4326")

    # Spatial join: koppel elke laadpaal aan een provincie op basis van coördinaten
    gdf = gdf.sjoin(provincies_gdf, how='left', predicate='within')
    gdf['Provincie'] = gdf['Provincie'].fillna('Onbekend')

    return pd.DataFrame(gdf.drop(columns=['geometry', 'index_right'], errors='ignore'))

df = load_data()

# ===== FILTERS OP PAGINA =====
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    filter_type = st.radio("Bekijken per:", ["Alle locaties", "Provincie", "Gemeente"])

with col2:
    filtered_df = df.copy()

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

# ===== KAART =====
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

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position='[lon, lat]',
        get_fill_color='[34, 197, 94, 200]',
        get_radius=5,
        radius_min_pixels=4,
        radius_max_pixels=12,
        pickable=True,
    )

    view = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lng,
        zoom=zoom,
        pitch=0,
    )

    tooltip = {
        "html": "<b>{title}</b><br>{address}<br>{town}<br>Vermogen: {power} kW<br>Provincie: {provincie}",
        "style": {"backgroundColor": "#1e293b", "color": "white", "fontSize": "13px", "padding": "8px"}
    }

    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view,
            tooltip=tooltip,
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        ),
        use_container_width=True,
        height=1000
    )