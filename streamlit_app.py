import streamlit as st
import pandas as pd
import pydeck as pdk
import geopandas as gpd
import json
from shapely.geometry import Point

# ===== PAGINA INSTELLINGEN =====
st.set_page_config(page_title="Laadpalen Nederland", page_icon="⚡", layout="wide")
st.title("⚡ Laadpalen Nederland")

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

    return (
        pd.DataFrame(gdf.drop(columns=['geometry', 'index_right'], errors='ignore')),
        provincies_gdf
    )
@st.cache_data
def load_voertuigen():
    df = pd.read_csv('rdw_voertuigen_clean.csv')
    df["datum_tenaamstelling"] = pd.to_datetime(df["datum_tenaamstelling"], errors="coerce")
    df["jaar_maand"] = df["datum_tenaamstelling"].dt.to_period("M").dt.to_timestamp()
    return df
df, provincies_gdf = load_data()
df_voer = load_voertuigen()

# ===== FILTERS OP PAGINA =====
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

    # ===== MARKERS LAAG =====
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

    # ===== PROVINCIEGRENZEN LAAG =====
    if filter_type == "Provincie" and gekozen:
        provincie_shape = provincies_gdf[provincies_gdf['Provincie'] == gekozen]

        if not provincie_shape.empty:
            geojson_data = json.loads(provincie_shape.to_json())

            grens_layer = pdk.Layer(
                "GeoJsonLayer",
                data=geojson_data,
                stroked=True,
                filled=True,
                get_line_color=[220, 38, 38, 255],    # rood
                get_fill_color=[220, 38, 38, 20],     # licht rood transparant
                line_width_min_pixels=3,
            )
            layers.append(grens_layer)

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
            layers=layers,
            initial_view_state=view,
            tooltip=tooltip,
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        ),
        use_container_width=True,
        height=1000
    )
# ===== VOERTUIGREGISTRATIES GRAFIEK =====
st.divider()
st.subheader("📈 Cumulatieve voertuigregistraties per brandstoftype")

alle_brandstoffen = sorted(df_voer["brandstof_omschrijving"].dropna().unique())
geselecteerd = st.multiselect(
    "Filter op brandstoftype",
    options=alle_brandstoffen,
    default=["Benzine", "Diesel", "Elektriciteit", "Benzine/Elektriciteit"]
)

df_gefilterd = df_voer[df_voer["brandstof_omschrijving"].isin(geselecteerd)]

df_groep = (
    df_gefilterd
    .groupby(["jaar_maand", "brandstof_omschrijving"])
    .size()
    .reset_index(name="aantal")
    .sort_values("jaar_maand")
)
df_groep["cumulatief"] = df_groep.groupby("brandstof_omschrijving")["aantal"].cumsum()

# Bereken percentage per maand t.o.v. totaal alle brandstoffen
totaal_per_maand = df_groep.groupby("jaar_maand")["cumulatief"].transform("sum")
df_groep["percentage"] = (df_groep["cumulatief"] / totaal_per_maand * 100).round(2)
if df_groep.empty:
    st.warning("Geen data beschikbaar voor de geselecteerde brandstoffen.")
else:
    import plotly.express as px
    fig = px.line(
    df_groep,
    x="jaar_maand",
    y="percentage",          # ← was "cumulatief"
    color="brandstof_omschrijving",
    labels={
        "jaar_maand": "Datum",
        "percentage": "Aandeel (%)",          # ← aangepast
        "brandstof_omschrijving": "Brandstoftype"
    },
    template="plotly_white"
)
    fig.update_traces(mode="lines", line_shape="spline")
    fig.update_layout(
        legend_title="Brandstoftype",
        xaxis_title="Datum",
        yaxis_title="Cumulatief aantal",
        hovermode="x unified",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)