import streamlit as st
import pandas as pd
import pydeck as pdk

# ===== GEMEENTE -> PROVINCIE MAPPING =====
gemeente_provincie = {
    "groningen": "Groningen", "delfzijl": "Groningen", "appingedam": "Groningen",
    "loppersum": "Groningen", "bedum": "Groningen", "winsum": "Groningen",
    "eemsmond": "Groningen", "de marne": "Groningen", "leek": "Groningen",
    "grootegast": "Groningen", "marum": "Groningen", "zuidhorn": "Groningen",
    "hoogezand-sappemeer": "Groningen", "slochteren": "Groningen", "menterwolde": "Groningen",
    "veendam": "Groningen", "pekela": "Groningen", "stadskanaal": "Groningen",
    "bellingwedde": "Groningen", "vlagtwedde": "Groningen", "oldambt": "Groningen",
    "westerkwartier": "Groningen", "midden-groningen": "Groningen", "het hogeland": "Groningen",
    "eemsdelta": "Groningen",
    "leeuwarden": "Friesland", "sneek": "Friesland", "harlingen": "Friesland",
    "franekeradeel": "Friesland", "heerenveen": "Friesland", "opsterland": "Friesland",
    "ooststellingwerf": "Friesland", "weststellingwerf": "Friesland",
    "súdwest-fryslân": "Friesland", "de fryske marren": "Friesland",
    "noardeast-fryslân": "Friesland", "waadhoeke": "Friesland", "smallingerland": "Friesland",
    "achtkarspelen": "Friesland", "tytsjerksteradiel": "Friesland",
    "assen": "Drenthe", "emmen": "Drenthe", "coevorden": "Drenthe",
    "hoogeveen": "Drenthe", "meppel": "Drenthe", "westerveld": "Drenthe",
    "de wolden": "Drenthe", "midden-drenthe": "Drenthe", "borger-odoorn": "Drenthe",
    "aa en hunze": "Drenthe", "noordenveld": "Drenthe", "tynaarlo": "Drenthe",
    "zwolle": "Overijssel", "deventer": "Overijssel", "almelo": "Overijssel",
    "enschede": "Overijssel", "hengelo": "Overijssel", "oldenzaal": "Overijssel",
    "losser": "Overijssel", "dinkelland": "Overijssel", "tubbergen": "Overijssel",
    "wierden": "Overijssel", "hellendoorn": "Overijssel", "twenterand": "Overijssel",
    "rijssen-holten": "Overijssel", "borne": "Overijssel", "haaksbergen": "Overijssel",
    "hof van twente": "Overijssel", "steenwijkerland": "Overijssel",
    "zwartewaterland": "Overijssel", "kampen": "Overijssel", "olst-wijhe": "Overijssel",
    "raalte": "Overijssel", "ommen": "Overijssel", "hardenberg": "Overijssel",
    "dalfsen": "Overijssel", "staphorst": "Overijssel",
    "arnhem": "Gelderland", "nijmegen": "Gelderland", "apeldoorn": "Gelderland",
    "zutphen": "Gelderland", "doetinchem": "Gelderland", "winterswijk": "Gelderland",
    "aalten": "Gelderland", "oost gelre": "Gelderland", "oude ijsselstreek": "Gelderland",
    "montferland": "Gelderland", "zevenaar": "Gelderland", "duiven": "Gelderland",
    "westervoort": "Gelderland", "rheden": "Gelderland", "renkum": "Gelderland",
    "wageningen": "Gelderland", "ede": "Gelderland", "barneveld": "Gelderland",
    "nijkerk": "Gelderland", "putten": "Gelderland", "ermelo": "Gelderland",
    "harderwijk": "Gelderland", "nunspeet": "Gelderland", "elburg": "Gelderland",
    "oldebroek": "Gelderland", "hattem": "Gelderland", "heerde": "Gelderland",
    "epe": "Gelderland", "brummen": "Gelderland", "voorst": "Gelderland",
    "bronckhorst": "Gelderland", "doesburg": "Gelderland", "overbetuwe": "Gelderland",
    "lingewaard": "Gelderland", "heumen": "Gelderland", "wijchen": "Gelderland",
    "maasdriel": "Gelderland", "zaltbommel": "Gelderland", "tiel": "Gelderland",
    "buren": "Gelderland", "culemborg": "Gelderland", "neder-betuwe": "Gelderland",
    "west betuwe": "Gelderland", "berkelland": "Gelderland", "lochem": "Gelderland",
    "utrecht": "Utrecht", "amersfoort": "Utrecht", "zeist": "Utrecht",
    "nieuwegein": "Utrecht", "houten": "Utrecht", "ijsselstein": "Utrecht",
    "lopik": "Utrecht", "montfoort": "Utrecht", "oudewater": "Utrecht",
    "woerden": "Utrecht", "de ronde venen": "Utrecht", "stichtse vecht": "Utrecht",
    "de bilt": "Utrecht", "bunnik": "Utrecht", "wijk bij duurstede": "Utrecht",
    "rhenen": "Utrecht", "veenendaal": "Utrecht", "utrechtse heuvelrug": "Utrecht",
    "soest": "Utrecht", "baarn": "Utrecht", "bunschoten": "Utrecht",
    "leusden": "Utrecht", "eemnes": "Utrecht",
    "amsterdam": "Noord-Holland", "haarlem": "Noord-Holland", "alkmaar": "Noord-Holland",
    "zaanstad": "Noord-Holland", "haarlemmermeer": "Noord-Holland", "purmerend": "Noord-Holland",
    "hoorn": "Noord-Holland", "enkhuizen": "Noord-Holland", "medemblik": "Noord-Holland",
    "hollands kroon": "Noord-Holland", "den helder": "Noord-Holland", "texel": "Noord-Holland",
    "schagen": "Noord-Holland", "bergen": "Noord-Holland", "heiloo": "Noord-Holland",
    "castricum": "Noord-Holland", "beverwijk": "Noord-Holland", "heemskerk": "Noord-Holland",
    "velsen": "Noord-Holland", "bloemendaal": "Noord-Holland", "zandvoort": "Noord-Holland",
    "heemstede": "Noord-Holland", "amstelveen": "Noord-Holland", "diemen": "Noord-Holland",
    "waterland": "Noord-Holland", "edam-volendam": "Noord-Holland",
    "blaricum": "Noord-Holland", "hilversum": "Noord-Holland", "huizen": "Noord-Holland",
    "laren": "Noord-Holland", "weesp": "Noord-Holland", "gooise meren": "Noord-Holland",
    "den haag": "Zuid-Holland", "'s-gravenhage": "Zuid-Holland", "rotterdam": "Zuid-Holland",
    "dordrecht": "Zuid-Holland", "leiden": "Zuid-Holland", "delft": "Zuid-Holland",
    "zoetermeer": "Zuid-Holland", "westland": "Zuid-Holland",
    "alphen aan den rijn": "Zuid-Holland", "gouda": "Zuid-Holland",
    "schiedam": "Zuid-Holland", "vlaardingen": "Zuid-Holland", "maassluis": "Zuid-Holland",
    "capelle aan den ijssel": "Zuid-Holland", "ridderkerk": "Zuid-Holland",
    "barendrecht": "Zuid-Holland", "lansingerland": "Zuid-Holland",
    "nissewaard": "Zuid-Holland", "goeree-overflakkee": "Zuid-Holland",
    "hoeksche waard": "Zuid-Holland", "leidschendam-voorburg": "Zuid-Holland",
    "pijnacker-nootdorp": "Zuid-Holland", "wassenaar": "Zuid-Holland",
    "rijswijk": "Zuid-Holland", "midden-delfland": "Zuid-Holland",
    "krimpenerwaard": "Zuid-Holland", "zuidplas": "Zuid-Holland",
    "waddinxveen": "Zuid-Holland", "bodegraven-reeuwijk": "Zuid-Holland",
    "gorinchem": "Zuid-Holland", "alblasserdam": "Zuid-Holland",
    "hendrik-ido-ambacht": "Zuid-Holland", "sliedrecht": "Zuid-Holland",
    "papendrecht": "Zuid-Holland", "zwijndrecht": "Zuid-Holland",
    "katwijk": "Zuid-Holland", "teylingen": "Zuid-Holland", "nieuwkoop": "Zuid-Holland",
    "molenlanden": "Zuid-Holland",
    "middelburg": "Zeeland", "vlissingen": "Zeeland", "goes": "Zeeland",
    "terneuzen": "Zeeland", "hulst": "Zeeland", "sluis": "Zeeland",
    "veere": "Zeeland", "schouwen-duiveland": "Zeeland", "kapelle": "Zeeland",
    "borsele": "Zeeland", "reimerswaal": "Zeeland", "noord-beveland": "Zeeland",
    "tilburg": "Noord-Brabant", "eindhoven": "Noord-Brabant", "breda": "Noord-Brabant",
    "'s-hertogenbosch": "Noord-Brabant", "helmond": "Noord-Brabant",
    "roosendaal": "Noord-Brabant", "oss": "Noord-Brabant",
    "bergen op zoom": "Noord-Brabant", "waalwijk": "Noord-Brabant",
    "meierijstad": "Noord-Brabant", "boxtel": "Noord-Brabant",
    "sint-michielsgestel": "Noord-Brabant", "bernheze": "Noord-Brabant",
    "gemert-bakel": "Noord-Brabant", "deurne": "Noord-Brabant",
    "geldrop-mierlo": "Noord-Brabant", "best": "Noord-Brabant",
    "veldhoven": "Noord-Brabant", "valkenswaard": "Noord-Brabant",
    "dongen": "Noord-Brabant", "gilze en rijen": "Noord-Brabant",
    "oisterwijk": "Noord-Brabant", "loon op zand": "Noord-Brabant",
    "heusden": "Noord-Brabant", "altena": "Noord-Brabant",
    "moerdijk": "Noord-Brabant", "halderberge": "Noord-Brabant",
    "rucphen": "Noord-Brabant", "zundert": "Noord-Brabant",
    "drimmelen": "Noord-Brabant", "geertruidenberg": "Noord-Brabant",
    "steenbergen": "Noord-Brabant", "woensdrecht": "Noord-Brabant",
    "land van cuijk": "Noord-Brabant",
    "maastricht": "Limburg", "venlo": "Limburg", "sittard-geleen": "Limburg",
    "heerlen": "Limburg", "roermond": "Limburg", "venray": "Limburg",
    "weert": "Limburg", "echt-susteren": "Limburg", "maasgouw": "Limburg",
    "roerdalen": "Limburg", "leudal": "Limburg", "nederweert": "Limburg",
    "peel en maas": "Limburg", "horst aan de maas": "Limburg", "beesel": "Limburg",
    "gennep": "Limburg", "valkenburg aan de geul": "Limburg",
    "eijsden-margraten": "Limburg", "gulpen-wittem": "Limburg", "vaals": "Limburg",
    "beekdaelen": "Limburg", "brunssum": "Limburg", "kerkrade": "Limburg",
    "landgraaf": "Limburg", "simpelveld": "Limburg", "voerendaal": "Limburg",
    "beek": "Limburg", "meerssen": "Limburg", "stein": "Limburg",
    "lelystad": "Flevoland", "almere": "Flevoland", "zeewolde": "Flevoland",
    "dronten": "Flevoland", "noordoostpolder": "Flevoland", "urk": "Flevoland",
}

# ===== PAGINA INSTELLINGEN =====
st.set_page_config(page_title="Laadpalen Nederland", page_icon="⚡", layout="wide")
st.title("⚡ Laadpalen Nederland")

# ===== DATA LADEN =====
@st.cache_data
def load_data():
    df = pd.read_csv('open_charge_map_NL.csv')
    df.columns = df.columns.str.strip()
    df['AddressInfo_Latitude'] = pd.to_numeric(df['AddressInfo_Latitude'], errors='coerce')
    df['AddressInfo_Longitude'] = pd.to_numeric(df['AddressInfo_Longitude'], errors='coerce')
    df = df.dropna(subset=['AddressInfo_Latitude', 'AddressInfo_Longitude'])

    df['title_norm']   = df['AddressInfo_Title'].astype(str).str.strip().str.lower()
    df['address_norm'] = df['AddressInfo_AddressLine1'].astype(str).str.strip().str.lower()
    df['town_norm']    = df['AddressInfo_Town'].astype(str).str.strip().str.lower()

    df = df.drop_duplicates(subset=['title_norm', 'address_norm', 'town_norm',
                                     'AddressInfo_Latitude', 'AddressInfo_Longitude'])

    df['Provincie'] = df['town_norm'].map(gemeente_provincie).fillna('Onbekend')
    return df

df = load_data()

# ===== SIDEBAR FILTERS =====
with st.sidebar:
    st.header("🔍 Filter")

    filter_type = st.radio("Bekijken per:", ["Alle locaties", "Provincie", "Gemeente"])

    filtered_df = df.copy()

    if filter_type == "Provincie":
        gekozen = st.selectbox("Kies een provincie:", sorted(df['Provincie'].unique()))
        filtered_df = df[df['Provincie'] == gekozen]

    elif filter_type == "Gemeente":
        gekozen = st.selectbox("Kies een gemeente:", sorted(df['AddressInfo_Town'].dropna().unique()))
        filtered_df = df[df['AddressInfo_Town'] == gekozen]

    st.metric("Laadpalen zichtbaar", len(filtered_df))

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
        get_fill_color='[34, 197, 94, 200]',  # groen
        get_radius=15,
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

    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        tooltip=tooltip,
        map_style="satellite",
        height=1200,
    ))