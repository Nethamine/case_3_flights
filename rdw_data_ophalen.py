import requests
import pandas as pd

# ── CONFIG ──────────────────────────────────────────────
URL_VOERTUIGEN = "https://opendata.rdw.nl/resource/m9d7-ebf2.json"
URL_BRANDSTOF = "https://opendata.rdw.nl/resource/8ys7-d773.json"
HEADERS = {}
LIMIT = 1000
MAX_RECORDS = 1000
# ────────────────────────────────────────────────────────


def haal_data_op(url, select, max_records):
    """Haalt data op met paginering tot max_records."""
    all_data = []
    offset = 0

    while len(all_data) < max_records:
        params = {
            "$limit": LIMIT,
            "$offset": offset,
            "$select": select
        }
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        data = response.json()

        if not data:
            break

        all_data.extend(data)
        offset += LIMIT
        print(f"  {len(all_data)} records opgehaald...")

    return all_data[:max_records]


# ── STAP 1: Voertuigen ophalen ───────────────────────────
print("Voertuigen ophalen...")
voertuigen = haal_data_op(
    URL_VOERTUIGEN,
    select="kenteken,datum_tenaamstelling,merk,voertuigsoort,eerste_kleur",
    max_records=MAX_RECORDS
)
df_voertuigen = pd.DataFrame(voertuigen)
print(f"✓ {len(df_voertuigen)} voertuigen opgehaald.\n")


# ── STAP 2: Brandstof ophalen voor alle kentekens ────────
print("Brandstofdata ophalen...")
kentekens = df_voertuigen["kenteken"].tolist()

brandstof_data = []
for i in range(0, len(kentekens), LIMIT):
    batch = kentekens[i:i + LIMIT]

    # Bouw een WHERE-query: kenteken IN ('XX001', 'XX002', ...)
    kenteken_lijst = ",".join(f"'{k}'" for k in batch)
    params = {
        "$where": f"kenteken in({kenteken_lijst})",
        "$select": "kenteken,brandstof_omschrijving",
        "$limit": LIMIT
    }
    response = requests.get(URL_BRANDSTOF, headers=HEADERS, params=params)
    response.raise_for_status()
    data = response.json()
    brandstof_data.extend(data)
    print(f"  {len(brandstof_data)} brandstof records opgehaald...")

df_brandstof = pd.DataFrame(brandstof_data)
print(f"✓ {len(df_brandstof)} brandstof records opgehaald.\n")


# ── STAP 3: Samenvoegen ──────────────────────────────────
print("Datasets samenvoegen...")
df = df_voertuigen.merge(df_brandstof, on="kenteken", how="left")

# Datum omzetten
df["datum_tenaamstelling"] = pd.to_datetime(
    df["datum_tenaamstelling"], errors="coerce")

print(f"✓ Samengevoegd: {len(df)} rijen, kolommen: {df.columns.tolist()}\n")


# ── STAP 4: Opslaan ──────────────────────────────────────
df.to_csv("rdw_voertuigen.csv", index=False)
print("✓ CSV opgeslagen als 'rdw_voertuigen.csv'!")
