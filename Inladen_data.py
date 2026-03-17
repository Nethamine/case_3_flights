import requests
import pandas as pd
import time

# ===== INSTELLINGEN =====
API_KEY       = "216e21a9-472f-4d17-9b9e-0c2b1e81053a"
COUNTRY       = "NL"
MAX_RESULTS   = 9999       # ← API heeft max ~8136 NL-locaties, 9999 = alles ophalen
OUTPUT_FILE   = "open_charge_map_NL.csv"

# ===== NUTTIGE KOLOMMEN =====
useful_columns = [
    "ID",
    "AddressInfo_Title",
    "AddressInfo_AddressLine1",
    "AddressInfo_Town",
    "AddressInfo_Postcode",
    "AddressInfo_Latitude",
    "AddressInfo_Longitude",
    "NumberOfPoints",
    "StatusType_IsOperational",
    "Connections_0_ConnectionType_FormalName",
    "Connections_0_PowerKW",
    "Connections_1_ConnectionType_FormalName",
    "Connections_1_PowerKW",
    "Connections_2_ConnectionType_FormalName",
    "Connections_2_PowerKW",
    "Connections_3_ConnectionType_FormalName",
    "Connections_3_PowerKW"
]

# ===== HULPFUNCTIES =====
def log(msg, indent=0):
    prefix = "  " * indent
    print(f"{prefix}{msg}")

# ===== START =====
print("=" * 50)
print("  Open Charge Map — Data ophalen")
print("=" * 50)
print()
print("-" * 50)
log(f"Land          : {COUNTRY}")
log(f"Max resultaten: {MAX_RESULTS}")
log(f"Output bestand: {OUTPUT_FILE}")
print("-" * 50)
print()

# ===== DATA OPHALEN =====
log("Stap 1/4 — Data ophalen van API...")
start_time = time.time()

url = (
    f"https://api.openchargemap.io/v3/poi/"
    f"?output=json&countrycode={COUNTRY}"
    f"&maxresults={MAX_RESULTS}&key={API_KEY}"
)

while True:
    try:
        response = requests.get(url, timeout=60)
    except requests.exceptions.RequestException as e:
        log(f"⚠  Verbindingsfout: {e}, opnieuw proberen in 10s...", indent=1)
        time.sleep(10)
        continue

    if response.status_code == 429:
        log("⚠  Rate limit bereikt, wachten 60 seconden...", indent=1)
        for i in range(60, 0, -5):
            print(f"\r  ⏳ Nog {i} seconden wachten...", end="", flush=True)
            time.sleep(5)
        print()
        continue

    if response.status_code != 200:
        log(f"✗  HTTP fout: {response.status_code}, opnieuw proberen in 10s...", indent=1)
        time.sleep(10)
        continue

    break

all_data = response.json()
log(f"✓  {len(all_data)} records opgehaald in {time.time() - start_time:.1f}s", indent=1)
print()

# ===== JSON FLATTEN =====
log("Stap 2/4 — JSON structuur platslaan...")
start = time.time()

def flatten_json(y, parent_key='', sep='_'):
    items = []
    if isinstance(y, dict):
        for k, v in y.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.extend(flatten_json(v, new_key, sep=sep).items())
    elif isinstance(y, list):
        for i, v in enumerate(y):
            new_key = f"{parent_key}{sep}{i}"
            items.extend(flatten_json(v, new_key, sep=sep).items())
    else:
        items.append((parent_key, y))
    return dict(items)

flattened_data = [flatten_json(item) for item in all_data]
log(f"✓  Platslaan klaar in {time.time() - start:.1f}s", indent=1)
print()

# ===== DATAFRAME MAKEN =====
log("Stap 3/4 — DataFrame samenstellen en opschonen...")
start = time.time()

df = pd.DataFrame(flattened_data)
log(f"Rijen voor deduplicatie : {len(df)}", indent=1)
log(f"Kolommen gevonden       : {len(df.columns)}", indent=1)

df = df[[col for col in useful_columns if col in df.columns]]
log(f"Kolommen na filter      : {len(df.columns)}", indent=1)

missing = [col for col in useful_columns if col not in df.columns]
if missing:
    log(f"⚠  Niet gevonden        : {', '.join(missing)}", indent=1)

df = df.astype(str).fillna("N/A")

before = len(df)
df = df.drop_duplicates(subset=["ID"])
log(f"Duplicaten verwijderd   : {before - len(df)}", indent=1)
log(f"✓  Unieke locaties      : {len(df)}", indent=1)
log(f"✓  DataFrame klaar in {time.time() - start:.1f}s", indent=1)
print()

# ===== CSV OPSLAAN =====
log("Stap 4/4 — CSV opslaan...")
start = time.time()
df.to_csv(OUTPUT_FILE, index=False)
log(f"✓  Opgeslagen als '{OUTPUT_FILE}' in {time.time() - start:.1f}s", indent=1)
print()

# ===== SAMENVATTING =====
print("=" * 50)
print("  Klaar!")
print("=" * 50)
log(f"Locaties opgeslagen : {len(df)}")
log(f"Kolommen            : {len(df.columns)}")
log(f"Output bestand      : {OUTPUT_FILE}")
log(f"Totale tijd         : {time.time() - start_time:.1f}s")
print("=" * 50)