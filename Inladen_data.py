import requests
import pandas as pd
import time

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

def log(msg, indent=0):
    prefix = "  " * indent
    print(f"{prefix}{msg}")

def progress_bar(current, total, width=30):
    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    pct = current / total * 100
    return f"[{bar}] {pct:.1f}% ({current}/{total})"

# ===== START =====
print("=" * 50)
print("  Open Charge Map — Data ophalen")
print("=" * 50)
print()

# ===== INSTELLINGEN =====
API_KEY = "216e21a9-472f-4d17-9b9e-0c2b1e81053a"
COUNTRY = "NL"
BATCH_SIZE = 10000
OUTPUT_FILE = "open_charge_map_NL.csv"

# ===== MAX RESULTS INSTELLEN =====
print("Hoeveel locaties wil je ophalen?")
print("  - Voer een getal in (bijv. 500)")
print("  - Of druk op Enter voor alle beschikbare data")
print()

raw = input("Max resultaten: ").strip()

if raw == "":
    MAX_RESULTS = None
    print("→ Alle beschikbare data wordt opgehaald.")
else:
    try:
        MAX_RESULTS = int(raw)
        if MAX_RESULTS <= 0:
            raise ValueError
        print(f"→ Max {MAX_RESULTS} locaties worden opgehaald.")
    except ValueError:
        print("⚠  Ongeldige invoer, standaard 10.000 wordt gebruikt.")
        MAX_RESULTS = 10000

print()
print("-" * 50)
log(f"Land          : {COUNTRY}")
log(f"Max resultaten: {MAX_RESULTS if MAX_RESULTS else 'Alles'}")
log(f"Batch grootte : {BATCH_SIZE}")
log(f"Output bestand: {OUTPUT_FILE}")
print("-" * 50)
print()

# ===== DATA OPHALEN =====
all_data = []
seen_ids = set()
offset = 0
batch_num = 0
start_time = time.time()

log("Stap 1/4 — Data ophalen van API...")
print()

while True:
    if MAX_RESULTS is not None:
        remaining = MAX_RESULTS - len(all_data)
        if remaining <= 0:
            break
        limit = min(BATCH_SIZE, remaining)
    else:
        limit = BATCH_SIZE

    batch_num += 1
    url = f"https://api.openchargemap.io/v3/poi/?output=json&countrycode={COUNTRY}&maxresults={limit}&offset={offset}&key={API_KEY}"

    log(f"Batch {batch_num} — offset {offset}, ophalen {limit} records...", indent=1)

    retries = 0
    while True:
        response = requests.get(url)

        if response.status_code == 429:
            retries += 1
            log(f"⚠  Rate limit bereikt (poging {retries}), wachten 2 minuten...", indent=2)
            for i in range(120, 0, -10):
                print(f"\r  ⏳ Nog {i} seconden wachten...", end="", flush=True)
                time.sleep(10)
            print()
            continue

        if response.status_code != 200:
            log(f"✗  HTTP fout: {response.status_code}", indent=2)
            break

        break

    if response.status_code != 200:
        break

    data = response.json()

    if len(data) == 0:
        log("✓  Geen data meer teruggekeerd, klaar met ophalen.", indent=1)
        break

    new_items = 0
    for item in data:
        item_id = item.get("ID")
        if item_id not in seen_ids:
            seen_ids.add(item_id)
            all_data.append(item)
            new_items += 1

    dupes = len(data) - new_items
    if MAX_RESULTS:
        log(f"✓  {new_items} nieuw  |  {dupes} duplicaat  |  {progress_bar(len(all_data), MAX_RESULTS)}", indent=2)
    else:
        log(f"✓  {new_items} nieuw  |  {dupes} duplicaat  |  Totaal: {len(all_data)}", indent=2)

    if new_items == 0:
        log("⚠  Hele batch was duplicaat, stoppen.", indent=1)
        break

    offset += limit

elapsed = time.time() - start_time
print()
print("-" * 50)
log(f"✓  Ophalen klaar in {elapsed:.1f}s — {len(all_data)} unieke locaties")
print("-" * 50)
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

flattened_data = []
total = len(all_data)
for i, item in enumerate(all_data):
    flattened_data.append(flatten_json(item))
    if (i + 1) % 500 == 0 or (i + 1) == total:
        print(f"\r  {progress_bar(i + 1, total)}", end="", flush=True)

print()
log(f"✓  Platslaan klaar in {time.time() - start:.1f}s", indent=1)
print()

# ===== DATAFRAME MAKEN =====
log("Stap 3/4 — DataFrame samenstellen en opschonen...")
start = time.time()

df = pd.DataFrame(flattened_data)
log(f"Kolommen gevonden : {len(df.columns)}", indent=1)

df = df[[col for col in useful_columns if col in df.columns]]
log(f"Kolommen na filter: {len(df.columns)}", indent=1)

missing = [col for col in useful_columns if col not in df.columns]
if missing:
    log(f"⚠  Niet gevonden  : {', '.join(missing)}", indent=1)

df = df.astype(str).fillna("N/A")

before = len(df)
df = df.drop_duplicates(subset=["ID"])
after = len(df)
if before != after:
    log(f"⚠  Extra check: {before - after} duplicaten verwijderd", indent=1)
else:
    log("✓  Geen duplicaten gevonden in DataFrame", indent=1)

log(f"✓  DataFrame klaar in {time.time() - start:.1f}s — {len(df)} rijen, {len(df.columns)} kolommen", indent=1)
print()

# ===== CSV OPSLAAN =====
log("Stap 4/4 — CSV opslaan...")
start = time.time()
df.to_csv(OUTPUT_FILE, index=False)
log(f"✓  Opgeslagen als '{OUTPUT_FILE}' in {time.time() - start:.1f}s", indent=1)
print()

# ===== SAMENVATTING =====
total_time = time.time() - start_time
print("=" * 50)
print("  Klaar!")
print("=" * 50)
log(f"Locaties opgeslagen : {len(df)}")
log(f"Kolommen            : {len(df.columns)}")
log(f"Output bestand      : {OUTPUT_FILE}")
log(f"Totale tijd         : {total_time:.1f}s")
print("=" * 50)