import requests
import pandas as pd
import time

# ===== INSTELLINGEN =====
API_KEY = "216e21a9-472f-4d17-9b9e-0c2b1e81053a"
COUNTRY = "NL"
MAX_RESULTS = 1000
BATCH_SIZE = 100
OUTPUT_FILE = "open_charge_map_NL.csv"

# ===== NUTTIGE KOLLOMS =====
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

print("Max datapunten:", MAX_RESULTS)

# ===== DATA OPHALEN =====
all_data = []
offset = 0

while True:

    if MAX_RESULTS is not None:
        remaining = MAX_RESULTS - len(all_data)
        if remaining <= 0:
            break
        limit = min(BATCH_SIZE, remaining)
    else:
        limit = BATCH_SIZE

    url = f"https://api.openchargemap.io/v3/poi/?output=json&countrycode={COUNTRY}&maxresults={limit}&offset={offset}&key={API_KEY}"

    print("Request:", url)

    while True:
        response = requests.get(url)

        if response.status_code == 429:
            print("Rate limit bereikt, wachten 2 minuten...")
            time.sleep(120)
            continue

        if response.status_code != 200:
            print("Fout:", response.status_code)
            break

        break

    if response.status_code != 200:
        break

    data = response.json()

    if len(data) == 0:
        break

    all_data.extend(data)
    print("Totaal opgehaald:", len(all_data))
    offset += limit

# ===== JSON FLATTEN =====
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

# ===== DATAFRAME MAKEN =====
df = pd.DataFrame(flattened_data)

# Alleen nuttige kollomen behouden
df = df[[col for col in useful_columns if col in df.columns]]

# Vervang lege waarden met "N/A"
df = df.astype(str).fillna("N/A")

print("Aantal kolommen:", len(df.columns))
print("Aantal rijen:", len(df))

# ===== CSV OPSLAAN =====
df.to_csv(OUTPUT_FILE, index=False)
print("CSV opgeslagen als:", OUTPUT_FILE)