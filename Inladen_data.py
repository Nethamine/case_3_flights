import requests
import pandas as pd
import time


# ===== INSTELLINGEN =====

API_KEY = "216e21a9-472f-4d17-9b9e-0c2b1e81053a"
COUNTRY = "NL"
MAX_RESULTS = 10000
BATCH_SIZE = 100
OUTPUT_FILE = "open_charge_map_full.csv"

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


# ===== JSON NORMALIZEN =====

df = pd.json_normalize(all_data)

print("Aantal kolommen:", len(df.columns))
print("Aantal rijen:", len(df))


# ===== CONNECTIONS NORMALIZEN =====

connections = pd.json_normalize(
    all_data,
    record_path="Connections",
    meta=["ID"],
    errors="ignore"
)

connections.rename(columns={"ID": "ChargePointID"}, inplace=True)


# ===== DATA SAMENVOEGEN =====

df_final = df.merge(connections, left_on="ID", right_on="ChargePointID", how="left")


# ===== CSV OPSLAAN =====

df_final.to_csv(OUTPUT_FILE, index=False)

print("CSV opgeslagen als:", OUTPUT_FILE)