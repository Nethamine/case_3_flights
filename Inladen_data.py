import requests
import pandas as pd


print("===== INSTELLINGEN =====")

API_KEY = "216e21a9-472f-4d17-9b9e-0c2b1e81053a"
COUNTRY = "NL"
MAX_RESULTS = None
BATCH_SIZE = 100
OUTPUT_FILE = "open_charge_map_NL.csv"

print("Max datapunten:", MAX_RESULTS)


print("===== DATA OPHALEN =====")

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

    url = f"https://api.openchargemap.io/v3/poi/?output=json&countrycode={COUNTRY}&maxresults={limit}&offset={offset}&compact=true&verbose=false&key={API_KEY}"

    print("Request:", url)

    response = requests.get(url)

    if response.status_code != 200:
        print("Fout:", response.status_code)
        break

    data = response.json()

    if len(data) == 0:
        break

    all_data.extend(data)

    print("Totaal opgehaald:", len(all_data))

    offset += limit


print("===== DATA STRUCTUREREN =====")

records = []

for station in all_data:

    record = {
        "ID": station.get("ID"),
        "Name": station.get("AddressInfo", {}).get("Title"),
        "Address": station.get("AddressInfo", {}).get("AddressLine1"),
        "City": station.get("AddressInfo", {}).get("Town"),
        "Latitude": station.get("AddressInfo", {}).get("Latitude"),
        "Longitude": station.get("AddressInfo", {}).get("Longitude"),
        "NumberOfPoints": station.get("NumberOfPoints"),
        "Operator": station.get("OperatorInfo", {}).get("Title")
    }

    records.append(record)


print("===== DATAFRAME MAKEN =====")

df = pd.DataFrame(records)

print(df.head())
print("Totaal stations:", len(df))


print("===== CSV OPSLAAN =====")

df.to_csv(OUTPUT_FILE, index=False)

print("CSV opgeslagen als:", OUTPUT_FILE)