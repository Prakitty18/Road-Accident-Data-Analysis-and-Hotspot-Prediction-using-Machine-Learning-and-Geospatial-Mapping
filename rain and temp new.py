import ee
import pandas as pd
import os
from datetime import datetime, timedelta

# Authenticate and initialize Earth Engine
ee.Authenticate()
ee.Initialize(project='ee-bhupendrarulekiller14')

# Telangana bounding box geometry
telangana_geometry = ee.Geometry.Polygon([[
    [78.1982025944165, 17.55150404384812],
    [78.20438799735059, 17.42280495436174],
    [78.2534451288316, 17.31619470087244],
    [78.31341265855271, 17.23448116666578],
    [78.42851123168857, 17.17229007100023],
    [78.57892665881893, 17.20770260273578],
    [78.79273539895105, 17.28955966195588],
    [78.99088936429234, 17.35375392812317],
    [79.18144245065606, 17.46435525448978],
    [79.38011292664341, 17.55419332675298],
    [79.56251763116626, 17.64803177013101],
    [79.75810004708121, 17.73712579442823],
    [79.84534742600411, 17.84011053269573],
    [79.86678159042476, 17.9724250576151],
    [79.80895739458876, 18.14346862583921],
    [79.67784966809954, 18.20994000063095],
    [79.54978557291138, 18.2392501122425],
    [79.34074728175705, 18.20819394949277],
    [79.01713900704206, 18.08602415822451],
    [78.68668083868219, 17.93733982461329],
    [78.38825549185415, 17.75863482542706],
    [78.1982025944165, 17.55150404384812]
]])


# Dataset sources
datasets = {
    "rainfall": "ECMWF/ERA5_LAND/HOURLY",
    "temperature": "ECMWF/ERA5_LAND/HOURLY"
}

# Helper: fetch rainfall for a date
def fetch_rainfall(date):
    try:
        collection = ee.ImageCollection(datasets["rainfall"]) \
            .filterDate(date, (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")) \
            .select('total_precipitation')

        if collection.size().getInfo() == 0:
            print(f"[Rainfall Warning] No images found for {date}")
            return -9999

        daily_total_image = collection.sum()  # For precipitation: SUM the hourly values
        value = daily_total_image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=telangana_geometry,
            scale=5000,
            maxPixels=1e13
        ).get('total_precipitation')

        val_m = value.getInfo() if value else -9999
        return val_m * 1000 if val_m != -9999 else -9999  # convert from meters to mm
    except Exception as e:
        print(f"[Rainfall Error] {date}: {e}")
        return -9999

# Helper: fetch average temperature for a date
def fetch_temperature(date):
    try:
        start_date = date
        end_date = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        
        collection = ee.ImageCollection(datasets["temperature"]) \
            .filterDate(start_date, end_date) \
            .filterBounds(telangana_geometry)
        
        if collection.size().getInfo() == 0:
            return -9999, -9999

        max_temp_image = collection.max()
        min_temp_image = collection.min()

        max_value = max_temp_image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=telangana_geometry,
            scale=5000,
            maxPixels=1e13
        ).get("temperature_2m")

        min_value = min_temp_image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=telangana_geometry,
            scale=5000,
            maxPixels=1e13
        ).get("temperature_2m")

        max_temp_k = max_value.getInfo() if max_value else -9999
        min_temp_k = min_value.getInfo() if min_value else -9999

        max_temp_c = max_temp_k - 273.15 if max_temp_k != -9999 else -9999
        min_temp_c = min_temp_k - 273.15 if min_temp_k != -9999 else -9999

        return max_temp_c, min_temp_c

    except Exception as e:
        print(f"[Temperature Error] {date}: {e}")
        return -9999, -9999


# Generate all dates in 2022
start_date = datetime(2025, 1, 1)
end_date = datetime(2026, 1, 1)
dates_2022 = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end_date - start_date).days)]

# Loop over each date and fetch weather data
results = []
for date in dates_2022:
    print(f"Processing {date}...")
    rainfall = fetch_rainfall(date)
    max_temp, min_temp = fetch_temperature(date)
    results.append({
        "Date": date,
        "Rainfall_mm": rainfall,
        "Max_temp_celsius": max_temp,
        "Min_temp_celsius": min_temp,
    })

# Save to CSV
df = pd.DataFrame(results)

downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
file_path = os.path.join(downloads_path, "telangana_weather_202255_new.csv")
df.to_csv(file_path, index=False)
print(f"Saved to {file_path}")

print("Saved as telangana_weather_2024.csv")