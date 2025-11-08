import ee
import pandas as pd
import os
from datetime import datetime, timedelta
import calendar
import time

# Authenticate and initialize Earth Engine
try:
    ee.Initialize(project='ee-bhupendrarulekiller14')
    print("Earth Engine initialized successfully")
except Exception as e:
    print(f"Earth Engine initialization: {e}")
    print("Attempting to initialize without project...")
    try:
        ee.Initialize()
        print("Earth Engine initialized successfully")
    except Exception as e2:
        print(f"Failed to initialize Earth Engine: {e2}")
        print("Please run: ee.Authenticate() first")
        raise

# Dataset sources
datasets = {
    "rainfall": "ECMWF/ERA5_LAND/HOURLY",
    "temperature": "ECMWF/ERA5_LAND/HOURLY"
}

# Month name to number mapping
MONTH_MAP = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

# Helper: fetch rainfall for a specific point and date
def fetch_rainfall_point(latitude, longitude, date):
    """
    Fetch rainfall for a specific latitude/longitude point on a given date.
    Returns rainfall in mm.
    """
    try:
        point = ee.Geometry.Point([longitude, latitude])
        
        collection = ee.ImageCollection(datasets["rainfall"]) \
            .filterDate(date, (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")) \
            .select('total_precipitation')

        if collection.size().getInfo() == 0:
            print(f"[Rainfall Warning] No images found for {date} at ({latitude}, {longitude})")
            return -9999.0

        daily_total_image = collection.sum()  # For precipitation: SUM the hourly values
        value = daily_total_image.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=11132,  # ~1 degree at equator, good for point extraction
            maxPixels=1e13
        ).get('total_precipitation')

        val_m = value.getInfo() if value else None
        if val_m is None or val_m == -9999:
            return -9999.0
        return float(val_m * 1000)  # convert from meters to mm
    except Exception as e:
        print(f"[Rainfall Error] {date} at ({latitude}, {longitude}): {e}")
        return -9999.0

# Helper: fetch temperature for a specific point and date
def fetch_temperature_point(latitude, longitude, date):
    """
    Fetch temperature for a specific latitude/longitude point on a given date.
    Returns (max_temp_c, min_temp_c).
    """
    try:
        point = ee.Geometry.Point([longitude, latitude])
        start_date = date
        end_date = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        
        collection = ee.ImageCollection(datasets["temperature"]) \
            .filterDate(start_date, end_date) \
            .select('temperature_2m')
        
        if collection.size().getInfo() == 0:
            print(f"[Temperature Warning] No images found for {date} at ({latitude}, {longitude})")
            return -9999.0, -9999.0

        max_temp_image = collection.max()
        min_temp_image = collection.min()

        max_value = max_temp_image.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=11132,
            maxPixels=1e13
        ).get("temperature_2m")

        min_value = min_temp_image.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=11132,
            maxPixels=1e13
        ).get("temperature_2m")

        max_temp_k = max_value.getInfo() if max_value else None
        min_temp_k = min_value.getInfo() if min_value else None

        if max_temp_k is None or max_temp_k == -9999:
            max_temp_c = -9999.0
        else:
            max_temp_c = float(max_temp_k - 273.15)
            
        if min_temp_k is None or min_temp_k == -9999:
            min_temp_c = -9999.0
        else:
            min_temp_c = float(min_temp_k - 273.15)

        return max_temp_c, min_temp_c

    except Exception as e:
        print(f"[Temperature Error] {date} at ({latitude}, {longitude}): {e}")
        return -9999.0, -9999.0


def construct_date(year, month_name):
    """
    Construct a date string from year and month name.
    Uses the 15th of the month as a representative date.
    """
    month_num = MONTH_MAP.get(month_name, 1)
    # Use 15th of the month as representative date
    day = 15
    # Check if the date is valid (e.g., Feb 30 doesn't exist)
    try:
        date_obj = datetime(int(year), month_num, day)
        return date_obj.strftime("%Y-%m-%d")
    except ValueError:
        # If day doesn't exist (e.g., Feb 30), use last day of month
        last_day = calendar.monthrange(int(year), month_num)[1]
        date_obj = datetime(int(year), month_num, last_day)
        return date_obj.strftime("%Y-%m-%d")


def add_weather_data_to_accidents(input_csv, output_csv=None):
    """
    Read accident data CSV and add temperature and rainfall columns.
    """
    print(f"Reading accident data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    print(f"Total rows: {len(df)}")
    
    # Initialize new columns as float to avoid dtype warnings
    df['Rainfall_mm'] = -9999.0
    df['Max_temp_celsius'] = -9999.0
    df['Min_temp_celsius'] = -9999.0
    
    # Process each row
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"Processing row {idx + 1}/{len(df)}...")
            # Save progress periodically
            if output_csv and idx > 0:
                df.to_csv(output_csv, index=False)
                print(f"  Progress saved to {output_csv}")
        
        try:
            # Extract data
            year = row['Year']
            month = row['Month']
            lat = row['Latitude']
            lon = row['Longitude']
            
            # Skip if lat/lon are invalid
            if pd.isna(lat) or pd.isna(lon):
                print(f"Row {idx + 1}: Missing latitude/longitude, skipping...")
                continue
            
            # Construct date
            date_str = construct_date(year, month)
            
            # Fetch weather data
            rainfall = fetch_rainfall_point(lat, lon, date_str)
            max_temp, min_temp = fetch_temperature_point(lat, lon, date_str)
            
            # Update dataframe
            df.at[idx, 'Rainfall_mm'] = rainfall
            df.at[idx, 'Max_temp_celsius'] = max_temp
            df.at[idx, 'Min_temp_celsius'] = min_temp
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error processing row {idx + 1}: {e}")
            continue
    
    # Save to output file
    if output_csv is None:
        output_csv = input_csv.replace('.csv', '_with_weather.csv')
    
    print(f"\nSaving results to {output_csv}...")
    df.to_csv(output_csv, index=False)
    print(f"Done! Saved {len(df)} rows with weather data.")
    
    return df


if __name__ == "__main__":
    # Path to the processed accident data
    input_file = os.path.join(
        os.path.dirname(__file__),
        "data",
        "processed_accident_data.csv"
    )
    
    # Output file (will be created in same directory)
    output_file = os.path.join(
        os.path.dirname(__file__),
        "data",
        "processed_accident_data_with_weather.csv"
    )
    
    add_weather_data_to_accidents(input_file, output_file)

