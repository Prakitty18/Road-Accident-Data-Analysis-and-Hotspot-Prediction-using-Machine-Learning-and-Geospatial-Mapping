"""Enrich accident records with rainfall and temperature derived from ERA5.

This script reuses the helper logic from `rain and temp new.py`, but fetches
weather data for each accident location (latitude/longitude) and month from the
`processed_accident_data.csv` file. The enriched dataframe is written to a new
CSV file alongside the original dataset.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import ee
import pandas as pd


# Dataset sources (ERA5 hourly)
DATASETS = {
    "rainfall": "ECMWF/ERA5_LAND/HOURLY",
    "temperature": "ECMWF/ERA5_LAND/HOURLY",
}


@dataclass(frozen=True)
class WeatherKey:
    """Cache key for repeated weather lookups."""

    year: int
    month: int
    latitude: float
    longitude: float


def month_name_to_number(month_name: str) -> int:
    """Convert textual month (e.g. 'January') to its numeric index."""

    try:
        return datetime.strptime(month_name.strip(), "%B").month
    except ValueError as exc:
        raise ValueError(f"Invalid month value: {month_name!r}") from exc


def month_date_range(year: int, month: int) -> Tuple[str, str]:
    """Return ISO start (inclusive) and end (exclusive) dates for a month."""

    start = datetime(year=year, month=month, day=1)
    if month == 12:
        end = datetime(year=year + 1, month=1, day=1)
    else:
        end = datetime(year=year, month=month + 1, day=1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def fetch_rainfall_mm(point: ee.Geometry, start_date: str, end_date: str) -> float:
    """Fetch total rainfall (mm) for the geometry and date range."""

    collection = (
        ee.ImageCollection(DATASETS["rainfall"])
        .filterDate(start_date, end_date)
        .filterBounds(point)
        .select("total_precipitation")
    )

    if collection.size().getInfo() == 0:
        return float("nan")

    # Sum hourly precipitation (meters) across the month and convert to mm.
    monthly_total = collection.sum()
    value = monthly_total.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point.buffer(5000),
        scale=5000,
        maxPixels=int(1e13),
    ).get("total_precipitation")

    if value is None:
        return float("nan")

    precipitation_m = value.getInfo()
    return precipitation_m * 1000 if precipitation_m is not None else float("nan")


def fetch_temperature_c(
    point: ee.Geometry, start_date: str, end_date: str
) -> Tuple[float, float]:
    """Fetch maximum and minimum 2m air temperature (°C) for the period."""

    collection = (
        ee.ImageCollection(DATASETS["temperature"])
        .filterDate(start_date, end_date)
        .filterBounds(point)
        .select("temperature_2m")
    )

    if collection.size().getInfo() == 0:
        return float("nan"), float("nan")

    max_image = collection.max()
    min_image = collection.min()

    def reduce_temperature(image: ee.Image) -> float:
        value = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point.buffer(5000),
            scale=5000,
            maxPixels=int(1e13),
        ).get("temperature_2m")
        if value is None:
            return float("nan")
        temp_k = value.getInfo()
        if temp_k is None:
            return float("nan")
        return temp_k - 273.15

    return reduce_temperature(max_image), reduce_temperature(min_image)


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add rainfall and temperature columns derived from Earth Engine."""

    required_columns = {"Year", "Month", "Latitude", "Longitude"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            "Input dataframe missing required columns: " + ", ".join(sorted(missing))
        )

    cache: Dict[WeatherKey, Tuple[float, float, float]] = {}
    rainfall_values = []
    max_temps = []
    min_temps = []

    for idx, row in df.iterrows():
        year = int(row["Year"])
        month = month_name_to_number(str(row["Month"]))
        lat = float(row["Latitude"])
        lon = float(row["Longitude"])

        key = WeatherKey(year=year, month=month, latitude=lat, longitude=lon)
        if key not in cache:
            start_date, end_date = month_date_range(year, month)
            point = ee.Geometry.Point([lon, lat])
            rainfall = fetch_rainfall_mm(point, start_date, end_date)
            max_temp, min_temp = fetch_temperature_c(point, start_date, end_date)
            cache[key] = (rainfall, max_temp, min_temp)

        rainfall, max_temp, min_temp = cache[key]
        rainfall_values.append(rainfall)
        max_temps.append(max_temp)
        min_temps.append(min_temp)

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} / {len(df)} records")

    df = df.copy()
    df["Monthly_Rainfall_mm"] = rainfall_values
    df["Monthly_Max_temp_celsius"] = max_temps
    df["Monthly_Min_temp_celsius"] = min_temps
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "/home/bhupi/prakriti/btp/"
            "Road-Accident-Data-Analysis-and-Hotspot-Prediction-using-Machine-Learning-and-Geospatial-Mapping/"
            "data/processed_accident_data.csv"
        ),
        help="Path to the accident CSV to enrich",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/bhupi/prakriti/btp/"
            "Road-Accident-Data-Analysis-and-Hotspot-Prediction-using-Machine-Learning-and-Geospatial-Mapping/"
            "data/processed_accident_data_with_weather.csv"
        ),
        help="Path where the enriched CSV will be written",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Authenticating with Earth Engine …")
    ee.Authenticate()
    ee.Initialize(project="ee-bhupendrarulekiller14")

    print(f"Loading accident data from {args.input}")
    df = pd.read_csv(args.input)

    enriched_df = enrich_dataframe(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    enriched_df.to_csv(args.output, index=False)
    print(f"Enriched dataset saved to {args.output}")


if __name__ == "__main__":
    main()


