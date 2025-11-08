import argparse
import math
import sys
import time
from typing import Dict, Tuple, Any

import pandas as pd

from hereapi import fetch_congestion_factor, HereTrafficError


def _safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def enrich_with_congestion(
    df: pd.DataFrame,
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    sleep_seconds: float = 0.2,
) -> pd.DataFrame:
    """
    Enrich DataFrame with HERE Traffic API data for current time.
    Adds columns: jamFactor, currentSpeedKmh, freeFlowSpeedKmh, 
    speedUncappedKmh, confidence, roadClosure, segmentId
    """
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"Input DataFrame must contain '{lat_col}' and '{lon_col}' columns")

    cache: Dict[Tuple[float, float], Dict[str, Any]] = {}
    
    # Initialize lists for all columns
    jam_factors = []
    current_speeds = []
    free_flow_speeds = []
    speed_uncapped = []
    confidences = []
    road_closures = []
    segment_ids = []

    total_rows = len(df)
    print(f"Processing {total_rows} rows...")

    for idx, row in df.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{total_rows} rows...")

        lat = _safe_float(row.get(lat_col))
        lon = _safe_float(row.get(lon_col))

        if math.isnan(lat) or math.isnan(lon):
            jam_factors.append(math.nan)
            current_speeds.append(math.nan)
            free_flow_speeds.append(math.nan)
            speed_uncapped.append(math.nan)
            confidences.append(math.nan)
            road_closures.append(math.nan)
            segment_ids.append(None)
            continue

        key = (round(lat, 6), round(lon, 6))
        if key in cache:
            cached = cache[key]
            jam_factors.append(cached.get("jamFactor"))
            current_speeds.append(cached.get("currentSpeedKmh"))
            free_flow_speeds.append(cached.get("freeFlowSpeedKmh"))
            speed_uncapped.append(cached.get("speedUncappedKmh"))
            confidences.append(cached.get("confidence"))
            road_closures.append(cached.get("roadClosure"))
            segment_ids.append(cached.get("segmentId"))
            continue

        # Fetch from API (uses current time by default)
        try:
            result = fetch_congestion_factor(latitude=lat, longitude=lon)
            cached_result = {
                "jamFactor": result.get("jamFactor"),
                "currentSpeedKmh": result.get("currentSpeedKmh"),
                "freeFlowSpeedKmh": result.get("freeFlowSpeedKmh"),
                "speedUncappedKmh": result.get("speedUncappedKmh"),
                "confidence": result.get("confidence"),
                "roadClosure": result.get("roadClosure"),
                "segmentId": result.get("segmentId"),
            }
        except (HereTrafficError, Exception) as e:
            print(f"Warning: Failed to fetch data for ({lat}, {lon}): {e}")
            cached_result = {
                "jamFactor": math.nan,
                "currentSpeedKmh": math.nan,
                "freeFlowSpeedKmh": math.nan,
                "speedUncappedKmh": math.nan,
                "confidence": math.nan,
                "roadClosure": None,
                "segmentId": None,
            }

        cache[key] = cached_result
        jam_factors.append(cached_result.get("jamFactor"))
        current_speeds.append(cached_result.get("currentSpeedKmh"))
        free_flow_speeds.append(cached_result.get("freeFlowSpeedKmh"))
        speed_uncapped.append(cached_result.get("speedUncappedKmh"))
        confidences.append(cached_result.get("confidence"))
        road_closures.append(cached_result.get("roadClosure"))
        segment_ids.append(cached_result.get("segmentId"))

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    df = df.copy()
    df["jamFactor"] = jam_factors
    df["currentSpeedKmh"] = current_speeds
    df["freeFlowSpeedKmh"] = free_flow_speeds
    df["speedUncappedKmh"] = speed_uncapped
    df["confidence"] = confidences
    df["roadClosure"] = road_closures
    df["segmentId"] = segment_ids
    
    print(f"Completed! Added {len(cache)} unique location queries.")
    return df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add HERE Traffic API data (congestion, speed, etc.) to processed_accident_data.csv"
    )
    parser.add_argument(
        "--input",
        default="data/processed_accident_data.csv",
        help="Path to input CSV (default: data/processed_accident_data.csv)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output CSV (default: overwrites input file)",
    )
    parser.add_argument(
        "--lat-col",
        default="Latitude",
        help="Latitude column name (default: Latitude)",
    )
    parser.add_argument(
        "--lon-col",
        default="Longitude",
        help="Longitude column name (default: Longitude)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Sleep between API calls in seconds to avoid rate limits (default: 0.5)",
    )
    args = parser.parse_args()

    print(f"Loading CSV from: {args.input}")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows")
    
    enriched = enrich_with_congestion(
        df, lat_col=args.lat_col, lon_col=args.lon_col, sleep_seconds=args.sleep
    )
    
    output_path = args.output if args.output else args.input
    print(f"Saving enriched data to: {output_path}")
    enriched.to_csv(output_path, index=False)
    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())


