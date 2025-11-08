import argparse
import math
import sys
import time
from typing import Dict, Tuple

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
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"Input DataFrame must contain '{lat_col}' and '{lon_col}' columns")

    cache: Dict[Tuple[float, float], float] = {}
    jam_factors = []

    for _, row in df.iterrows():
        lat = _safe_float(row.get(lat_col))
        lon = _safe_float(row.get(lon_col))

        if math.isnan(lat) or math.isnan(lon):
            jam_factors.append(math.nan)
            continue

        key = (round(lat, 6), round(lon, 6))
        if key in cache:
            jam_factors.append(cache[key])
            continue

        try:
            result = fetch_congestion_factor(latitude=lat, longitude=lon)
            jf = result.get("jamFactor")
        except HereTrafficError:
            jf = math.nan
        except Exception:
            jf = math.nan

        cache[key] = jf
        jam_factors.append(jf)

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    df = df.copy()
    df["congestion_factor"] = jam_factors
    return df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add HERE Traffic congestion factor to processed_accident_data.csv"
    )
    parser.add_argument(
        "--input",
        default="data/processed_accident_data.csv",
        help="Path to input CSV (default: data/processed_accident_data.csv)",
    )
    parser.add_argument(
        "--output",
        default="data/processed_accident_data_with_congestion.csv",
        help="Path to output CSV (default: data/processed_accident_data_with_congestion.csv)",
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
        default=0.2,
        help="Sleep between API calls in seconds to avoid rate limits (default: 0.2)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    enriched = enrich_with_congestion(
        df, lat_col=args.lat_col, lon_col=args.lon_col, sleep_seconds=args.sleep
    )
    enriched.to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())


