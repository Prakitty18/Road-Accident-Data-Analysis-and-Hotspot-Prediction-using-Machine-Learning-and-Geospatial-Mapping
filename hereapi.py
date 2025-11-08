"""
Lightweight HERE Traffic API client to fetch congestion (jam factor) for a
given latitude/longitude and time.

Notes
- Uses HERE Traffic v7 Flow endpoint for realtime data.
- Historical/past time retrieval requires HERE Traffic Analytics/Archive,
  which is not covered by the public v7 Flow endpoint. This module will
  raise a clear error if a non-realtime timestamp is requested.
"""

import os
import time
import datetime as dt
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Correct endpoint for HERE Traffic API v7
HERE_TRAFFIC_FLOW_V7 = "https://data.traffic.hereapi.com/v7/flow"
HARDCODED_API_KEY = "4bl8MFy38CUDbnx5KPtwdmbYxWW7xoP7trBt50q5Qyw"

# Create a session with retry strategy
_session = None


def _get_session():
    """Get or create a requests session with retry strategy."""
    global _session
    if _session is None:
        _session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        _session.mount("https://", adapter)
        _session.mount("http://", adapter)
    return _session


class HereTrafficError(Exception):
    pass


def _read_api_key(explicit_api_key: Optional[str]) -> str:
    if explicit_api_key and explicit_api_key.strip():
        return explicit_api_key.strip()
    env_key = os.environ.get("HERE_API_KEY", "").strip()
    if env_key:
        return env_key
    # Fallback to hardcoded key provided by user
    if HARDCODED_API_KEY:
        return HARDCODED_API_KEY
    raise HereTrafficError(
        "HERE_API_KEY is not set and no api_key was provided."
    )


def _is_near_now(ts: dt.datetime, tolerance_seconds: int = 300) -> bool:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    now = dt.datetime.now(dt.timezone.utc)
    return abs((now - ts).total_seconds()) <= tolerance_seconds


def fetch_congestion_factor(
    latitude: float,
    longitude: float,
    when: Optional[dt.datetime] = None,
    radius_meters: int = 250,
    api_key: Optional[str] = None,
    timeout_seconds: int = 15,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Fetch congestion for a point using HERE Traffic Flow v7.

    Returns a dict with keys:
      - jamFactor: float in [0, 10]
      - currentSpeedKmh, freeFlowSpeedKmh, speedUncappedKmh
      - confidence: float (0..1) when available
      - roadClosure: bool when available
      - segmentId: str (provider segment id)
      - raw: original minimal slice of response kept for reference

    Historical timestamps are not supported by the realtime Flow API. If
    'when' is provided and not close to now (±5 minutes), an error is raised.
    """
    if when is not None and not _is_near_now(when):
        raise HereTrafficError(
            "HERE Traffic v7 Flow provides realtime data only. For historical "
            "timestamps, enable Traffic Analytics/Archive and use its endpoints."
        )

    key = _read_api_key(api_key)

    params = {
        "in": f"circle:{latitude:.6f},{longitude:.6f};r={max(50, min(radius_meters, 2000))}",
        "locationReferencing": "shape",  # Required parameter per API docs
        "apiKey": key,
    }

    session = _get_session()
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            resp = session.get(
                HERE_TRAFFIC_FLOW_V7, 
                params=params, 
                timeout=timeout_seconds,
                headers={"User-Agent": "Python-HERE-Traffic-Client/1.0"}
            )
            
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                if not results:
                    raise HereTrafficError("No traffic results returned for the given location.")

                # Choose the first result (closest segment)
                first = results[0]
                current_flow = first.get("currentFlow", {})
                location = first.get("location", {})

                # Extract flow data - speeds are in m/s, convert to km/h
                speed_ms = current_flow.get("speed")
                speed_uncapped_ms = current_flow.get("speedUncapped")
                free_flow_ms = current_flow.get("freeFlow")
                
                # Convert m/s to km/h (multiply by 3.6)
                current_speed_kmh = speed_ms * 3.6 if speed_ms is not None else None
                speed_uncapped_kmh = speed_uncapped_ms * 3.6 if speed_uncapped_ms is not None else None
                free_flow_speed_kmh = free_flow_ms * 3.6 if free_flow_ms is not None else None

                jam_factor = current_flow.get("jamFactor")
                confidence = current_flow.get("confidence")
                traversability = current_flow.get("traversability", "open")
                road_closure = traversability != "open" if traversability else None
                
                # Get segment ID from location if available
                segment_id = location.get("id") or first.get("id")

                return {
                    "jamFactor": jam_factor,
                    "currentSpeedKmh": current_speed_kmh,
                    "freeFlowSpeedKmh": free_flow_speed_kmh,
                    "speedUncappedKmh": speed_uncapped_kmh,
                    "confidence": confidence,
                    "roadClosure": road_closure,
                    "traversability": traversability,
                    "segmentId": segment_id,
                    "raw": first,
                    "fetchedAt": int(time.time()),
                }
            elif resp.status_code == 429:
                # Rate limited - wait longer
                wait_time = (2 ** attempt) * 2  # Exponential backoff: 2, 4, 8 seconds
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                    continue
                raise HereTrafficError(
                    f"HERE Traffic rate limit exceeded (429). Please wait before retrying."
                )
            else:
                raise HereTrafficError(
                    f"HERE Traffic error {resp.status_code}: {resp.text[:300]}"
                )
                
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 1  # Exponential backoff: 1, 2, 4 seconds
                time.sleep(wait_time)
                continue
        except requests.RequestException as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 1
                time.sleep(wait_time)
                continue
    
    # If we get here, all retries failed
    raise HereTrafficError(
        f"Network error calling HERE Traffic after {max_retries} attempts: {last_exception}"
    ) from last_exception


__all__ = ["fetch_congestion_factor", "HereTrafficError"]


