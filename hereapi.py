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


HERE_TRAFFIC_FLOW_V7 = "https://traffic.hereapi.com/v7/flow"
HARDCODED_API_KEY = "4bl8MFy38CUDbnx5KPtwdmbYxWW7xoP7trBt50q5Qyw"


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
    timeout_seconds: int = 10,
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
        "apiKey": key,
        "units": "metric",
        # spatial=reduced returns nearest segment summary
        "spatial": "reduced",
    }

    try:
        resp = requests.get(HERE_TRAFFIC_FLOW_V7, params=params, timeout=timeout_seconds)
    except requests.RequestException as exc:
        raise HereTrafficError(f"Network error calling HERE Traffic: {exc}") from exc

    if resp.status_code != 200:
        raise HereTrafficError(
            f"HERE Traffic error {resp.status_code}: {resp.text[:300]}"
        )

    data = resp.json()
    results = data.get("results", [])
    if not results:
        raise HereTrafficError("No traffic results returned for the given location.")

    # Choose the first result (closest/reduced)
    first = results[0]
    flow = first.get("flow", {})

    jam_factor = flow.get("jamFactor")
    current_speed = flow.get("currentSpeed")
    free_flow_speed = flow.get("freeFlowSpeed")
    speed_uncapped = flow.get("speedUncapped")
    confidence = flow.get("confidence")
    road_closure = flow.get("roadClosure")
    segment_id = first.get("id") or first.get("mapSegmentId")

    return {
        "jamFactor": jam_factor,
        "currentSpeedKmh": current_speed,
        "freeFlowSpeedKmh": free_flow_speed,
        "speedUncappedKmh": speed_uncapped,
        "confidence": confidence,
        "roadClosure": road_closure,
        "segmentId": segment_id,
        "raw": first,
        "fetchedAt": int(time.time()),
    }


__all__ = ["fetch_congestion_factor", "HereTrafficError"]


