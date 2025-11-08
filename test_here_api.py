#!/usr/bin/env python3
"""Test script to verify HERE API connectivity and endpoint."""

import sys
import requests
from hereapi import fetch_congestion_factor, HereTrafficError

def test_connectivity():
    """Test basic connectivity to HERE API endpoints."""
    print("Testing connectivity to HERE API endpoints...")
    
    # Test DNS resolution
    endpoints = [
        "https://traffic.hereapi.com",
        "https://geocode.search.hereapi.com",
        "https://router.hereapi.com",
    ]
    
    for endpoint in endpoints:
        try:
            resp = requests.get(f"{endpoint}/", timeout=5)
            print(f"✓ {endpoint} - Reachable (status: {resp.status_code})")
        except requests.exceptions.ConnectionError as e:
            print(f"✗ {endpoint} - Connection failed: {e}")
        except requests.exceptions.Timeout:
            print(f"✗ {endpoint} - Timeout")
        except Exception as e:
            print(f"✗ {endpoint} - Error: {e}")

def test_traffic_api():
    """Test the traffic API with a known location."""
    print("\nTesting Traffic Flow API with a sample location...")
    print("Location: Berlin, Germany (52.5200, 13.4050)")
    
    try:
        result = fetch_congestion_factor(
            latitude=52.5200,
            longitude=13.4050,
            timeout_seconds=15
        )
        print("✓ API call successful!")
        print(f"  Jam Factor: {result.get('jamFactor')}")
        print(f"  Current Speed: {result.get('currentSpeedKmh')} km/h")
        print(f"  Free Flow Speed: {result.get('freeFlowSpeedKmh')} km/h")
        return True
    except HereTrafficError as e:
        print(f"✗ API Error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_connectivity()
    success = test_traffic_api()
    sys.exit(0 if success else 1)

