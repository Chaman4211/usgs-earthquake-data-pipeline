"""Fetch the USGS GeoJSON earthquake feed and convert it to a DataFrame."""

import pandas as pd
import requests

BASE = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary"


def fetch_feed(feed: str = "all_week") -> dict:
    """Fetch a USGS GeoJSON feed and return its parsed dictionary."""
    url = f"{BASE}/{feed}.geojson"

    response = requests.get(url, timeout=15)
    response.raise_for_status()

    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError(
            f"The USGS response for feed '{feed}' is not valid JSON. "
            "Check that the feed name is correct."
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(f"The USGS feed '{feed}' did not return a JSON object.")

    return payload


def geojson_to_df(payload: dict) -> pd.DataFrame:
    """Convert GeoJSON features into one row per earthquake event."""
    rows = []

    for feature in payload.get("features", []):
        row = dict(feature.get("properties") or {})

        # Use the event ID from the top-level GeoJSON feature.
        row["id"] = feature.get("id")

        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates") or []

        # GeoJSON coordinates are [longitude, latitude, depth in km].
        row["lon"] = coordinates[0] if len(coordinates) > 0 else None
        row["lat"] = coordinates[1] if len(coordinates) > 1 else None
        row["depth_km"] = coordinates[2] if len(coordinates) > 2 else None

        rows.append(row)

    return pd.DataFrame(rows)
