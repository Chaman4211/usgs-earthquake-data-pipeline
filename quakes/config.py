"""Shared paths and machine-learning configuration."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FALLBACK_PATH = ROOT / "data" / "fallback" / "all_week.geojson"
STREAM_DIR = ROOT / "data" / "stream"
PROCESSED_DIR = ROOT / "data" / "processed"

SEED = 42
TARGET = "big_quake"

# Numeric predictors known or derived from information available at prediction time.
NUMERIC: list[str] = [
    "lon",
    "lat",
    "depth_km",
    "hour",
    "dayofweek",
    "update_lag_hours",
    "is_reviewed",
    "nst_missing",
    "abs_lat",
    "is_shallow",
    "nst",
    "dmin",
    "rms",
    "gap",
]

# Categorical predictors.
NOMINAL: list[str] = [
    "region",
    "magType",
    "net",
]

# Remove magnitude-derived fields and impact/report fields that could leak
# magnitude information. Keep this list aligned with the lab handout.
LEAKY: list[str] = [
    "mag",
    "big_quake",
    "title",
    "sig",
    "mmi",
    "cdi",
    "felt",
    "alert",
    "tsunami",
]
