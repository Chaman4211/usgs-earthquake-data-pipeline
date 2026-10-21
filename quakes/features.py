"""Feature engineering for earthquake classification."""

import pandas as pd

from quakes import config


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour and day of week using UTC event time."""
    out = df.copy()
    event_time = pd.to_datetime(out["time"], utc=True, errors="coerce")
    out["hour"] = event_time.dt.hour
    out["dayofweek"] = event_time.dt.dayofweek
    return out


def add_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add event update lag and data-quality indicators."""
    out = df.copy()

    event_time = pd.to_datetime(out["time"], utc=True, errors="coerce")
    updated_time = pd.to_datetime(out["updated"], utc=True, errors="coerce")
    out["update_lag_hours"] = (
        (updated_time - event_time).dt.total_seconds() / 3600
    )

    status = out["status"].astype("string").str.strip().str.lower()
    out["is_reviewed"] = (status == "reviewed").astype("int8")
    out["nst_missing"] = out["nst"].isna().astype("int8")

    return out


def add_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add absolute latitude and a shallow-event indicator."""
    out = df.copy()
    out["abs_lat"] = pd.to_numeric(out["lat"], errors="coerce").abs()

    depth = pd.to_numeric(out["depth_km"], errors="coerce")
    out["is_shallow"] = (depth < 70).astype("int8")

    return out


def group_rare(s: pd.Series, top_k: int = 15) -> pd.Series:
    """Keep the top_k most common values and group the rest as Other."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    counts = s.value_counts(dropna=True)
    common_values = set(counts.head(top_k).index)

    result = s.astype("object").copy()
    keep_mask = result.isin(common_values)
    result.loc[~keep_mask] = "Other"
    return result


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add big_quake: 1 when magnitude is at least 4.5, otherwise 0."""
    out = df.copy()
    out["big_quake"] = (pd.to_numeric(out["mag"], errors="coerce") >= 4.5).astype("int8")
    return out


def drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns configured as target-leaking."""
    return df.drop(columns=config.LEAKY, errors="ignore").copy()
