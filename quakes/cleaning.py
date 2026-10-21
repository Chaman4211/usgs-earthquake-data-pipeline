"""Functions for cleaning earthquake data."""

import pandas as pd


def epoch_ms_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 'time' and 'updated' epoch milliseconds to UTC datetimes."""
    out = df.copy()

    for column in ("time", "updated"):
        if column in out.columns:
            out[column] = pd.to_datetime(
                out[column], unit="ms", utc=True, errors="coerce"
            )

    return out


def dedupe_latest(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per event ID, using the greatest updated value."""
    if "id" not in df.columns or "updated" not in df.columns:
        return df.copy()

    out = df.copy()
    out = out.sort_values("updated", kind="stable")
    out = out.drop_duplicates(subset="id", keep="last")
    return out.reset_index(drop=True)


def keep_earthquakes(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows whose normalized type is 'earthquake'."""
    out = df.copy()
    if "type" not in out.columns:
        return out.iloc[0:0].copy()

    normalized_type = out["type"].astype("string").str.strip().str.lower()
    return out.loc[normalized_type == "earthquake"].copy()


def extract_region(place: pd.Series) -> pd.Series:
    """Return text after the last comma, or the full place if no comma exists."""
    region = place.astype("string").str.split(",").str[-1].str.strip()
    return region.fillna("Unknown").replace("", "Unknown")


def iqr_outlier_mask(s: pd.Series, k: float = 1.5) -> pd.Series:
    """Mark values outside the interquartile range bounds."""
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr

    return ((s < lower_bound) | (s > upper_bound)).fillna(False)


def drop_missing_target(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where magnitude, the target, is missing."""
    return df.dropna(subset=["mag"]).copy()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the main cleaning steps and add a region column."""
    out = epoch_ms_to_datetime(df)
    out = dedupe_latest(out)
    out = keep_earthquakes(out)
    out = drop_missing_target(out)

    if "place" in out.columns:
        out["region"] = extract_region(out["place"])
    else:
        out["region"] = "Unknown"

    return out
