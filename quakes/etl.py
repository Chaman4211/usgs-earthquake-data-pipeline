"""Extract, transform, and load earthquake data with logging."""

import argparse
import json
import logging
import sqlite3

import pandas as pd

from . import cleaning, config, features
from .fetch import fetch_feed, geojson_to_df

log = logging.getLogger("quakes.etl")

DB_PATH = config.PROCESSED_DIR / "quakes.db"
LOG_PATH = config.ROOT / "logs" / "etl.log"


def extract(feed: str = "all_week") -> pd.DataFrame:
    """Fetch the USGS feed, falling back to the supplied GeoJSON file."""
    try:
        payload = fetch_feed(feed)
        df = geojson_to_df(payload)
        log.info("Fetched live USGS feed %s: %d rows", feed, len(df))
        return df
    except Exception as exc:
        log.warning("Live feed failed: %s", exc)

    if not config.FALLBACK_PATH.is_file():
        raise FileNotFoundError(
            f"Fallback data file not found: {config.FALLBACK_PATH}"
        )

    with config.FALLBACK_PATH.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    df = geojson_to_df(payload)
    log.info("Loaded fallback data: %d rows", len(df))
    return df


def read_stream_files() -> pd.DataFrame:
    """Read and concatenate all non-empty JSONL stream files."""
    frames = []

    for path in sorted(config.STREAM_DIR.glob("*.jsonl")):
        if path.stat().st_size == 0:
            continue

        frame = pd.read_json(path, lines=True)
        if not frame.empty:
            frames.append(frame)
            log.info("Read %d stream rows from %s", len(frame), path.name)

    if not frames:
        log.info("No stream records found")
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True, sort=False)
    log.info("Total stream rows: %d", len(result))
    return result


def transform(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean data, engineer features, retain targets, and remove leakage."""
    log.info("Transform started: %d input rows", len(raw))

    # Reuse the Day 1 cleaning pipeline.
    out = cleaning.clean(raw)
    log.info("After cleaning: %d rows", len(out))

    out = features.add_time_features(out)
    out = features.add_quality_features(out)
    out = features.add_location_features(out)

    if "region" in out.columns:
        out["region"] = features.group_rare(out["region"], top_k=15)

    # Preserve both training targets before removing leakage columns.
    out = features.add_target(out)
    targets = out[["mag", config.TARGET]].copy()

    out = features.drop_leaky_columns(out)
    out["mag"] = targets["mag"]
    out[config.TARGET] = targets[config.TARGET]

    # SQLite cannot directly store timezone-aware datetime objects.
    for column in out.columns:
        if isinstance(out[column].dtype, pd.DatetimeTZDtype):
            out[column] = out[column].astype("string")
        elif pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = out[column].astype("string")

    log.info("After feature engineering: %d rows, %d columns",
             len(out), len(out.columns))
    return out


def load_sqlite(df: pd.DataFrame, db_path, table: str = "quakes") -> None:
    """Insert rows into SQLite, replacing existing records with matching IDs."""
    if df.empty:
        log.warning("No rows to load")
        return

    if "id" not in df.columns:
        raise ValueError("DataFrame must contain an 'id' column")

    if not table.replace("_", "").isalnum():
        raise ValueError(f"Invalid SQLite table name: {table!r}")

    out = df.copy()

    # Convert datetime columns to text before SQLite insertion.
    for column in out.columns:
        if isinstance(out[column].dtype, pd.DatetimeTZDtype):
            out[column] = out[column].astype("string")
        elif pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = out[column].astype("string")

    db_path = str(db_path)
    with sqlite3.connect(db_path) as connection:
        # Create the table if necessary.
        out.iloc[0:0].to_sql(table, connection, if_exists="append", index=False)

        ids = out["id"].dropna().unique().tolist()
        if ids:
            placeholders = ",".join("?" for _ in ids)
            connection.execute(
                f'DELETE FROM "{table}" WHERE id IN ({placeholders})',
                ids,
            )

        out.to_sql(table, connection, if_exists="append", index=False)

    log.info("Loaded %d rows into %s", len(out), db_path)


def run_etl(mode: str = "batch") -> None:
    """Extract, optionally merge stream records, transform, and load."""
    if mode not in {"batch", "merge"}:
        raise ValueError("mode must be 'batch' or 'merge'")

    raw = extract()
    log.info("Extract stage: %d rows", len(raw))

    if mode == "merge":
        stream = read_stream_files()
        if not stream.empty:
            raw = pd.concat([raw, stream], ignore_index=True, sort=False)
        log.info("After stream merge: %d rows", len(raw))

    transformed = transform(raw)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    load_sqlite(transformed, DB_PATH)

    log.info("ETL completed successfully: %d transformed rows", len(transformed))


def main() -> None:
    """Configure file and console logging, then run the selected ETL mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["batch", "merge"],
        default="batch",
        help="batch fetches the feed; merge also includes saved stream files",
    )
    args = parser.parse_args()

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )

    run_etl(mode=args.mode)


if __name__ == "__main__":
    main()
