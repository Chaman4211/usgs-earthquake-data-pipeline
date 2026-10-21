"""Run the earthquake preprocessing pipeline end to end."""

import argparse
import json

import joblib
import pandas as pd

from quakes import cleaning, config, features
from quakes.fetch import fetch_feed, geojson_to_df
from quakes.transform import build_preprocessor, split_data


def load_batch_data() -> pd.DataFrame:
    """Fetch the live feed, or use the instructor's fallback file."""
    try:
        payload = fetch_feed("all_week")
        df = geojson_to_df(payload)
        print("Batch source: live USGS feed")
        return df
    except Exception as exc:
        print(f"Live feed unavailable: {exc}")

        if not config.FALLBACK_PATH.is_file():
            raise RuntimeError(
                "Cannot load earthquake data: the live feed failed and the "
                f"fallback file is missing: {config.FALLBACK_PATH}\n"
                "Restore the instructor-provided all_week.geojson file "
                "or check your network connection."
            ) from exc

        with config.FALLBACK_PATH.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        print(f"Batch source: fallback file ({config.FALLBACK_PATH})")
        return geojson_to_df(payload)


def load_stream_data() -> pd.DataFrame:
    """Load all available JSONL stream files."""
    frames = []

    for path in sorted(config.STREAM_DIR.glob("*.jsonl")):
        if path.stat().st_size == 0:
            continue

        frame = pd.read_json(path, lines=True)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True, sort=False)


def count_updated_events(df: pd.DataFrame) -> int:
    """Count event IDs with more than one distinct numeric update timestamp."""
    if not {"id", "updated"}.issubset(df.columns):
        return 0

    updates = df[["id", "updated"]].copy()
    numeric = pd.to_numeric(updates["updated"], errors="coerce")
    parsed = pd.to_datetime(numeric, unit="ms", utc=True, errors="coerce")

    non_numeric = numeric.isna()
    if non_numeric.any():
        parsed.loc[non_numeric] = pd.to_datetime(
            updates.loc[non_numeric, "updated"],
            utc=True,
            errors="coerce",
        )

    updates["_updated_parsed"] = parsed
    updates = updates.dropna(subset=["id", "_updated_parsed"])

    versions_per_event = updates.groupby("id")["_updated_parsed"].nunique()
    return int((versions_per_event > 1).sum())


def main() -> None:
    """Fetch, clean, engineer, split, preprocess, report, and save."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scaler",
        choices=["standard", "minmax", "robust"],
        default="robust",
        help="Scaler for numeric features (default: robust).",
    )
    args = parser.parse_args()

    batch_df = load_batch_data()
    raw_rows = len(batch_df)

    stream_df = load_stream_data()
    stream_rows = len(stream_df)

    frames = [batch_df]
    if not stream_df.empty:
        frames.append(stream_df)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    changed_events = count_updated_events(combined)

    # Keep the newest version of each event before cleaning.
    combined = cleaning.epoch_ms_to_datetime(combined)
    after_datetime = len(combined)

    combined = cleaning.dedupe_latest(combined)
    after_dedupe = len(combined)

    combined = cleaning.keep_earthquakes(combined)
    after_type_filter = len(combined)

    cleaned = cleaning.drop_missing_target(combined)
    after_target_filter = len(cleaned)

    if "place" in cleaned.columns:
        cleaned["region"] = cleaning.extract_region(cleaned["place"])
    else:
        cleaned["region"] = "Unknown"

    cleaned_rows = len(cleaned)

    engineered = features.add_time_features(cleaned)
    engineered = features.add_quality_features(engineered)
    engineered = features.add_location_features(engineered)

    if "region" in engineered.columns:
        engineered["region"] = cleaning.extract_region(
            engineered["place"]
        ) if "place" in engineered.columns else engineered["region"]
        engineered["region"] = features.group_rare(
            engineered["region"], top_k=15
        )

    # Create the target before removing magnitude-related leakage columns.
    engineered = features.add_target(engineered)
    target = engineered[config.TARGET].copy()
    model_df = features.drop_leaky_columns(engineered)
    model_df[config.TARGET] = target

    X_train, X_test, y_train, y_test = split_data(model_df)

    # Fit preprocessing on training data only.
    preprocessor = build_preprocessor(scaler=args.scaler)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Convert sparse one-hot output to dense arrays for CSV export.
    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()

    feature_names = preprocessor.get_feature_names_out()

    train_output = pd.DataFrame(
        X_train_processed, columns=feature_names, index=X_train.index
    )
    train_output[config.TARGET] = y_train

    test_output = pd.DataFrame(
        X_test_processed, columns=feature_names, index=X_test.index
    )
    test_output[config.TARGET] = y_test

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_output.to_csv(config.PROCESSED_DIR / "train.csv", index=False)
    test_output.to_csv(config.PROCESSED_DIR / "test.csv", index=False)
    joblib.dump(
        preprocessor, config.PROCESSED_DIR / "preprocessor.joblib"
    )

    print("\n===== EARTHQUAKE PREPROCESSING REPORT =====")
    print(f"Raw batch rows: {raw_rows}")
    print(f"Stream rows merged: {stream_rows}")
    print(f"Events with multiple updated versions: {changed_events}")
    print(f"Rows after datetime conversion: {after_datetime}")
    print(f"Rows after deduplication: {after_dedupe}")
    print(f"Rows after earthquake-type filter: {after_type_filter}")
    print(f"Rows after missing-target removal: {after_target_filter}")
    print(f"Rows after cleaning: {cleaned_rows}")
    print(f"Scaler: {args.scaler}")
    print(f"Training shape: {train_output.shape}")
    print(f"Testing shape: {test_output.shape}")
    print(f"Training positive-class rate: {y_train.mean():.4f}")
    print(f"Testing positive-class rate: {y_test.mean():.4f}")
    print(f"Saved files to: {config.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
