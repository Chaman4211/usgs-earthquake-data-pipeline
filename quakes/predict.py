"""Predict earthquake magnitude and significant-earthquake class."""

import json

import joblib
import pandas as pd

from . import config


FEATURES = config.NUMERIC + config.NOMINAL


def _prepare_input(values):
    """Build one raw feature row for the saved sklearn pipeline."""
    row = {name: values.get(name) for name in FEATURES}

    # Derive engineered features from the supplied raw values.
    if row.get("lat") is not None:
        row["abs_lat"] = abs(float(row["lat"]))

    if row.get("depth_km") is not None:
        row["is_shallow"] = int(float(row["depth_km"]) < 70)

    row["nst_missing"] = int(row.get("nst") is None)

    return pd.DataFrame([row], columns=FEATURES)


def predict_magnitude(values):
    """Return the predicted earthquake magnitude."""
    model_path = config.ROOT / "models" / "regression_linear.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            "Model not found. Train it with: "
            "python -m quakes.train --task regression --model linear"
        )

    model = joblib.load(model_path)
    X = _prepare_input(values)
    return float(model.predict(X)[0])


def predict_big_quake(values):
    """Return predicted class, probability, and threshold."""
    model_path = config.ROOT / "models" / "classification_logistic.joblib"
    report_path = config.ROOT / "reports" / "classification_logistic.json"

    if not model_path.exists() or not report_path.exists():
        raise FileNotFoundError(
            "Train the classifier first with: "
            "python -m quakes.train --task classification --model logistic"
        )

    model = joblib.load(model_path)
    with report_path.open(encoding="utf-8") as file:
        report = json.load(file)

    threshold = report["model_test_metrics"].get("threshold", 0.5)
    X = _prepare_input(values)
    probability = float(model.predict_proba(X)[0, 1])
    predicted_class = int(probability >= threshold)

    return {
        "class": predicted_class,
        "label": "Significant" if predicted_class else "Not significant",
        "probability": probability,
        "threshold": threshold,
    }


def main():
    """Interactively enter feature values and display predictions."""
    print("EARTHQUAKE PREDICTION")
    print("Press Enter to use each example value.")
    print("These example values are for demonstration only.\n")

    numeric_defaults = {
        "lon": -118.0,
        "lat": 34.0,
        "depth_km": 10.0,
        "hour": 12,
        "dayofweek": 2,
        "update_lag_hours": 0.0,
        "is_reviewed": 0,
        "nst": 20,
        "dmin": 0.1,
        "rms": 0.5,
        "gap": 100.0,
    }

    values = {}
    for name, default in numeric_defaults.items():
        raw = input(f"{name} [{default}]: ").strip()
        try:
            values[name] = float(raw) if raw else default
        except ValueError:
            print(f"Please enter a number for {name}.")
            return

    for name, default in {
        "region": "California",
        "magType": "ml",
        "net": "ci",
    }.items():
        raw = input(f"{name} [{default}]: ").strip()
        values[name] = raw or default

    try:
        magnitude = predict_magnitude(values)
        classification = predict_big_quake(values)
    except (ValueError, KeyError, FileNotFoundError) as error:
        print(f"\nPrediction error: {error}")
        return

    print("\n===== PREDICTION RESULTS =====")
    print(f"Predicted magnitude: {magnitude:.3f}")
    print(f"Predicted class: {classification['label']}")
    print(
        "Probability of significant class: "
        f"{classification['probability']:.1%}"
    )
    print(f"Classification threshold: {classification['threshold']:.2f}")


if __name__ == "__main__":
    main()
