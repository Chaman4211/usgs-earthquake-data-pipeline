"""Train, evaluate, and save earthquake prediction models."""

import argparse
import json
import sqlite3

import joblib
import pandas as pd
from sklearn.model_selection import cross_val_predict, train_test_split

from . import config
from .evaluate import (
    best_threshold,
    classification_report_dict,
    cross_validate_model,
    regression_report,
)
from .models import make_classifier, make_regressor


def load_data(db_path=None):
    """Load ETL output from SQLite."""
    path = db_path or (config.PROCESSED_DIR / "quakes.db")
    with sqlite3.connect(path) as conn:
        df = pd.read_sql_query("SELECT * FROM quakes", conn)
    if df.empty:
        raise ValueError(f"No training data found in {path}")
    return df


def main() -> None:
    """Train a baseline and selected model, evaluate, and save results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=["regression", "classification"],
        required=True,
    )
    parser.add_argument(
        "--model",
        choices=["dummy", "linear", "ridge", "poly", "logistic"],
        required=True,
    )
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--db-path", default=None)
    args = parser.parse_args()

    df = load_data(args.db_path)

    target = "mag" if args.task == "regression" else config.TARGET
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} is missing")

    # Neither target may be used as a predictor.
    X = df.drop(columns=["mag", config.TARGET], errors="ignore").copy()
    y = pd.to_numeric(df[target], errors="coerce")

    valid = y.notna()
    X = X.loc[valid].copy()
    y = y.loc[valid].copy()

    # Remove predictors with no usable values.
    X = X.dropna(axis=1, how="all")

    if args.task == "classification":
        y = y.astype(int)
        if y.nunique() != 2:
            raise ValueError("Classification requires both target classes")

    if args.task == "regression":
        if args.model == "logistic":
            raise ValueError("Use a regression model for the regression task")
        stratify = None
    else:
        if args.model not in {"dummy", "logistic"}:
            raise ValueError(
                "Classification supports only --model dummy or --model logistic"
            )
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=config.SEED,
        stratify=stratify,
    )

    if args.task == "regression":
        baseline = make_regressor("dummy")
        model = make_regressor(
            args.model,
            degree=args.degree,
            alpha=args.alpha,
        )
        scoring = "neg_root_mean_squared_error"
    else:
        baseline = make_classifier("dummy")
        model = make_classifier(
            args.model,
            class_weight="balanced" if args.balanced else None,
        )
        scoring = "f1"

    baseline.fit(X_train, y_train)
    model.fit(X_train, y_train)

    baseline_predictions = baseline.predict(X_test)
    model_predictions = model.predict(X_test)

    if args.task == "regression":
        baseline_report = regression_report(y_test, baseline_predictions)
        model_report = regression_report(y_test, model_predictions)
    else:
        baseline_probabilities = baseline.predict_proba(X_test)[:, 1]
        model_probabilities = model.predict_proba(X_test)[:, 1]

        baseline_report = classification_report_dict(
            y_test, baseline_predictions, baseline_probabilities
        )

        # Select a threshold from out-of-fold training predictions only.
        cv_folds = min(5, int(y_train.value_counts().min()))
        if cv_folds < 2:
            raise ValueError(
                "Not enough examples in the minority class for cross-validation"
            )

        oof_probabilities = cross_val_predict(
            model,
            X_train,
            y_train,
            cv=cv_folds,
            method="predict_proba",
        )[:, 1]

        threshold = best_threshold(y_train, oof_probabilities)
        threshold_predictions = (model_probabilities >= threshold).astype(int)

        model_report = classification_report_dict(
            y_test, threshold_predictions, model_probabilities
        )
        model_report["threshold"] = float(threshold)
        model_report["default_threshold"] = 0.5

    # Cross-validation uses training data only.
    cv_folds = 5
    if args.task == "classification":
        cv_folds = min(5, int(y_train.value_counts().min()))

    cv_result = cross_validate_model(
        model, X_train, y_train, scoring=scoring, cv=cv_folds
    )

    if args.task == "regression":
        cv_result["mean_rmse"] = float(-cv_result["mean"])
        cv_result["std_rmse"] = float(cv_result["std"])
        cv_result["fold_rmse"] = [float(-score) for score in cv_result["scores"]]

    config.ROOT.joinpath("models").mkdir(parents=True, exist_ok=True)
    config.ROOT.joinpath("reports").mkdir(parents=True, exist_ok=True)

    model_path = (
        config.ROOT / "models" / f"{args.task}_{args.model}.joblib"
    )
    report_path = (
        config.ROOT / "reports" / f"{args.task}_{args.model}.json"
    )

    joblib.dump(model, model_path)

    report = {
        "task": args.task,
        "model": args.model,
        "target": target,
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "baseline_test_metrics": baseline_report,
        "model_test_metrics": model_report,
        "cross_validation": cv_result,
        "model_path": str(model_path),
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)

    print("\n===== TRAINING REPORT =====")
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"\nSaved model: {model_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
