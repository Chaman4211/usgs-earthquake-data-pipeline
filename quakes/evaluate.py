"""Evaluation metrics and cross-validation helpers."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score


def regression_report(y_true, y_pred):
    """Return regression metrics as ordinary Python floats."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def classification_report_dict(y_true, y_pred, y_proba):
    """Return classification metrics as ordinary Python floats."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)

    try:
        roc_auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        # ROC AUC is undefined when y_true contains only one class.
        roc_auc = float("nan")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc),
    }


def best_threshold(y_true, y_proba, metric="f1"):
    """Select a threshold from 0.05 to 0.95 using the requested metric."""
    metric_functions = {
        "f1": f1_score,
        "precision": precision_score,
        "recall": recall_score,
    }

    if metric not in metric_functions:
        raise ValueError("metric must be 'f1', 'precision', or 'recall'")

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if y_true.shape[0] != y_proba.shape[0]:
        raise ValueError("y_true and y_proba must have the same length")

    thresholds = np.arange(0.05, 1.00, 0.05)
    scorer = metric_functions[metric]

    best_value = -1.0
    best_cutoff = 0.5

    for threshold in thresholds:
        predictions = (y_proba >= threshold).astype(int)
        value = scorer(y_true, predictions, zero_division=0)

        # Keep the first threshold in case of a tie.
        if value > best_value:
            best_value = float(value)
            best_cutoff = float(threshold)

    return best_cutoff


def cross_validate_model(pipe, X, y, scoring, cv=5):
    """Return cross-validation mean, standard deviation, and fold scores."""
    scores = cross_val_score(pipe, X, y, scoring=scoring, cv=cv)

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "scores": [float(score) for score in scores],
    }
