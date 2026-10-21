"""Split earthquake data and build preprocessing pipelines."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
)

from . import config


def split_data(df: pd.DataFrame, test_size: float = 0.2):
    """Return X_train, X_test, y_train, y_test using stratified splitting."""
    if config.TARGET not in df.columns:
        raise ValueError(f"Target column {config.TARGET!r} is missing.")

    y = df[config.TARGET].copy()
    X = df.drop(columns=[config.TARGET, "mag"], errors="ignore").copy()

    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=config.SEED,
        stratify=stratify,
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor(
    scaler: str = "robust", degree: int = 1
) -> ColumnTransformer:
    """Build numeric and categorical preprocessing pipelines."""
    scalers = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }

    if scaler not in scalers:
        raise ValueError(
            f"Unknown scaler {scaler!r}. Choose from {list(scalers)}."
        )
    if not isinstance(degree, int) or isinstance(degree, bool) or degree < 1:
        raise ValueError("degree must be a positive integer.")

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]

    if degree > 1:
        numeric_steps.append(
            (
                "polynomial",
                PolynomialFeatures(degree=degree, include_bias=False),
            )
        )

    numeric_steps.append(("scaler", scalers[scaler]()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    nominal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, config.NUMERIC),
            ("nominal", nominal_pipeline, config.NOMINAL),
        ],
        remainder="drop",
    )
