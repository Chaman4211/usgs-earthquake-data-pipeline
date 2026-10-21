"""Scikit-learn-compatible estimators and earthquake model factories."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.pipeline import Pipeline

from .transform import build_preprocessor


class GDLinearRegression(BaseEstimator, RegressorMixin):
    """Vectorized linear regression trained using gradient descent."""

    def __init__(self, lr: float = 0.1, n_iter: int = 1000):
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2 or X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have matching sample counts.")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must contain at least one sample and feature.")
        if self.lr <= 0 or self.n_iter < 1:
            raise ValueError("lr must be positive and n_iter must be >= 1.")

        self.n_features_in_ = X.shape[1]
        self.coef_ = np.zeros(X.shape[1], dtype=float)
        self.intercept_ = 0.0
        self.loss_history_ = []

        n_samples = X.shape[0]

        for _ in range(self.n_iter):
            predictions = X @ self.coef_ + self.intercept_
            errors = predictions - y
            self.loss_history_.append(float(np.mean(errors ** 2) / 2))

            self.coef_ -= self.lr * (X.T @ errors) / n_samples
            self.intercept_ -= self.lr * float(np.mean(errors))

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_


def make_regressor(
    kind: str = "linear", degree: int = 2, alpha: float = 1.0
):
    """Create a preprocessing and regression pipeline."""
    kind = kind.lower()

    if kind == "dummy":
        estimator = DummyRegressor(strategy="mean")
        polynomial_degree = 1
    elif kind == "linear":
        estimator = LinearRegression()
        polynomial_degree = 1
    elif kind == "ridge":
        estimator = Ridge(alpha=alpha)
        polynomial_degree = 1
    elif kind == "poly":
        if degree < 1:
            raise ValueError("degree must be at least 1.")
        estimator = Ridge(alpha=alpha)
        polynomial_degree = degree
    else:
        raise ValueError(f"Unknown regressor kind: {kind!r}")

    return Pipeline(
        [
            (
                "prep",
                build_preprocessor(degree=polynomial_degree),
            ),
            ("model", estimator),
        ]
    )


def make_classifier(
    kind: str = "logistic", C: float = 1.0, class_weight=None
):
    """Create a preprocessing and classification pipeline."""
    kind = kind.lower()

    if kind == "dummy":
        estimator = DummyClassifier(strategy="prior")
    elif kind == "logistic":
        estimator = LogisticRegression(
            C=C,
            class_weight=class_weight,
            max_iter=2000,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown classifier kind: {kind!r}")

    return Pipeline(
        [
            ("prep", build_preprocessor()),
            ("model", estimator),
        ]
    )
