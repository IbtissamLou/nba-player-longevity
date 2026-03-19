from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.model_pipeline.metrics import compute_classification_metrics
from ml.model_pipeline.evaluate import evaluate_with_threshold


def test_compute_classification_metrics_basic():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.4, 0.2, 0.8])

    metrics = compute_classification_metrics(y_true, y_pred, y_proba=y_proba)

    assert "f1" in metrics
    assert metrics["f1"] >= 0.0
    assert "classification_report" in metrics
    assert isinstance(metrics["classification_report"], dict)


def test_evaluate_with_threshold_end_to_end():
    # Build a tiny pipeline on synthetic data
    rng = np.random.RandomState(0)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=200),
            "x2": rng.normal(size=200),
        }
    )
    y = (X["x1"] + X["x2"] > 0).astype(int)

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression()),
        ]
    )

    # Simple split
    X_train, X_test = X.iloc[:150], X.iloc[150:]
    y_train, y_test = y.iloc[:150], y.iloc[150:]

    pipe.fit(X_train, y_train)

    # Try threshold 0.4
    result = evaluate_with_threshold(pipe, X_test, y_test, threshold=0.4)

    assert "threshold_used" in result
    assert "metrics_default" in result
    assert "metrics_tuned" in result
    assert result["threshold_used"] == 0.4