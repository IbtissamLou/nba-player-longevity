#metrics & threshold  ✅
from __future__ import annotations

"""
Evaluation on held-out test set using a tuned threshold.
"""

from typing import Dict, Any

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline

from .metrics import compute_classification_metrics


def evaluate_with_threshold(
    pipeline: ImbPipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> Dict[str, Any]:
    """
    - Runs predict_proba on X_test
    - Applies custom threshold
    - Computes metrics for default 0.5 and tuned threshold
    """
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred_default = pipeline.predict(X_test)
    y_pred_tuned = (y_proba >= threshold).astype(int)

    metrics_default = compute_classification_metrics(
        y_test, y_pred_default, y_proba=y_proba
    )
    metrics_tuned = compute_classification_metrics(
        y_test, y_pred_tuned, y_proba=y_proba
    )

    return {
        "threshold_used": float(threshold),
        "metrics_default": metrics_default,
        "metrics_tuned": metrics_tuned,
    }