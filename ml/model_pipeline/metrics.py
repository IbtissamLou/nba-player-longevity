#scoring functions  ✅
# METRICS USED FOR MODEL EVALUATION (F1 SCORE - PRECISION SCORE - ROC SCORE - CLASSIFICATION REPORT)
from __future__ import annotations

"""
Metric helpers used during evaluation.

This keeps metric logic in one place so we don't duplicate it across
training / evaluation / monitoring.
"""

from typing import Dict, Any

import numpy as np
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
)


def compute_classification_metrics(
    y_true,
    y_pred,
    y_proba=None,
    *,
    average: str = "binary",
) -> Dict[str, Any]:
    """Return a dictionary of core metrics + classification report."""
    metrics: Dict[str, Any] = {}

    metrics["f1"] = float(f1_score(y_true, y_pred, average=average))

    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["roc_auc"] = None

        try:
            metrics["average_precision"] = float(
                average_precision_score(y_true, y_proba)
            )
        except ValueError:
            metrics["average_precision"] = None

    metrics["classification_report"] = classification_report(
        y_true, y_pred, output_dict=True
    )

    return metrics