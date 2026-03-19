from __future__ import annotations

import pandas as pd
from typing import Dict, Any
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


class MetricDecayMonitor:
    """
    Tracks performance decay over time.
    Requires predictions + ground truth.
    """

    def __init__(self):
        pass

    def compute_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
    ) -> Dict[str, float]:

        return {
            "f1": float(f1_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred)),
            "recall": float(recall_score(y_true, y_pred)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
        }

    def detect_decay(
        self,
        reference_metrics: Dict[str, float],
        current_metrics: Dict[str, float],
        threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Detect performance degradation.
        """

        decay = {}

        for metric in reference_metrics:
            diff = reference_metrics[metric] - current_metrics[metric]

            decay[metric] = {
                "reference": reference_metrics[metric],
                "current": current_metrics[metric],
                "drop": diff,
                "alert": diff > threshold,
            }

        return decay