from __future__ import annotations

from typing import Dict, Any, List

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    balanced_accuracy_score,
)


def _compute_metric_for_threshold(
    metric: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute a scalar score for a given threshold.

    Supported options:
      - "f1"                : F1-score for positive class
      - "recall"            : Recall for positive class
      - "precision"         : Precision for positive class
      - "balanced_accuracy" : Average recall over both classes
      - "balanced"          : Custom weighted mix (F1, recall, precision)
    """
    if metric == "f1":
        return f1_score(y_true, y_pred)

    elif metric == "recall":
        return recall_score(y_true, y_pred)

    elif metric == "precision":
        return precision_score(y_true, y_pred)

    elif metric == "balanced_accuracy":
        # This is the cleanest “balance between classes” metric:
        # (recall_0 + recall_1) / 2
        return balanced_accuracy_score(y_true, y_pred)

    elif metric == "balanced":
        f1 = f1_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)

        w_f1 = 0.4
        w_rec = 0.4
        w_prec = 0.2

        return w_f1 * f1 + w_rec * rec + w_prec * prec

    else:
        raise ValueError(
            f"Unsupported metric for threshold tuning: {metric}. "
            f"Use one of: 'f1', 'recall', 'precision', 'balanced', 'balanced_accuracy'."
        )


def tune_threshold_cv(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    feature_transformer,
    estimator,
    smote_cfg: Dict[str, Any],
    n_splits: int,
    random_state: int,
    metric: str = "f1",
    n_thresholds: int = 101,
) -> Dict[str, Any]:
    """
    Tune decision threshold using cross-validated probabilities.

    """
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    X = X.copy()
    y = y.copy()

    all_proba: List[np.ndarray] = []
    all_true: List[np.ndarray] = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        ft = clone(feature_transformer)
        est = clone(estimator)

        X_tr_tf = ft.fit_transform(X_tr, y_tr)
        X_va_tf = ft.transform(X_va)

        X_fit, y_fit = X_tr_tf, y_tr
        if smote_cfg.get("enabled", True) and smote_cfg.get("method", "smote") == "smote":
            smote_params = dict(smote_cfg.get("smote", {}))
            sm = SMOTE(random_state=random_state, **smote_params)
            X_fit, y_fit = sm.fit_resample(X_tr_tf, y_tr)

        est.fit(X_fit, y_fit)

        if not hasattr(est, "predict_proba"):
            raise ValueError(
                "Estimator must support predict_proba for threshold tuning."
            )

        probs = est.predict_proba(X_va_tf)[:, 1]
        all_proba.append(probs)
        all_true.append(y_va.values)

    y_true_all = np.concatenate(all_true)
    y_proba_all = np.concatenate(all_proba)

    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    scores: List[float] = []

    for t in thresholds:
        y_pred = (y_proba_all >= t).astype(int)
        s = _compute_metric_for_threshold(metric, y_true_all, y_pred)
        scores.append(float(s))

    scores_np = np.array(scores)
    best_idx = int(np.argmax(scores_np))
    best_threshold = float(thresholds[best_idx])
    best_score = float(scores_np[best_idx])

    return {
        "best_threshold": best_threshold,
        "best_score": best_score,
        "metric": metric,
        "thresholds": thresholds.tolist(),
        "scores": scores,
    }