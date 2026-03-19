
"""
K-fold cross-validation with:
- Feature transformer (skewness-based transforms + scaling)
- Optional SMOTE for class imbalance
- Metric functions (f1, recall, precision, accuracy, roc_auc, etc.)

Important: we do NOT wrap the feature pipeline inside an imblearn.Pipeline
Instead, we:
  - fit/transform with feature_transformer manually
  - apply SMOTE on the transformed X_train
  - fit the estimator on resampled data
"""

from __future__ import annotations

from typing import Dict, Callable, Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE


def run_kfold_cv(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    feature_transformer,
    estimator,
    metric_fns: Dict[str, Callable],
    n_splits: int = 5,
    random_state: int = 42,
    shuffle: bool = True,
    smote_cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Run K-fold CV for a single model (estimator) with:
      - feature pipeline
      - optional SMOTE
      - probability-based metrics (roc_auc, average_precision, etc.)

    Parameters
    ----------
    X : DataFrame
        Features (already filtered to selected feature columns).
    y : Series
        Binary target (0/1).
    feature_transformer :
        A fitted-then-cloneable sklearn Pipeline (from build_feature_pipeline).
    estimator :
        An sklearn classifier (RF, XGB, etc.).
    metric_fns : dict[str, callable]
        Mapping metric_name -> function(y_true, y_pred or y_proba).
        For prob-based metrics we will pass probabilities.
    n_splits : int
        Number of folds.
    random_state : int
        Seed for StratifiedKFold and SMOTE.
    shuffle : bool
        Whether to shuffle splits.
    smote_cfg : dict or None
        Config for SMOTE:
          {
            "enabled": bool,
            "method": "smote",
            "smote": {"k_neighbors": 5, ...},
            "min_positive_rate": 0.05,  # (not used here but kept for consistency)
          }

    Returns
    -------
    dict with:
      - "fold_scores": {metric_name: [scores_per_fold]}
      - "mean_scores": {metric_name: float}
    """
    if smote_cfg is None:
        smote_cfg = {"enabled": False}

    cv = StratifiedKFold(  #Ensures each fold has a similar proportion of TARGET_5Yrs = 1.
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )

    metric_names = list(metric_fns.keys())
    fold_scores: Dict[str, list[float]] = {m: [] for m in metric_names}

    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # --- 1) Clone and fit feature transformer only on train fold ---
        ft = clone(feature_transformer) #clone ensures a fresh transformer per fold (no cross-fold contamination)
        ft.fit(X_tr, y_tr) #fit on X_tr only → avoids data leakage from validation into transforms
        X_tr_ft = ft.transform(X_tr)
        X_val_ft = ft.transform(X_val)

        # --- 2)  SMOTE on transformed training data -------------
        if smote_cfg.get("enabled", True): #Oversamples the minority class inside the current train fold only
            smote_params = smote_cfg.get("smote", {})
            smote = SMOTE(random_state=random_state, **smote_params)
            X_tr_ft, y_tr = smote.fit_resample(X_tr_ft, y_tr)

        # --- 3) Clone and train estimator -------------------------------
        clf = clone(estimator)
        clf.fit(X_tr_ft, y_tr)

        # --- 4) Predictions / probabilities -----------------------------
        if hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(X_val_ft)[:, 1]
        elif hasattr(clf, "decision_function"):
            # Some models (e.g. SVM) only expose decision_function
            y_proba = clf.decision_function(X_val_ft)
        else:
            # Fall back to hard predictions only
            y_proba = None

        y_pred = clf.predict(X_val_ft)

        # --- 5) Metrics per fold ----------------------------------------
        for name, fn in metric_fns.items():
            # Heuristic: probability-based metrics by name
            if name in ("roc_auc", "average_precision") and y_proba is not None:
                score = fn(y_val, y_proba)
            else:
                score = fn(y_val, y_pred)
            fold_scores[name].append(float(score))

    mean_scores = { #Simple mean across folds: this is CV estimate for each metric.
        name: float(np.mean(vals)) if len(vals) > 0 else float("nan")
        for name, vals in fold_scores.items()
    }

    return {
        "fold_scores": fold_scores,
        "mean_scores": mean_scores,
    }