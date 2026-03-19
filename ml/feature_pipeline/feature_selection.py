from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif


@dataclass
class SelectionConfig:
    use_mutual_information: bool
    mi_top_k: int
    mi_min_score: float

    use_model_importance: bool
    rf_top_k: int

    prune_correlated: bool
    corr_threshold: float

    always_keep: Tuple[str, ...] = ()


def _numeric_feature_matrix(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Extract numeric feature matrix X from df[feature_cols].
    Drops columns that are not numeric or are entirely NaN.
    """
    valid_cols: List[str] = []
    matrices: List[np.ndarray] = []

    for c in feature_cols:
        if c not in df.columns:
            continue
        s = df[c]
        if not np.issubdtype(s.dtype, np.number):
            continue
        if s.dropna().empty:
            continue
        valid_cols.append(c)
        matrices.append(s.to_numpy().reshape(-1, 1))

    if not valid_cols:
        raise ValueError("No valid numeric feature columns found for selection.")

    X = np.concatenate(matrices, axis=1)
    return X, valid_cols


def compute_mutual_information_scores(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> Dict[str, float]:
    """
    Compute mutual information between each feature and the target.
    """
    X, cols = _numeric_feature_matrix(df, feature_cols)
    y = df[target_col].to_numpy()

    mi = mutual_info_classif(X, y, discrete_features=False, random_state=42)
    return {c: float(v) for c, v in zip(cols, mi)}


def compute_rf_importance_scores(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> Dict[str, float]:
    """
    Fit a small RandomForest to estimate feature importance.
    """
    X, cols = _numeric_feature_matrix(df, feature_cols)
    y = df[target_col].to_numpy()

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y)

    importances = rf.feature_importances_
    return {c: float(v) for c, v in zip(cols, importances)}


def prune_correlated_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    scores: Dict[str, float],
    corr_threshold: float,
) -> List[str]:
    """
    Remove redundant features based on correlation.

    Strategy:
      - compute correlation matrix
      - if |corr(i, j)| > threshold, drop the one with lower score
    """
    # Work only with columns that exist and are numeric
    cols = [c for c in feature_cols if c in df.columns]
    corr = df[cols].corr().abs()

    cols_to_keep = set(cols)

    # Iterate over upper triangle of corr matrix
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            if c1 not in cols_to_keep or c2 not in cols_to_keep:
                continue
            if corr.loc[c1, c2] >= corr_threshold:
                # Drop the lower-scoring feature
                s1 = scores.get(c1, 0.0)
                s2 = scores.get(c2, 0.0)
                if s1 >= s2:
                    cols_to_keep.discard(c2)
                else:
                    cols_to_keep.discard(c1)

    return sorted(cols_to_keep)


def run_feature_selection(
    df: pd.DataFrame,
    *,
    target_col: str,
    candidate_features: List[str],
    cfg: SelectionConfig,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Perform feature selection and return:
      - reduced dataframe (only selected features + target)
      - selection report dict (scores, decisions)
    """
    report: Dict = {
        "candidate_features": candidate_features,
        "always_keep": list(cfg.always_keep),
        "steps": {},
    }

    current_features = [c for c in candidate_features if c in df.columns]

    # ---- 1) Mutual information filter ----
    mi_scores: Dict[str, float] = {}
    mi_selected: List[str] = []

    if cfg.use_mutual_information:
        mi_scores = compute_mutual_information_scores(df, current_features, target_col)
        # Sort by MI descending
        sorted_by_mi = sorted(mi_scores.items(), key=lambda kv: kv[1], reverse=True)

        # Apply top-k and min-score filters
        mi_selected = [
            f for f, score in sorted_by_mi
            if score >= cfg.mi_min_score
        ][: cfg.mi_top_k]

        report["steps"]["mutual_information"] = {
            "scores": mi_scores,
            "selected": mi_selected,
        }
    else:
        mi_selected = current_features

    # ---- 2) RandomForest importance filter ----
    rf_scores: Dict[str, float] = {}
    rf_selected: List[str] = []

    if cfg.use_model_importance:
        rf_scores = compute_rf_importance_scores(df, mi_selected, target_col)
        sorted_by_rf = sorted(rf_scores.items(), key=lambda kv: kv[1], reverse=True)
        rf_selected = [f for f, _ in sorted_by_rf][: cfg.rf_top_k]

        report["steps"]["rf_importance"] = {
            "scores": rf_scores,
            "selected": rf_selected,
        }
    else:
        rf_selected = mi_selected

    # Combine scores for correlation pruning (prefer MI, then RF)
    combined_scores: Dict[str, float] = {}
    for f in rf_selected:
        combined_scores[f] = mi_scores.get(f, 0.0) + rf_scores.get(f, 0.0)

    selected_after_scores = rf_selected

    # ---- 3) Correlation pruning ----
    if cfg.prune_correlated:
        pruned = prune_correlated_features(
            df,
            selected_after_scores,
            scores=combined_scores,
            corr_threshold=cfg.corr_threshold,
        )
        report["steps"]["correlation_pruning"] = {
            "before": selected_after_scores,
            "after": pruned,
            "corr_threshold": cfg.corr_threshold,
        }
        selected_after_scores = pruned

    # ---- 4) Always-keep features ----
    final_features = sorted(set(selected_after_scores) | set(cfg.always_keep))

    report["final_features"] = final_features

    # Return dataframe with only selected features + target
    selected_cols = final_features + [target_col]
    df_selected = df[selected_cols].copy()

    return df_selected, report