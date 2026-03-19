from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


@dataclass
class FeatureSelectionConfig:
    enabled: bool = True
    top_k: int = 15
    min_importance: float = 0.0
    always_keep: Tuple[str, ...] = ()
    n_estimators: int = 300
    max_depth: int | None = None
    min_samples_leaf: int = 1
    n_jobs: int = -1


def _parse_fs_cfg(fs_cfg: FeatureSelectionConfig | Dict[str, Any]) -> FeatureSelectionConfig:
    """
    Allow passing either a dataclass or a plain dict from YAML.
    """
    if isinstance(fs_cfg, FeatureSelectionConfig):
        return fs_cfg
    # map possible YAML keys to dataclass fields
    return FeatureSelectionConfig(
        enabled=fs_cfg.get("enabled", True),
        top_k=int(fs_cfg.get("top_k", 15)),
        min_importance=float(fs_cfg.get("min_importance", 0.0)),
        always_keep=tuple(fs_cfg.get("always_keep", [])),
        n_estimators=int(fs_cfg.get("n_estimators", fs_cfg.get("rf_n_estimators", 300))),
        max_depth=fs_cfg.get("max_depth", fs_cfg.get("rf_max_depth", None)),
        min_samples_leaf=int(fs_cfg.get("min_samples_leaf", fs_cfg.get("rf_min_samples_leaf", 1))),
        n_jobs=int(fs_cfg.get("n_jobs", fs_cfg.get("rf_n_jobs", -1))),
    )


def rf_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    all_features: List[str],
    fs_cfg: FeatureSelectionConfig | Dict[str, Any],
    random_state: int = 42,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Train a RandomForest on TRAIN ONLY and select a subset of features.

    Rules:
      - If fs_cfg.enabled is False: return all_features (no selection).
      - Otherwise:
          * compute feature importances
          * keep union of:
              - top_k most important features
              - features with importance >= min_importance
              - always_keep list
          * if everything gets dropped by thresholds, fall back to:
              - top_k (if possible), else all_features

    Returns:
      selected_features: list[str]  (ordered as in all_features)
      details: dict with importances, thresholds, and any fallback reason
    """
    cfg = _parse_fs_cfg(fs_cfg)

    # If feature selection is disabled, just return all features.
    if not cfg.enabled:
        return list(all_features), {
            "enabled": False,
            "reason": "feature selection disabled in config",
            "selected_features": list(all_features),
        }

    # Ensure we only work with the given feature subset
    X_fs = X_train[all_features]

    rf = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        n_jobs=cfg.n_jobs,
        random_state=random_state,
    )
    rf.fit(X_fs, y_train)

    importances = rf.feature_importances_
    imp_series = pd.Series(importances, index=all_features).sort_values(ascending=False)

    # --- Selection rules ---
    selected_set: set[str] = set()

    # 1) Top-k
    if cfg.top_k is not None and cfg.top_k > 0:
        top_k_feats = imp_series.head(cfg.top_k).index.tolist()
        selected_set.update(top_k_feats)

    # 2) Importance threshold
    if cfg.min_importance > 0.0:
        high_imp_feats = imp_series[imp_series >= cfg.min_importance].index.tolist()
        selected_set.update(high_imp_feats)

    # 3) Always-keep features (ensure they are in the final set)
    always_keep_set = {f for f in cfg.always_keep if f in all_features}
    selected_set.update(always_keep_set)

    # Fallback logic if everything somehow got dropped
    fallback_reason = None
    if not selected_set:
        fallback_reason = "no features passed thresholds; falling back to top_k or all_features"
        if cfg.top_k is not None and cfg.top_k > 0:
            selected_set = set(imp_series.head(cfg.top_k).index.tolist())
        else:
            selected_set = set(all_features)

    # Preserve the original column order
    selected_features = [f for f in all_features if f in selected_set]

    details: Dict[str, Any] = {
        "enabled": True,
        "top_k": cfg.top_k,
        "min_importance": cfg.min_importance,
        "always_keep": list(always_keep_set),
        "feature_importances_sorted": imp_series.to_dict(),
        "selected_features": selected_features,
    }
    if fallback_reason:
        details["fallback_reason"] = fallback_reason

    return selected_features, details