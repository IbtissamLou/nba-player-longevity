#FEATURE PIPELINE  ✅
# Scaling + per-feature transform
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    PowerTransformer,
    StandardScaler,
    MinMaxScaler,
)


def _load_skewness_map(
    skewness_report_path: str | None,
    selected_features: List[str],
    #default_transform: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Load skewness metadata for each selected feature.

    Expected CSV columns:
      - feature
      - skewness
      - zero_ratio
      - category
      - suggestion   (values like 'none', 'log1p', 'yeo_johnson')

    If a feature is missing or file missing, we fall back to `default_transform`.
    """
    skew_map: Dict[str, Dict[str, Any]] = {}

    df_skew = pd.read_csv(skewness_report_path)
    if "feature" not in df_skew.columns:
        raise ValueError("skewness_report.csv must contain a 'feature' column")

    df_skew = df_skew.set_index("feature")

    for f in selected_features:
        if f in df_skew.index:
            row = df_skew.loc[f]
            suggestion = str(row.get("suggestion"))
            skew_map[f] = {
                    "skewness": float(row.get("skewness")),
                    "zero_ratio": float(row.get("zero_ratio")),
                    "category": row.get("category"),
                    "suggestion": suggestion,
                }
        else:
            # Feature not present in report
            skew_map[f] = {
                   "skewness": np.nan,
                    "zero_ratio": np.nan,
                    "category": "missing_in_report",
                    "suggestion": "none",
                }

    return skew_map


def build_feature_pipeline(
    *,
    selected_features: List[str],
    X_train: pd.DataFrame,
    eng_cfg: Dict[str, Any],
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Build the *training-time* feature engineering pipeline.

    Steps:
      1. Read skewness_report.csv and decide per-feature transform:
         'none', 'log1p', or 'yeo_johnson'.
      2. Build a ColumnTransformer that applies those transforms.
      3. Optionally apply scaling (standard or minmax) on top.

    Returns:
      pipeline: sklearn.Pipeline ready to .fit(X_train, y)
      details:  dict describing what has been configured
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame for column-based transforms.")
    
    skew_path = eng_cfg.get("skewness_report_path") or eng_cfg.get("skewness_report_csv")

    if skew_path is None:
        raise ValueError(
            "Missing skewness report path in eng_cfg. "
            "Expected one of: 'skewness_report_path' or 'skewness_report_csv'."
        )
    
    scaler_name = eng_cfg.get("scaler", "standard")

    # ------------------------------------------------------------------
    # 1) Skewness → transform decision per feature
    # ------------------------------------------------------------------
    skew_map = _load_skewness_map(skew_path, selected_features)

    # feature -> "none" / "log1p" / "yeo_johnson"
    transform_map: Dict[str, str] = {
        f: skew_map[f]["suggestion"] for f in selected_features
    }

    log_cols = [f for f, t in transform_map.items() if t == "log1p"]
    yeo_cols = [f for f, t in transform_map.items() if t == "yeo_johnson"]
    passthrough_cols = [f for f, t in transform_map.items() if t not in ("log1p", "yeo_johnson")]

    # ------------------------------------------------------------------
    # 2) ColumnTransformer assembly
    # ------------------------------------------------------------------
    transformers = []

    if log_cols:
        transformers.append(
            (
                "log1p",
                FunctionTransformer(np.log1p, validate=False),
                log_cols,
            )
        )

    if yeo_cols:
        transformers.append(
            (
                "yeo_johnson",
                PowerTransformer(method="yeo-johnson"),
                yeo_cols,
            )
        )

    if passthrough_cols:
        transformers.append(
            (
                "passthrough",
                "passthrough",
                passthrough_cols,
            )
        )

    col_tf = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # ------------------------------------------------------------------
    # 3) Scaling
    # ------------------------------------------------------------------
    scaler_name = (scaler_name or "none").lower()

    if scaler_name == "standard":
        scaler = StandardScaler()
    elif scaler_name == "minmax":
        scaler = MinMaxScaler()
    else:  # "none" or anything unexpected
        scaler = "passthrough"

    steps = [("cols", col_tf)]
    if scaler != "passthrough":
        steps.append(("scaler", scaler))

    pipeline = Pipeline(steps)

    # ------------------------------------------------------------------
    # 4) Details for debugging/tests
    # ------------------------------------------------------------------
    groups = {
        "log1p": log_cols,
        "yeo_johnson": yeo_cols,
        "none": passthrough_cols,
    }

    # what tests expect: dict(feature -> transform)
    transform_plan = dict(transform_map)

    details = {
        "transform_map": transform_map,
        "transform_plan": transform_plan,
        "groups": groups,
        "scaling": scaler_name,
        "skew_map": skew_map,
    }

    return pipeline, details