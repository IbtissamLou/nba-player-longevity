"""
Anomaly detection utilities for the Data Cycle.

This module flags suspicious rows that are statistically unusual (but not necessarily invalid).
It complements:
- schema validation (structure/type)
- rule validation (logical constraints)
- statistical checks (dataset-level drift)

Output is designed for debugging and gating before training/retraining.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AnomalyConfig:
    """
    Configuration for anomaly detection (quantile-based).

    - q_low/q_high define the acceptable range per numeric feature based on data quantiles.
    - min_non_null controls whether a column has enough values to compute quantiles reliably.
    - exclude_cols prevents flagging label/id columns.
    - max_outlier_rate_* supports gating policies (warn/fail).
    """
    enabled: bool = True
    q_low: float = 0.01
    q_high: float = 0.99
    min_non_null: int = 10
    exclude_cols: tuple[str, ...] = ("TARGET_5Yrs",)

    max_outlier_rate_warn: float = 0.02
    max_outlier_rate_fail: float = 0.10


def flag_quantile_outliers(
    df: pd.DataFrame,
    cfg: AnomalyConfig,
) -> tuple[pd.DataFrame, dict]:
    """
    Flag row-level anomalies using per-column quantile bounds.

    For each numeric column c:
      outlier_c = (c < quantile(q_low)) OR (c > quantile(q_high))

    Returns:
      df_out:
        original df + columns:
          - outlier_any (bool)
          - outlier_count (int): number of columns that flagged this row
          - outlier_cols (str): comma-separated list of columns that flagged this row
      report:
        summary with bounds, per-column outlier rates, total outlier rate, and gating status
    """
    if not cfg.enabled:
        df_out = df.copy()
        df_out["outlier_any"] = False
        df_out["outlier_count"] = 0
        df_out["outlier_cols"] = ""
        return df_out, {
            "enabled": False,
            "status": "pass",
            "details": ["Anomaly detection disabled."],
            "outlier_rate": 0.0,
            "per_column_outlier_rate": {},
            "bounds": {},
        }

    df_out = df.copy()

    # Choose numeric columns, excluding configured cols
    num_cols = df_out.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in set(cfg.exclude_cols)]

    flags: dict[str, pd.Series] = {}
    bounds: dict[str, dict] = {}

    for c in num_cols:
        s = df_out[c].dropna()

        # skip columns with too few values to estimate quantiles reliably
        if len(s) < cfg.min_non_null:
            continue

        lo = float(s.quantile(cfg.q_low))
        hi = float(s.quantile(cfg.q_high))

        bounds[c] = {"q_low": cfg.q_low, "q_high": cfg.q_high, "low": lo, "high": hi}

        # boolean flag per row
        flags[c] = (df_out[c] < lo) | (df_out[c] > hi)

    if not flags:
        df_out["outlier_any"] = False
        df_out["outlier_count"] = 0
        df_out["outlier_cols"] = ""
        report = {
            "enabled": True,
            "status": "pass",
            "details": ["No numeric columns eligible for quantile outlier detection."],
            "outlier_rate": 0.0,
            "per_column_outlier_rate": {},
            "bounds": bounds,
        }
        return df_out, report

    flag_df = pd.DataFrame(flags)  # columns = numeric cols, values = bool outlier per row

    # Row-level aggregation
    df_out["outlier_count"] = flag_df.sum(axis=1).astype(int)
    df_out["outlier_any"] = df_out["outlier_count"] > 0

    # Which columns triggered (debug-friendly)
    def _cols_triggered(row: pd.Series) -> str:
        cols = row.index[row.values].tolist()
        return ",".join(cols)

    df_out["outlier_cols"] = flag_df.apply(_cols_triggered, axis=1)

    # Summary metrics
    outlier_rate = float(df_out["outlier_any"].mean())
    per_col_rate = {c: float(flag_df[c].mean()) for c in flag_df.columns}

    # Gating policy
    status = "pass"
    details: list[str] = []
    if outlier_rate >= cfg.max_outlier_rate_warn:
        status = "warn"
        details.append(f"Outlier rate {outlier_rate:.2%} >= warn threshold {cfg.max_outlier_rate_warn:.2%}.")
    if outlier_rate >= cfg.max_outlier_rate_fail:
        status = "fail"
        details.append(f"Outlier rate {outlier_rate:.2%} >= fail threshold {cfg.max_outlier_rate_fail:.2%}.")

    report = {
        "enabled": True,
        "status": status,
        "details": details or ["Outlier rate within expected thresholds."],
        "outlier_rate": outlier_rate,
        "per_column_outlier_rate": per_col_rate,
        "bounds": bounds,
        "thresholds": {
            "warn": cfg.max_outlier_rate_warn,
            "fail": cfg.max_outlier_rate_fail,
        },
    }

    return df_out, report
