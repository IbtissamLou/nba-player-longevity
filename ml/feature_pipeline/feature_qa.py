from __future__ import annotations

"""
Feature QA (quality assurance) checks.

This is NOT EDA: it is a quality gate on the feature dataset.
Even if raw data passed Data Cycle validation, engineered features can still be:
- mostly missing
- constant or near-constant
- contain inf/NaN (division bugs)
- dominated by weird values

We output a JSON report and return status: pass / warn / fail.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureQAConfig:
    enabled: bool = True
    missing_warn: float = 0.05
    missing_fail: float = 0.30

    constant_unique_ratio_warn: float = 0.01
    constant_unique_ratio_fail: float = 0.001

    inf_fail: bool = True


def run_feature_qa(
    df: pd.DataFrame,
    *,
    exclude_cols: tuple[str, ...],
    cfg: FeatureQAConfig,
    out_path: str = "ml/reports/feature_qa.json",
) -> dict[str, Any]:
    """
    Run lightweight QA checks on engineered features.

    exclude_cols: columns not treated as features (target, id, etc.)
    """
    report: dict[str, Any] = {
        "enabled": cfg.enabled,
        "status": "pass",
        "details": [],
        "missingness": {},
        "constant_features": [],
        "near_constant_features": [],
        "inf_features": [],
        "nan_features": [],
        "thresholds": {
            "missing_warn": cfg.missing_warn,
            "missing_fail": cfg.missing_fail,
            "constant_unique_ratio_warn": cfg.constant_unique_ratio_warn,
            "constant_unique_ratio_fail": cfg.constant_unique_ratio_fail,
            "inf_fail": cfg.inf_fail,
        },
    }

    if not cfg.enabled:
        return report

    # Feature columns only 
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # --------------------
    # (1) Missingness
    # --------------------
    miss = df[feature_cols].isna().mean().to_dict()
    report["missingness"] = {k: float(v) for k, v in miss.items()}

    worst_missing = max(miss.values()) if miss else 0.0
    if worst_missing >= cfg.missing_fail:
        report["status"] = "fail"
        report["details"].append(f"Missingness fail: max missing ratio={worst_missing:.2%}")
    elif worst_missing >= cfg.missing_warn and report["status"] != "fail":
        report["status"] = "warn"
        report["details"].append(f"Missingness warn: max missing ratio={worst_missing:.2%}")

    # --------------------
    # (2) Constant / near-constant
    # --------------------
    # unique_ratio = (#unique values) / (#rows)
    n = max(int(df.shape[0]), 1)
    unique_ratios = {c: float(df[c].nunique(dropna=True) / n) for c in feature_cols}

    const_fail = [c for c, r in unique_ratios.items() if r <= cfg.constant_unique_ratio_fail]
    const_warn = [c for c, r in unique_ratios.items()
                  if cfg.constant_unique_ratio_fail < r <= cfg.constant_unique_ratio_warn]

    if const_fail:
        report["constant_features"] = const_fail
        report["status"] = "fail"
        report["details"].append(f"Constant features detected (fail): {const_fail[:10]}")

    if const_warn and report["status"] != "fail":
        report["near_constant_features"] = const_warn
        report["status"] = "warn"
        report["details"].append(f"Near-constant features detected (warn): {const_warn[:10]}")

    # --------------------
    # (3) NaN / Inf checks on numeric features
    # --------------------
    num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    nan_cols = [c for c in num_cols if df[c].isna().any()]
    inf_cols = [c for c in num_cols if np.isinf(df[c].to_numpy()).any()]

    report["nan_features"] = nan_cols
    report["inf_features"] = inf_cols

    if cfg.inf_fail and inf_cols:
        report["status"] = "fail"
        report["details"].append(f"Inf values detected in: {inf_cols[:10]}")

    # Save artifact
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    return report
