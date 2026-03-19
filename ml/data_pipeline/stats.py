from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def build_profile(
    df: pd.DataFrame,
    target_col: str,
    *,
    quantiles: tuple[float, float, float] = (0.01, 0.50, 0.99),
) -> dict:
    """
    Build a machine-readable dataset profile used for statistical checks and drift comparison.

    The profile is deterministic and lightweight:
    - dataset shape
    - missing ratios
    - target distribution
    - per-numeric-column summary statistics (mean/std/quantiles)
    - per-column unique ratio (helps detect constant/near-constant features)
    """
    q1, q2, q3 = quantiles

    profile = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "missing_ratio": df.isna().mean().to_dict(),
        "target_distribution": df[target_col].value_counts(dropna=False, normalize=True).to_dict()
        if target_col in df.columns
        else None,
        "numeric_summary": {},
        "unique_ratio": {}, 
    }

    # unique ratio for all columns (including Name)
    n = len(df) if len(df) else 1
    for c in df.columns:
        profile["unique_ratio"][c] = _safe_float(df[c].nunique(dropna=False) / n)

    # numeric summary
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        s = df[c].dropna()
        profile["numeric_summary"][c] = {
            "mean": _safe_float(s.mean()) if len(s) else None,
            "std": _safe_float(s.std()) if len(s) else None,
            f"p{int(q1*100):02d}": _safe_float(s.quantile(q1)) if len(s) else None,
            f"p{int(q2*100):02d}": _safe_float(s.quantile(q2)) if len(s) else None,
            f"p{int(q3*100):02d}": _safe_float(s.quantile(q3)) if len(s) else None,
        }

    return profile


def run_statistical_checks(
    df: pd.DataFrame,
    profile: dict,
    *,
    target_col: str,
    missingness_warn_threshold: float = 0.20,
    missingness_fail_threshold: float = 0.50,
    outlier_zscore_threshold: float = 4.0,
    outlier_rate_warn: float = 0.05,
    constant_unique_ratio_threshold: float = 0.01,
) -> dict:
    """
    Run statistical checks on the current dataset (single-batch checks).

    Returns a structured dict with:
    - status: "pass" | "warn" | "fail"
    - details: list of messages
    - metrics: useful numbers for debugging
    """
    details: list[str] = []
    status = "pass"
    metrics: dict = {}

    # ---- 1) Missingness checks ----
    missing_ratio = profile.get("missing_ratio", {})
    warn_cols = [c for c, r in missing_ratio.items() if r is not None and r >= missingness_warn_threshold]
    fail_cols = [c for c, r in missing_ratio.items() if r is not None and r >= missingness_fail_threshold]

    if warn_cols:
        details.append(f"High missingness (>= {missingness_warn_threshold:.0%}) in columns: {warn_cols}")
        status = "warn"
    if fail_cols:
        details.append(f"Severe missingness (>= {missingness_fail_threshold:.0%}) in columns: {fail_cols}")
        status = "fail"

    metrics["missingness_warn_cols"] = warn_cols
    metrics["missingness_fail_cols"] = fail_cols

    # ---- 2) Constant / near-constant columns ----
    unique_ratio = profile.get("unique_ratio", {})
    constant_cols = [c for c, r in unique_ratio.items() if r is not None and r <= constant_unique_ratio_threshold]
    if constant_cols:
        details.append(f"Constant/near-constant columns detected (unique_ratio <= {constant_unique_ratio_threshold}): {constant_cols}")
        status = "warn" if status != "fail" else status
    metrics["constant_cols"] = constant_cols

    # ---- 3) Outlier rate check (z-score based, numeric columns only) ----
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_rates = {}
    for c in num_cols:
        s = df[c].dropna()
        if len(s) < 10:
            continue
        std = s.std()
        if std == 0 or np.isnan(std):
            continue
        z = (s - s.mean()) / std
        out_rate = float((np.abs(z) > outlier_zscore_threshold).mean())
        outlier_rates[c] = out_rate

    # warn if any feature has too many extreme outliers
    high_outlier_cols = [c for c, r in outlier_rates.items() if r >= outlier_rate_warn]
    if high_outlier_cols:
        details.append(
            f"High outlier rate detected (|z| > {outlier_zscore_threshold} for >= {outlier_rate_warn:.0%} rows) in: {high_outlier_cols}"
        )
        status = "warn" if status != "fail" else status

    metrics["outlier_rates"] = outlier_rates
    metrics["high_outlier_cols"] = high_outlier_cols

    # ---- 4) Target distribution sanity----
    if target_col in df.columns:
        dist = profile.get("target_distribution")
        if isinstance(dist, dict):
            # If one class dominates too heavily, it may indicate labeling / filtering issues
            max_share = max(dist.values()) if dist else None
            metrics["target_max_share"] = max_share
            if max_share is not None and max_share >= 0.95:
                details.append("Target distribution is extremely imbalanced (>=95% in one class). Verify labeling/filtering.")
                status = "warn" if status != "fail" else status

    return {"status": status, "details": details, "metrics": metrics}



#### -------- DRIFT COMPARAISON ---------- #####
def compare_profiles(
    current: dict,
    baseline: dict,
    *,
    numeric_drift_warn: float = 0.20,
    numeric_drift_fail: float = 0.50,
    compare_quantile: str = "p50",
) -> dict:
    """
    Compare two profiles for drift.

    Drift rule :
    - For each numeric column, compute relative change in median (or selected quantile)
    - rel_change = |cur - base| / (|base| + eps)

    Returns drift status + per-column drift metrics.
    """
    eps = 1e-8
    details: list[str] = []
    status = "pass"
    drift = {}

    cur_num = current.get("numeric_summary", {})
    base_num = baseline.get("numeric_summary", {})

    shared_cols = sorted(set(cur_num.keys()) & set(base_num.keys()))
    for c in shared_cols:
        cur_val = cur_num[c].get(compare_quantile)
        base_val = base_num[c].get(compare_quantile)
        if cur_val is None or base_val is None:
            continue

        rel = abs(cur_val - base_val) / (abs(base_val) + eps)
        drift[c] = float(rel)

    warn_cols = [c for c, r in drift.items() if r >= numeric_drift_warn]
    fail_cols = [c for c, r in drift.items() if r >= numeric_drift_fail]

    if warn_cols:
        details.append(f"Numeric drift warning: {len(warn_cols)} columns exceed {numeric_drift_warn:.0%} relative drift on {compare_quantile}.")
        status = "warn"
    if fail_cols:
        details.append(f"Numeric drift failure: {len(fail_cols)} columns exceed {numeric_drift_fail:.0%} relative drift on {compare_quantile}.")
        status = "fail"

    return {
        "status": status,
        "details": details,
        "drift_relative_change": drift,
        "warn_cols": warn_cols,
        "fail_cols": fail_cols,
        "compare_quantile": compare_quantile,
        "thresholds": {"warn": numeric_drift_warn, "fail": numeric_drift_fail},
    }
