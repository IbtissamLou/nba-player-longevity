from __future__ import annotations

"""
Leakage guardrails for the Feature Cycle.

Why this exists:
- Leakage can make offline results look amazing while production performance collapses.
- These checks are designed to fail fast on obvious leakage patterns:
  * IDs used as features
  * accidental target inclusion
  * target proxy features (high correlation with label)
  * duplicate columns from joins/copying

This module is intentionally model-agnostic and runs before training.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LeakageConfig:
    """Config object so leakage behavior is explicit and testable."""
    enabled: bool = True

    # Label column
    target_col: str = "TARGET_5Yrs"

    # Identifiers (drop from feature set to prevent memorization)
    id_cols: tuple[str, ...] = ("Name",)

    # Columns never allowed as features
    forbidden_feature_cols: tuple[str, ...] = ("TARGET_5Yrs",)

    # Controls whether we automatically drop id columns from the feature artifact
    drop_id_cols: bool = True

    # Detect “proxy leakage” by suspiciously high correlation with target
    check_target_proxy: bool = True
    max_abs_target_corr_warn: float = 0.85
    max_abs_target_corr_fail: float = 0.95

    # Detect exact duplicate columns (common join bug)
    check_duplicate_columns: bool = True

    required_cols: tuple[str, ...] = ()


def _to_serializable(x: Any) -> Any:
    """Make numpy types JSON serializable (for reports)."""
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def build_leakage_report(df: pd.DataFrame, cfg: LeakageConfig) -> dict[str, Any]:
    """
    Build a leakage report with a simple status:
      - pass: no issues found
      - warn: suspicious patterns but not guaranteed leakage
      - fail: strong evidence of leakage or broken data contract

    """
    report: dict[str, Any] = {
        "enabled": cfg.enabled,
        "status": "pass",
        "details": [],
        "target_col": cfg.target_col,
        "id_cols": list(cfg.id_cols),
        "forbidden_feature_cols": list(cfg.forbidden_feature_cols),
        "thresholds": {
            "max_abs_target_corr_warn": cfg.max_abs_target_corr_warn,
            "max_abs_target_corr_fail": cfg.max_abs_target_corr_fail,
        },
        "high_corr_to_target": {},    # filled if proxy candidates found
        "duplicate_columns": [],      # filled if duplicates detected
    }

    if not cfg.enabled:
        return report

    # ---- (0) Required columns contract check (fail fast) ----
    missing_required = [c for c in cfg.required_cols if c not in df.columns]
    if missing_required:
        report["status"] = "fail"
        report["details"].append(f"Missing required columns: {missing_required}")
        return report

    # ---- (1) Ensure target exists ----
    if cfg.target_col not in df.columns:
        report["status"] = "fail"
        report["details"].append(f"Target column '{cfg.target_col}' not found.")
        return report

    # ---- (2) Duplicate columns detection (join/copy bug) ----
    if cfg.check_duplicate_columns:
        # We hash each column content; identical hashes => identical column values (very likely duplicates).
        hashes: dict[int, str] = {}
        dup_pairs: list[tuple[str, str]] = []

        for col in df.columns:
            h = int(pd.util.hash_pandas_object(df[col], index=False).sum())
            if h in hashes:
                dup_pairs.append((hashes[h], col))
            else:
                hashes[h] = col

        if dup_pairs:
            report["duplicate_columns"] = [{"col_a": a, "col_b": b} for a, b in dup_pairs]
            # Duplicates are suspicious and often indicate a pipeline bug.
            report["status"] = "warn"
            report["details"].append(f"Found duplicate columns (possible join/copy bug): {dup_pairs}")

    # ---- (3) Target proxy detection using correlation ----
    if cfg.check_target_proxy:
        y = df[cfg.target_col]

        # Only meaningful if target is binary {0,1} 
        if not set(y.dropna().unique()).issubset({0, 1}):
            report["status"] = "fail"
            report["details"].append("Target is not binary {0,1}; cannot run proxy checks safely.")
            return report

        # Check numeric features only; correlation is most meaningful there
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_cols if c != cfg.target_col]

        high_corr: dict[str, dict[str, float]] = {}
        for c in num_cols:
            s = df[c]

            # Skip constant or empty columns
            if s.isna().all() or s.nunique(dropna=True) < 2:
                continue

            corr = float(pd.Series(s).corr(pd.Series(y)))
            if np.isnan(corr):
                continue

            abs_corr = abs(corr)
            if abs_corr >= cfg.max_abs_target_corr_warn:
                high_corr[c] = {"corr": corr, "abs_corr": abs_corr}

        if high_corr:
            report["high_corr_to_target"] = high_corr

            max_abs = max(v["abs_corr"] for v in high_corr.values())
            if max_abs >= cfg.max_abs_target_corr_fail:
                report["status"] = "fail"
                report["details"].append(
                    f"Target proxy detected: max_abs_corr={max_abs:.3f} >= fail threshold "
                    f"({cfg.max_abs_target_corr_fail})."
                )
            else:
                # Only warn if we didn’t already fail
                if report["status"] != "fail":
                    report["status"] = "warn"
                report["details"].append(
                    f"Potential target proxies found: max_abs_corr={max_abs:.3f} >= warn threshold "
                    f"({cfg.max_abs_target_corr_warn})."
                )

    return report


def apply_leakage_guards(
    df: pd.DataFrame,
    cfg: LeakageConfig,
    *,
    write_report_path: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Apply leakage guardrails and optionally write a JSON report.

    Behavior:
    - builds leakage report
    - if status == fail -> raise ValueError (fail-fast)
    - drops ID columns if configured (prevents memorization leakage)
    - removes forbidden feature columns (except target, which we keep for y)
    """
    report = build_leakage_report(df, cfg)

    # Persist report so the pipeline is auditable/debuggable
    if write_report_path:
        out = Path(write_report_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=_to_serializable))

    if not cfg.enabled:
        return df, report

    # Fail-fast on strong leakage evidence
    if report["status"] == "fail":
        raise ValueError("Leakage guard failed:\n- " + "\n- ".join(report["details"]))

    df_safe = df.copy()

    # Drop identifier columns so the model cannot memorize player identity
    if cfg.drop_id_cols:
        drop_cols = [c for c in cfg.id_cols if c in df_safe.columns]
        if drop_cols:
            df_safe = df_safe.drop(columns=drop_cols)

    # Remove forbidden columns except the target (target is needed to create y later)
    for col in cfg.forbidden_feature_cols:
        if col in df_safe.columns and col != cfg.target_col:
            df_safe = df_safe.drop(columns=[col])

    return df_safe, report
