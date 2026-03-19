from __future__ import annotations

"""
Correlation checks for Feature Cycle.

Purpose:
- detect redundant features (high correlation)
- improve interpretability (especially for linear models)
- catch data issues (duplicate-like columns)

IMPORTANT:
- This module is for *reporting* and *quality checks*.
- It does not drop features automatically.
  Dropping should be a modeling decision (training cycle) or done via a controlled pipeline transformer.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class CorrelationConfig:
    enabled: bool = True
    threshold: float = 0.95              # report pairs above this absolute correlation
    method: str = "pearson"              # pearson|spearman
    exclude_cols: tuple[str, ...] = ("TARGET_5Yrs",)
    max_features_heatmap: int = 25       # keep visualization readable
    save_heatmap: bool = True


def compute_correlation_matrix(
    df: pd.DataFrame,
    cfg: CorrelationConfig,
) -> pd.DataFrame:
    """
    Compute correlation matrix on numeric columns only,
    excluding configured columns (target, IDs, etc.).
    """
    # Select numeric columns only (correlation only makes sense for numeric data here)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude target or any user-defined columns
    num_cols = [c for c in num_cols if c not in cfg.exclude_cols]

    if len(num_cols) < 2:
        # Not enough columns to compute correlation
        return pd.DataFrame()

    corr = df[num_cols].corr(method=cfg.method)
    return corr


def extract_high_corr_pairs(corr: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Extract pairs of columns with abs(correlation) >= threshold.
    We take only the upper triangle to avoid duplicates like (A,B) and (B,A).
    """
    if corr.empty:
        return pd.DataFrame(columns=["feature_a", "feature_b", "corr", "abs_corr"])

    # Mask lower triangle + diagonal
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    corr_upper = corr.where(mask)

    rows = []
    for col in corr_upper.columns:
        s = corr_upper[col].dropna()
        for idx, value in s.items():
            abs_corr = abs(value)
            if abs_corr >= threshold:
                rows.append(
                    {
                        "feature_a": idx,
                        "feature_b": col,
                        "corr": float(value),
                        "abs_corr": float(abs_corr),
                    }
                )

    out = pd.DataFrame(rows).sort_values("abs_corr", ascending=False).reset_index(drop=True)
    return out


def save_heatmap(
    corr: pd.DataFrame,
    out_path: Path,
    *,
    max_features: int = 25,
) -> None:
    """
    Save a correlation heatmap as a PNG.

    We keep it readable by limiting to max_features:
    - if there are too many columns, show the first N (sorted by variance)
    """
    if corr.empty:
        raise ValueError("Correlation matrix is empty; cannot generate heatmap.")

    # Reduce size if too many features
    corr_to_plot = corr.copy()
    if corr_to_plot.shape[0] > max_features:
        corr_to_plot = corr_to_plot.iloc[:max_features, :max_features]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Feature Correlation Heatmap", pad=10)

    im = ax.imshow(corr_to_plot.values, aspect="auto")

    ax.set_xticks(range(corr_to_plot.shape[1]))
    ax.set_yticks(range(corr_to_plot.shape[0]))
    ax.set_xticklabels(corr_to_plot.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr_to_plot.index, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def run_correlation_checks(
    df: pd.DataFrame,
    cfg: CorrelationConfig,
    *,
    out_matrix_csv: str,
    out_pairs_csv: str,
    out_heatmap_png: str | None = None,
) -> dict[str, Any]:
    """
    Run correlation checks and write outputs.

    Returns a small summary dict for your pipeline logs/reports.
    """
    summary = {
        "enabled": cfg.enabled,
        "status": "pass",
        "method": cfg.method,
        "threshold": cfg.threshold,
        "n_features_used": 0,
        "n_high_corr_pairs": 0,
        "outputs": {
            "matrix_csv": out_matrix_csv,
            "pairs_csv": out_pairs_csv,
            "heatmap_png": out_heatmap_png,
        },
        "details": [],
    }

    if not cfg.enabled:
        return summary

    corr = compute_correlation_matrix(df, cfg)
    if corr.empty:
        summary["status"] = "warn"
        summary["details"].append("Not enough numeric columns to compute correlation.")
        return summary

    summary["n_features_used"] = int(corr.shape[0])

    # Save full correlation matrix
    Path(out_matrix_csv).parent.mkdir(parents=True, exist_ok=True)
    corr.to_csv(out_matrix_csv)

    # Extract + save high-correlation pairs
    pairs = extract_high_corr_pairs(corr, cfg.threshold)
    pairs.to_csv(out_pairs_csv, index=False)

    summary["n_high_corr_pairs"] = int(pairs.shape[0])

    if summary["n_high_corr_pairs"] > 0:
        # Correlation isn't always "bad", but it's important to review.
        summary["status"] = "warn"
        summary["details"].append(
            f"Found {summary['n_high_corr_pairs']} feature pairs with abs(corr) >= {cfg.threshold}."
        )

    # Save heatmap (optional)
    if cfg.save_heatmap and out_heatmap_png:
        save_heatmap(corr, Path(out_heatmap_png), max_features=cfg.max_features_heatmap)

    return summary
