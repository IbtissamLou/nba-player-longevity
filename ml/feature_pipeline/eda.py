from __future__ import annotations

"""
EDA reporting (Feature Cycle).

This module is intentionally separated from sklearn Pipelines.

Why:
- EDA is for understanding + reporting (plots, summaries).
- It should NOT mutate training features automatically.
- Transformations like PowerTransformer/log should be applied in the TRAINING pipeline
  (fit on train only, reused at inference).

Outputs:
- target balance plot
- numeric distributions plot grid
- skewness report + transformation suggestions
- univariate feature scoring report (ranking only)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import skew
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif


# ----------------------------
# Config objects (typed + explicit)
# ----------------------------

@dataclass(frozen=True)
class SkewConfig:
    high_threshold: float = 1.0
    moderate_threshold: float = 0.5
    zero_ratio_threshold: float = 0.30


@dataclass(frozen=True)
class ScoringConfig:
    enabled: bool = True
    k_best: int = 10
    method: str = "f_classif"  # "f_classif" or "mutual_info"


@dataclass(frozen=True)
class EDAConfig:
    enabled: bool = True
    out_dir: str = "ml/reports/eda"
    exclude_cols: tuple[str, ...] = ("Name", "TARGET_5Yrs")
    target_col: str = "TARGET_5Yrs"
    skew: SkewConfig = SkewConfig()
    scoring: ScoringConfig = ScoringConfig()


# ----------------------------
# Plot helpers
# ----------------------------

def plot_target_balance(df: pd.DataFrame, target_col: str, out_path: Path) -> dict[str, float]:
    """
    Plot class balance (target distribution).

    Why:
    - classification performance is strongly affected by imbalance
    - this plot is part of every ML baseline report
    """
    # Calculate normalized distribution
    dist = df[target_col].value_counts(normalize=True).sort_index() * 100.0

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=dist.index.astype(str), y=dist.values, ax=ax)
    ax.set_title("Target distribution (%)")
    ax.set_xlabel(target_col)
    ax.set_ylabel("Frequency (%)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    return {str(k): float(v) for k, v in dist.to_dict().items()}


def plot_numeric_distributions(df: pd.DataFrame, numeric_cols: list[str], out_path: Path, max_cols: int = 20) -> None:
    """
    Plot distributions for numeric variables in one grid.

    Why:
    - quick scan for skewness, weird spikes, and outliers
    """
    cols = numeric_cols[:max_cols]  # limit to keep plot readable
    if not cols:
        return

    n = len(cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(cols):
        ax = axes[i]
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


# ----------------------------
# Skewness + transformation suggestion (report only)
# ----------------------------

def compute_skewness_report(df: pd.DataFrame, cols: list[str], cfg: SkewConfig) -> pd.DataFrame:
    """
    Compute skewness and suggest transformations (report only).

    IMPORTANT:
    We do NOT apply transformations here.
    We only classify and suggest based on skewness/zero inflation.

    Transformations must be applied later in the training pipeline.
    """
    rep_rows = []
    n = len(df)

    for col in cols:
        s = df[col].dropna()
        if s.nunique() < 2:
            rep_rows.append(
                {"feature": col, "skewness": None, "zero_ratio": None, "category": "constant_or_empty", "suggestion": "drop_or_ignore"}
            )
            continue

        sk = float(skew(s))
        zero_ratio = float((df[col] == 0).sum() / n)

        # classify skewness intensity
        if sk >= cfg.high_threshold and  zero_ratio > 0:
            category = "high_pos_zero_skew"
            suggestion = "log1p"
        elif sk >= cfg.high_threshold and  zero_ratio < 0:
            category = "high_pos_skew"
            suggestion = "yeo_johnson"
        elif sk <= -cfg.high_threshold:
            category = "high_neg_skew"
            suggestion = "yeo_johnson"
        elif abs(sk) >= cfg.moderate_threshold:
            category = "moderate_skew"
            suggestion = "yeo_johnson"
        else:
            category = "low_skew"
            suggestion = "none"

        # special case: too many zeros -> often needs log1p or robust handling
        if zero_ratio >= cfg.zero_ratio_threshold and abs(sk) >= cfg.high_threshold:
            category = "zero_inflated_high_skew"
            suggestion = "log1p"

        rep_rows.append(
            {
                "feature": col,
                "skewness": sk,
                "zero_ratio": zero_ratio,
                "category": category,
                "suggestion": suggestion,
            }
        )

    return pd.DataFrame(rep_rows).sort_values(by=["category", "feature"]).reset_index(drop=True)


# ----------------------------
# Univariate scoring (selection report only)
# ----------------------------

def compute_feature_scores(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    cfg: ScoringConfig,
) -> pd.DataFrame:
    """
    Compute univariate feature scores to rank features.

    IMPORTANT:
    - This is NOT feature selection in production.
    - It's an EDA ranking tool to understand which features correlate with the target.
    """
    if not cfg.enabled or not feature_cols:
        return pd.DataFrame()

    X = df[feature_cols]
    y = df[target_col].astype(int)

    # Choose scoring function
    if cfg.method == "mutual_info":
        # mutual information works for non-linear relationships too
        scores = mutual_info_classif(X, y, random_state=42)
        out = pd.DataFrame({"feature": feature_cols, "score": scores})
        out = out.sort_values("score", ascending=False).reset_index(drop=True)
        return out

    # default: f_classif (ANOVA)
    selector = SelectKBest(score_func=f_classif, k=min(cfg.k_best, len(feature_cols)))
    selector.fit(X, y)

    f_scores = selector.scores_
    p_vals = selector.pvalues_

    out = pd.DataFrame(
        {"feature": feature_cols, "f_score": f_scores, "p_value": p_vals}
    ).sort_values("f_score", ascending=False).reset_index(drop=True)

    return out


# ----------------------------
# Main EDA runner (writes artifacts)
# ----------------------------

def run_eda(df: pd.DataFrame, cfg: EDAConfig) -> dict[str, Any]:
    """
    Run EDA and write artifacts.

    Outputs (in cfg.out_dir):
    - target_balance.png
    - numeric_distributions.png
    - skewness_report.csv
    - feature_scores.csv
    - eda_summary.json (high-level metadata)
    """
    summary: dict[str, Any] = {"enabled": cfg.enabled, "status": "pass", "details": []}
    if not cfg.enabled:
        return summary

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Identify numeric feature columns to analyze ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in cfg.exclude_cols]

    if cfg.target_col not in df.columns:
        summary["status"] = "fail"
        summary["details"].append(f"Target column missing: {cfg.target_col}")
        return summary

    # (1) Target balance plot
    balance = plot_target_balance(df, cfg.target_col, out_dir / "target_balance.png")
    summary["target_balance_pct"] = balance

    # (2) Distributions grid
    plot_numeric_distributions(df, numeric_cols, out_dir / "numeric_distributions.png", max_cols=20)

    # (3) Skewness report (and transformation suggestions)
    skew_report = compute_skewness_report(df, numeric_cols, cfg.skew)
    skew_report.to_csv(out_dir / "skewness_report.csv", index=False)

    # (4) Feature score ranking (uses target; EDA only)
    scores = compute_feature_scores(df, cfg.target_col, numeric_cols, cfg.scoring)
    if not scores.empty:
        scores.to_csv(out_dir / "feature_scores.csv", index=False)

    # (5) Summary JSON for quick inspection
    summary["n_rows"] = int(df.shape[0])
    summary["n_cols"] = int(df.shape[1])
    summary["n_numeric_features_analyzed"] = int(len(numeric_cols))
    summary["outputs"] = {
        "target_balance_png": str(out_dir / "target_balance.png"),
        "numeric_distributions_png": str(out_dir / "numeric_distributions.png"),
        "skewness_report_csv": str(out_dir / "skewness_report.csv"),
        "feature_scores_csv": str(out_dir / "feature_scores.csv") if (out_dir / "feature_scores.csv").exists() else None,
    }

    return summary
