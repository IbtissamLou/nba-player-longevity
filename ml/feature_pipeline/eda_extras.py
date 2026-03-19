from __future__ import annotations

"""
EDA Extras: visual reports that complement EDA outputs.

This module generates:
- Missingness heatmap (visual)
- Class-conditional distributions (target=0 vs target=1)
- Simple model-based feature importance (tiny RF baseline)
- PCA explained variance plot

All of these are REPORTS (not preprocessing).
Transformations suggested by EDA must be implemented later in Training Cycle pipelines.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier


@dataclass(frozen=True)
class MissingnessHeatmapConfig:
    enabled: bool = True
    max_cols: int = 30


@dataclass(frozen=True)
class ClassConditionalConfig:
    enabled: bool = True
    max_cols: int = 12  # plot top N numeric features


@dataclass(frozen=True)
class FeatureImportanceConfig:
    enabled: bool = True
    model: str = "random_forest"
    n_estimators: int = 50
    max_depth: int | None = 5
    top_k: int = 15


@dataclass(frozen=True)
class PCAConfig:
    enabled: bool = True
    n_components: int = 10


@dataclass(frozen=True)
class EDAExtrasConfig:
    enabled: bool = True
    out_dir: str = "ml/reports/eda"
    target_col: str = "TARGET_5Yrs"
    exclude_cols: tuple[str, ...] = ("Name", "TARGET_5Yrs")

    missingness_heatmap: MissingnessHeatmapConfig = MissingnessHeatmapConfig()
    class_conditional: ClassConditionalConfig = ClassConditionalConfig()
    feature_importance: FeatureImportanceConfig = FeatureImportanceConfig()
    pca: PCAConfig = PCAConfig()


# ------------------------
# 1) Missingness Heatmap
# ------------------------

def plot_missingness_bar(df: pd.DataFrame, cols: list[str], out_path: Path) -> pd.DataFrame:
    """
    Always-useful missingness summary:
    - shows % missing per feature (sorted)
    - if everything is ~0%, we'll see it clearly
    """
    if not cols:
        return pd.DataFrame()

    miss = df[cols].isna().mean().sort_values(ascending=False) * 100.0
    miss_df = miss.reset_index()
    miss_df.columns = ["feature", "missing_pct"]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=miss_df.head(30), x="missing_pct", y="feature", ax=ax)
    ax.set_title("Missingness per Feature (%) — Top 30")
    ax.set_xlabel("Missing (%)")
    ax.set_ylabel("")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    return miss_df


def plot_missingness_heatmap(df: pd.DataFrame, cols: list[str], out_path: Path, *, max_rows: int = 300) -> None:
    """
    Missingness heatmap (only useful if there is missingness).
    - plots ONLY columns that have missing > 0
    - samples rows for readability
    - uses a strong contrast colormap so missing stands out
    """
    if not cols:
        return

    miss_ratio = df[cols].isna().mean()
    cols_with_missing = miss_ratio[miss_ratio > 0].sort_values(ascending=False).index.tolist()

    # If no missingness -> skip heatmap (it will be a black rectangle)
    if not cols_with_missing:
        return

    # Sample rows for readability (otherwise huge dataset makes unreadable blocks)
    plot_df = df[cols_with_missing].isna()
    if plot_df.shape[0] > max_rows:
        plot_df = plot_df.sample(n=max_rows, random_state=42)

    # Convert boolean to int (0/1) for a clearer heatmap
    plot_mat = plot_df.astype(int)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        plot_mat,
        cbar=True,
        ax=ax,
        vmin=0,
        vmax=1,
        cmap="viridis",
        yticklabels=False,
    )
    ax.set_title("Missingness Heatmap (1 = missing, 0 = present)")
    ax.set_xlabel("Features")
    ax.set_ylabel("Sampled rows")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


# ------------------------
# 2) Class-Conditional Distributions
# ------------------------

def plot_class_conditional_distributions(
    df: pd.DataFrame,
    target_col: str,
    cols: list[str],
    out_path: Path,
    *,
    max_cols: int = 12,
) -> None:
    """
    Small-multiples plot: one subplot per feature.
    Much clearer than a single violin plot with mixed scales.
    """
    cols = cols[:max_cols]
    if not cols:
        return

    plot_df = df[[target_col] + cols].copy()
    plot_df[target_col] = plot_df[target_col].astype(int)

    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(cols):
        ax = axes[i]

        # Two distributions: target=0 and target=1
        for label in [0, 1]:
            s = plot_df.loc[plot_df[target_col] == label, col].dropna()
            if len(s) > 2:
                sns.kdeplot(s, ax=ax, label=str(label), fill=False)
            else:
                # fallback for tiny samples
                sns.histplot(s, ax=ax, bins=20, stat="density", label=str(label), element="step", fill=False)

        ax.set_title(col)
        ax.legend(title=target_col)
        ax.set_xlabel("")
        ax.set_ylabel("density")

    # Turn off unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


# ------------------------
# 3) Feature Importance (Tiny Baseline Model)
# ------------------------

def plot_feature_importance_rf(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    out_path: Path,
    *,
    n_estimators: int = 50,
    max_depth: int | None = 5,
    top_k: int = 15,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Train a tiny RandomForest just to rank features (EDA only).
    - Not tuned
    - Not used as final model
    - Helps identify strongest signals quickly
    """
    X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0.0)
    y = df[target_col].astype(int)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X, y)

    importances = pd.DataFrame(
        {"feature": X.columns, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False)

    top = importances.head(top_k)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top, x="importance", y="feature", ax=ax)
    ax.set_title("Feature Importance (Tiny RandomForest baseline)")
    ax.set_xlabel("importance")
    ax.set_ylabel("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    return importances


# ------------------------
# 4) PCA Variance Plot
# ------------------------

def plot_pca_explained_variance(
    df: pd.DataFrame,
    feature_cols: list[str],
    out_path: Path,
    *,
    n_components: int = 10,
) -> pd.DataFrame:
    """
    PCA explained variance (EDA only).
    Shows how much variance is captured by first k components.
    """
    X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0.0)

    n_components = min(n_components, X.shape[1])
    if n_components < 2:
        return pd.DataFrame()

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X)

    var = pd.DataFrame(
        {
            "component": list(range(1, n_components + 1)),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative": np.cumsum(pca.explained_variance_ratio_),
        }
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(var["component"], var["cumulative"], marker="o")
    ax.set_title("PCA Cumulative Explained Variance")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    return var


# ------------------------
# Main Runner
# ------------------------

def run_eda_extras(df: pd.DataFrame, cfg: EDAExtrasConfig) -> dict[str, Any]:
    """
    Run extra EDA plots and return paths + small summary.
    """
    summary: dict[str, Any] = {"enabled": cfg.enabled, "status": "pass", "outputs": {}}
    if not cfg.enabled:
        return summary

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Select numeric feature columns (exclude target + IDs)
    feature_cols = [c for c in df.columns if c not in cfg.exclude_cols]
    num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    # (1) Missingness heatmap
    if cfg.missingness_heatmap.enabled:
        cols = num_cols[: cfg.missingness_heatmap.max_cols]

        # Always generate the bar chart
        bar_path = out_dir / "missingness_bar.png"
        miss_df = plot_missingness_bar(df, cols, bar_path)
        miss_df.to_csv(out_dir / "missingness_summary.csv", index=False)
        summary["outputs"]["missingness_bar_png"] = str(bar_path)
        summary["outputs"]["missingness_summary_csv"] = str(out_dir / "missingness_summary.csv")

        # Heatmap only if there is missingness
        heat_path = out_dir / "missingness_heatmap.png"
        plot_missingness_heatmap(df, cols, heat_path, max_rows=300)
        if heat_path.exists():
            summary["outputs"]["missingness_heatmap_png"] = str(heat_path)

    # (2) Class conditional distributions
    if cfg.class_conditional.enabled and cfg.target_col in df.columns:
        cols = num_cols[: cfg.class_conditional.max_cols]
        out_path = out_dir / "class_conditional_distributions.png"
        plot_class_conditional_distributions(
            df,
            cfg.target_col,
            cols,
            out_path,
            max_cols=cfg.class_conditional.max_cols,
        )
        summary["outputs"]["class_conditional_png"] = str(out_path)


    # (3) Feature importance
    if cfg.feature_importance.enabled and cfg.target_col in df.columns and len(num_cols) > 1:
        out_path = out_dir / "feature_importance.png"
        importances = plot_feature_importance_rf(
            df,
            cfg.target_col,
            num_cols,
            out_path,
            n_estimators=cfg.feature_importance.n_estimators,
            max_depth=cfg.feature_importance.max_depth,
            top_k=cfg.feature_importance.top_k,
        )
        # Save the full importance table
        importances.to_csv(out_dir / "feature_importance.csv", index=False)
        summary["outputs"]["feature_importance_png"] = str(out_path)
        summary["outputs"]["feature_importance_csv"] = str(out_dir / "feature_importance.csv")

    # (4) PCA variance plot
    if cfg.pca.enabled and len(num_cols) > 2:
        out_path = out_dir / "pca_explained_variance.png"
        var = plot_pca_explained_variance(df, num_cols, out_path, n_components=cfg.pca.n_components)
        if not var.empty:
            var.to_csv(out_dir / "pca_explained_variance.csv", index=False)
            summary["outputs"]["pca_variance_png"] = str(out_path)
            summary["outputs"]["pca_variance_csv"] = str(out_dir / "pca_explained_variance.csv")

    return summary
