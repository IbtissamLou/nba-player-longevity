from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Config
# -----------------------------
BASELINE_PROFILE_PATH = Path("ml/reports/baseline_profile.json")
CURRENT_PROFILE_PATH = Path("ml/reports/data_profile.json")


PROCESSED_DATA_PATH = Path("ml/data/processed/nba_validated.parquet")

OUT_DIR = Path("ml/reports/visuals")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Helpers
# -----------------------------
def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text())


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


@dataclass(frozen=True)
class DriftRow:
    feature: str
    baseline_mean: float | None
    current_mean: float | None
    mean_delta: float | None
    mean_delta_pct: float | None

    baseline_p50: float | None
    current_p50: float | None
    p50_delta: float | None
    p50_delta_pct: float | None

    baseline_p99: float | None
    current_p99: float | None
    p99_delta: float | None
    p99_delta_pct: float | None

    baseline_std: float | None
    current_std: float | None
    std_delta: float | None
    std_delta_pct: float | None

    baseline_missing: float | None
    current_missing: float | None
    missing_delta: float | None


def build_distribution_shift_summary(
    baseline_profile: dict[str, Any],
    current_profile: dict[str, Any],
) -> pd.DataFrame:
    base_num = baseline_profile.get("numeric_summary", {})
    cur_num = current_profile.get("numeric_summary", {})

    base_missing = baseline_profile.get("missing_ratio", {})
    cur_missing = current_profile.get("missing_ratio", {})

    all_features = sorted(set(base_num.keys()) | set(cur_num.keys()))

    rows: list[DriftRow] = []
    for f in all_features:
        b = base_num.get(f, {})
        c = cur_num.get(f, {})

        b_mean, c_mean = _safe_float(b.get("mean")), _safe_float(c.get("mean"))
        b_p50, c_p50 = _safe_float(b.get("p50")), _safe_float(c.get("p50"))
        b_p99, c_p99 = _safe_float(b.get("p99")), _safe_float(c.get("p99"))
        b_std, c_std = _safe_float(b.get("std")), _safe_float(c.get("std"))

        def delta(a: float | None, d: float | None) -> float | None:
            if a is None or d is None:
                return None
            return d - a

        def delta_pct(a: float | None, d: float | None) -> float | None:
            if a is None or d is None:
                return None
            if abs(a) < 1e-12:
                return None
            return (d - a) / a * 100.0

        b_miss = _safe_float(base_missing.get(f))
        c_miss = _safe_float(cur_missing.get(f))

        rows.append(
            DriftRow(
                feature=f,
                baseline_mean=b_mean,
                current_mean=c_mean,
                mean_delta=delta(b_mean, c_mean),
                mean_delta_pct=delta_pct(b_mean, c_mean),
                baseline_p50=b_p50,
                current_p50=c_p50,
                p50_delta=delta(b_p50, c_p50),
                p50_delta_pct=delta_pct(b_p50, c_p50),
                baseline_p99=b_p99,
                current_p99=c_p99,
                p99_delta=delta(b_p99, c_p99),
                p99_delta_pct=delta_pct(b_p99, c_p99),
                baseline_std=b_std,
                current_std=c_std,
                std_delta=delta(b_std, c_std),
                std_delta_pct=delta_pct(b_std, c_std),
                baseline_missing=b_miss,
                current_missing=c_miss,
                missing_delta=delta(b_miss, c_miss),
            )
        )

    df = pd.DataFrame([r.__dict__ for r in rows])

    # Add an "overall_shift_score" to rank features (simple + practical):
    # - absolute % change in mean 
    # - absolute % change in std 
    # - absolute change in missingness
    # This is not "official" drift math, but great for Data Cycle reporting.
    def _abs_or_zero(x: Any) -> float:
        return float(abs(x)) if pd.notna(x) else 0.0

    df["shift_score"] = (
        df["mean_delta_pct"].map(_abs_or_zero)
        + df["std_delta_pct"].map(_abs_or_zero)
        + (df["missing_delta"].map(_abs_or_zero) * 100.0)
    )

    df = df.sort_values("shift_score", ascending=False).reset_index(drop=True)
    return df


def save_table_as_image(df: pd.DataFrame, out_path: Path, title: str, max_rows: int = 12) -> None:
    show = df.head(max_rows).copy()

    # Round for readability
    for c in show.columns:
        if show[c].dtype.kind in "fc":
            show[c] = show[c].round(3)

    fig, ax = plt.subplots(figsize=(14, 0.6 * (len(show) + 2)))
    ax.axis("off")
    ax.set_title(title, pad=10)

    table = ax.table(
        cellText=show.values,
        colLabels=show.columns.tolist(),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_drift_bar(
    df_shift: pd.DataFrame,
    out_path: Path,
    metric_col: str = "shift_score",
    top_n: int = 12,
    title: str = "Drift Chart (Top Features by Shift Score)",
) -> None:
    top = df_shift.head(top_n).copy()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(title, pad=10)

    x = np.arange(len(top))
    y = top[metric_col].astype(float).values
    ax.bar(x, y)

    ax.set_xticks(x)
    ax.set_xticklabels(top["feature"].tolist(), rotation=45, ha="right")
    ax.set_ylabel(metric_col)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_overlay_histograms(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str],
    out_dir: Path,
    bins: int = 30,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in features:
        if f not in baseline_df.columns or f not in current_df.columns:
            continue
        if not np.issubdtype(baseline_df[f].dtype, np.number):
            continue

        b = baseline_df[f].dropna()
        c = current_df[f].dropna()
        if len(b) < 5 or len(c) < 5:
            continue

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title(f"Distribution Overlay — {f}", pad=10)

        ax.hist(b.values, bins=bins, alpha=0.5, label="baseline")
        ax.hist(c.values, bins=bins, alpha=0.5, label="current")
        ax.legend()

        plt.tight_layout()
        out_path = out_dir / f"drift_overlay_{f}.png"
        plt.savefig(out_path, dpi=200)
        plt.close(fig)


def main() -> None:
    baseline_profile = _read_json(BASELINE_PROFILE_PATH)
    current_profile = _read_json(CURRENT_PROFILE_PATH)

    # 1) Distribution Shift Summary
    shift_df = build_distribution_shift_summary(baseline_profile, current_profile)

    # Save as CSV for analysis
    csv_path = OUT_DIR / "distribution_shift_summary.csv"
    shift_df.to_csv(csv_path, index=False)

    # Save as image table (top features)
    table_img_path = OUT_DIR / "distribution_shift_summary.png"
    save_table_as_image(
        shift_df[
            [
                "feature",
                "mean_delta_pct",
                "std_delta_pct",
                "p50_delta_pct",
                "p99_delta_pct",
                "missing_delta",
                "shift_score",
            ]
        ],
        table_img_path,
        title="Distribution Shift Summary (Baseline vs Current) — Top Features",
        max_rows=12,
    )

    # 2) Drift chart (bar plot)
    drift_chart_path = OUT_DIR / "drift_chart.png"
    plot_drift_bar(
        shift_df,
        drift_chart_path,
        metric_col="shift_score",
        top_n=12,
        title="Drift Chart — Top Features by Shift Score (Baseline vs Current)",
    )

    print(f"Saved: {csv_path}")
    print(f"Saved: {table_img_path}")
    print(f"Saved: {drift_chart_path}")


if __name__ == "__main__":
    main()
