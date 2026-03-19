from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ml.data_pipeline.stats import build_profile


# -----------------------------
# Paths 
# -----------------------------
RAW_PATH = Path("ml/data/raw/nba.csv")
CLEAN_PATH = Path("ml/data/processed/nba_validated.parquet")  
TARGET_COL = "TARGET_5Yrs"

OUT_DIR = Path("ml/reports/visuals/raw_vs_cleaned")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Helpers
# -----------------------------
def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def build_shift_summary(raw_profile: dict[str, Any], clean_profile: dict[str, Any]) -> pd.DataFrame:
    raw_num = raw_profile.get("numeric_summary", {})
    clean_num = clean_profile.get("numeric_summary", {})

    raw_missing = raw_profile.get("missing_ratio", {})
    clean_missing = clean_profile.get("missing_ratio", {})

    features = sorted(set(raw_num.keys()) | set(clean_num.keys()))

    rows = []
    for f in features:
        r = raw_num.get(f, {})
        c = clean_num.get(f, {})

        r_mean, c_mean = _safe_float(r.get("mean")), _safe_float(c.get("mean"))
        r_std, c_std = _safe_float(r.get("std")), _safe_float(c.get("std"))
        r_p50, c_p50 = _safe_float(r.get("p50")), _safe_float(c.get("p50"))
        r_p99, c_p99 = _safe_float(r.get("p99")), _safe_float(c.get("p99"))

        r_miss = _safe_float(raw_missing.get(f))
        c_miss = _safe_float(clean_missing.get(f))

        def delta(a: float | None, b: float | None) -> float | None:
            if a is None or b is None:
                return None
            return b - a

        def delta_pct(a: float | None, b: float | None) -> float | None:
            if a is None or b is None:
                return None
            if abs(a) < 1e-12:
                return None
            return (b - a) / a * 100.0

        row = {
            "feature": f,

            "raw_mean": r_mean,
            "clean_mean": c_mean,
            "mean_delta": delta(r_mean, c_mean),
            "mean_delta_pct": delta_pct(r_mean, c_mean),

            "raw_std": r_std,
            "clean_std": c_std,
            "std_delta": delta(r_std, c_std),
            "std_delta_pct": delta_pct(r_std, c_std),

            "raw_p50": r_p50,
            "clean_p50": c_p50,
            "p50_delta": delta(r_p50, c_p50),
            "p50_delta_pct": delta_pct(r_p50, c_p50),

            "raw_p99": r_p99,
            "clean_p99": c_p99,
            "p99_delta": delta(r_p99, c_p99),
            "p99_delta_pct": delta_pct(r_p99, c_p99),

            "raw_missing": r_miss,
            "clean_missing": c_miss,
            "missing_delta": delta(r_miss, c_miss),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Simple shift score to rank changes :
    # combine abs(mean% shift) + abs(std% shift) + abs(missing shift)
    def _abs_or_zero(x: Any) -> float:
        return float(abs(x)) if pd.notna(x) else 0.0

    df["shift_score"] = (
        df["mean_delta_pct"].map(_abs_or_zero)
        + df["std_delta_pct"].map(_abs_or_zero)
        + df["missing_delta"].map(_abs_or_zero) * 100.0
    )

    df = df.sort_values("shift_score", ascending=False).reset_index(drop=True)
    return df


def save_table_image(df: pd.DataFrame, out_path: Path, title: str, max_rows: int = 12) -> None:
    show = df.head(max_rows).copy()
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


def plot_drift_bar(df_shift: pd.DataFrame, out_path: Path, top_n: int = 12) -> None:
    top = df_shift.head(top_n).copy()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title("Raw → Cleaned Drift Chart (Top Features by Shift Score)", pad=10)

    x = np.arange(len(top))
    y = top["shift_score"].astype(float).values

    ax.bar(x, y)
    ax.set_xticks(x)
    ax.set_xticklabels(top["feature"].tolist(), rotation=45, ha="right")
    ax.set_ylabel("shift_score")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_overlay_histograms(raw_df: pd.DataFrame, clean_df: pd.DataFrame, features: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in features:
        if f not in raw_df.columns or f not in clean_df.columns:
            continue
        if not np.issubdtype(raw_df[f].dtype, np.number):
            continue

        r = raw_df[f].dropna()
        c = clean_df[f].dropna()
        if len(r) < 5 or len(c) < 5:
            continue

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title(f"Distribution Overlay — {f} (Raw vs Cleaned)", pad=10)

        # Same bins for fair comparison
        all_vals = pd.concat([r, c], ignore_index=True)
        bins = min(40, max(10, int(np.sqrt(len(all_vals)))))

        ax.hist(r.values, bins=bins, alpha=0.5, label="raw")
        ax.hist(c.values, bins=bins, alpha=0.5, label="cleaned")
        ax.legend()

        plt.tight_layout()
        out_path = out_dir / f"overlay_{f}.png"
        plt.savefig(out_path, dpi=200)
        plt.close(fig)


def main() -> None:
    # ---- Load datasets ----
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_PATH}")
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {CLEAN_PATH}")

    raw_df = pd.read_csv(RAW_PATH)

    if CLEAN_PATH.suffix.lower() == ".parquet":
        clean_df = pd.read_parquet(CLEAN_PATH)
    else:
        clean_df = pd.read_csv(CLEAN_PATH)

    # ---- Build profiles ----
    raw_profile = build_profile(raw_df, target_col=TARGET_COL)
    clean_profile = build_profile(clean_df, target_col=TARGET_COL)

    # Save profiles for traceability
    (OUT_DIR / "raw_profile.json").write_text(json.dumps(raw_profile, indent=2))
    (OUT_DIR / "clean_profile.json").write_text(json.dumps(clean_profile, indent=2))

    # ---- 1) Distribution shift summary ----
    shift_df = build_shift_summary(raw_profile, clean_profile)

    csv_path = OUT_DIR / "distribution_shift_summary.csv"
    shift_df.to_csv(csv_path, index=False)

    table_path = OUT_DIR / "distribution_shift_summary.png"
    save_table_image(
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
        table_path,
        title="Distribution Shift Summary (Raw vs Cleaned) — Top Features",
        max_rows=12,
    )

    # ---- 2) Drift chart ----
    drift_chart_path = OUT_DIR / "drift_chart.png"
    plot_drift_bar(shift_df, drift_chart_path, top_n=12)

    # ---- 3) Histogram overlays for top shifted features ----
    top_features = shift_df["feature"].head(4).tolist()  # best for LinkedIn
    overlays_dir = OUT_DIR / "overlays"
    plot_overlay_histograms(raw_df, clean_df, top_features, overlays_dir)

    print("\n✅ Generated visuals:")
    print(f"- {table_path}")
    print(f"- {drift_chart_path}")
    print(f"- {overlays_dir} (overlay PNGs)")
    print(f"- {csv_path}")
    print("\nTop shifted features used for overlays:", top_features)


if __name__ == "__main__":
    main()
