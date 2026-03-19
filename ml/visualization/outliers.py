from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ml.data_pipeline.anomaly import flag_quantile_outliers, AnomalyConfig

PROCESSED_PATH = Path("ml/data/processed/nba_validated.parquet")  # adjust if needed
OUT_DIR = Path("ml/reports/visuals")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Load data ----
df = pd.read_parquet(PROCESSED_PATH)

# ---- Run anomaly detection (to get bounds + report) ----
cfg = AnomalyConfig(
    enabled=True,
    q_low=0.01,
    q_high=0.99,
    min_non_null=10,
    exclude_cols=("TARGET_5Yrs",),
    max_outlier_rate_warn=0.15,
    max_outlier_rate_fail=0.25,
)

df_out, report = flag_quantile_outliers(df, cfg)

print("Anomaly status:", report.get("status"))
print("Outlier rate:", report.get("outlier_rate"))

bounds = report.get("bounds", {})
if not bounds:
    raise ValueError("No bounds found in anomaly report. Ensure flag_quantile_outliers returns 'bounds'.")

# ---- Build flag matrix using bounds (no need for outlier_<col> columns) ----
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in cfg.exclude_cols and c in bounds]

if not num_cols:
    raise ValueError("No numeric columns found for heatmap after filtering/exclusions.")

# outlier flags per column
flags = {}
for c in num_cols:
    lo = bounds[c]["low"]
    hi = bounds[c]["high"]
    flags[c] = (df[c] < lo) | (df[c] > hi)

flag_df = pd.DataFrame(flags).astype(int)

# pick top N features by outlier rate for readability
outlier_rate_per_col = flag_df.mean().sort_values(ascending=False)
top_features = outlier_rate_per_col.head(12).index.tolist()

# limit rows so heatmap is readable
max_rows = 120
heat_df = flag_df[top_features].head(max_rows)

# ---- Plot heatmap ----
fig, ax = plt.subplots(figsize=(12, 5))
ax.set_title("Outliers Heatmap (Top Features) — 1 = Outlier, 0 = Normal", pad=10)

im = ax.imshow(heat_df.values, aspect="auto")

ax.set_yticks([])
ax.set_xticks(range(len(top_features)))
ax.set_xticklabels(top_features, rotation=45, ha="right")

plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

out_path = OUT_DIR / "outliers_heatmap.png"
plt.tight_layout()
plt.savefig(out_path, dpi=200)
plt.show()

print(f"Saved: {out_path}")
