from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ml.data_pipeline.stats import build_profile

# ---- Paths (update if your filenames differ) ----
PROCESSED_PATH = Path("ml/data/processed/nba_validated.parquet")  # or your processed artifact
OUT_DIR = Path("ml/reports/visuals")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Load data ----
df = pd.read_parquet(PROCESSED_PATH)

# ---- Build profile (same function you use in pipeline) ----
profile = build_profile(df, target_col="TARGET_5Yrs")

# ---- Select columns to display in the table (pick the most important features) ----
cols_to_show = [
    "GP", "MIN", "PTS", "FG%", "3P%", "FT%", "REB", "AST", "STL", "TOV"
]
cols_to_show = [c for c in cols_to_show if c in profile["numeric_summary"]]

# Build a small table DataFrame
rows = []
for c in cols_to_show:
    s = profile["numeric_summary"][c]
    rows.append(
        {
            "feature": c,
            "mean": s["mean"],
            "std": s["std"],
            "p01": s["p01"],
            "p50": s["p50"],
            "p99": s["p99"],
        }
    )

summary_df = pd.DataFrame(rows)

# Optional: round for readability
summary_df_rounded = summary_df.copy()
for col in ["mean", "std", "p01", "p50", "p99"]:
    summary_df_rounded[col] = summary_df_rounded[col].astype(float).round(3)

# ---- Plot as a matplotlib table ----
fig, ax = plt.subplots(figsize=(12, 3.5))
ax.axis("off")
ax.set_title("Statistical Profile Summary (Processed Dataset)", pad=10)

table = ax.table(
    cellText=summary_df_rounded.values,
    colLabels=summary_df_rounded.columns,
    cellLoc="center",
    loc="center",
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

out_path = OUT_DIR / "stat_profile_summary.png"
plt.tight_layout()
plt.savefig(out_path, dpi=200)
plt.show()

print(f"Saved: {out_path}")
