"""
Data ingestion module for the NBA project.

This file is responsible for loading the raw dataset from its source (CSV file) and performing
 only minimal sanity checks (file existence, non-empty data, and required columns).
   It intentionally does NOT apply cleaning, transformations, feature engineering, or any training-related 
   logic—those belong to later stages of the ML lifecycle. The output of this module is a raw pandas DataFrame
     that feeds the downstream cleaning and validation steps.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = [
    "Name","GP","MIN","PTS","FGM","FGA","FG%","3P Made","3PA","3P%",
    "FTM","FTA","FT%","OREB","DREB","REB","AST","STL","BLK","TOV","TARGET_5Yrs"
]

def ingest_csv(raw_path: str) -> pd.DataFrame:
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")

    try:
        df = pd.read_csv(raw_path)
    except pd.errors.EmptyDataError as e:
        raise ValueError("Raw dataset is empty") from e

    if df.empty:
        raise ValueError("Raw dataset is empty")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df

