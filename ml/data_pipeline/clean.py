"""
Deterministic data cleaning module.

This module applies dataset-level cleaning operations that are safe to perform
before model training. All behavior is parameterized via configuration to
avoid hidden logic and ensure reproducibility across runs and environments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows deterministically."""
    return df.drop_duplicates().reset_index(drop=True)


def impute_3p_pct(
    df: pd.DataFrame,
    *,
    made_col: str = "3P Made",
    att_col: str = "3PA",
    pct_col: str = "3P%",
) -> pd.DataFrame:
    """
    Impute missing 3P% values using domain logic:

    3P% = (3P Made / 3PA) * 100 if 3PA > 0 else 0

    """
    df = df.copy()

    mask = df[pct_col].isna()
    if mask.any():
        df.loc[mask, pct_col] = df.loc[mask].apply(
            lambda row: (row[made_col] / row[att_col] * 100) if row[att_col] > 0 else 0,
            axis=1,
        )

    return df


def correct_impossible_values(
    df: pd.DataFrame,
    *,
    percent_cols: Iterable[str],
    percent_bounds: tuple[float, float],
    min_cap: float,
    non_negative_cols: Iterable[str],
) -> pd.DataFrame:
    """
    Correct clearly invalid or impossible values using deterministic rules:

    - Clip percentage columns to configured bounds # 0<values<100 
    - Cap minutes played (e.g., MIN <= 48) # values < 48 
    - Enforce non-negativity for selected numeric columns
    - Ensure made shots do not exceed attempts
    - Recompute total rebounds as OREB + DREB

    This function *corrects* values rather than dropping rows, preserving
    dataset size and ensuring stable downstream behavior.
    """
    df = df.copy()

    # ---- Percent columns ---- 
    lo, hi = percent_bounds
    for c in percent_cols:
        if c in df.columns:
            df[c] = df[c].clip(lo, hi)

    # ---- Minutes cap ----
    if "MIN" in df.columns:
        df["MIN"] = df["MIN"].clip(upper=min_cap)

    # ---- Non-negative enforcement ----
    for c in non_negative_cols:
        if c in df.columns:
            df[c] = df[c].clip(lower=0)

    # ---- Made <= Attempted constraints ----
    if {"FGM", "FGA"}.issubset(df.columns):
        df["FGM"] = np.minimum(df["FGM"], df["FGA"])

    if {"3P Made", "3PA"}.issubset(df.columns):
        df["3P Made"] = np.minimum(df["3P Made"], df["3PA"])

    if {"FTM", "FTA"}.issubset(df.columns):
        df["FTM"] = np.minimum(df["FTM"], df["FTA"])

    # ---- Rebounds identity ----
    if {"OREB", "DREB", "REB"}.issubset(df.columns):
        df["REB"] = df["OREB"] + df["DREB"]

    return df.reset_index(drop=True)


def run_cleaning(
    df: pd.DataFrame,
    *,
    do_drop_duplicates: bool,
    do_impute_3p_pct: bool,
    do_correct_impossible_values: bool,
    percent_cols: Iterable[str],
    percent_bounds: tuple[float, float],
    min_cap: float,
    non_negative_cols: Iterable[str],
) -> pd.DataFrame:
    """
    Orchestrates the data cleaning step based on configuration flags.

    This function contains no hardcoded domain values; all behavior is driven
    by config to ensure reproducibility and transparency.
    """
    if do_drop_duplicates:
        df = drop_duplicates(df)

    if do_impute_3p_pct:
        df = impute_3p_pct(df)

    if do_correct_impossible_values:
        df = correct_impossible_values(
            df,
            percent_cols=percent_cols,
            percent_bounds=percent_bounds,
            min_cap=min_cap,
            non_negative_cols=non_negative_cols,
        )

    return df
