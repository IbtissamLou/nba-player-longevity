"""
Schema validation + domain rule validation for the dataset.

- Schema validation (Pandera) enforces: expected columns, types, and basic constraints.
- Domain validation rules enforce: logical relationships between columns (e.g., made <= attempted).

This module is designed to fail fast to prevent silent data corruption from reaching training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from ml.data_pipeline.schemas.nba import NBA_SCHEMA, enforce_column_order, TARGET_COL


# -------------------------
# Schema validation
# -------------------------

def validate_schema(df: pd.DataFrame, *, enforce_order: bool = True) -> pd.DataFrame:
    """
    Validate a DataFrame against the canonical Pandera schema.

    What it guarantees:
    - All expected columns exist 
    - Column dtypes are correct 
    - Basic constraints embedded in schema 

    Args:
        df: Input dataframe.
        enforce_order: If True, reorders columns to the canonical order defined in the schema module.

    Returns:
        Validated (and possibly type-coerced) dataframe with canonical column order.
    """
    validated = NBA_SCHEMA.validate(df)
    return enforce_column_order(validated) if enforce_order else validated


# -------------------------
# Domain / business rules
# -------------------------

@dataclass(frozen=True) #Makes the config immutable
class ValidationConfig:
    """
    Configuration container for rule validation.

    This keeps rule thresholds centralized and testable.
    """
    enforce_ranges: bool = True
    enforce_made_leq_attempted: bool = True
    enforce_rebounds_identity: bool = True
    rebounds_tolerance: float = 0.0

    # configurable thresholds (avoid hardcoding)
    min_cap: float = 48.0
    percent_bounds: tuple[float, float] = (0.0, 100.0)
    non_negative_cols: tuple[str, ...] = ("PTS",)  


def validate_rules(df: pd.DataFrame, cfg: ValidationConfig) -> None:
    """
    Enforce domain/business logic rules.

    This function raises ValueError on failure (hard fail) to stop the pipeline
    when data violates logical constraints.

    Args:
        df: Validated dataframe .
        cfg: ValidationConfig containing rule toggles and thresholds.

    Raises:
        ValueError: If one or more rules fail.
    """
    errors: list[str] = []

    # --- Range checks ---
    if cfg.enforce_ranges:
        # MIN cap
        if "MIN" in df.columns and (df["MIN"] > cfg.min_cap).any():
            errors.append(f"MIN has values > {cfg.min_cap}")

        # Percent bounds
        lo, hi = cfg.percent_bounds
        for c in ("FG%", "3P%", "FT%"):
            if c in df.columns and ((df[c] < lo) | (df[c] > hi)).any():
                errors.append(f"{c} has values outside [{lo},{hi}]")

        # Non-negativity checks (configurable)
        for c in cfg.non_negative_cols:
            if c in df.columns and (df[c] < 0).any():
                errors.append(f"{c} has negative values")

    # --- Made <= Attempted checks ---
    if cfg.enforce_made_leq_attempted:
        pairs = [("FGM", "FGA"), ("3P Made", "3PA"), ("FTM", "FTA")]
        for made, att in pairs:
            if made in df.columns and att in df.columns and (df[made] > df[att]).any():
                errors.append(f"{made} > {att} exists")

    # --- Rebounds identity ---
    if cfg.enforce_rebounds_identity:
        required = {"OREB", "DREB", "REB"}
        if required.issubset(df.columns):
            diff = (df["OREB"] + df["DREB"] - df["REB"]).abs()
            if (diff > cfg.rebounds_tolerance).any():
                errors.append(f"OREB + DREB != REB (tolerance={cfg.rebounds_tolerance})")
        else:
            errors.append("Rebounds identity check requested but required columns are missing")


    # --- Target sanity ---
    if TARGET_COL in df.columns:
        if not df[TARGET_COL].isin([0, 1]).all():
            errors.append(f"{TARGET_COL} contains values other than 0/1")
    else:
        errors.append(f"Missing target column: {TARGET_COL}")

    if errors:
        raise ValueError("Validation rules failed:\n- " + "\n- ".join(errors))
