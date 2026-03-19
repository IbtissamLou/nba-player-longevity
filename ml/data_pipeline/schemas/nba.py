"""
Canonical dataset schema (contract) for the NBA tabular dataset.

This module enforces schema consistency (expected columns + types + constraints)
and exposes a single source of truth for feature/label column names.
"""

from __future__ import annotations

import pandera as pa
from pandera import Column, Check

# ---- Single source of truth for columns (prevents drift + misalignment) ----

ID_COL = "Name"
TARGET_COL = "TARGET_5Yrs"

FEATURE_COLS: list[str] = [ #a stable feature list
    "GP", "MIN", "PTS",
    "FGM", "FGA", "FG%",
    "3P Made", "3PA", "3P%",
    "FTM", "FTA", "FT%",
    "OREB", "DREB", "REB",
    "AST", "STL", "BLK", "TOV",
]

ALL_COLS: list[str] = [ID_COL] + FEATURE_COLS + [TARGET_COL] #a stable order


# ---- Pandera schema: columns + types + basic validity constraints ----
NBA_SCHEMA = pa.DataFrameSchema(
    columns={
        ID_COL: Column(str, nullable=True), #missing names are allowed

        "GP": Column(int, Check.ge(0)),
        "MIN": Column(float, Check.ge(0)),
        "PTS": Column(float, Check.ge(0)),

        "FGM": Column(float, Check.ge(0)),
        "FGA": Column(float, Check.ge(0)),
        "FG%": Column(float, Check.between(0, 100)),

        "3P Made": Column(float, Check.ge(0)),
        "3PA": Column(float, Check.ge(0)),
        "3P%": Column(float, Check.between(0, 100)),

        "FTM": Column(float, Check.ge(0)),
        "FTA": Column(float, Check.ge(0)),
        "FT%": Column(float, Check.between(0, 100)),

        "OREB": Column(float, Check.ge(0)),
        "DREB": Column(float, Check.ge(0)),
        "REB": Column(float, Check.ge(0)),

        "AST": Column(float, Check.ge(0)),
        "STL": Column(float, Check.ge(0)),
        "BLK": Column(float, Check.ge(0)),
        "TOV": Column(float, Check.ge(0)),

        TARGET_COL: Column(int, Check.isin([0, 1])),
    },
    coerce=True,   # cast types when possible (CSV often loads ints as floats)
    strict=True,   # fail fast if any column is missing or any extra column appears
)


def enforce_column_order(df):
    """
    Enforce a deterministic, canonical column order.

    This prevents silent feature misalignment (e.g., when converting to numpy
    or when training/inference pipelines assume a consistent order).
    """
    return df[ALL_COLS]
