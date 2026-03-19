from __future__ import annotations

"""
Feature schema validation.

Goal:
- Ensure engineered features exist and are within reasonable bounds.
- Catch NaNs/infs or broken calculations early.

This is a "feature contract" similar to the Data Cycle schema.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureRange:
    """Range check for one feature."""
    low: float
    high: float


@dataclass(frozen=True)
class FeatureSchemaConfig:
    """
    Holds expected engineered feature ranges.
    """
    feature_ranges: dict[str, FeatureRange]


def validate_feature_schema(df: pd.DataFrame, cfg: FeatureSchemaConfig) -> None:
    """
    Validate engineered features:
    - feature exists
    - numeric type
    - finite values (no inf)
    - within expected range
    """
    errors: list[str] = []

    for feat, fr in cfg.feature_ranges.items():
        if feat not in df.columns:
            errors.append(f"Missing engineered feature: {feat}")
            continue

        s = df[feat]

        # Must be numeric
        if not np.issubdtype(s.dtype, np.number):
            errors.append(f"{feat} must be numeric, got dtype={s.dtype}")
            continue

        # No inf
        if np.isinf(s.to_numpy()).any():
            errors.append(f"{feat} contains inf values")

        # Range check
        if s.isna().any():
            errors.append(f"{feat} contains NaNs (feature computation broken or missing inputs)")

        if ((s < fr.low) | (s > fr.high)).any():
            errors.append(f"{feat} outside expected range [{fr.low}, {fr.high}]")

    if errors:
        raise ValueError("Feature schema validation failed:\n- " + "\n- ".join(errors))
