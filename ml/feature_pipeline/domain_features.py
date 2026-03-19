from __future__ import annotations

"""
Domain feature engineering for NBA career longevity.
Add new features 

Design principles:
- deterministic (same input -> same output)
- safe math (no division by zero)
- interpretable features (ratios/usage proxies)
- config-driven enabling/disabling

Scaling/encoding belongs later in sklearn pipelines (training stage).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DomainFeatureConfig:
    """Configuration loaded from feature_spec.yaml."""
    epsilon: float = 1e-9
    enabled_features: tuple[str, ...] = (
        "PTS_per_MIN",
        "FGM_per_FGA",
        "FT_per_FTA",
        "ThreePA_rate",
        "FT_rate",
        "Usage_proxy",
        "AST_to_TOV",
        "REB_per_MIN",
        "OREB_share",
        "DREB_share",
    )


def _safe_div(numer: pd.Series, denom: pd.Series, eps: float) -> pd.Series:
    """Safe division to avoid division-by-zero; keeps output numeric and stable."""
    return numer / (denom + eps)


def add_domain_features(df: pd.DataFrame, cfg: DomainFeatureConfig) -> pd.DataFrame:
    """
    Add domain knowledge features to a dataframe.

    Input: df must contain raw NBA stat columns (PTS, MIN, FGA, etc.)
    Output: df with extra engineered features appended.
    """
    out = df.copy()
    eps = cfg.epsilon

    # --- Rate features (normalize by opportunity/time) ---
    if "PTS_per_MIN" in cfg.enabled_features:
        out["PTS_per_MIN"] = _safe_div(out["PTS"], out["MIN"], eps)

    if "FGM_per_FGA" in cfg.enabled_features:
        out["FGM_per_FGA"] = _safe_div(out["FGM"], out["FGA"], eps)

    if "FT_per_FTA" in cfg.enabled_features:
        out["FT_per_FTA"] = _safe_div(out["FTM"], out["FTA"], eps)

    # --- Style/shot profile ---
    if "ThreePA_rate" in cfg.enabled_features:
        out["ThreePA_rate"] = _safe_div(out["3PA"], out["FGA"], eps)

    if "FT_rate" in cfg.enabled_features:
        out["FT_rate"] = _safe_div(out["FTA"], out["FGA"], eps)

    # --- Usage proxy (common NBA analytics approximation) ---
    # More usage often correlates with role; role can correlate with career length.
    if "Usage_proxy" in cfg.enabled_features:
        out["Usage_proxy"] = out["FGA"] + 0.44 * out["FTA"] + out["TOV"]

    # --- Ball security / playmaking efficiency ---
    if "AST_to_TOV" in cfg.enabled_features:
        out["AST_to_TOV"] = _safe_div(out["AST"], out["TOV"], eps)

    # --- Rebounding normalized by playing time ---
    if "REB_per_MIN" in cfg.enabled_features:
        out["REB_per_MIN"] = _safe_div(out["REB"], out["MIN"], eps)

    # --- Rebound composition shares (should be between 0 and 1) ---
    if "OREB_share" in cfg.enabled_features:
        out["OREB_share"] = _safe_div(out["OREB"], out["REB"], eps).clip(0, 1)

    if "DREB_share" in cfg.enabled_features:
        out["DREB_share"] = _safe_div(out["DREB"], out["REB"], eps).clip(0, 1)

    return out
