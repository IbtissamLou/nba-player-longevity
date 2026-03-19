from __future__ import annotations

import pandas as pd

from ml.feature_pipeline.correlation import (
    CorrelationConfig,
    compute_correlation_matrix,
    extract_high_corr_pairs,
)


def test_extract_high_corr_pairs_detects_redundancy():
    # Create two highly correlated features
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [2, 4, 6, 8, 10],   # perfectly correlated with A
            "TARGET_5Yrs": [0, 1, 0, 1, 0],
        }
    )

    cfg = CorrelationConfig(enabled=True, threshold=0.95, exclude_cols=("TARGET_5Yrs",))
    corr = compute_correlation_matrix(df, cfg)

    pairs = extract_high_corr_pairs(corr, threshold=0.95)

    assert pairs.shape[0] >= 1
    assert {"feature_a", "feature_b", "corr", "abs_corr"}.issubset(pairs.columns)


def test_compute_correlation_matrix_excludes_target():
    df = pd.DataFrame(
        {"X": [1, 2, 3], "TARGET_5Yrs": [0, 1, 0]}
    )

    cfg = CorrelationConfig(exclude_cols=("TARGET_5Yrs",))
    corr = compute_correlation_matrix(df, cfg)

    assert "TARGET_5Yrs" not in corr.columns
    assert "TARGET_5Yrs" not in corr.index
