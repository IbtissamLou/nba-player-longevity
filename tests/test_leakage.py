from __future__ import annotations

"""
Leakage tests ensure the pipeline fails fast if:
- IDs are not removed
- a target proxy feature exists
- duplicate columns appear (possible join/copy bug)
"""

import pandas as pd
import pytest

from ml.feature_pipeline.leakage import LeakageConfig, apply_leakage_guards


def test_leakage_drops_id_cols():
    # Purpose: verify we drop identifiers like 'Name' to prevent memorization leakage.
    df = pd.DataFrame(
        {
            "Name": ["A", "B"],
            "PTS": [10.0, 12.0],
            "TARGET_5Yrs": [1, 0],
        }
    )

    cfg = LeakageConfig(
        enabled=True,
        target_col="TARGET_5Yrs",
        id_cols=("Name",),
        drop_id_cols=True,
        forbidden_feature_cols=("TARGET_5Yrs",),
        check_target_proxy=False,      # disable to isolate ID behavior
        check_duplicate_columns=False,
    )

    df_out, report = apply_leakage_guards(df, cfg)
    assert "Name" not in df_out.columns
    assert report["status"] in ("pass", "warn")


def test_leakage_fails_on_target_proxy():
    # Purpose: verify we fail fast when a feature directly leaks the target.
    # Here, leaky_feature == TARGET_5Yrs -> correlation = 1.0.
    df = pd.DataFrame(
        {
            "Name": ["A", "B", "C", "D"],
            "leaky_feature": [1, 0, 1, 0],
            "TARGET_5Yrs": [1, 0, 1, 0],
        }
    )

    cfg = LeakageConfig(
        enabled=True,
        target_col="TARGET_5Yrs",
        id_cols=("Name",),
        drop_id_cols=True,
        forbidden_feature_cols=("TARGET_5Yrs",),
        check_target_proxy=True,
        max_abs_target_corr_warn=0.8,
        max_abs_target_corr_fail=0.9,  # ensures corr=1.0 triggers fail
        check_duplicate_columns=False,
    )

    with pytest.raises(ValueError):
        apply_leakage_guards(df, cfg)


def test_leakage_warns_on_duplicate_columns():
    # Purpose: duplicate columns are usually a pipeline bug (bad join or accidental copy).
    # We expect at least a WARN, never PASS.
    df = pd.DataFrame(
        {
            "Name": ["A", "B", "C"],
            "X": [1, 2, 3],
            "X_DUP": [1, 2, 3],  # exact duplicate of X
            "TARGET_5Yrs": [0, 1, 0],
        }
    )

    cfg = LeakageConfig(
        enabled=True,
        target_col="TARGET_5Yrs",
        id_cols=("Name",),
        drop_id_cols=True,
        forbidden_feature_cols=("TARGET_5Yrs",),
        check_target_proxy=False,
        check_duplicate_columns=True,
    )

    _, report = apply_leakage_guards(df, cfg)
    assert report["status"] in ("warn", "fail")
    assert len(report["duplicate_columns"]) > 0
