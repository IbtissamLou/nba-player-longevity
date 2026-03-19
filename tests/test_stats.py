# tests/test_stats.py
#Checks: profile keys exist, and “checks” produce pass/warn in expected cases
from __future__ import annotations

import pandas as pd
from ml.data_pipeline.stats import build_profile, run_statistical_checks

def test_build_profile_has_expected_keys(sample_df: pd.DataFrame):
    #Ensures profiling output is stable and structured
    #Profiling is used later for drift detection and debugging.
    prof = build_profile(sample_df, target_col="TARGET_5Yrs")
    assert "rows" in prof and "cols" in prof
    assert "missing_ratio" in prof
    assert "numeric_summary" in prof

def test_statistical_checks_warn_on_missingness(sample_df: pd.DataFrame):
    #Uses intentionally strict threshold to reliably force a warning in a unit test
    prof = build_profile(sample_df, target_col="TARGET_5Yrs")
    checks = run_statistical_checks(
        sample_df,
        prof,
        target_col="TARGET_5Yrs",
        missingness_warn_threshold=0.01,  # intentionally low to force warn
        missingness_fail_threshold=0.9,
    )
    assert checks["status"] in ("warn", "fail")
