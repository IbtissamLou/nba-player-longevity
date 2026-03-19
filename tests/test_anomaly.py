# tests/test_anomaly.py
#Checks: flags are produced; report contains outlier rate
#Ensures anomaly detection outputs are:
#debug-friendly / consistent for JSON reporting
from __future__ import annotations

import pandas as pd
from ml.data_pipeline.anomaly import flag_quantile_outliers, AnomalyConfig

def test_anomaly_flags_exist(sample_df: pd.DataFrame):
    cfg = AnomalyConfig(
        enabled=True,
        q_low=0.01,
        q_high=0.99,
        min_non_null=2,
        exclude_cols=("TARGET_5Yrs",),
        max_outlier_rate_warn=0.0,   # force warn/fail for testing
        max_outlier_rate_fail=0.9,
    )
    df_out, report = flag_quantile_outliers(sample_df, cfg)
    assert "outlier_any" in df_out.columns
    assert "outlier_count" in df_out.columns
    assert "outlier_cols" in df_out.columns
    assert "outlier_rate" in report
