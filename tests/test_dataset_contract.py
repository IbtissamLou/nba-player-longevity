from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml
import json

from ml.data_pipeline.ingest import ingest_csv
from ml.data_pipeline.clean import run_cleaning
from ml.data_pipeline.validate import validate_schema, validate_rules, ValidationConfig
from ml.data_pipeline.stats import build_profile, run_statistical_checks, compare_profiles
from ml.data_pipeline.anomaly import flag_quantile_outliers, AnomalyConfig
from ml.data_pipeline.schemas.nba import ALL_COLS, TARGET_COL

#real dataset contract test
#Ensures that pipeline works on the real dataset snapshot


@pytest.mark.contract
def test_dataset_contract_raw_to_validated(config_path: str = "ml/configs/data.yaml"):
    """
    Dataset contract test (Data Cycle):

    Ensures that the REAL raw dataset snapshot can go through:
    ingest -> clean -> schema validation -> rule validation
    and still satisfy the canonical schema + business constraints.
    """

    cfg_file = Path(config_path) #Loads the actual ml/configs/data.yaml
    assert cfg_file.exists(), f"Missing config: {cfg_file}"

    cfg = yaml.safe_load(cfg_file.read_text())

    raw_path = Path(cfg["dataset"]["raw_path"]) #Reads real ml/data/raw/nba.csv
    assert raw_path.exists(), f"Missing raw dataset: {raw_path}"
    assert raw_path.stat().st_size > 0, "Raw dataset file is empty"

    # ---- 1) Ingest ---- 
    df = ingest_csv(str(raw_path))
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0, "Raw dataset has no rows"

    # ---- 2) Clean (config-driven) ----
    df = run_cleaning(
        df,
        do_drop_duplicates=cfg["cleaning"]["drop_duplicates"],
        do_impute_3p_pct=cfg["cleaning"]["impute_3p_pct"],
        do_correct_impossible_values=cfg["cleaning"]["correct_impossible_values"],
        percent_cols=cfg["cleaning"]["percent_cols"],
        percent_bounds=tuple(cfg["cleaning"]["percent_bounds"]),
        min_cap=cfg["cleaning"]["min_cap"],
        non_negative_cols=cfg["cleaning"]["non_negative_cols"],
    )

    assert df.shape[0] > 0, "Cleaning produced empty dataset"

    # ---- 3) Schema validation (strict) + canonical column order ----
    df = validate_schema(df, enforce_order=True)

    # Schema consistency: exact canonical columns in the expected order
    assert list(df.columns) == list(ALL_COLS), (
        "Schema contract broken: columns mismatch.\n"
        f"Expected: {ALL_COLS}\nGot: {list(df.columns)}"
    )

    # ---- 4) Validation rules ----
    vcfg = ValidationConfig(
        enforce_ranges=cfg["validation"]["enforce_ranges"],
        enforce_made_leq_attempted=cfg["validation"]["enforce_made_leq_attempted"],
        enforce_rebounds_identity=cfg["validation"]["enforce_rebounds_identity"],
        rebounds_tolerance=cfg["validation"]["rebounds_tolerance"],
        min_cap=float(cfg["cleaning"]["min_cap"]),
        percent_bounds=tuple(cfg["cleaning"]["percent_bounds"]),
        non_negative_cols=tuple(cfg["cleaning"]["non_negative_cols"]),
    )
    validate_rules(df, vcfg)  # should not raise

    # ---- 5) Target sanity ----
    assert TARGET_COL in df.columns
    assert df[TARGET_COL].isin([0, 1]).all(), "Target contains invalid values"

    # ---- 6) Stats profile + statistical checks ----
    save_profile = bool(cfg.get("stats", {}).get("save_profile", True))
    if save_profile:
        target_col = cfg["dataset"]["target_col"]
        profile = build_profile(df, target_col=target_col)

        # basic structure checks
        assert "rows" in profile and profile["rows"] > 0
        assert "missing_ratio" in profile
        assert "numeric_summary" in profile

        checks = run_statistical_checks(
            df,
            profile,
            target_col=target_col,
            missingness_warn_threshold=cfg["stats"]["missingness_warn_threshold"],
            missingness_fail_threshold=cfg["stats"].get("missingness_fail_threshold", 0.50),
            outlier_zscore_threshold=cfg["stats"].get("outlier_zscore_threshold", 4.0),
            outlier_rate_warn=cfg["stats"].get("outlier_rate_warn", 0.02),
            constant_unique_ratio_threshold=cfg["stats"].get("constant_unique_ratio_threshold", 0.01),
        )

        # Contract expectation: checks should not hard-fail in normal conditions
        assert checks["status"] in ("pass", "warn"), f"Statistical checks failed: {checks}"

    else:
        profile = None

    # ---- 7) Anomaly detection (structure + optional gating) ----
    #Monitoring checks: “is the data suspicious?” (warn/fail produces alert but doesn’t necessarily block training)
    anomaly_cfg = cfg.get("anomaly", {})
    if anomaly_cfg.get("enabled", False):
        an_cfg = AnomalyConfig(
            enabled=True,
            q_low=anomaly_cfg["q_low"],
            q_high=anomaly_cfg["q_high"],
            min_non_null=anomaly_cfg["min_non_null"],
            exclude_cols=tuple(anomaly_cfg["exclude_cols"]),
            max_outlier_rate_warn=anomaly_cfg["max_outlier_rate_warn"],
            max_outlier_rate_fail=anomaly_cfg["max_outlier_rate_fail"],
        )

        df_out, report = flag_quantile_outliers(df, an_cfg)

        assert "outlier_any" in df_out.columns
        assert "outlier_count" in df_out.columns
        assert "outlier_cols" in df_out.columns
        assert "outlier_rate" in report

        # Contract expectation: anomaly detection must produce a valid report.
        # "fail" is allowed here because outliers can be legitimate in real datasets.
        assert report["status"] in ("pass", "warn", "fail")
        assert "details" in report
        assert isinstance(report["outlier_rate"], float)


    # ---- 8) Drift detection (only if baseline exists) ----
    drift_cfg = cfg.get("drift", {})
    if drift_cfg.get("enabled", False) and profile is not None:
        baseline_path = drift_cfg.get("baseline_profile")
        if baseline_path:
            baseline_p = Path(baseline_path)
            if baseline_p.exists():
                baseline = json.loads(baseline_p.read_text())
                drift_report = compare_profiles(profile, baseline)

                # Contract expectation: drift can warn but should not fail for stable datasets
                assert drift_report["status"] in ("pass", "warn"), f"Drift failed: {drift_report}"
            else:
                # No baseline is an acceptable state for first run
                assert True

