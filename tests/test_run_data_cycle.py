# tests/test_run_data_cycle_integration.py
#This ensures the entire pipeline works end-to-end on a temp directory without touching the real data
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from ml.run_data_cycle import run

#Tests the full orchestration logic using fake raw data 
    #writing artifacts
    #running all steps in order
    #producing reports
    #generating manifest


def test_full_data_cycle_end_to_end(tmp_path: Path):
    # ---- Create fake raw data ----
    raw_dir = tmp_path / "ml" / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = raw_dir / "nba.csv"
    raw_csv.write_text(
        "Name,GP,MIN,PTS,FGM,FGA,FG%,3P Made,3PA,3P%,FTM,FTA,FT%,OREB,DREB,REB,AST,STL,BLK,TOV,TARGET_5Yrs\n"
        "A,10,30,12,5,10,50,1,3,,1,2,50,1,3,4,2,1,0,1,1\n"
        "B,5,60,-2,8,6,120,5,2,200,4,1,-10,2,2,4,1,0,0,0,0\n"
    )

    # ---- Config file for this isolated test run ----
    cfg = {
        "dataset": {
            "raw_path": str(raw_csv),
            "interim_path": str(tmp_path / "ml" / "data" / "interim" / "nba_clean.parquet"),
            "processed_path": str(tmp_path / "ml" / "data" / "processed" / "nba_validated.parquet"),
            "target_col": "TARGET_5Yrs",
            "id_col": "Name",
        },
        "cleaning": {
            "drop_duplicates": True,
            "impute_3p_pct": True,
            "correct_impossible_values": True,
            "percent_cols": ["FG%", "3P%", "FT%"],
            "percent_bounds": [0, 100],
            "min_cap": 48,
            "non_negative_cols": [
                "GP","MIN","PTS","FGM","FGA","3P Made","3PA","FTM","FTA",
                "OREB","DREB","REB","AST","STL","BLK","TOV"
            ],
        },
        "validation": {
            "enforce_ranges": True,
            "enforce_made_leq_attempted": True,
            "enforce_rebounds_identity": True,
            "rebounds_tolerance": 0.0,
        },
        "stats": {"save_profile": True, "missingness_warn_threshold": 0.2,"missingness_fail_threshold": 0.5,"outlier_zscore_threshold": 4.0,"outlier_rate_warn": 0.02,"constant_unique_ratio_threshold": 0.01},
        "anomaly": {
            "enabled": False,
            "q_low": 0.01,
            "q_high": 0.99,
            "min_non_null": 10,
            "exclude_cols": ["TARGET_5Yrs"],
            "max_outlier_rate_warn": 0.02,
            "max_outlier_rate_fail": 0.10,},
        "outputs": {"sort_by": ["Name"], "write_index": False},
        "run_metadata": {"save_manifest": True, "manifest_path": str(tmp_path / "ml" / "reports" / "run_manifest.json")},
        "reproducibility": {"random_state": 42},
    }

    cfg_path = tmp_path / "ml" / "configs" / "data.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg))

    # ---- Run pipeline ----
    run(str(cfg_path))

    # ---- Assert artifacts exist ----
    assert Path(cfg["dataset"]["interim_path"]).exists()
    assert Path(cfg["dataset"]["processed_path"]).exists()

    # ---- Assert processed dataset is readable ----
    df = pd.read_parquet(cfg["dataset"]["processed_path"])
    assert df.shape[0] > 0
    assert "TARGET_5Yrs" in df.columns

    # ---- Assert manifest exists and contains hashes ----
    manifest = json.loads(Path(cfg["run_metadata"]["manifest_path"]).read_text())
    assert manifest["config"]["sha256"]
    assert manifest["data_artifacts"]["raw"]["sha256"]
