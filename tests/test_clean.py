# tests/test_clean.py
#Checks: duplicates removed, imputation correct, impossible values corrected using config-driven behavior
#using sample_df from conftest.py
#Mirrors YAML config design
from __future__ import annotations

import pandas as pd
from ml.data_pipeline.clean import run_cleaning

def test_cleaning_drop_duplicates(sample_df: pd.DataFrame):
    out = run_cleaning(
        sample_df,
        do_drop_duplicates=True,
        do_impute_3p_pct=False,
        do_correct_impossible_values=False,
        percent_cols=["FG%", "3P%", "FT%"],
        percent_bounds=(0, 100),
        min_cap=48,
        non_negative_cols=["PTS"],
    )
    # A duplicated row removed (3 rows -> 2 rows)
    assert out.shape[0] == 2

def test_cleaning_impute_3p_pct(sample_df: pd.DataFrame):
    out = run_cleaning(
        sample_df,
        do_drop_duplicates=False,
        do_impute_3p_pct=True,
        do_correct_impossible_values=False,
        percent_cols=["FG%", "3P%", "FT%"],
        percent_bounds=(0, 100),
        min_cap=48,
        non_negative_cols=["PTS"],
    )
    # for row A: 3P% should become (1/3)*100 ~= 33.333
    val = out.loc[out["Name"] == "A", "3P%"].iloc[0]
    assert abs(val - (1.0 / 3.0 * 100.0)) < 1e-6

def test_cleaning_correct_impossible_values(sample_df: pd.DataFrame):
    out = run_cleaning(
        sample_df,
        do_drop_duplicates=False,
        do_impute_3p_pct=False,
        do_correct_impossible_values=True,
        percent_cols=["FG%", "3P%", "FT%"],
        percent_bounds=(0, 100),
        min_cap=48,
        non_negative_cols=["PTS", "GP", "MIN", "FGM", "FGA", "3P Made", "3PA", "FTM", "FTA", "REB", "OREB", "DREB"],
    )
    b = out[out["Name"] == "B"].iloc[0]
    assert b["MIN"] <= 48
    assert b["PTS"] >= 0
    assert b["FG%"] <= 100
    assert b["FGM"] <= b["FGA"]
    assert b["REB"] == b["OREB"] + b["DREB"]
