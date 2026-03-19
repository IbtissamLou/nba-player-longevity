# tests/test_validate.py
#Checks: schema enforces columns/types/order; rule validation fails on bad data, passes on cleaned data
from __future__ import annotations

import pytest
import pandas as pd

from ml.data_pipeline.clean import run_cleaning
from ml.data_pipeline.validate import validate_schema, validate_rules, ValidationConfig

def test_validate_schema_enforces_order(sample_df: pd.DataFrame):
    # after cleaning, schema should validate and ordering should be canonical
    cleaned = run_cleaning(
        sample_df,
        do_drop_duplicates=True,
        do_impute_3p_pct=True,
        do_correct_impossible_values=True,
        percent_cols=["FG%", "3P%", "FT%"],
        percent_bounds=(0, 100),
        min_cap=48,
        non_negative_cols=["PTS", "GP", "MIN", "FGM", "FGA", "3P Made", "3PA", "FTM", "FTA", "OREB", "DREB", "REB"],
    )
    validated = validate_schema(cleaned, enforce_order=True)
    # quick sanity: first column should be Name
    assert validated.columns[0] == "Name"

def test_validate_rules_fails_on_unclean_data(sample_df: pd.DataFrame):
    #Confirms validation rules catch impossible values
    #Ensures the pipeline fails early if cleaning didn’t happen.

    cfg = ValidationConfig(
        enforce_ranges=True,
        enforce_made_leq_attempted=True,
        enforce_rebounds_identity=True,
        rebounds_tolerance=0.0,
        min_cap=48,
        percent_bounds=(0, 100),
        non_negative_cols=("PTS",),
    )
    # sample_df contains impossible values
    with pytest.raises(ValueError):
        validate_rules(sample_df, cfg)

def test_validate_rules_passes_after_cleaning(sample_df: pd.DataFrame):
    cleaned = run_cleaning(
        sample_df,
        do_drop_duplicates=True,
        do_impute_3p_pct=True,
        do_correct_impossible_values=True,
        percent_cols=["FG%", "3P%", "FT%"],
        percent_bounds=(0, 100),
        min_cap=48,
        non_negative_cols=["PTS", "GP", "MIN", "FGM", "FGA", "3P Made", "3PA", "FTM", "FTA", "OREB", "DREB", "REB"],
    )
    validated = validate_schema(cleaned)

    cfg = ValidationConfig(
        enforce_ranges=True,
        enforce_made_leq_attempted=True,
        enforce_rebounds_identity=True,
        rebounds_tolerance=0.0,
        min_cap=48,
        percent_bounds=(0, 100),
        non_negative_cols=("PTS", "GP", "MIN"),
    )
    # should not raise
    validate_rules(validated, cfg)
