# tests/test_ingest.py
#Checks: file exists, columns read, types reasonable
#Uses tmp_path so we never depend on the real dataset.
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml.data_pipeline.ingest import ingest_csv, REQUIRED_COLUMNS


def _make_valid_row() -> dict:
    """Create one valid row that contains all required columns."""
    return {
        "Name": "A",
        "GP": 10,
        "MIN": 30.0,
        "PTS": 12.0,
        "FGM": 5.0,
        "FGA": 10.0,
        "FG%": 50.0,
        "3P Made": 1.0,
        "3PA": 3.0,
        "3P%": 33.3,
        "FTM": 1.0,
        "FTA": 2.0,
        "FT%": 50.0,
        "OREB": 1.0,
        "DREB": 3.0,
        "REB": 4.0,
        "AST": 2.0,
        "STL": 1.0,
        "BLK": 0.0,
        "TOV": 1.0,
        "TARGET_5Yrs": 1,
    }


def test_ingest_csv_reads_file_with_required_columns(tmp_path: Path):
    #Checks:
        #returned object is a DataFrame
        #one row exists
        #all required columns exist
#Ensures ingestion works for a correct dataset snapshot.
#Ensures the contract “these columns must exist” is enforced.
    csv = tmp_path / "nba.csv"

    df_in = pd.DataFrame([_make_valid_row()])
    df_in.to_csv(csv, index=False)

    df = ingest_csv(str(csv))

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 1

    # Required columns must exist
    assert set(REQUIRED_COLUMNS).issubset(df.columns)


def test_ingest_csv_raises_if_file_missing(tmp_path: Path):
#Calls ingest_csv() on a path that doesn’t exist
#Verifies it raises FileNotFoundError
#Prevents silent failures (like returning empty df or weird pandas error)
#Confirms the pipeline fails early with a clear error message.
    missing_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        ingest_csv(str(missing_path))


def test_ingest_csv_raises_if_dataset_empty(tmp_path: Path):
    #Writes an empty file nba.csv
    #Expects ingest_csv() to raise an “empty dataset” error
    
    csv = tmp_path / "nba.csv"
    csv.write_text("")  # empty file

    with pytest.raises(ValueError, match="empty"):
        ingest_csv(str(csv))


def test_ingest_csv_raises_if_required_columns_missing(tmp_path: Path):
    #Writes a minimal CSV that is intentionally missing many columns
    #Expects a ValueError("Missing required columns")
#schema consistency at the ingestion level
    csv = tmp_path / "nba.csv"
    # missing many required columns on purpose
    csv.write_text("Name,GP,MIN,TARGET_5Yrs\nA,10,30,1\n")

    with pytest.raises(ValueError, match="Missing required columns"):
        ingest_csv(str(csv))
