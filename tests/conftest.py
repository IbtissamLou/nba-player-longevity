#test every function on a small controlled dataset.
#Controlled synthetic dataset with intentional issues:
#duplicates / missing 3P% / impossible values (MIN > 48, PTS negative, FGM > FGA, etc.)

# tests/conftest.py
from __future__ import annotations

import pandas as pd
import pytest

@pytest.fixture
def sample_df() -> pd.DataFrame:
    # small dataset with a few issues to test cleaning & validation
    return pd.DataFrame(
        [
            {
                "Name": "A",
                "GP": 10, "MIN": 30.0, "PTS": 12.0,
                "FGM": 5.0, "FGA": 10.0, "FG%": 50.0,
                "3P Made": 1.0, "3PA": 3.0, "3P%": None,  # missing 3P%
                "FTM": 1.0, "FTA": 2.0, "FT%": 50.0,
                "OREB": 1.0, "DREB": 3.0, "REB": 3.0,  # wrong (should become 4)
                "AST": 2.0, "STL": 1.0, "BLK": 0.0, "TOV": 1.0,
                "TARGET_5Yrs": 1,
            },
            # duplicate row to test drop_duplicates
            {
                "Name": "A",
                "GP": 10, "MIN": 30.0, "PTS": 12.0,
                "FGM": 5.0, "FGA": 10.0, "FG%": 50.0,
                "3P Made": 1.0, "3PA": 3.0, "3P%": None,
                "FTM": 1.0, "FTA": 2.0, "FT%": 50.0,
                "OREB": 1.0, "DREB": 3.0, "REB": 3.0,
                "AST": 2.0, "STL": 1.0, "BLK": 0.0, "TOV": 1.0,
                "TARGET_5Yrs": 1,
            },
            # intentionally impossible values to test correction + rules
            {
                "Name": "B",
                "GP": 5, "MIN": 60.0, "PTS": -2.0,  # MIN too high, PTS negative
                "FGM": 8.0, "FGA": 6.0, "FG%": 120.0,  # made > attempts, FG% > 100
                "3P Made": 5.0, "3PA": 2.0, "3P%": 200.0,
                "FTM": 4.0, "FTA": 1.0, "FT%": -10.0,
                "OREB": 2.0, "DREB": 2.0, "REB": 100.0,
                "AST": 1.0, "STL": 0.0, "BLK": 0.0, "TOV": 0.0,
                "TARGET_5Yrs": 0,
            },
        ]
    )
