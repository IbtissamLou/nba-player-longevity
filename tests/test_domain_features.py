from __future__ import annotations

"""
Domain feature tests:
- ensure features are computed deterministically
- ensure safe division works (no NaN/inf)
- ensure ranges are reasonable
"""

import numpy as np
import pandas as pd
import pytest

from ml.feature_pipeline.domain_features import DomainFeatureConfig, add_domain_features
from ml.feature_pipeline.feature_schema import FeatureSchemaConfig, FeatureRange, validate_feature_schema


def test_add_domain_features_no_nan_inf():
    df = pd.DataFrame(
        [{
            "PTS": 12.0, "MIN": 30.0,
            "FGM": 5.0, "FGA": 10.0,
            "3PA": 3.0,
            "FTM": 1.0, "FTA": 2.0,
            "TOV": 1.0,
            "REB": 4.0, "OREB": 1.0, "DREB": 3.0,
            "AST": 2.0,
        }]
    )

    cfg = DomainFeatureConfig(epsilon=1e-9, enabled_features=("PTS_per_MIN", "FGM_per_FGA", "AST_to_TOV", "OREB_share"))
    out = add_domain_features(df, cfg)

    for col in ["PTS_per_MIN", "FGM_per_FGA", "AST_to_TOV", "OREB_share"]:
        assert col in out.columns
        assert not out[col].isna().any()
        assert not np.isinf(out[col].to_numpy()).any()


def test_feature_schema_validation_passes():
    df = pd.DataFrame(
        [{
            "PTS": 12.0, "MIN": 30.0,
            "FGM": 5.0, "FGA": 10.0,
            "3PA": 3.0,
            "FTM": 1.0, "FTA": 2.0,
            "TOV": 1.0,
            "REB": 4.0, "OREB": 1.0, "DREB": 3.0,
            "AST": 2.0,
        }]
    )

    cfg = DomainFeatureConfig(epsilon=1e-9, enabled_features=("PTS_per_MIN",))
    out = add_domain_features(df, cfg)

    schema_cfg = FeatureSchemaConfig(feature_ranges={"PTS_per_MIN": FeatureRange(0.0, 5.0)})
    validate_feature_schema(out, schema_cfg)  # should not raise


def test_feature_schema_validation_fails_out_of_range():
    df = pd.DataFrame(
        [{
            "PTS": 120.0, "MIN": 1.0,   # unrealistic PTS per minute
            "FGM": 1.0, "FGA": 1.0,
            "3PA": 0.0,
            "FTM": 0.0, "FTA": 0.0,
            "TOV": 1.0,
            "REB": 1.0, "OREB": 0.0, "DREB": 1.0,
            "AST": 0.0,
        }]
    )

    cfg = DomainFeatureConfig(epsilon=1e-9, enabled_features=("PTS_per_MIN",))
    out = add_domain_features(df, cfg)

    schema_cfg = FeatureSchemaConfig(feature_ranges={"PTS_per_MIN": FeatureRange(0.0, 5.0)})

    with pytest.raises(ValueError):
        validate_feature_schema(out, schema_cfg)
