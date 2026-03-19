#test feature selection

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import numpy as np

from ml.model_pipeline.feature_selection_rf import rf_feature_selection


def _make_toy_features(n_samples: int = 100) -> pd.DataFrame:
    rng = np.random.RandomState(0)
    df = pd.DataFrame(
        {
            "GP": rng.randint(1, 82, size=n_samples),
            "MIN": rng.uniform(5, 35, size=n_samples),
            "PTS": rng.uniform(0, 30, size=n_samples),
            "AST": rng.uniform(0, 8, size=n_samples),
            "REB": rng.uniform(0, 12, size=n_samples),
            "NOISE1": rng.normal(0, 1, size=n_samples),
            "NOISE2": rng.normal(0, 1, size=n_samples),
        }
    )
    return df


def _make_toy_target(df: pd.DataFrame) -> pd.Series:
    # simple rule: high minutes + points => more likely target=1
    return ((df["MIN"] > 20) & (df["PTS"] > 10)).astype(int)


def test_rf_feature_selection_basic_behavior():
    df = _make_toy_features()
    y = _make_toy_target(df)
    all_features: List[str] = list(df.columns)

    fs_cfg = {
        "enabled": True,
        "top_k": 3,
        "min_importance": 0.0,
        "always_keep": ["GP"],  # domain-important
        "n_estimators": 100,
        "max_depth": None,
        "n_jobs": -1,
    }

    selected, details = rf_feature_selection(
        X_train=df,
        y_train=y,
        all_features=all_features,
        fs_cfg=fs_cfg,
        random_state=42,
    )

    # Basic sanity checks
    assert len(selected) > 0
    assert set(selected).issubset(set(all_features))

    assert "GP" in selected

    assert len(selected) <= fs_cfg["top_k"] + 2

    # Details dict should contain useful info
    assert "feature_importances_sorted" in details
    assert "selected_features" in details