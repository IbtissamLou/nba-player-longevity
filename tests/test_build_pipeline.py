# tests/test_build_pipeline.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.model_pipeline.build_pipeline import build_feature_pipeline


def _make_skewness_report_csv(path: Path, features: list[str]) -> None:
    """
    Create a minimal skewness_report.csv compatible with build_feature_pipeline.

    Columns expected by build_feature_pipeline:
      - feature
      - skewness
      - zero_ratio
      - category
      - suggestion   (one of: 'none', 'log1p', 'yeo_johnson')
    """
    rows = []
    for f in features:
        rows.append(
            {
                "feature": f,
                "skewness": 0.0,
                "zero_ratio": 0.0,
                "category": "low_skew",
                "suggestion": "none",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_feature_pipeline_end_to_end(tmp_path: Path):
    # ---- Fake training data ----
    rng = np.random.RandomState(0)
    df = pd.DataFrame(
        {
            "GP": rng.randint(1, 82, size=100),
            "MIN": rng.uniform(5, 35, size=100),
            "PTS": rng.uniform(0, 30, size=100),
            "AST": rng.uniform(0, 8, size=100),
        }
    )

    X_train, _ = train_test_split(df, test_size=0.2, random_state=42)

    selected_features = ["GP", "MIN", "PTS", "AST"]

    # ---- Skewness report for these features ----
    skew_report_path = tmp_path / "skewness_report.csv"
    _make_skewness_report_csv(skew_report_path, selected_features)

    eng_cfg = {
        "skewness_report_path": str(skew_report_path),
        "default_transform": "none",
        "scaler": "standard",
    }

    pipeline, details = build_feature_pipeline(
        selected_features=selected_features,
        X_train=X_train,
        eng_cfg=eng_cfg,
    )

    # ---- Fit + transform should run without errors ----
    Xt = pipeline.fit_transform(X_train)

    # Same number of rows as input
    assert Xt.shape[0] == X_train.shape[0]
    # Same number of features after transform + scaling
    assert Xt.shape[1] == len(selected_features)

    # ---- Details structure: must match build_pipeline.py ----
    assert isinstance(details, dict)

    # transform_plan: dict feature -> method ('none', 'log1p', 'yeo_johnson')
    assert "transform_plan" in details
    assert isinstance(details["transform_plan"], dict)
    assert set(details["transform_plan"].keys()) == set(selected_features)

    # scaling info
    assert details.get("scaling") == "standard"

    # skew_map contains the raw skewness metadata
    assert "skew_map" in details
    assert isinstance(details["skew_map"], dict)
    assert set(details["skew_map"].keys()) == set(selected_features)