from __future__ import annotations

from typing import Dict, List
import pandas as pd

from app.model_loader import LoadedModelPackage
from app.schemas import (
    PredictResponse,
    BatchPredictRowResponse,
    BatchPredictResponse,
)


def _build_dataframe(
    rows: List[Dict[str, float]],
    expected_features: List[str],
) -> pd.DataFrame:
    """
    Build a DataFrame that exactly matches the feature schema expected
    by the packaged model.

    Parameters
    ----------
    rows:
        List of dictionaries, where each dictionary is one inference row:
        {
            "GP": 60,
            "MIN": 28.5,
            ...
        }

    expected_features:
        Ordered list of feature names from the model package manifest.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ordered exactly as expected by the model.

    Raises
    ------
    ValueError
        If required features are missing.
    """
    df = pd.DataFrame(rows)

    missing = [c for c in expected_features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected features: {missing}")

    return df[expected_features]


def predict_one(
    pkg: LoadedModelPackage,
    row: Dict[str, float],
) -> PredictResponse:
    """
    Real-time inference for a single row.

    Parameters
    ----------
    pkg:
        Loaded packaged model object.

    row:
        Dictionary of feature_name -> value.

    Returns
    -------
    PredictResponse
        Probability + binary prediction using the tuned threshold.
    """
    X = _build_dataframe([row], pkg.features)

    proba = float(pkg.predict_proba(X)[0])
    pred = int(proba >= pkg.threshold)

    return PredictResponse(
        probability=round(proba, 6),
        prediction=pred,
        threshold_used=pkg.threshold,
    )


def predict_batch(
    pkg: LoadedModelPackage,
    rows: List[Dict[str, float]],
) -> BatchPredictResponse:
    """
    Batch inference for multiple rows.

    Parameters
    ----------
    pkg:
        Loaded packaged model object.

    rows:
        List of dictionaries. Each dict contains one row of features.

    Returns
    -------
    BatchPredictResponse
        Predictions for all rows.
    """
    X = _build_dataframe(rows, pkg.features)

    probas = pkg.predict_proba(X)
    preds = (probas >= pkg.threshold).astype(int)

    results = [
        BatchPredictRowResponse(
            probability=round(float(p), 6),
            prediction=int(y),
        )
        for p, y in zip(probas, preds)
    ]

    return BatchPredictResponse(
        threshold_used=pkg.threshold,
        n_rows=len(results),
        predictions=results,
    )