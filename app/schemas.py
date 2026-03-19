from __future__ import annotations

from typing import Dict, List
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """
    Request for real-time inference.

    The keys of the dictionary must match the feature names
    expected by the packaged model.
    """

    features: Dict[str, float] = Field(
        ...,
        description="Dictionary of feature_name -> value matching the packaged model schema",
        example={
            "GP": 60,
            "MIN": 28.5,
            "PTS": 15.2,
            "FGM": 5.6,
            "FGA": 12.1,
        },
    )


class PredictResponse(BaseModel):
    """
    Response returned by the /predict endpoint.
    """

    probability: float = Field(
        ...,
        description="Predicted probability of the positive class",
        example=0.73,
    )

    prediction: int = Field(
        ...,
        description="Binary prediction after applying the tuned threshold",
        example=1,
    )

    threshold_used: float = Field(
        ...,
        description="Decision threshold used to convert probability to prediction",
        example=0.62,
    )


class BatchPredictRequest(BaseModel):
    """
    Request for batch inference.
    """

    rows: List[Dict[str, float]] = Field(
        ...,
        description="List of feature dictionaries",
    )


class BatchPredictRowResponse(BaseModel):
    """
    Prediction for a single row in batch inference.
    """

    probability: float
    prediction: int


class BatchPredictResponse(BaseModel):
    """
    Response returned by /predict_batch endpoint.
    """

    threshold_used: float
    n_rows: int
    predictions: List[BatchPredictRowResponse]