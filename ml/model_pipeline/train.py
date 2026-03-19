from __future__ import annotations

"""
Final training helpers.

Given a (still-unfitted) best_pipeline selected by model_selection,
fit it on training data and return the fitted object.
"""

import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline


def train_best_pipeline(
    pipeline: ImbPipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
):
    """Fit in-place and return the pipeline (for chaining)."""
    pipeline.fit(X_train, y_train)
    return pipeline