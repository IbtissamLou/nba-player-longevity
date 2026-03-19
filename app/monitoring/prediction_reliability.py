from __future__ import annotations

from typing import Any, Dict


class PredictionReliabilityStore:
    """
    Tracks reliability of model outputs.

    Main metrics:
    - prediction_validity_rate
    - invalid_prediction_count
    - positive_prediction_rate
    - average_probability
    """

    def __init__(self) -> None:
        self.total_predictions = 0
        self.invalid_prediction_count = 0
        self.positive_prediction_count = 0
        self.probability_sum = 0.0

    def record_prediction(
        self,
        *,
        prediction: int,
        probability: float,
        is_valid: bool,
    ) -> None:
        self.total_predictions += 1

        if not is_valid:
            self.invalid_prediction_count += 1
            return

        if prediction == 1:
            self.positive_prediction_count += 1

        self.probability_sum += float(probability)

    def get_metrics(self) -> Dict[str, Any]:
        valid_prediction_count = self.total_predictions - self.invalid_prediction_count

        prediction_validity_rate = (
            valid_prediction_count / self.total_predictions
            if self.total_predictions > 0
            else 0.0
        )

        positive_prediction_rate = (
            self.positive_prediction_count / valid_prediction_count
            if valid_prediction_count > 0
            else 0.0
        )

        average_probability = (
            self.probability_sum / valid_prediction_count
            if valid_prediction_count > 0
            else 0.0
        )

        return {
            "total_predictions": self.total_predictions,
            "invalid_prediction_count": self.invalid_prediction_count,
            "prediction_validity_rate": round(prediction_validity_rate, 4),
            "positive_prediction_rate": round(positive_prediction_rate, 4),
            "average_probability": round(average_probability, 4),
        }