from __future__ import annotations

import json
import pandas as pd
from typing import Dict, Any
from pathlib import Path


class PredictionMonitor:
    """
    Analyze prediction behavior from inference logs.
    """

    def __init__(self, log_path: str):
        self.log_path = Path(log_path)

    def load_predictions(self) -> pd.DataFrame:
        rows = []

        with open(self.log_path, "r") as f:
            for line in f:
                try:
                    event = json.loads(line)
                    if event.get("event_type") == "single_inference":
                        rows.append({
                            "prediction": event["prediction"],
                            "probability": event["probability"],
                            "timestamp": event["timestamp_utc"],
                        })
                except Exception:
                    continue

        if not rows:
            raise ValueError("No predictions found in logs")

        return pd.DataFrame(rows)

    def compute_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        total = len(df)

        positive_rate = df["prediction"].mean()

        avg_proba = df["probability"].mean()
        p50 = df["probability"].quantile(0.5)
        p90 = df["probability"].quantile(0.9)
        p95 = df["probability"].quantile(0.95)

        return {
            "total_predictions": total,
            "positive_prediction_rate": round(float(positive_rate), 4),
            "average_probability": round(float(avg_proba), 4),
            "p50_probability": round(float(p50), 4),
            "p90_probability": round(float(p90), 4),
            "p95_probability": round(float(p95), 4),
        }