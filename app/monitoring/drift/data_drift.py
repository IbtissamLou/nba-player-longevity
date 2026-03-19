from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Dict, Any

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


class DataDriftMonitor:
    """
    Detects drift between:
    - reference dataset (training)
    - current dataset (inference logs)
    """

    def __init__(self, reference_path: str):
        self.reference_path = Path(reference_path)

        if not self.reference_path.exists():
            raise FileNotFoundError(f"Reference dataset not found: {reference_path}")

        self.reference_df = self._load_reference()

    def _load_reference(self) -> pd.DataFrame:
        if self.reference_path.suffix == ".parquet":
            return pd.read_parquet(self.reference_path)
        elif self.reference_path.suffix == ".csv":
            return pd.read_csv(self.reference_path)
        else:
            raise ValueError("Unsupported reference format")

    def load_current_from_logs(self, log_path: str) -> pd.DataFrame:
        """
        Extract features from inference logs
        """
        rows = []

        with open(log_path, "r") as f:
            for line in f:
                try:
                    event = eval(line)
                    if event.get("event_type") == "single_inference":
                        rows.append(event["features"])
                except Exception:
                    continue

        if not rows:
            raise ValueError("No valid inference data found")

        return pd.DataFrame(rows)

    def run_drift_report(
        self,
        current_df: pd.DataFrame,
        output_path: str,
    ) -> Dict[str, Any]:
        """
        Run Evidently drift report
        """

        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=self.reference_df,
            current_data=current_df,
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        report.save_html(output_path)

        result = report.as_dict()

        return {
            "dataset_drift": result["metrics"][0]["result"]["dataset_drift"],
            "drifted_features": result["metrics"][0]["result"]["number_of_drifted_columns"],
            "total_features": result["metrics"][0]["result"]["number_of_columns"],
        }