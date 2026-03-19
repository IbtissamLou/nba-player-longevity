from app.monitoring.drift.data_drift import DataDriftMonitor
from app.monitoring.drift.concept_drift import ConceptDriftProxy

import pandas as pd


def run_monitoring():
    # ------------------------
    # 1. DATA DRIFT
    # ------------------------
    monitor = DataDriftMonitor(
        reference_path="ml/data/processed/nba_validated.parquet"
    )

    current_df = monitor.load_current_from_logs(
        log_path="logs/inference_log.jsonl"
    )

    drift_result = monitor.run_drift_report(
        current_df=current_df,
        output_path="reports/drift_report.html",
    )

    print("DATA DRIFT RESULT:")
    print(drift_result)


if __name__ == "__main__":
    run_monitoring()