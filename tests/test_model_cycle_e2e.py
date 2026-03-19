from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import yaml

from ml.run_model_cycle import run as run_model_cycle


def _make_toy_feature_dataset(path: Path):
    rng = np.random.RandomState(0)
    n = 200

    df = pd.DataFrame(
        {
            "Name": [f"Player_{i}" for i in range(n)],
            "GP": rng.randint(1, 82, size=n),
            "MIN": rng.uniform(5, 35, size=n),
            "PTS": rng.uniform(0, 30, size=n),
            "FGM": rng.uniform(0, 10, size=n),
            "FGA": rng.uniform(0, 20, size=n),
            "FG%": rng.uniform(30, 60, size=n),
            "3P Made": rng.uniform(0, 5, size=n),
            "3PA": rng.uniform(0, 10, size=n),
            "3P%": rng.uniform(25, 45, size=n),
            "FTM": rng.uniform(0, 5, size=n),
            "FTA": rng.uniform(0, 6, size=n),
            "FT%": rng.uniform(60, 90, size=n),
            "OREB": rng.uniform(0, 3, size=n),
            "DREB": rng.uniform(0, 7, size=n),
            "REB": rng.uniform(0, 10, size=n),
            "AST": rng.uniform(0, 8, size=n),
            "STL": rng.uniform(0, 3, size=n),
            "BLK": rng.uniform(0, 3, size=n),
            "TOV": rng.uniform(0, 4, size=n),
        }
    )

    df["TARGET_5Yrs"] = ((df["MIN"] > 20) & (df["PTS"] > 10)).astype(int)

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_run_model_cycle_e2e(tmp_path: Path):
    # ---- 1) Create toy features parquet ----
    features_path = tmp_path / "ml" / "data" / "processed" / "toy_features.parquet"
    _make_toy_feature_dataset(features_path)

    # ---- 2) Minimal model config for this test ----
    model_cfg = {
        "dataset": {
            "processed_path": str(features_path),
            "target_col": "TARGET_5Yrs",
            "id_col": "Name",
        },
        "features": {
            "all_base_features": [
                "GP",
                "MIN",
                "PTS",
                "FGM",
                "FGA",
                "FG%",
                "3P Made",
                "3PA",
                "3P%",
                "FTM",
                "FTA",
                "FT%",
                "OREB",
                "DREB",
                "REB",
                "AST",
                "STL",
                "BLK",
                "TOV",
            ],
            "all_engineered_features": [],
            "selected_for_training": {"use": "all"},
            "transformations": {
                "skewness_report_csv": str(
                    tmp_path / "ml" / "reports" / "skewness_report.csv"
                )
            },
            "scaling": {"enabled": True, "scaler": "standard"},
            "selection": {
                "enabled": False,  # keep it simple for this e2e test
                "method": "rf_importance",
                "top_k": 10,
                "corr_prune": False,
                "corr_threshold": 0.9,
                "always_keep": ["GP", "MIN", "PTS"],
            },
        },
        "reproducibility": {
            "random_state": 42,
            "test_size": 0.2,
            "val_size": 0.2,
        },
        "class_imbalance": {
            "enabled": True,
            "method": "smote",
            "smote": {"k_neighbors": 3},
            "min_positive_rate": 0.05,
        },
        "model": {
            "candidates": [
                {
                    "name": "rf",
                    "type": "random_forest",
                    "params": {
                        "n_estimators": 50,
                        "max_depth": 5,
                        "min_samples_leaf": 2,
                        "n_jobs": -1,
                    },
                }
            ]
        },
        "cv": {"enabled": True, "n_splits": 3, "shuffle": True},
        "metrics": {
            "primary": "f1",
            "extra": ["recall", "precision", "accuracy"],
        },
        "threshold_optimization": {
            "enabled": True,
            "metric": "f1",
        },
        "outputs": {
            "model_dir": str(tmp_path / "ml" / "models"),
            "best_model_filename": "nba_best_model.joblib",
            "metrics_report": str(tmp_path / "ml" / "reports" / "model_metrics.json"),
            "cv_report": str(tmp_path / "ml" / "reports" / "model_cv_report.json"),
            "candidate_report": str(
                tmp_path / "ml" / "reports" / "model_candidates.json"
            ),
            "shap_dir": str(tmp_path / "ml" / "reports" / "shap"),
        },
        "registry": {
            "mlflow_tracking_uri": "mlruns",
            "experiment_name": "nba_test_e2e",
            "register_best": False,
        },
        "run_metadata": {
            "save_manifest": True,
            "manifest_path": str(
                tmp_path / "ml" / "reports" / "model_run_manifest.json"
            ),
        },
    }

    cfg_path = tmp_path / "ml" / "configs" / "model.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(model_cfg))

    # Minimal skewness report (no transforms) so build_pipeline doesn't fail
    skew_report_path = (
        tmp_path / "ml" / "reports" / "skewness_report.csv"
    )
    skew_report_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for col in model_cfg["features"]["all_base_features"]:
        rows.append(
            {
                "feature": col,
                "skewness": 0.0,
                "zero_ratio": 0.0,
                "category": "low_skew",
                "suggestion": "none",
            }
        )

    pd.DataFrame(rows).to_csv(skew_report_path, index=False)

    # ---- 3) Run the model cycle ----
    run_model_cycle(str(cfg_path))

    # ---- 4) Assert artifacts exist ----
    model_dir = Path(model_cfg["outputs"]["model_dir"])
    model_path = model_dir / model_cfg["outputs"]["best_model_filename"]
    assert model_path.exists()

    metrics_path = Path(model_cfg["outputs"]["metrics_report"])
    assert metrics_path.exists()

    metrics = json.loads(metrics_path.read_text())
    assert "evaluation" in metrics
    assert "metrics_tuned" in metrics["evaluation"]
    assert "f1" in metrics["evaluation"]["metrics_tuned"]