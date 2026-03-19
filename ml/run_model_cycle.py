# Orchestrates the full MODEL CYCLE:
# 1) Load processed features
# 2) Feature selection (RF-based, optional)
# 3) Build feature pipeline (skewness-based transforms + scaling)
# 4) Model selection (RF/BRF/XGB) with CV + optional Optuna + threshold tuning
# 5) Final training on full train split
# 6) Evaluation on hold-out test split
# 7) Persist best pipeline + metrics reports + manifest

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import json

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from ml.utils.reproducibility import set_global_seed
from ml.model_pipeline.build_pipeline import build_feature_pipeline
from ml.model_pipeline.feature_selection_rf import rf_feature_selection
from ml.model_pipeline.model_selection import select_best_model
from ml.model_pipeline.train import train_best_pipeline
from ml.model_pipeline.evaluate import evaluate_with_threshold
from ml.validation.validation_gates import run_validation_gates
from ml.packaging.package_model import create_model_package
from ml.model_pipeline.registry.mlflow_utils import log_and_register_with_mlflow
from ml.model_pipeline.model_card import generate_model_card


def _safe_get(cfg: dict, keys: List[str], default: Any) -> Any:
    """
    Helper for nested config access:
    _safe_get(cfg, ["block", "field"], default).
    """
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def run(config_path: str = "ml/configs/model.yaml") -> None:
    """
    High-level orchestration entrypoint for the MODEL CYCLE.

    This function does NOT care about model details directly. Instead it:
      - Reads configuration
      - Loads processed feature dataset
      - Applies feature selection (RF-based) if enabled
      - Builds the feature engineering pipeline
      - Calls model selection logic (CV + optuna + threshold tuning)
      - Trains the best pipeline and evaluates on a hold-out test set
      - Saves model + metrics + selection reports
    """
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Model config not found: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text())

    # ------------------------------------------------------------------
    # 0) REPRODUCIBILITY SETTINGS
    # ------------------------------------------------------------------
    repro_cfg = cfg.get("reproducibility", {})
    random_state = int(repro_cfg.get("random_state", 42))
    test_size = float(repro_cfg.get("test_size", 0.2))
    set_global_seed(random_state)

    # ------------------------------------------------------------------
    # 1) DATASET PATHS + FEATURES 
    # ------------------------------------------------------------------
    data_cfg = cfg["dataset"]
    features_path = Path(data_cfg["processed_path"])
    target_col = data_cfg["target_col"]

    if not features_path.exists():
        raise FileNotFoundError(f"Features parquet not found: {features_path}")

    df = pd.read_parquet(features_path)

    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not in features dataset")

    # Optional: drop rows with missing target
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    feat_cfg = cfg["features"]

    # base + engineered feature lists
    all_base = feat_cfg.get("all_base_features", [])
    all_eng = feat_cfg.get("all_engineered_features", [])

    # Which features to use for training?
    sel_cfg = feat_cfg.get("selected_for_training", {"use": "all"})
    use_mode = sel_cfg.get("use", "all")

    if use_mode == "all":
        input_cols = list(all_base) + list(all_eng)
    elif use_mode == "base":
        input_cols = list(all_base)
    elif use_mode == "engineered":
        input_cols = list(all_eng)
    elif use_mode == "explicit":
        input_cols = list(sel_cfg.get("cols", []))
    else:
        raise ValueError(f"Unknown features.selected_for_training.use: {use_mode}")

    if not input_cols:
        raise ValueError("No feature columns selected for training.")

    all_features = input_cols

    # ------------------------------------------------------------------
    # 2) TRAIN / TEST SPLIT 
    # ------------------------------------------------------------------
    X = df[all_features].copy()
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # ------------------------------------------------------------------
    # 3) FEATURE SELECTION 
    # ------------------------------------------------------------------
    fs_cfg = feat_cfg.get("feature_selection", {})
    fs_enabled = bool(fs_cfg.get("enabled", True))

    if fs_enabled:
        selected_features, fs_details = rf_feature_selection(
            X_train=X_train,
            y_train=y_train,
            all_features=all_features,
            fs_cfg=fs_cfg,
            random_state=random_state,
        )
    else:
        selected_features = list(all_features)
        fs_details = {
            "enabled": False,
            "selected_features": selected_features,
            "reason": "Feature selection disabled in config",
        }

    # where to write reports
    out_cfg = cfg["outputs"]
    model_dir = Path(out_cfg["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    (model_dir / "feature_selection_report.json").write_text(
        json.dumps(fs_details, indent=2)
    )

    # ------------------------------------------------------------------
    # 4) FEATURE ENGINEERING PIPELINE (TRANSFORMS + SCALING) 
    # ------------------------------------------------------------------
    fe_cfg = feat_cfg.get("transformations", {})

    feature_transformer, fe_details = build_feature_pipeline(
        selected_features=selected_features,
        X_train=X_train,
        eng_cfg=fe_cfg,
    )

    (model_dir / "feature_engineering_report.json").write_text(
        json.dumps(fe_details, indent=2)
    )

    # ------------------------------------------------------------------
    # 5) CLASS IMBALANCE HANDLING CONFIG (SMOTE) 
    # ------------------------------------------------------------------
    imb_cfg = cfg.get("class_imbalance", {})
    smote_cfg = {
        "enabled": bool(imb_cfg.get("enabled", True)),
        "method": imb_cfg.get("method", "smote"),
        "smote": imb_cfg.get("smote", {"k_neighbors": 5}),
        "min_positive_rate": imb_cfg.get("min_positive_rate", 0.05),
    }

    # ------------------------------------------------------------------
    # 6) CV, METRICS, AND OPTUNA CONFIG 
    # ------------------------------------------------------------------
    cv_cfg = cfg.get("cv", {})
    n_splits = int(cv_cfg.get("n_splits", 5))
    cv_shuffle = bool(cv_cfg.get("shuffle", True))

    metrics_cfg = cfg.get("metrics", {})
    primary_metric = metrics_cfg.get("primary")
    extra_metrics = metrics_cfg.get("extra", ["recall", "precision", "accuracy","roc_auc"])
    metrics_list = [primary_metric] + [m for m in extra_metrics if m != primary_metric]
    scoring = {m: m for m in metrics_list}

    model_cfg = cfg["model"]
    candidates_cfg = model_cfg["candidates"]

    opt_cfg = model_cfg.get("optuna", {"enabled": True, "n_trials": 25})

    # ------------------------------------------------------------------
    # 7) MODEL SELECTION (BASELINE CV + OPTUNA + THRESHOLD TUNING) 
    # ------------------------------------------------------------------
    selection = select_best_model(
        X_train=X_train[selected_features],
        y_train=y_train,
        feature_transformer=feature_transformer,
        candidates_cfg=candidates_cfg,
        scoring=scoring,
        primary_metric=primary_metric,
        n_splits=n_splits,
        random_state=random_state,
        cv_shuffle=cv_shuffle,
        smote_cfg=smote_cfg,
        opt_cfg=opt_cfg,
        out_dir=model_dir,
    )

    best_pipeline = selection["best_pipeline"]
    best_threshold = selection["best_threshold"]


    (model_dir / "model_selection_results.json").write_text(
        json.dumps(selection["all_results"], indent=2)
    )

    # ------------------------------------------------------------------
    # 8) FINAL TRAINING ON TRAIN SPLIT 
    # ------------------------------------------------------------------
    best_pipeline = train_best_pipeline(
        best_pipeline,
        X_train[selected_features],
        y_train,
    )

    # ------------------------------------------------------------------
    # 9) HOLD-OUT TEST EVALUATION 
    # ------------------------------------------------------------------
    eval_results = evaluate_with_threshold(
        best_pipeline,
        X_test[selected_features],
        y_test,
        threshold=best_threshold,
    )
      # ------------------------------------------------------------------
    # 10) SAVE MODEL + METRICS 
    # ------------------------------------------------------------------
    model_filename = out_cfg["best_model_filename"]
    metrics_path = Path(out_cfg["metrics_report"])
    cv_metrics_path = Path(out_cfg["cv_report"])
    candidate_report_path = Path(
        out_cfg.get("candidate_report", model_dir / "model_candidates.json")
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    cv_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_report_path.parent.mkdir(parents=True, exist_ok=True)

    # persist trained pipeline locally (for debugging / reproducibility)
    model_path = model_dir / model_filename
    joblib.dump(best_pipeline, model_path)

    # Main summary
    selection_summary = {
        "best_model_name": selection["best_model_name"],
        "best_variant": selection["best_variant"],
        "comparison_score_train_cv": selection["comparison_score"],
        "best_threshold": best_threshold,
    }
    metrics_out = {
        "selection_summary": selection_summary,
        "evaluation": eval_results,
    }
    metrics_path.write_text(json.dumps(metrics_out, indent=2))

    # CV metrics per model/variant
    cv_metrics_path.write_text(json.dumps(selection["all_results"], indent=2))

    # candidate report
    candidate_report_path.write_text(json.dumps(selection["all_results"], indent=2))

    # ------------------------------------------------------------------
    # 12) MODEL PACKAGING (handoff for deployment) 
    # ------------------------------------------------------------------
    package_root = Path(out_cfg.get("package_root", model_dir / "packages"))
   
    package_info = create_model_package(
        package_root=package_root,
        model_path=model_path,
        threshold=best_threshold,
        features=selected_features,
        target_col=target_col,
        metrics=metrics_out,
        cfg_path=cfg_path,
    )

    print(f"✅ Model packaged: {package_info['package_dir']}")

    # ------------------------------------------------------------------
    # 13) VALIDATION GATES 
    # ------------------------------------------------------------------
    validation_gates_cfg = cfg.get("validation_gates")

    if validation_gates_cfg:
        report_1 = eval_results["metrics_tuned"]["classification_report"]["1"]

        gate_metrics = {
            "f1": float(report_1["f1-score"]),
            "recall": float(report_1["recall"]),
            # "previous_f1": 0.75,  # optional later if comparing to previous model
        }

        gate_results = run_validation_gates(
            metrics=gate_metrics,
            gates_cfg=validation_gates_cfg,
        )

        (model_dir / "validation_gates.json").write_text(
            json.dumps(gate_results, indent=2)
        )

        if not gate_results["passed"]:
            print("❌ Validation gates failed, model will NOT be promoted.")
            return
    else:
        gate_results = {
            "passed": True,
            "checks": {},
            "skipped": True,
            "reason": "No validation_gates section found in config",
        }

        (model_dir / "validation_gates.json").write_text(
            json.dumps(gate_results, indent=2)
        )

    # ------------------------------------------------------------------
    # 12) MLflow MODEL REGISTRY 
    # ------------------------------------------------------------------
    registry_cfg = cfg.get("registry", {}) #stage promotion is done automatically at the end of the cycle.
    if registry_cfg.get("register_best", False):
        tracking_uri = registry_cfg.get("mlflow_tracking_uri", "mlruns")
        experiment_name = registry_cfg.get("experiment_name", "nba_career_prediction")
        model_name = registry_cfg.get("model_name", "nba_career_model")
        stage = registry_cfg.get("stage", "Staging")

        mlflow_result = log_and_register_with_mlflow(
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            model_name=model_name,
            stage=stage,
            best_pipeline=best_pipeline,
            selection_summary=selection_summary,
            eval_results=eval_results,
            extra_params={
                "n_selected_features": len(selected_features),
                "selected_features": selected_features,
            },
        )

        # Persist registry outcome into a small JSON sidecar
        (model_dir / "mlflow_registry_result.json").write_text(
            json.dumps(mlflow_result, indent=2)
        )
        if mlflow_result.get("registered", False):
            print("✅ Model registered correctly")
        else:
            print(f"⚠️ Model registry skipped: {mlflow_result.get('reason')}")

    # ------------------------------------------------------------------
    # 13) RUN MANIFEST FOR MODEL CYCLE 
    # ------------------------------------------------------------------
    run_meta = cfg.get("run_metadata", {})
    if run_meta.get("save_manifest", True):
        manifest_path = Path(
            run_meta.get("manifest_path", model_dir / "model_run_manifest.json")
        )
        manifest = {
            "config_path": str(cfg_path),
            "features_path": str(features_path),
            "model_path": str(model_path),
            "random_state": random_state,
            "test_size": test_size,
            "selected_features": selected_features,
            "primary_metric": primary_metric,
            "best_model_name": selection["best_model_name"],
            "best_variant": selection["best_variant"],
            "best_threshold": best_threshold,
            "mlflow_registry_enabled": bool(registry_cfg.get("register_best", False)),
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))
    
    # ------------------------------------------------------------------
    # 15) MODEL CARD 
    # ------------------------------------------------------------------
    model_card_path = Path(out_cfg.get("model_card_path",str(model_dir / "model_card.md")))

    model_card_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        generate_model_card(
            cfg=cfg,
            selection= {
                "best_model_name" : selection["best_model_name"],
                "best_variant" : selection["best_variant"],
                "comparison_score" : selection["comparison_score"],
                "best_threshold" : best_threshold,
            },
            eval_results=eval_results,
            manifest=manifest,
            output_path=model_card_path,
        )

        print(f"📄 Model card generated: {model_card_path}")

    except Exception as e:
        print(f"⚠️ Model card generation skipped: {e}")

if __name__ == "__main__":
    run()