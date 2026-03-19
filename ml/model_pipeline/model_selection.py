"""
Model selection logic for the NBA project.

Responsibilities:
- Build metric functions from names (f1, recall, precision, accuracy, roc_auc, etc.)
- For each model candidate:
    - Run baseline K-fold CV
    - Optionally run Optuna-based hyperparameter search
    - Optionally tune threshold on validation
- Compare models on a primary metric 
- Return the best pipeline + threshold + detailed results
"""

from __future__ import annotations

from typing import Dict, Any, List
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
import xgboost as xgb
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
)

from ml.model_pipeline.scoring_kfold import run_kfold_cv
from ml.model_pipeline.scoring_optim import tune_threshold_cv
from ml.model_pipeline.scoring_optuna import optuna_search_for_candidate


def _build_estimator(candidate_cfg: Dict[str, Any], random_state: int) -> Any:
    """
    Factory to build an estimator from config.
    Extend this to support BRF, XGB, etc.
    """
    ctype = candidate_cfg["type"]
    params = dict(candidate_cfg.get("params", {}))

    # Always inject random_state if supported
    params.setdefault("random_state", random_state)

    if ctype == "random_forest":
        return RandomForestClassifier(**params)
    
    if ctype == "balanced_random_forest":
        return BalancedRandomForestClassifier(**params)
    
    if ctype == "xgboost":
        return  xgb.XGBClassifier(**params)


    raise ValueError(f"Unknown model type: {ctype}")


def _metric_functions(metric_names: List[str]) -> Dict[str, Any]:
    """
    Map metric names to functions(y_true, y_pred or y_proba).
    """
    mapping = {}
    for name in metric_names:
        if name == "f1":
            mapping[name] = f1_score
        elif name == "recall":
            mapping[name] = recall_score
        elif name == "precision":
            mapping[name] = precision_score
        elif name == "accuracy":
            mapping[name] = accuracy_score
        elif name == "roc_auc":
            mapping[name] = roc_auc_score
        elif name == "average_precision":
            mapping[name] = average_precision_score
        elif name == "balanced_accuracy":
            # BEST METRIC FOR IMBALANCED DATA
            mapping[name] = balanced_accuracy_score
        else:
            raise ValueError(f"Unsupported metric: {name}")
    return mapping


def select_best_model(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_transformer,
    candidates_cfg: List[Dict[str, Any]],
    scoring: Dict[str, str],
    primary_metric: str,
    n_splits: int,
    random_state: int,
    cv_shuffle: bool,
    smote_cfg: Dict[str, Any],
    opt_cfg: Dict[str, Any],
    out_dir: Path,
) -> Dict[str, Any]:
    """
    High-level model selection over multiple candidates.

    Returns dict with:
      - "best_pipeline"
      - "best_threshold"
      - "best_model_name"
      - "best_variant" (e.g. "baseline", "optuna")
      - "comparison_score"
      - "all_results" (per model/variant metrics)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build metric functions from scoring dict keys
    metric_names = list(scoring.keys())
    metric_fns = _metric_functions(metric_names)

    all_results: Dict[str, Any] = {}
    best_model_name = None
    best_variant = None
    best_score = -np.inf
    best_threshold = 0.5
    best_estimator_params: Dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Loop over model candidates
    # ------------------------------------------------------------------
    for cand in candidates_cfg:
        name = cand["name"]
        estimator = _build_estimator(cand, random_state=random_state)

        # --------------------------------------------------------------
        # 1) Baseline CV with default params + threshold=0.5 
        # --------------------------------------------------------------
        baseline_cv = run_kfold_cv(
            X=X_train,
            y=y_train,
            feature_transformer=feature_transformer,
            estimator=estimator,
            metric_fns=metric_fns,
            n_splits=n_splits,
            random_state=random_state,
            shuffle=cv_shuffle,
            smote_cfg=smote_cfg,
        )

        baseline_mean = baseline_cv["mean_scores"][primary_metric]

        all_results[f"{name}_baseline"] = {
            "variant": "baseline",
            "mean_scores": baseline_cv["mean_scores"],
            "fold_scores": baseline_cv["fold_scores"],
        }

        # Track best so far
        if baseline_mean > best_score:
            best_score = baseline_mean
            best_model_name = name
            best_variant = "baseline"
            best_estimator_params = cand.get("params", {})

        # --------------------------------------------------------------
        # 2) Optuna search (hyperparameters) 
        # --------------------------------------------------------------
        if opt_cfg.get("enabled", True):
            opt_res = optuna_search_for_candidate(
                name=name,
                base_candidate_cfg=cand,
                X_train=X_train,
                y_train=y_train,
                feature_transformer=feature_transformer,
                metric_fns=metric_fns,
                primary_metric=primary_metric,
                n_splits=n_splits,
                random_state=random_state,
                cv_shuffle=cv_shuffle,
                smote_cfg=smote_cfg,
                n_trials=int(opt_cfg.get("n_trials", 25)),
                out_dir=out_dir,
            )

            all_results[f"{name}_optuna"] = {
                "variant": "optuna",
                "mean_scores": opt_res["mean_scores"],
                "fold_scores": opt_res["fold_scores"],
                "best_params": opt_res["best_params"],
            }

            opt_mean = opt_res["mean_scores"][primary_metric]

            if opt_mean > best_score:
                best_score = opt_mean
                best_model_name = name
                best_variant = "optuna"
                best_estimator_params = opt_res["best_params"]

    # ------------------------------------------------------------------
    # At this point, we know which model name/variant is best by CV.
    # We now:
    #   - Rebuild the best estimator with chosen params 
    #   - Tune threshold using tune_threshold_cv 
    #   - Build a SINGLE final pipeline (features + best_estimator) 
    # ------------------------------------------------------------------
    if best_model_name is None:
        raise RuntimeError("No best model selected. Check configuration and CV.")

    best_cand = next(c for c in candidates_cfg if c["name"] == best_model_name)

    # merge base params + tuned params 
    final_params = dict(best_cand.get("params", {}))
    if best_estimator_params is not None:
        final_params.update(best_estimator_params)
    final_params.setdefault("random_state", random_state)

    final_estimator = _build_estimator(
        {"name": best_cand["name"], "type": best_cand["type"], "params": final_params},
        random_state=random_state,
    )

    #For classification, run cross-validated threshold optimization to pick the decision boundary that maximizes
    #  our primary metric, based on out-of-fold probabilities.
    #  That threshold is then used consistently in test evaluation and later in production.
    thresh_res = tune_threshold_cv(
        X=X_train,
        y=y_train,
        feature_transformer=feature_transformer,
        estimator=final_estimator,
        smote_cfg=smote_cfg,
        n_splits=n_splits,
        random_state=random_state,
        metric=primary_metric,
    )

    best_threshold = thresh_res["best_threshold"]

    # Final pipeline used for training & inference (no SMOTE in inference)
    best_pipeline = Pipeline(
        steps=[
            ("features", feature_transformer),
            ("model", final_estimator),
        ]
    )

    # save threshold search trace
    (out_dir / "threshold_tuning.json").write_text(json.dumps(thresh_res, indent=2))

    return {
        "best_pipeline": best_pipeline,
        "best_threshold": best_threshold,
        "best_model_name": best_model_name,
        "best_variant": best_variant,
        "comparison_score": best_score,
        "all_results": all_results,
    }