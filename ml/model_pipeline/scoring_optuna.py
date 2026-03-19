from __future__ import annotations

from typing import Dict, Any
from pathlib import Path

import numpy as np
import pandas as pd
import json

from sklearn.base import clone

from ml.model_pipeline.scoring_kfold import run_kfold_cv
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

# Optuna is optional
try:
    import optuna  # type: ignore
except ImportError:  # pragma: no cover
    optuna = None


def _build_estimator_from_candidate(
    cand_cfg: Dict[str, Any],
    random_state: int,
):
    """
    Local (non-circular) version of the estimator builder.
    Mirrors the logic from model_selection._build_estimator but kept
    here to avoid circular imports.
    """

    name = cand_cfg.get("name", "")
    model_type = cand_cfg["type"]
    params = dict(cand_cfg.get("params", {}))
    params.setdefault("random_state", random_state)

    if model_type == "random_forest":
        return RandomForestClassifier(**params)
    elif model_type == "xgboost":
        params.setdefault("use_label_encoder", False)
        params.setdefault("eval_metric", "logloss")
        return XGBClassifier(**params)
    elif model_type == "balanced_random_forest":
        return BalancedRandomForestClassifier(**params)
    else:
        raise ValueError(f"Unsupported model type for Optuna search: {model_type}")
    

def optuna_search_for_candidate(
    *,
    name: str,
    base_candidate_cfg: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_transformer,
    metric_fns: Dict[str, Any],
    primary_metric: str,
    n_splits: int,
    random_state: int,
    cv_shuffle: bool,
    smote_cfg: Dict[str, Any],
    n_trials: int,
    out_dir: Path,
) -> Dict[str, Any]:
    """
    Hyperparameter search wrapper for a single candidate using Optuna.

    IMPORTANT:
    - Optuna is treated as OPTIONAL. If it's not installed, this function
      falls back to a simple CV run using the base_candidate_cfg params
      and returns that as the "optuna" result.
    - This keeps tests and pipelines working without forcing Optuna as
      a hard dependency.
    """

    out_dir.mkdir(parents=True, exist_ok=True)


    if optuna is None: 
        # Just evaluate the base candidate once with CV.
        estimator = _build_estimator_from_candidate(
            base_candidate_cfg,
            random_state=random_state,
        )

        cv_res = run_kfold_cv(
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

        return {
            "mean_scores": cv_res["mean_scores"],
            "fold_scores": cv_res["fold_scores"],
            "best_params": dict(base_candidate_cfg.get("params", {})),
            "n_trials": 0,
            "optuna_used": False,
        }

    # ------------------------------------------------------------------
    # Otherwise: real Optuna search (simplified search space).
    # ------------------------------------------------------------------

    model_type = base_candidate_cfg["type"]
    base_params = dict(base_candidate_cfg.get("params", {}))

    def objective(trial: "optuna.trial.Trial") -> float: # type: ignore
        params = dict(base_params)

        if model_type == "random_forest":
            params["n_estimators"] = trial.suggest_int("n_estimators", 100, 1000, step=100)
            params["max_depth"] = trial.suggest_categorical("max_depth", [5, 10,15,None])
            params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1,10)
            params["min_samples_split"] = trial.suggest_int("min_samples_split", 2,20)
            params["max_features"] = trial.suggest_categorical("max_features", ["sqrt","log2",None])

        elif model_type == "xgboost":
            params["n_estimators"] = trial.suggest_int("n_estimators", 100, 500, step=100)
            params["max_depth"] = trial.suggest_int("max_depth", 3, 10)
            params["learning_rate"] = trial.suggest_float(
                "learning_rate", 0.01, 1, log=True
            )
        elif model_type == "balanced_random_forest":
            params["n_estimators"] = trial.suggest_int("n_estimators", 100, 1000, step=100)
            params["max_depth"] = trial.suggest_categorical("max_depth", [5, 10,15,None])
            params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1,10)
            params["min_samples_split"] = trial.suggest_int("min_samples_split", 2,20)
            params["max_features"] = trial.suggest_categorical("max_features", ["sqrt","log2",None])

        estimator = _build_estimator_from_candidate(
            {"name": name, "type": model_type, "params": params},
            random_state=random_state,
        )

        cv_res = run_kfold_cv(
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

        # Maximize primary_metric
        return cv_res["mean_scores"][primary_metric]

    study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), #Used Optuna pruners to cut off bad trials early
)
    study.optimize(objective, n_trials=n_trials)

    # Rebuild estimator with best trial params
    best_params = dict(base_params)
    best_params.update(study.best_trial.params)

    best_estimator = _build_estimator_from_candidate(
        {"name": name, "type": model_type, "params": best_params},
        random_state=random_state,
    )

    # Evaluate best estimator once more for full metrics
    cv_res_best = run_kfold_cv( #Runs run_kfold_cv one more time to get stable metrics for that tuned model.
        X=X_train,
        y=y_train,
        feature_transformer=feature_transformer,
        estimator=best_estimator,
        metric_fns=metric_fns,
        n_splits=n_splits,
        random_state=random_state,
        shuffle=cv_shuffle,
        smote_cfg=smote_cfg,
    )

    summary = {
    "best_trial": {
        "params": study.best_trial.params,
        "value": study.best_trial.value,
        "number": study.best_trial.number,
    },
    "n_trials": len(study.trials),
    }
    (out_dir / f"{name}_optuna_study_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
  

    return {
        "mean_scores": cv_res_best["mean_scores"],
        "fold_scores": cv_res_best["fold_scores"],
        "best_params": best_params,
        "n_trials": n_trials,
        "optuna_used": True,
    }