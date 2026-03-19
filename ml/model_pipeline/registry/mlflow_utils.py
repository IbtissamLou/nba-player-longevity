from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np

#take the best pipeline trained, log it to MLflow, register it into the Model Registry, and optionally move it to a stage

def _flatten_params(d: Dict[str, Any]) -> Dict[str, str]:
    """
    Flatten a nested dict into simple string params suitable for mlflow.log_params.

    Anything non-scalar is stringified so we don't accidentally break MLflow.
    """
    flat: Dict[str, str] = {}

    def _rec(prefix: str, obj: Any):
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_prefix = f"{prefix}.{k}" if prefix else k
                _rec(new_prefix, v)
        else:
            # Accept simple scalars, stringify everything else
            if isinstance(obj, (str, int, float, bool)) or obj is None:
                flat[prefix] = str(obj)
            else:
                flat[prefix] = str(obj)

    _rec("", d)
    return flat


def log_and_register_with_mlflow(
    *,
    tracking_uri: str,
    experiment_name: str,
    model_name: str,
    stage: str,
    best_pipeline,
    selection_summary: Dict[str, Any],
    eval_results: Dict[str, Any],
    extra_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Log the best model + metrics to MLflow and register it in the Model Registry.

    Returns a small dict with:
      - registered: bool
      - run_id: Optional[str]
      - model_version: Optional[str]
      - stage: Optional[str]
      - reason: Optional[str] (if not registered)
    """
    try:
        import mlflow
        import mlflow.sklearn  # type: ignore
    except ImportError:
        print("⚠️ MLflow is not installed. Skipping model registry step.")
        return {
            "registered": False,
            "reason": "mlflow_not_installed",
            "run_id": None,
            "model_version": None,
            "stage": None,
        }

    # ------------------------------------------------------------------
    # 1) Setup tracking + experiment
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # ------------------------------------------------------------------
    # 2) Start run, log params + metrics + model
    # ------------------------------------------------------------------
    with mlflow.start_run(run_name=f"{model_name}_train") as run:
        run_id = run.info.run_id

        # ---- Params: selection summary + extras  ----
        params_to_log = {
            "selection": selection_summary,
        }
        if extra_params:
            params_to_log["extra"] = extra_params

        flat_params = _flatten_params(params_to_log)
        mlflow.log_params(flat_params)

        # ---- Metrics: use tuned metrics from our eval results ----
        tuned = eval_results.get("metrics_tuned", {})
        for key, value in tuned.items():
            if isinstance(value, (int, float)) and not (
                isinstance(value, float) and np.isnan(value)
            ):
                mlflow.log_metric(key, float(value))

        # ---- Log model + register in registry ----
        mlflow.sklearn.log_model( 
            best_pipeline,
            artifact_path="model",
            registered_model_name=model_name,
        )

    # ------------------------------------------------------------------
    # 3) Resolve latest model version and optionally move to a stage
    # ------------------------------------------------------------------
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    versions = client.get_latest_versions(model_name)
    if not versions:
        return {
            "registered": False,
            "reason": "no_versions_found",
            "run_id": run_id,
            "model_version": None,
            "stage": None,
        }

    latest_version = versions[-1]
    model_version = latest_version.version

    final_stage = None
    if stage and stage.lower() != "none":
       
        try:
            client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage=stage,
                archive_existing=True,
            )
        except TypeError:
            # Older / different MLflow version: call without archive_existing
            client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage=stage,
            )
        final_stage = stage

    return {
        "registered": True,
        "run_id": run_id,
        "model_version": model_version,
        "stage": final_stage,
        "reason": None,
    }