from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional
import datetime
import json


def _safe_get(d: Dict[str, Any], keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def generate_model_card(
    *,
    cfg: Dict[str, Any],
    selection: Dict[str, Any],
    eval_results: Dict[str, Any],
    manifest: Optional[Dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Generate a Markdown model card for the NBA career prediction model.

    - cfg: full model config (parsed model.yaml)
    - selection: output from select_best_model(...)
    - eval_results: output from evaluate_with_threshold(...)
    - manifest: optional run metadata dict from run_model_cycle
    - output_path: where to write MODEL_CARD.md
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # Basic info from config
    # --------------------------------------------------------------
    dataset_cfg = cfg.get("dataset", {})
    feat_cfg = cfg.get("features", {})
    model_cfg = cfg.get("model", {})
    metrics_cfg = cfg.get("metrics", {})
    repro_cfg = cfg.get("reproducibility", {})

    target_col = dataset_cfg.get("target_col", "TARGET_5Yrs")
    id_col = dataset_cfg.get("id_col", "Name")
    all_base = feat_cfg.get("all_base_features", [])
    all_eng = feat_cfg.get("all_engineered_features", [])

    primary_metric = metrics_cfg.get("primary", "f1")
    extra_metrics = metrics_cfg.get("extra", [])

    best_model_name = selection.get("best_model_name", "unknown")
    best_variant = selection.get("best_variant", "baseline")
    comparison_score = selection.get("comparison_score", None)
    best_threshold = selection.get("best_threshold", 0.5)

    # evaluation metrics (handle both flat dict or {"metrics": {...}})
    metrics_block = eval_results.get("metrics", eval_results)
    thresh_used = eval_results.get("threshold", best_threshold)

    # manifest info if available
    run_random_state = None
    features_path = None
    model_path = None
    if manifest:
        run_random_state = manifest.get("random_state", None)
        features_path = manifest.get("features_path", None)
        model_path = manifest.get("model_path", None)

    now_utc = datetime.datetime.utcnow().isoformat() + "Z"

    # --------------------------------------------------------------
    # Build Markdown lines
    # --------------------------------------------------------------
    lines = []

    # Title
    lines.append("# Model Card – NBA 5-Year Career Prediction")
    lines.append("")
    lines.append(f"_Generated on: {now_utc}_")
    lines.append("")

    # 1. Model Overview
    lines.append("## 1. Model Overview")
    lines.append("")
    lines.append(
        "This model predicts whether an NBA player is likely to stay in the league "
        "for at least 5 years (`TARGET_5Yrs`). It is built as part of an end-to-end "
        "ML engineering project with full data, feature, model, deployment, monitoring, "
        "and retraining cycles."
    )
    lines.append("")
    lines.append(f"- **Target column**: `{target_col}`")
    lines.append(f"- **ID column**: `{id_col}`")
    lines.append(f"- **Best model**: `{best_model_name}` (variant: `{best_variant}`)")
    lines.append(f"- **Primary selection metric**: `{primary_metric}`")
    if comparison_score is not None:
        lines.append(f"- **Best CV score ({primary_metric})**: `{comparison_score:.4f}`")
    lines.append(f"- **Decision threshold used**: `{thresh_used:.3f}`")
    lines.append("")

    # 2. Data
    lines.append("## 2. Data")
    lines.append("")
    lines.append("### 2.1 Dataset")
    lines.append("")
    lines.append(
        "- Source: Historical NBA player stats (per-player season-level features).\n"
        "- Each row corresponds to a single player season with aggregated statistics."
    )
    lines.append("")
    if features_path:
        lines.append(f"- **Processed features file**: `{features_path}`")
    lines.append("")
    lines.append("### 2.2 Features")
    lines.append("")
    lines.append("- **Base features used** (raw box-score stats, minutes, shooting, etc.):")
    if all_base:
        lines.append("  - " + ", ".join(f"`{c}`" for c in all_base))
    else:
        lines.append("  - (Not explicitly listed in config)")
    lines.append("")
    if all_eng:
        lines.append("- **Engineered features** (ratios, interactions, domain-based features):")
        lines.append("  - " + ", ".join(f"`{c}`" for c in all_eng))
        lines.append("")
    else:
        lines.append("- **Engineered features**: none or handled upstream in feature cycle.")
        lines.append("")

    # 3. Training & Evaluation
    lines.append("## 3. Training & Evaluation")
    lines.append("")
    test_size = repro_cfg.get("test_size", 0.2)
    lines.append(f"- Train/test split: `{1.0 - test_size:.2f} / {test_size:.2f}` stratified by target.")
    if run_random_state is not None:
        lines.append(f"- Global random_state for reproducibility: `{run_random_state}`")
    else:
        lines.append(f"- random_state (from config): `{repro_cfg.get('random_state', 42)}`")
    lines.append("")
    lines.append("- **Class imbalance handling**: SMOTE applied inside CV/threshold tuning.")
    lines.append("- **Feature pipeline**: skewness-based transformation + scaling (from feature cycle).")
    lines.append("")

    lines.append("### 3.1 Test Metrics")
    lines.append("")
    if metrics_block:
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for m, v in metrics_block.items():
            try:
                lines.append(f"| `{m}` | `{float(v):.4f}` |")
            except Exception:
                continue
        lines.append("")
    else:
        lines.append("_No evaluation metrics available in eval_results._")
        lines.append("")

    # 4. Model Behaviour & Limitations
    lines.append("## 4. Model Behaviour & Limitations")
    lines.append("")
    lines.append(
        "- The model tends to be more confident on players with consistent minutes and scoring volume.\n"
        "- Predictions are **probabilistic**: threshold tuning was used to balance recall and precision "
        "for the positive class (players staying ≥5 years).\n"
        "- Performance may degrade on eras or play styles not represented in the training data "
        "(e.g., very old seasons or future strategic shifts in the NBA)."
    )
    lines.append("")
    lines.append("**Known limitations:**")
    lines.append("")
    lines.append("- Model is trained purely on box-score style statistics; it does **not** account for injuries,")
    lines.append("  contract situations, or off-court factors.")
    lines.append("- It should **not** be used for real-world player decisions (contracts, drafting, etc.).")
    lines.append("- It is an educational / portfolio project demonstrating ML engineering best practices.")
    lines.append("")

    # 5. Fairness & Ethical Considerations
    lines.append("## 5. Fairness & Ethical Considerations")
    lines.append("")
    lines.append(
        "This model is trained on sports performance data and does not explicitly encode sensitive attributes. "
        "However, like any predictive model on human careers, it could implicitly learn patterns that correlate "
        "with non-performance factors. It should therefore be treated as a **technical demo only**, not a tool "
        "for decision-making about real people."
    )
    lines.append("")

    # 6. Versioning & Reproducibility
    lines.append("## 6. Versioning & Reproducibility")
    lines.append("")
    registry_cfg = cfg.get("registry", {})
    experiment_name = registry_cfg.get("experiment_name", "nba_career_model")
    tracking_uri = registry_cfg.get("mlflow_tracking_uri", "mlruns")
    lines.append(f"- **MLflow tracking URI**: `{tracking_uri}`")
    lines.append(f"- **MLflow experiment**: `{experiment_name}`")
    lines.append(
        "- Each model training run logs parameters, metrics, and artifacts to MLflow, "
        "and is optionally registered in the MLflow Model Registry."
    )
    lines.append("")
    if model_path:
        lines.append(f"- Local packaged model path: `{model_path}`")
    lines.append("")
    lines.append(
        "Reproducibility is supported via:\n"
        "- Fixed random seed in `model.yaml`\n"
        "- DVC-tracked data + feature pipelines\n"
        "- Run manifest JSON linking config, data, and model artifacts\n"
        "- MLflow experiment + model registry"
    )
    lines.append("")

    # 7. How to Use the Model
    lines.append("## 7. How to Use This Model")
    lines.append("")
    lines.append("### 7.1 Programmatic (Python)")
    lines.append("")
    lines.append("```python")
    lines.append("import joblib")
    lines.append("import pandas as pd")
    lines.append("")
    lines.append("# Load saved pipeline")
    lines.append("pipe = joblib.load('ml/models/nba_best_model.joblib')")
    lines.append("")
    lines.append("# X_new must contain the same feature columns used in training")
    lines.append("proba = pipe.predict_proba(X_new)[:, 1]")
    lines.append("pred = (proba >= THRESHOLD).astype(int)")
    lines.append("```")
    lines.append("")
    lines.append("### 7.2 MLflow (Production-style)")
    lines.append("")
    lines.append("```python")
    lines.append("import mlflow.pyfunc")
    lines.append("")
    lines.append("model = mlflow.pyfunc.load_model('models:/nba_career_model/Production')")
    lines.append("preds = model.predict(X_new)")
    lines.append("```")
    lines.append("")

    # 8. Change Log
    lines.append("## 8. Change Log")
    lines.append("")
    lines.append("- This model card is auto-generated at the end of `run_model_cycle.py`.")
    lines.append("- It reflects the latest training run and evaluation metrics at generation time.")
    lines.append("")

    # --------------------------------------------------------------
    # Write file
    # --------------------------------------------------------------
    output_path.write_text("\n".join(lines), encoding="utf-8")