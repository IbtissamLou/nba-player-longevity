from __future__ import annotations
from typing import Dict, Any

def run_validation_gates(
    metrics: Dict[str, Any],
    gates_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Decide if the model is allowed to be promoted / registered.

    metrics      -> typically eval_results["metrics_tuned"]
    gates_cfg    -> from model.yaml, e.g.:

    validation_gates:
      min_f1: 0.70
      min_recall: 0.65
      max_drop_vs_previous: 0.05   # optional, needs previous_f1 in metrics
    """
    results: Dict[str, Any] = {"passed": True, "checks": {}}

    f1 = float(metrics.get("f1"))
    recall = float(metrics.get("recall"))
    previous_f1 = metrics.get("previous_f1", None)

    # 1) Minimum F1 gate
    if "min_f1" in gates_cfg:
        ok = f1 >= float(gates_cfg["min_f1"])
        results["checks"]["min_f1"] = {
            "ok": ok,
            "value": f1,
            "threshold": float(gates_cfg["min_f1"]),
        }
        results["passed"] &= ok

    # 2) Minimum recall gate
    if "min_recall" in gates_cfg:
        ok = recall >= float(gates_cfg["min_recall"])
        results["checks"]["min_recall"] = {
            "ok": ok,
            "value": recall,
            "threshold": float(gates_cfg["min_recall"]),
        }
        results["passed"] &= ok

    # 3) Regression gate vs previous model (optional)
    if "max_drop_vs_previous" in gates_cfg and previous_f1 is not None:
        drop = float(previous_f1) - f1
        ok = drop <= float(gates_cfg["max_drop_vs_previous"])
        results["checks"]["max_drop_vs_previous"] = {
            "ok": ok,
            "value": drop,
            "threshold": float(gates_cfg["max_drop_vs_previous"]),
            "previous_f1": float(previous_f1),
            "current_f1": f1,
        }
        results["passed"] &= ok

    return results