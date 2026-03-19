"""
End-to-end Data Cycle runner (reproducible entrypoint).

This module runs the full data pipeline in a deterministic way:
ingest -> clean -> validate -> write artifacts -> write reports/manifest.
All behavior is configuration-driven via YAML to avoid hidden parameters.
"""

from __future__ import annotations

from ml.utils.reproducibility import set_global_seed

import logging

logger = logging.getLogger(__name__)
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from ml.data_pipeline.ingest import ingest_csv
from ml.data_pipeline.clean import run_cleaning
from ml.data_pipeline.validate import validate_schema, validate_rules, ValidationConfig
from ml.data_pipeline.stats import run_statistical_checks, build_profile
from ml.data_pipeline.anomaly import flag_quantile_outliers, AnomalyConfig
from ml.data_pipeline.stats import compare_profiles


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str: #proves which exact dataset file was used
    """Compute SHA256 hash of a file (used to track data version deterministically)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str: #proves which exact configuration produced these outputs.
    """Compute SHA256 hash for text (used to hash config content deterministically)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_get(cfg: dict, keys: list[str], default): #Makes future changes less risky / add optional config fields without breaking code
    """Small helper to read optional nested config fields safely."""
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _write_parquet(df: pd.DataFrame, out_path: str, *, sort_by: list[str] | None, write_index: bool) -> None: #making the output deterministic
    """Write parquet deterministically (stable ordering helps reproducibility and diffing)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    df_to_save = df
    if sort_by:
        existing = [c for c in sort_by if c in df_to_save.columns]
        if existing:
            df_to_save = df_to_save.sort_values(existing).reset_index(drop=True)

    df_to_save.to_parquet(out, index=write_index)



def run(config_path: str = "ml/configs/data.yaml") -> None:
    # -------- (0) Load config (single source of truth) -------------------------- 
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    config_text = config_file.read_text()
    cfg = yaml.safe_load(config_text)

    reports_dir = Path(_safe_get(cfg, ["reports", "dir"], "ml/reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)

    profile_path = reports_dir / "data_profile.json"
    checks_path = reports_dir / "data_checks.json"
    anomaly_path = reports_dir / "anomaly_report.json"
    drift_path = reports_dir / "drift_report.json"   # future
    validation_summary_path = reports_dir / "data_validation_summary.json"  


    # -------- (0.1) Set global seed -----------------------------------------------
    set_global_seed(cfg["reproducibility"]["random_state"])

    # Required dataset settings
    raw_path = cfg["dataset"]["raw_path"]
    interim_path = cfg["dataset"]["interim_path"]
    processed_path = cfg["dataset"]["processed_path"]
    target_col = cfg["dataset"]["target_col"]

    # reproducibility/output settings 
    sort_by = _safe_get(cfg, ["outputs", "sort_by"], None)      
    write_index = bool(_safe_get(cfg, ["outputs", "write_index"], False))
    save_profile = bool(_safe_get(cfg, ["stats", "save_profile"], True))

    # Manifest settings (for traceability) #track versions 
    save_manifest = bool(_safe_get(cfg, ["run_metadata", "save_manifest"], True))
    manifest_path = _safe_get(cfg, ["run_metadata", "manifest_path"], "ml/reports/run_manifest.json")

    # -------- (1) Ingest ------------------------------------------------------------
    df = ingest_csv(raw_path) #Reads raw CSV

    # -------- (2) Clean (data-cycle safe + deterministic) ---------------------------
    df = run_cleaning(
        df,
        do_drop_duplicates=cfg["cleaning"]["drop_duplicates"],
        do_impute_3p_pct=cfg["cleaning"]["impute_3p_pct"],
        do_correct_impossible_values=cfg["cleaning"]["correct_impossible_values"],
        percent_cols=cfg["cleaning"]["percent_cols"],
        percent_bounds=tuple(cfg["cleaning"]["percent_bounds"]),
        min_cap=cfg["cleaning"]["min_cap"],
        non_negative_cols=cfg["cleaning"]["non_negative_cols"],
    )

    # Save interim artifact  
    _write_parquet(df, interim_path, sort_by=sort_by, write_index=write_index) #after cleaning

    # -------- (3) Validate (schema then rules) ------------------------------------------

    df = validate_schema(df)  # enforces columns/types/ranges from schema contract

    cfg_rules = ValidationConfig( #rules 
        enforce_ranges=cfg["validation"]["enforce_ranges"],
        enforce_made_leq_attempted=cfg["validation"]["enforce_made_leq_attempted"],
        enforce_rebounds_identity=cfg["validation"]["enforce_rebounds_identity"],
        rebounds_tolerance=cfg["validation"]["rebounds_tolerance"],
        min_cap=cfg["cleaning"]["min_cap"],
        percent_bounds=tuple(cfg["cleaning"]["percent_bounds"]),
        non_negative_cols=tuple(cfg["cleaning"]["non_negative_cols"]),
    )

    validate_rules(df, cfg_rules)

    # Save processed artifact
    _write_parquet(df, processed_path, sort_by=sort_by, write_index=write_index) #after validation 

    # -------- (4) Stats report (dataset fingerprint) ---------------------------------------
    if save_profile:
        profile = build_profile(df, target_col=target_col)
        profile_path.write_text(json.dumps(profile, indent=2))

        checks = run_statistical_checks(
            df,
            profile,
            target_col=target_col,
            missingness_warn_threshold=cfg["stats"]["missingness_warn_threshold"],
            missingness_fail_threshold=cfg["stats"].get("missingness_fail_threshold", 0.50),
            outlier_zscore_threshold=cfg["stats"].get("outlier_zscore_threshold", 4.0),
            outlier_rate_warn=cfg["stats"].get("outlier_rate_warn", 0.02),
            constant_unique_ratio_threshold=cfg["stats"].get("constant_unique_ratio_threshold", 0.01),
        )

        checks_path.write_text(json.dumps(checks, indent=2))

        if checks["status"] == "fail":
            raise ValueError("Statistical checks failed. Aborting pipeline.")
    else:
        profile = None  # so drift/anomaly logic can handle it safely

    # -------- (4.5) Drift detection (compare to baseline profile) -------------------------
    drift_enabled = bool(_safe_get(cfg, ["drift", "enabled"], False))
    baseline_profile_path = _safe_get(cfg, ["drift", "baseline_profile"], None)
    fail_on_drift = bool(_safe_get(cfg, ["drift", "fail_on_drift"], True))

    drift_report = None

    if drift_enabled:
        if profile is None:
            drift_report = {
                "enabled": True,
                "status": "skipped",
                "reason": "Profile not generated (stats.save_profile=false).",
            }
            drift_path.write_text(json.dumps(drift_report, indent=2))

        else:
            if not baseline_profile_path:
                drift_report = {
                    "enabled": True,
                    "status": "skipped",
                    "reason": "No drift.baseline_profile configured.",
                }
                drift_path.write_text(json.dumps(drift_report, indent=2))

            else:
                baseline_p = Path(baseline_profile_path)
                if not baseline_p.exists():
                    drift_report = {
                        "enabled": True,
                        "status": "no_baseline",
                        "reason": f"Baseline profile not found at {baseline_p}.",
                        "hint": "Generate baseline by copying a known-good data_profile.json to this path.",
                    }
                    drift_path.write_text(json.dumps(drift_report, indent=2))

                else:
                    baseline = json.loads(baseline_p.read_text())
                    drift_report = compare_profiles(profile, baseline)
                    drift_path.write_text(json.dumps(drift_report, indent=2))

                    if fail_on_drift and drift_report.get("status") == "fail":
                        raise ValueError("Drift too high — aborting pipeline.")


    # -------- (5) Anomaly detection -----------------------------------------------------------

    an_cfg = AnomalyConfig(
    enabled=cfg["anomaly"]["enabled"],
    q_low=cfg["anomaly"]["q_low"],
    q_high=cfg["anomaly"]["q_high"],
    min_non_null=cfg["anomaly"]["min_non_null"],
    exclude_cols=tuple(cfg["anomaly"]["exclude_cols"]),
    max_outlier_rate_warn=cfg["anomaly"]["max_outlier_rate_warn"],
    max_outlier_rate_fail=cfg["anomaly"]["max_outlier_rate_fail"],)

    df_with_flags, anomaly_report = flag_quantile_outliers(df, an_cfg)

    # Always write the report to inspect it later
    anomaly_path.write_text(json.dumps(anomaly_report, indent=2))

    # Decide whether anomaly should block the pipeline (default: False)
    fail_pipeline_on_anomaly = bool(_safe_get(cfg, ["anomaly", "fail_pipeline"], False))

    status = anomaly_report.get("status", "unknown")
    details = anomaly_report.get("details", [])

    if status in ("warn", "fail"):
        msg = (
            f"[ANOMALY] status={status} | outlier_rate={anomaly_report.get('outlier_rate'):.2%} | "
            f"details={details}"
        )
        if status == "fail":
            # strong signal, but still allow continuation unless configured otherwise
            logger.error(msg)
            if fail_pipeline_on_anomaly:
                raise ValueError("Anomaly detection failed and anomaly.fail_pipeline=true. Aborting pipeline.")
        else:
            logger.warning(msg)
    else:
        logger.info(f"[ANOMALY] status={status} | outlier_rate={anomaly_report.get('outlier_rate'):.2%}")


    # -------- (6) Run manifest (reproducibility trace) -------------------------------------------
    # This is the “audit log” that makes debugging + retraining explainable.
    if save_manifest:
        raw_p = Path(raw_path)
        interim_p = Path(interim_path)
        processed_p = Path(processed_path)

        manifest = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "config": {
                "path": str(config_file),
                "sha256": _sha256_text(config_text),
            },
            "data_artifacts": {
                "raw": {
                    "path": str(raw_p),
                    "sha256": _sha256_file(raw_p) if raw_p.exists() else None,
                },
                "interim": {
                    "path": str(interim_p),
                    "sha256": _sha256_file(interim_p) if interim_p.exists() else None,
                },
                "processed": {
                    "path": str(processed_p),
                    "sha256": _sha256_file(processed_p) if processed_p.exists() else None,
                },
            },
            "dataset_shape": {
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1]),
            },
            
            "reports": {
            "data_profile": str(profile_path) if profile_path.exists() else None,
            "data_checks": str(checks_path) if checks_path.exists() else None,
            "anomaly_report": str(anomaly_path) if anomaly_path.exists() else None,
            "drift_report": str(drift_path) if drift_path.exists() else None,
            "data_validation_summary": str(validation_summary_path) if validation_summary_path.exists() else None,
        },

        }

        manifest_out = Path(manifest_path)
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
        manifest_out.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ml/configs/data.yaml")
    args = parser.parse_args()

    run(args.config)

