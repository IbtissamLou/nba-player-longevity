from __future__ import annotations

"""
Feature Cycle runner:

Stage order (production-style):
1) load validated dataset (from Data Cycle)
2) leakage guards (drop IDs, detect target proxies)
3) domain feature engineering (interpretable features)
4) feature schema validation (ranges + finiteness)
5) save feature dataset artifact + reports
"""

from pathlib import Path

import pandas as pd
import yaml
import json

from ml.feature_pipeline.leakage import LeakageConfig, apply_leakage_guards
from ml.feature_pipeline.domain_features import DomainFeatureConfig, add_domain_features
from ml.feature_pipeline.feature_schema import FeatureSchemaConfig, FeatureRange, validate_feature_schema
from ml.feature_pipeline.correlation import CorrelationConfig, run_correlation_checks
from ml.feature_pipeline.eda import EDAConfig, SkewConfig, ScoringConfig, run_eda
import json
from ml.feature_pipeline.feature_qa import FeatureQAConfig, run_feature_qa
from ml.feature_pipeline.eda_extras import EDAExtrasConfig, MissingnessHeatmapConfig,ClassConditionalConfig,FeatureImportanceConfig,PCAConfig,run_eda_extras
from ml.feature_pipeline.versioning import FeatureManifestConfig, write_feature_manifest



def _safe_get(cfg: dict, keys: list[str], default):
    """Safely read nested config values; avoids breaking code if config evolves."""
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def run( feature_cfg_path: str = "ml/configs/feature.yaml", feature_spec_path: str = "ml/configs/feature_spec.yaml") -> None:
    # ---- (0) Load configs ----
    cfg_file = Path(feature_cfg_path)
    if not cfg_file.exists():
        raise FileNotFoundError(f"Missing feature config: {cfg_file}")
    cfg = yaml.safe_load(cfg_file.read_text())

    spec_file = Path(feature_spec_path)
    if not spec_file.exists():
        raise FileNotFoundError(f"Missing feature spec: {spec_file}")
    spec = yaml.safe_load(spec_file.read_text())

    # ---- (1) Load dataset (output of Data Cycle) ------------------------
    input_path = Path(cfg["dataset"]["input_path"])
    if not input_path.exists():
        raise FileNotFoundError(f"Missing processed dataset: {input_path}")

    target_col = cfg["dataset"]["target_col"]
    id_cols = tuple(cfg["dataset"].get("id_cols", []))

    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    # ---- (2) Leakage guards ----------------------------------------------
    leak_cfg = LeakageConfig(
        enabled=bool(_safe_get(cfg, ["leakage", "enabled"], True)),
        target_col=target_col,
        id_cols=id_cols,
        forbidden_feature_cols=tuple(_safe_get(cfg, ["leakage", "forbidden_feature_cols"], [target_col])),
        drop_id_cols=bool(_safe_get(cfg, ["leakage", "drop_id_cols"], True)),
        check_target_proxy=bool(_safe_get(cfg, ["leakage", "check_target_proxy"], True)),
        max_abs_target_corr_warn=float(_safe_get(cfg, ["leakage", "max_abs_target_corr_warn"], 0.85)),
        max_abs_target_corr_fail=float(_safe_get(cfg, ["leakage", "max_abs_target_corr_fail"], 0.95)),
        check_duplicate_columns=bool(_safe_get(cfg, ["leakage", "check_duplicate_columns"], True)),
        required_cols=tuple(_safe_get(cfg, ["leakage", "required_cols"], [])),
    )

    report_path = _safe_get(cfg, ["outputs", "leakage_report_path"], "ml/reports/leakage_report.json")
    df_safe, report = apply_leakage_guards(df, leak_cfg, write_report_path=report_path)

    # ---- (3) Domain feature engineering (config-driven) -------------------  
    eps = float(spec["base"].get("epsilon", 1e-9))
    enabled_feats = tuple(
        [k for k, v in spec["features"].items() if v.get("enabled", True)]
    )

    # Ensure required raw inputs exist
    required_raw = spec.get("required_raw_cols", [])
    missing_raw = [c for c in required_raw if c not in df_safe.columns]
    if missing_raw:
        raise ValueError(f"Missing raw columns needed for domain features: {missing_raw}")

    dom_cfg = DomainFeatureConfig(epsilon=eps, enabled_features=enabled_feats)
    df_feat = add_domain_features(df_safe, dom_cfg)

    # ---- (4) Feature schema validation (ranges from spec) ------------------ FOR NEW FEATURE 
    ranges = {}
    for feat_name, feat_meta in spec["features"].items():
        if not feat_meta.get("enabled", True):
            continue
        lo, hi = feat_meta.get("range", [None, None])
        if lo is None or hi is None:
            continue
        ranges[feat_name] = FeatureRange(float(lo), float(hi))

    schema_cfg = FeatureSchemaConfig(feature_ranges=ranges)
    validate_feature_schema(df_feat, schema_cfg)

    # ----- (5) Feature QA (quality gate) -------------------------------------
    qa_cfg = FeatureQAConfig(
        enabled=bool(_safe_get(cfg, ["feature_qa", "enabled"], True)),
        missing_warn=float(_safe_get(cfg, ["feature_qa", "missing_warn"], 0.05)),
        missing_fail=float(_safe_get(cfg, ["feature_qa", "missing_fail"], 0.30)),
        constant_unique_ratio_warn=float(_safe_get(cfg, ["feature_qa", "constant_unique_ratio_warn"], 0.01)),
        constant_unique_ratio_fail=float(_safe_get(cfg, ["feature_qa", "constant_unique_ratio_fail"], 0.001)),
        inf_fail=bool(_safe_get(cfg, ["feature_qa", "inf_fail"], True)),
    )

    exclude_for_qa = tuple(_safe_get(cfg, ["eda", "exclude_cols"], ["Name", target_col]))
    qa_report = run_feature_qa(
        df_feat,
        exclude_cols=exclude_for_qa,
        cfg=qa_cfg,
        out_path="ml/reports/feature_qa.json",
    )

    print("✅ Feature QA status:", qa_report["status"])
    for d in qa_report.get("details", []):
        print("-", d)

    # Fail-fast only on QA FAIL (warn continues)
    if qa_report["status"] == "fail":
        raise ValueError("Feature QA failed — aborting Feature Cycle.")


    # ---- (5) EDA reporting  -----------------------------------------------
    eda_cfg = EDAConfig(
        enabled=bool(_safe_get(cfg, ["eda", "enabled"], True)),
        out_dir=str(_safe_get(cfg, ["eda", "out_dir"], "ml/reports/eda")),
        exclude_cols=tuple(_safe_get(cfg, ["eda", "exclude_cols"], ["Name", target_col])),
        target_col=target_col,
        skew=SkewConfig(
            high_threshold=float(_safe_get(cfg, ["eda", "skew", "high_threshold"], 1.0)),
            moderate_threshold=float(_safe_get(cfg, ["eda", "skew", "moderate_threshold"], 0.5)),
            zero_ratio_threshold=float(_safe_get(cfg, ["eda", "skew", "zero_ratio_threshold"], 0.30)),
        ),
        scoring=ScoringConfig(
            enabled=bool(_safe_get(cfg, ["eda", "feature_scoring", "enabled"], True)),
            k_best=int(_safe_get(cfg, ["eda", "feature_scoring", "k_best"], 10)),
            method=str(_safe_get(cfg, ["eda", "feature_scoring", "method"], "f_classif")),
        ),
    )

    eda_summary = run_eda(df_feat, eda_cfg)

    # Save EDA summary as JSON artifact
    eda_summary_path = Path(eda_cfg.out_dir) / "eda_summary.json"
    eda_summary_path.write_text(json.dumps(eda_summary, indent=2))
    print("✅ EDA status:", eda_summary["status"])

    # ---- (5.1) Correlation checks (reporting step) -----------------------------------------------
    corr_cfg = CorrelationConfig(
        enabled=bool(_safe_get(cfg, ["correlation", "enabled"], True)),
        threshold=float(_safe_get(cfg, ["correlation", "threshold"], 0.95)),
        method=str(_safe_get(cfg, ["correlation", "method"], "pearson")),
        exclude_cols=tuple(_safe_get(cfg, ["correlation", "exclude_cols"], [target_col])),
        max_features_heatmap=int(_safe_get(cfg, ["correlation", "max_features_heatmap"], 25)),
        save_heatmap=bool(_safe_get(cfg, ["correlation", "save_heatmap"], True)),
    )

    corr_summary = run_correlation_checks(
        df_feat,
        corr_cfg,
        out_matrix_csv=_safe_get(cfg, ["outputs", "correlation_matrix_csv"], "ml/reports/feature_correlation.csv"),
        out_pairs_csv=_safe_get(cfg, ["outputs", "correlation_pairs_csv"], "ml/reports/high_correlation_pairs.csv"),
        out_heatmap_png=_safe_get(cfg, ["outputs", "correlation_heatmap_png"], "ml/reports/feature_correlation.png"),
    )

    print("✅ Correlation checks:", corr_summary["status"])
    for d in corr_summary.get("details", []):
        print("-", d)

    # ---- (5.2) EDA Extras (visual reports) -------------------------------------------------------
    
    extras_cfg = EDAExtrasConfig(
        enabled=bool(_safe_get(cfg, ["eda_extras", "enabled"], True)),
        out_dir=str(_safe_get(cfg, ["eda_extras", "out_dir"], "ml/reports/eda")),
        target_col=target_col,
        exclude_cols=exclude_for_qa,
        missingness_heatmap=MissingnessHeatmapConfig(
            enabled=bool(_safe_get(cfg, ["eda_extras", "missingness_heatmap", "enabled"], True)),
            max_cols=int(_safe_get(cfg, ["eda_extras", "missingness_heatmap", "max_cols"], 30)),
        ),
        class_conditional=ClassConditionalConfig(
            enabled=bool(_safe_get(cfg, ["eda_extras", "class_conditional", "enabled"], True)),
            max_cols=int(_safe_get(cfg, ["eda_extras", "class_conditional", "max_cols"], 12)),
        ),
        feature_importance=FeatureImportanceConfig(
            enabled=bool(_safe_get(cfg, ["eda_extras", "feature_importance", "enabled"], True)),
            model=str(_safe_get(cfg, ["eda_extras", "feature_importance", "model"], "random_forest")),
            n_estimators=int(_safe_get(cfg, ["eda_extras", "feature_importance", "n_estimators"], 50)),
            max_depth=int(_safe_get(cfg, ["eda_extras", "feature_importance", "max_depth"], 5)),
            top_k=int(_safe_get(cfg, ["eda_extras", "feature_importance", "top_k"], 15)),
        ),
        pca=PCAConfig(
            enabled=bool(_safe_get(cfg, ["eda_extras", "pca", "enabled"], True)),
            n_components=int(_safe_get(cfg, ["eda_extras", "pca", "n_components"], 10)),
        ),
    )

    extras_summary = run_eda_extras(df_feat, extras_cfg)

    # Save summary JSON so it's traceable
    extras_summary_path = Path(extras_cfg.out_dir) / "eda_extras_summary.json"
    extras_summary_path.write_text(json.dumps(extras_summary, indent=2))
    print("✅ EDA Extras generated:", extras_summary.get("outputs", {}))



    # ---- (6) Save feature artifact ----------------------------------------------------------
    out_path = Path(cfg["outputs"]["features_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".parquet":
        df_feat.to_parquet(out_path, index=False)
    else:
        df_feat.to_csv(out_path, index=False)

    print(f"✅ Leakage status: {report['status']}")
    print(f"✅ Features artifact saved: {out_path}")
    print(f"✅ Leakage report saved: {report_path}")
    #print(f"✅ Domain features added: {list(enabled_feats)}")


    # ---  Save features artifact ------------------------------------------------------
    
    out_path = Path(cfg["outputs"]["features_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Deterministic ordering helps reproducibility + diffing
    if id_cols in  df_feat.columns:
         df_feat =  df_feat.sort_values(id_cols).reset_index(drop=True)

    df_feat.to_parquet(out_path, index=False)

    # ---  Feature manifest (traceability) ----------------------------------------------
    manifest_cfg = FeatureManifestConfig(
        save_manifest=bool(cfg.get("feature_versioning", {}).get("save_manifest", True)),
        manifest_path=str(cfg.get("feature_versioning", {}).get("manifest_path", "ml/reports/feature_manifest.json")),
    )

    manifest = write_feature_manifest(
        feature_path=str(out_path),
        config_path=str(cfg_file),
        config_text = cfg_file.read_text(),
        input_processed_path=str(input_path) if input_path else None,
        df_features= df_feat,
        cfg=manifest_cfg,
        extra_reports={
            "feature_qa": "ml/reports/feature_qa.json",
            "eda_dir": cfg.get("eda_extras", {}).get("out_dir", "ml/reports/eda"),
        },
    )

    print("✅ Features saved:", out_path)
    print("✅ Feature manifest:", manifest_cfg.manifest_path)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ml/configs/feature.yaml")
    args = parser.parse_args()

    run(args.config)

