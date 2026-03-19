from __future__ import annotations

"""
Feature versioning utilities.

Goal:
- make feature artifacts traceable + reproducible
- record which config + raw/processed inputs produced this feature dataset
- integrate cleanly with DVC (which versions the data files themselves)

This module DOES NOT call DVC commands directly.
Why:
- keeping pipeline pure makes it portable (CI/CD, Airflow, GitHub Actions)
- DVC is invoked via CLI in workflows (dvc repro / dvc commit)
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Deterministic fingerprint for a data artifact."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    """Deterministic fingerprint for config/code text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FeatureManifestConfig:
    save_manifest: bool = True
    manifest_path: str = "ml/reports/feature_manifest.json"


def write_feature_manifest(
    *,
    feature_path: str,
    config_path: str,
    config_text: str,
    input_processed_path: str | None,
    df_features: pd.DataFrame,
    cfg: FeatureManifestConfig,
    extra_reports: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    """
    Create a feature manifest (audit log) for traceability.

    This complements DVC:
    - DVC versions the file itself.
    - manifest explains "what created this file".
    """
    manifest: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "path": config_path,
            "sha256": sha256_text(config_text),
        },
        "inputs": {
            "processed_data": None,
        },
        "features": {
            "path": feature_path,
            "sha256": None,
            "rows": int(df_features.shape[0]),
            "cols": int(df_features.shape[1]),
            "columns": list(df_features.columns),
        },
        "reports": extra_reports or {},
    }

    feat_p = Path(feature_path)
    if feat_p.exists():
        manifest["features"]["sha256"] = sha256_file(feat_p)

    if input_processed_path:
        in_p = Path(input_processed_path)
        manifest["inputs"]["processed_data"] = {
            "path": input_processed_path,
            "sha256": sha256_file(in_p) if in_p.exists() else None,
        }

    if cfg.save_manifest:
        out = Path(cfg.manifest_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2))

    return manifest
