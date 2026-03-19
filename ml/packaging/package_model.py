from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime
import json
import shutil


def create_model_package(
    *,
    package_root: Path,
    model_path: Path,
    threshold: float,
    features: List[str],
    target_col: str,
    metrics: Dict[str, Any],
    cfg_path: Path,
) -> Dict[str, Any]:
    """
    Create a self-contained 'model package' directory that API or batch
    jobs can load.

    Contents:
      - model.joblib 
      - package_manifest.json (metadata: threshold, features, metrics, config path)
    """

    #ts is a UTC timestamp like 20250301T212045Z
    #package_id becomes something like nba_model_20250301T212045Z
    #pkg_dir is the folder ml/packages/nba_model_2025...
    #Using timestamp makes each training run produce a new immutable package
    package_root.mkdir(parents=True, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    package_id = f"nba_model_{ts}"
    pkg_dir = package_root / package_id
    pkg_dir.mkdir(parents=True, exist_ok=True)


    # Copy the trained model into the package
    packaged_model_path = pkg_dir / "model.joblib"
    shutil.copy2(model_path, packaged_model_path)

    # Package manifest = the contract for serving
    manifest = {
        "package_id": package_id,
        "created_at_utc": ts,
        "model_path": str(packaged_model_path),
        "threshold": float(threshold),
        "features": list(features),
        "target_col": target_col,
        "metrics": metrics,           # full eval summary from model cycle
        "config_path": str(cfg_path), # which config produced this model
    }

    (pkg_dir / "package_manifest.json").write_text(json.dumps(manifest, indent=2))

    return {
        "package_id": package_id,
        "package_dir": str(pkg_dir),
        "manifest_path": str(pkg_dir / "package_manifest.json"),
        "model_path": str(packaged_model_path),
    }