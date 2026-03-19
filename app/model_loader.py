from __future__ import annotations

import json
import joblib
import pandas as pd

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class LoadedModelPackage:
    package_dir: Path
    model: Any
    manifest: Dict[str, Any]
    threshold: float
    features: List[str]
    target_col: str

    def predict_proba(self, X: pd.DataFrame):
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)


def _resolve_latest_package(packages_root: Path) -> Path:
    """
    Resolve the newest packaged model folder.

    Example folders:
        nba_model_20260313T000101Z
        nba_model_20260312T193000Z

    Returns the most recent one.
    """
    if not packages_root.exists():
        raise FileNotFoundError(f"Packages directory not found: {packages_root}")

    candidates = [
        p for p in packages_root.iterdir()
        if p.is_dir() and p.name.startswith("nba_model_")
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No packaged models found in {packages_root}"
        )

    # newest timestamped folder
    candidates.sort(reverse=True)

    return candidates[0]


def load_model_package(package_dir: str | Path) -> LoadedModelPackage:
    """
    Load packaged model + manifest from the deployment package directory.

    Supports two modes:

    1️⃣ Direct path:
        /app/ml/packaging/packages/nba_model_20260313T000101Z

    2️⃣ "latest" alias:
        /app/ml/packaging/packages/latest

    In case (2) the loader automatically finds the newest
    timestamped package folder.
    """

    package_dir = Path(package_dir)

    # --------------------------------------------------
    # Resolve "latest"
    # --------------------------------------------------
    if package_dir.name == "latest":
        packages_root = package_dir.parent
        resolved = _resolve_latest_package(packages_root)

        print(f"📦 Resolved 'latest' package -> {resolved.name}")

        package_dir = resolved

    # --------------------------------------------------
    # Validate files
    # --------------------------------------------------
    manifest_path = package_dir / "package_manifest.json"
    model_path = package_dir / "model.joblib"

    if not package_dir.exists():
        raise FileNotFoundError(f"Package directory not found: {package_dir}")

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # --------------------------------------------------
    # Load artifacts
    # --------------------------------------------------
    manifest = json.loads(manifest_path.read_text())
    model = joblib.load(model_path)

    threshold = float(manifest["threshold"])
    features = list(manifest["features"])
    target_col = str(manifest["target_col"])

    print(f"✅ Loaded model package from: {package_dir}")
    print(f"📊 Features: {len(features)} | Threshold: {threshold}")

    return LoadedModelPackage(
        package_dir=package_dir,
        model=model,
        manifest=manifest,
        threshold=threshold,
        features=features,
        target_col=target_col,
    )