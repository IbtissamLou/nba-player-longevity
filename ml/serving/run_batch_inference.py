from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from app.model_loader import load_model_package


def run_batch_inference(
    *,
    package_dir: str,
    input_path: str,
    output_path: str,
):
    pkg = load_model_package(package_dir)

    input_path = Path(input_path)
    output_path = Path(output_path)

    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    missing = [c for c in pkg.features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected features in batch input: {missing}")

    X = df[pkg.features].copy()
    probas = pkg.predict_proba(X)
    preds = (probas >= pkg.threshold).astype(int)

    out = df.copy()
    out["prediction_proba"] = probas
    out["prediction"] = preds
    out["threshold_used"] = pkg.threshold

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        out.to_parquet(output_path, index=False)
    else:
        out.to_csv(output_path, index=False)

    print(f"Batch inference completed. Output saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    run_batch_inference(
        package_dir=args.package_dir,
        input_path=args.input_path,
        output_path=args.output_path,
    )