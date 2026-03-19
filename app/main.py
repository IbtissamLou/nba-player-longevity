from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.model_loader import load_model_package
from app.schemas import (
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
)
from app.predict import predict_one, predict_batch
from app.validation import validate_feature_payload, validate_prediction_output

from app.monitoring.inference_logger import InferenceLogger, generate_request_id
from app.monitoring.metrics_store import MetricsStore
from app.monitoring.request_metrics import MetricsMiddleware
from app.monitoring.prediction_reliability import PredictionReliabilityStore
from app.monitoring.data_reliability import DataReliabilityStore


PACKAGE_DIR = os.getenv("MODEL_PACKAGE_DIR", "ml/packaging/packages/latest")
INFERENCE_LOG_PATH = os.getenv("INFERENCE_LOG_PATH", "logs/inference_log.jsonl")
FAILURE_LOG_PATH = os.getenv("FAILURE_LOG_PATH", "logs/failures.jsonl")

app = FastAPI(
    title="NBA Career Prediction API",
    description="Predict whether an NBA player will stay at least 5 years in the league.",
    version="2.0.0",
)

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

model_package: Any = None

inference_logger = InferenceLogger(
    log_path=INFERENCE_LOG_PATH,
    failure_log_path=FAILURE_LOG_PATH,
)

metrics_store = MetricsStore()
prediction_reliability_store = PredictionReliabilityStore()
data_reliability_store = DataReliabilityStore()

app.add_middleware(MetricsMiddleware, metrics_store=metrics_store)


@app.on_event("startup")
def startup_event():
    global model_package
    try:
        model_package = load_model_package(PACKAGE_DIR)
        print(f"✅ Loaded model package from: {model_package.package_dir}")
    except Exception as e:
        print(f"❌ Failed to load model package: {e}")
        raise


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    if model_package is None:
        raise HTTPException(status_code=500, detail="Model package not loaded")

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "features": model_package.features,
            "threshold": model_package.threshold,
            "package_id": model_package.manifest.get("package_id"),
        },
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_package is not None,
    }


@app.get("/model-info")
def model_info():
    if model_package is None:
        raise HTTPException(status_code=500, detail="Model package not loaded")

    return {
        "package_id": model_package.manifest.get("package_id"),
        "created_at_utc": model_package.manifest.get("created_at_utc"),
        "threshold": model_package.threshold,
        "n_features": len(model_package.features),
        "features": model_package.features,
    }


@app.get("/metrics")
def get_metrics():
    """
    Combined monitoring endpoint:
    - system reliability
    - prediction reliability
    - data reliability
    """
    return {
        "system_reliability": metrics_store.get_metrics(),
        "prediction_reliability": prediction_reliability_store.get_metrics(),
        "data_reliability": data_reliability_store.get_metrics(),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    if model_package is None:
        raise HTTPException(status_code=500, detail="Model package not loaded")

    request_id = generate_request_id()
    start = time.perf_counter()

    # ------------------------------
    # Data reliability checks
    # ------------------------------
    is_valid_payload, counts, payload_error = validate_feature_payload(
        payload.features,
        model_package.features,
    )

    data_reliability_store.record_payload(
        is_valid=is_valid_payload,
        missing_count=counts["missing_count"],
        unexpected_count=counts["unexpected_count"],
        invalid_value_count=counts["invalid_value_count"],
    )

    if not is_valid_payload:
        latency_ms = (time.perf_counter() - start) * 1000.0

        inference_logger.log_single_inference(
            request_id=request_id,
            package_id=model_package.manifest.get("package_id", "unknown"),
            threshold=model_package.threshold,
            features=payload.features,
            prediction=-1,
            probability=-1.0,
            latency_ms=latency_ms,
            status="failed",
            error_message=payload_error,
        )

        raise HTTPException(status_code=400, detail=payload_error)

    try:
        result = predict_one(model_package, payload.features)
        latency_ms = (time.perf_counter() - start) * 1000.0

        # ------------------------------
        # Prediction reliability checks
        # ------------------------------
        is_valid_prediction, prediction_error = validate_prediction_output(
            result.prediction,
            result.probability,
        )

        prediction_reliability_store.record_prediction(
            prediction=result.prediction,
            probability=result.probability,
            is_valid=is_valid_prediction,
        )

        if not is_valid_prediction:
            inference_logger.log_single_inference(
                request_id=request_id,
                package_id=model_package.manifest.get("package_id", "unknown"),
                threshold=model_package.threshold,
                features=payload.features,
                prediction=result.prediction,
                probability=result.probability,
                latency_ms=latency_ms,
                status="failed",
                error_message=prediction_error,
            )
            raise HTTPException(status_code=500, detail=prediction_error)

        inference_logger.log_single_inference(
            request_id=request_id,
            package_id=model_package.manifest.get("package_id", "unknown"),
            threshold=model_package.threshold,
            features=payload.features,
            prediction=result.prediction,
            probability=result.probability,
            latency_ms=latency_ms,
            status="success",
        )

        return result

    except HTTPException:
        raise

    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000.0

        inference_logger.log_single_inference(
            request_id=request_id,
            package_id=model_package.manifest.get("package_id", "unknown"),
            threshold=model_package.threshold,
            features=payload.features,
            prediction=-1,
            probability=-1.0,
            latency_ms=latency_ms,
            status="failed",
            error_message=str(e),
        )

        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_batch", response_model=BatchPredictResponse)
def predict_batch_endpoint(payload: BatchPredictRequest):
    if model_package is None:
        raise HTTPException(status_code=500, detail="Model package not loaded")

    request_id = generate_request_id()
    start = time.perf_counter()

    # ------------------------------
    # Data reliability checks for all rows
    # ------------------------------
    for row in payload.rows:
        is_valid_payload, counts, payload_error = validate_feature_payload(
            row,
            model_package.features,
        )

        data_reliability_store.record_payload(
            is_valid=is_valid_payload,
            missing_count=counts["missing_count"],
            unexpected_count=counts["unexpected_count"],
            invalid_value_count=counts["invalid_value_count"],
        )

        if not is_valid_payload:
            latency_ms = (time.perf_counter() - start) * 1000.0

            inference_logger.log_batch_inference(
                request_id=request_id,
                package_id=model_package.manifest.get("package_id", "unknown"),
                threshold=model_package.threshold,
                n_rows=len(payload.rows),
                latency_ms=latency_ms,
                status="failed",
                error_message=payload_error,
            )

            raise HTTPException(status_code=400, detail=payload_error)

    try:
        result = predict_batch(model_package, payload.rows)
        latency_ms = (time.perf_counter() - start) * 1000.0

        # ------------------------------
        # Prediction reliability checks for all outputs
        # ------------------------------
        for row_pred in result.predictions:
            is_valid_prediction, _ = validate_prediction_output(
                row_pred.prediction,
                row_pred.probability,
            )

            prediction_reliability_store.record_prediction(
                prediction=row_pred.prediction,
                probability=row_pred.probability,
                is_valid=is_valid_prediction,
            )

        inference_logger.log_batch_inference(
            request_id=request_id,
            package_id=model_package.manifest.get("package_id", "unknown"),
            threshold=model_package.threshold,
            n_rows=result.n_rows,
            latency_ms=latency_ms,
            status="success",
        )

        metrics_store.record_request(
            latency_ms=latency_ms,
            success=True,
            request_type="batch",
            n_rows=result.n_rows,
        )

        return result

    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000.0

        inference_logger.log_batch_inference(
            request_id=request_id,
            package_id=model_package.manifest.get("package_id", "unknown"),
            threshold=model_package.threshold,
            n_rows=len(payload.rows),
            latency_ms=latency_ms,
            status="failed",
            error_message=str(e),
        )

        metrics_store.record_request(
            latency_ms=latency_ms,
            success=False,
            request_type="batch",
            n_rows=len(payload.rows),
        )

        raise HTTPException(status_code=400, detail=str(e))