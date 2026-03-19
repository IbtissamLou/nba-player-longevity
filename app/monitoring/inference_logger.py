from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_request_id() -> str:
    return str(uuid4())


class InferenceLogger:
    """
    Structured JSONL logger for inference events.

    Writes:
    - all inference events to the main inference log
    - failed events also to a dedicated failure log
    """

    def __init__(
        self,
        log_path: str | Path,
        failure_log_path: str | Path = "logs/failures.jsonl",
    ):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.failure_log_path = Path(failure_log_path)
        self.failure_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _append_jsonl(self, path: Path, event: Dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def log_event(self, event: Dict[str, Any]) -> None:
        self._append_jsonl(self.log_path, event)

        if event.get("status") == "failed":
            self._append_jsonl(self.failure_log_path, event)

    def log_single_inference(
        self,
        *,
        request_id: str,
        package_id: str,
        threshold: float,
        features: Dict[str, float],
        prediction: int,
        probability: float,
        latency_ms: float,
        status: str = "success",
        error_message: Optional[str] = None,
    ) -> None:
        event = {
            "event_type": "single_inference",
            "timestamp_utc": utc_now_iso(),
            "request_id": request_id,
            "package_id": package_id,
            "threshold_used": threshold,
            "features": features,
            "prediction": prediction,
            "probability": probability,
            "latency_ms": round(latency_ms, 3),
            "status": status,
            "error_message": error_message,
        }
        self.log_event(event)

    def log_batch_inference(
        self,
        *,
        request_id: str,
        package_id: str,
        threshold: float,
        n_rows: int,
        latency_ms: float,
        status: str = "success",
        error_message: Optional[str] = None,
    ) -> None:
        event = {
            "event_type": "batch_inference",
            "timestamp_utc": utc_now_iso(),
            "request_id": request_id,
            "package_id": package_id,
            "threshold_used": threshold,
            "n_rows": n_rows,
            "latency_ms": round(latency_ms, 3),
            "status": status,
            "error_message": error_message,
        }
        self.log_event(event)