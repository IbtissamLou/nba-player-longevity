#Middleware for FastAPI

from __future__ import annotations

import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.monitoring.metrics_store import MetricsStore


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Tracks request-level operational metrics:
    - success / failure
    - latency
    - throughput

    It does not know prediction results, only request behavior.
    """

    def __init__(self, app, metrics_store: MetricsStore):
        super().__init__(app)
        self.metrics_store = metrics_store

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()

        try:
            response = await call_next(request)
            success = response.status_code < 400
            return response

        except Exception:
            success = False
            raise

        finally:
            latency_ms = (time.perf_counter() - start) * 1000.0
            self.metrics_store.record_request(
                latency_ms=latency_ms,
                success=success,
                request_type="single",   # default at middleware level
                n_rows=1,
            )