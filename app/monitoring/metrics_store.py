#This keeps aggregated metrics in memory + saves them

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Dict, Any
import math


def utc_now():
    return datetime.now(timezone.utc)


def percentile(values: list[float], q: float) -> float:
    """
    Compute percentile without external dependency.
    q should be in [0, 100].
    """
    if not values:
        return 0.0

    vals = sorted(values)
    k = (len(vals) - 1) * (q / 100.0)
    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return float(vals[int(k)])

    d0 = vals[f] * (c - k)
    d1 = vals[c] * (k - f)
    return float(d0 + d1)


class MetricsStore:
    """
    In-memory metrics store for:
    - reliability: success rate, error rate
    - latency: avg, p95
    - throughput: requests/min
    - usage: total predictions and batch volume (cost proxy)
    """

    def __init__(self, max_history: int = 5000):
        self.latencies = deque(maxlen=max_history)
        self.request_timestamps = deque(maxlen=max_history)

        self.total_requests = 0
        self.success_count = 0
        self.error_count = 0

        # usage / cost proxy
        self.single_inference_count = 0
        self.batch_request_count = 0
        self.total_batch_rows = 0

    def record_request(
        self,
        *,
        latency_ms: float,
        success: bool,
        request_type: str = "single",
        n_rows: int = 1,
    ) -> None:
        self.total_requests += 1
        self.latencies.append(float(latency_ms))
        self.request_timestamps.append(utc_now())

        if success:
            self.success_count += 1
        else:
            self.error_count += 1

        if request_type == "single":
            self.single_inference_count += 1
        elif request_type == "batch":
            self.batch_request_count += 1
            self.total_batch_rows += int(n_rows)

    def get_metrics(self) -> Dict[str, Any]:
        latencies_list = list(self.latencies)

        avg_latency = (
            sum(latencies_list) / len(latencies_list)
            if latencies_list
            else 0.0
        )

        p95_latency = percentile(latencies_list, 95)

        success_rate = (
            self.success_count / self.total_requests
            if self.total_requests > 0
            else 0.0
        )

        error_rate = (
            self.error_count / self.total_requests
            if self.total_requests > 0
            else 0.0
        )

        now = utc_now()
        requests_last_minute = [
            ts for ts in self.request_timestamps
            if (now - ts).total_seconds() <= 60
        ]
        throughput_rpm = len(requests_last_minute)

        avg_rows_per_batch = (
            self.total_batch_rows / self.batch_request_count
            if self.batch_request_count > 0
            else 0.0
        )

        return {
            # reliability
            "total_requests": self.total_requests,
            "success_rate": round(success_rate, 4),
            "error_rate": round(error_rate, 4),

            # latency
            "avg_latency_ms": round(avg_latency, 3),
            "p95_latency_ms": round(p95_latency, 3),

            # throughput
            "throughput_requests_per_min": throughput_rpm,

            # usage / cost proxy
            "single_inference_count": self.single_inference_count,
            "batch_request_count": self.batch_request_count,
            "total_batch_rows": self.total_batch_rows,
            "avg_rows_per_batch": round(avg_rows_per_batch, 3),
        }