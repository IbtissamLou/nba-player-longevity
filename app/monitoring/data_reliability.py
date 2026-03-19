from __future__ import annotations

from typing import Any, Dict


class DataReliabilityStore:
    """
    Tracks reliability of incoming model inputs.

    Main metrics:
    - schema_validity_rate
    - invalid_payload_rate
    - total_missing_features
    - total_unexpected_features
    - total_invalid_values
    """

    def __init__(self) -> None:
        self.total_payloads = 0
        self.valid_payload_count = 0
        self.invalid_payload_count = 0

        self.total_missing_features = 0
        self.total_unexpected_features = 0
        self.total_invalid_values = 0

    def record_payload(
        self,
        *,
        is_valid: bool,
        missing_count: int = 0,
        unexpected_count: int = 0,
        invalid_value_count: int = 0,
    ) -> None:
        self.total_payloads += 1

        if is_valid:
            self.valid_payload_count += 1
        else:
            self.invalid_payload_count += 1

        self.total_missing_features += int(missing_count)
        self.total_unexpected_features += int(unexpected_count)
        self.total_invalid_values += int(invalid_value_count)

    def get_metrics(self) -> Dict[str, Any]:
        schema_validity_rate = (
            self.valid_payload_count / self.total_payloads
            if self.total_payloads > 0
            else 0.0
        )

        invalid_payload_rate = (
            self.invalid_payload_count / self.total_payloads
            if self.total_payloads > 0
            else 0.0
        )

        return {
            "total_payloads": self.total_payloads,
            "valid_payload_count": self.valid_payload_count,
            "invalid_payload_count": self.invalid_payload_count,
            "schema_validity_rate": round(schema_validity_rate, 4),
            "invalid_payload_rate": round(invalid_payload_rate, 4),
            "total_missing_features": self.total_missing_features,
            "total_unexpected_features": self.total_unexpected_features,
            "total_invalid_values": self.total_invalid_values,
        }