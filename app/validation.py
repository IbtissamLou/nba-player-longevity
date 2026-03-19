from __future__ import annotations

import math
from typing import Dict, List, Tuple


def validate_feature_payload(
    features: Dict[str, float],
    expected_features: List[str],
) -> Tuple[bool, Dict[str, int], str | None]:
    """
    Validate inference payload against expected feature schema.

    Checks:
    - missing features
    - unexpected features
    - non-finite numeric values (NaN / inf)

    Returns
    -------
    is_valid, counts, error_message
    """
    provided_keys = set(features.keys())
    expected_keys = set(expected_features)

    missing = sorted(expected_keys - provided_keys)
    unexpected = sorted(provided_keys - expected_keys)

    invalid_value_count = 0
    invalid_value_features = []

    for key, value in features.items():
        try:
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                invalid_value_count += 1
                invalid_value_features.append(key)
        except Exception:
            invalid_value_count += 1
            invalid_value_features.append(key)

    is_valid = (
        len(missing) == 0
        and len(unexpected) == 0
        and invalid_value_count == 0
    )

    counts = {
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "invalid_value_count": invalid_value_count,
    }

    if is_valid:
        return True, counts, None

    parts = []
    if missing:
        parts.append(f"Missing features: {missing}")
    if unexpected:
        parts.append(f"Unexpected features: {unexpected}")
    if invalid_value_features:
        parts.append(f"Invalid numeric values for: {invalid_value_features}")

    return False, counts, " | ".join(parts)


def validate_prediction_output(
    prediction: int,
    probability: float,
) -> tuple[bool, str | None]:
    """
    Validate model outputs.

    Checks:
    - prediction in {0, 1}
    - probability is finite
    - probability in [0, 1]
    """
    if prediction not in {0, 1}:
        return False, f"Invalid prediction value: {prediction}"

    try:
        p = float(probability)
    except Exception:
        return False, f"Probability is not numeric: {probability}"

    if not math.isfinite(p):
        return False, f"Probability is not finite: {probability}"

    if p < 0.0 or p > 1.0:
        return False, f"Probability out of range [0,1]: {probability}"

    return True, None