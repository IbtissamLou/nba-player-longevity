"""
Global reproducibility utilities.

This module centralizes all randomness control to ensure deterministic
behavior across Python, NumPy, and ML libraries.
"""

from __future__ import annotations

import os
import random
import numpy as np


def set_global_seed(seed: int) -> None:
    """
    Set random seeds for all commonly used libraries.

    This should be called ONCE at the very start of any pipeline
    (data cycle, training, evaluation).
    """
    if seed is None:
        return

    # Python built-in randomness
    random.seed(seed)

    # NumPy randomness
    np.random.seed(seed)

    # Python hash seed 
    os.environ["PYTHONHASHSEED"] = str(seed)
