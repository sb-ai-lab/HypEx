"""Shared optimal CUPED coefficient helper.

theta = Cov(X, Y) / Var(X), computed with centered arithmetic to
avoid catastrophic cancellation when the covariate has a large mean.
"""

from __future__ import annotations

import numpy as np


def cuped_theta(y: np.ndarray, x: np.ndarray) -> float:
    """Compute the variance-optimal CUPED coefficient.

    NaN pairs are excluded before computation so that a single
    missing value affects only its own row, not the entire column.

    Args:
        y: Target values (array-like).
        x: Covariate / prediction values (array-like).

    Returns:
        Optimal theta.  ``0.0`` when the covariate variance is
        negligible or no valid observations remain.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    # Exclude NaN pairs so one missing value doesn't poison theta.
    valid: np.ndarray = np.isfinite(y) & np.isfinite(x)
    if not valid.any():
        return 0.0
    y = y[valid]
    x = x[valid]

    x_centered = x - np.mean(x)
    var_x = float(np.mean(x_centered**2))

    # Guard: variance must exceed floating-point rounding noise.
    eps = np.finfo(float).eps
    noise_floor = (1e3 * eps * float(np.max(np.abs(x)))) ** 2 if len(x) > 0 else 0.0
    if not np.isfinite(var_x) or var_x <= noise_floor:
        return 0.0

    cov_xy = float(np.mean((y - np.mean(y)) * x_centered))
    return cov_xy / var_x
