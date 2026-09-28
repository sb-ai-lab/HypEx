"""Shared optimal CUPED coefficient helper.

theta = Cov(X, Y) / Var(X), computed with centered arithmetic to
avoid catastrophic cancellation when the covariate has a large mean.
"""
from __future__ import annotations

import numpy as np


# Финальная корректная версия:
def cuped_theta(y, x) -> float:
    """Compute the variance-optimal CUPED coefficient.

    Uses centered arithmetic to avoid catastrophic cancellation when
    the covariate has a large mean.  The guard threshold is based on
    floating-point rounding noise rather than raw E[x²], which
    prevents false deactivation at large additive offsets.

    Args:
        y: Target values (array-like).
        x: Covariate / prediction values (array-like).

    Returns:
        Optimal theta.  ``0.0`` when the covariate variance is
        negligible relative to floating-point noise.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    x_centered = x - x.mean()
    var_x = float(np.mean(x_centered ** 2))

    # Guard: variance must exceed floating-point rounding noise.
    # Rounding noise ≈ (eps * max|x|)², with safety factor 1e3.
    eps = np.finfo(float).eps
    noise_floor = (1e3 * eps * float(np.max(np.abs(x)))) ** 2 if len(x) > 0 else 0.0
    if not np.isfinite(var_x) or var_x <= noise_floor:
        return 0.0

    cov_xy = float(np.mean((y - y.mean()) * x_centered))
    return cov_xy / var_x
