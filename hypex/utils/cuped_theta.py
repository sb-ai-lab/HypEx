"""Shared optimal CUPED coefficient helper.

theta = Cov(X, Y) / Var(X), computed with centered arithmetic to
avoid catastrophic cancellation when the covariate has a large mean.
"""
from __future__ import annotations

import numpy as np


def cuped_theta(y, x) -> float:
    """Compute the variance-optimal CUPED coefficient.

    Args:
        y: Target values (array-like).
        x: Covariate / prediction values (array-like).

    Returns:
        Optimal theta.  ``0.0`` when the covariate variance is
        negligible relative to its magnitude.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    x_centered = x - x.mean()
    var_x = float(np.mean(x_centered ** 2))
    scale = float(np.mean(x ** 2))
    if not np.isfinite(var_x) or var_x <= 1e-12 * max(scale, 1.0):
        return 0.0
    cov_xy = float(np.mean((y - y.mean()) * x_centered))
    return cov_xy / var_x
