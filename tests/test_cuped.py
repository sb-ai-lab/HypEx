"""Regression tests for the CUPED variance-reduction transformer.

CUPED adjusts a target Y using a pre-experiment covariate X via
``Y_cuped = Y - theta * (X - E[X])`` where the optimal coefficient is
``theta = Cov(X, Y) / Var(X)``. With this theta the achievable variance
reduction equals the squared correlation ``rho**2``.

A previous implementation used ``theta = Cov(X, Y) / (std_x * std_y) = rho``,
which is only correct when ``std_x == std_y``. These tests pin the correct
behaviour, especially when the covariate and target are on different scales.
"""

import numpy as np
import pandas as pd
import pytest

from hypex.dataset import Dataset, PreTargetRole, TargetRole
from hypex.transformers.cuped import CUPEDTransformer


def _variance_reduction(original: np.ndarray, adjusted: np.ndarray) -> float:
    """Fraction of variance removed by the adjustment (1.0 == all variance)."""
    return 1 - adjusted.var() / original.var()


def _make_dataset(x: np.ndarray, y: np.ndarray) -> Dataset:
    df = pd.DataFrame({"x": x, "y": y})
    return Dataset(roles={"x": PreTargetRole(), "y": TargetRole()}, data=df)


@pytest.mark.parametrize(
    "x_scale, y_scale",
    [
        (1.0, 1.0),   # equal scale: the old (buggy) theta also happened to work
        (1.0, 5.0),   # target on a larger scale than the covariate
        (8.0, 1.0),   # covariate on a much larger scale than the target
    ],
)
def test_cuped_reaches_rho_squared_reduction(x_scale, y_scale):
    """CUPED with the correct theta must reach ~rho**2 variance reduction
    regardless of the relative scales of X and Y."""
    rng = np.random.default_rng(42)
    n = 100_000
    latent = rng.normal(0, 1, n)
    x = x_scale * (latent + rng.normal(0, 0.5, n))
    y = y_scale * (latent + rng.normal(0, 0.5, n))

    rho = np.corrcoef(x, y)[0, 1]
    expected_reduction = rho**2

    ds = _make_dataset(x, y)
    adjusted = CUPEDTransformer._inner_function(ds, {"y": "x"})
    y_orig = np.asarray(ds["y"].data).ravel()
    y_cuped = np.asarray(adjusted["y_cuped"].data).ravel()

    actual_reduction = _variance_reduction(y_orig, y_cuped)

    # Achieved reduction must match the theoretical optimum (rho**2) closely.
    assert actual_reduction == pytest.approx(expected_reduction, abs=1e-3)
    # And it must never increase variance.
    assert actual_reduction >= -1e-9


def test_cuped_theta_equals_cov_over_var():
    """The implied theta must equal Cov(X, Y) / Var(X), not Cov / (std_x*std_y)."""
    rng = np.random.default_rng(0)
    n = 50_000
    x = rng.normal(0, 1.0, n)
    y = 5.0 * x + 3.0 * rng.normal(0, 1.0, n)  # std_x != std_y on purpose

    ds = _make_dataset(x, y)
    adjusted = CUPEDTransformer._inner_function(ds, {"y": "x"})
    y_orig = np.asarray(ds["y"].data).ravel()
    y_cuped = np.asarray(adjusted["y_cuped"].data).ravel()

    # Recover the theta actually applied: y_cuped = y - theta * (x - mean_x).
    theta_applied = np.polyfit(x - x.mean(), y_orig - y_cuped, 1)[0]
    theta_expected = np.cov(x, y, bias=True)[0, 1] / x.var()

    assert theta_applied == pytest.approx(theta_expected, rel=1e-6)


def test_cuped_constant_covariate_is_noop():
    """A constant (zero-variance) covariate yields theta=0, leaving Y unchanged."""
    n = 1000
    x = np.full(n, 7.0)
    y = np.random.default_rng(1).normal(0, 1, n)

    ds = _make_dataset(x, y)
    adjusted = CUPEDTransformer._inner_function(ds, {"y": "x"})
    y_orig = np.asarray(ds["y"].data).ravel()
    y_cuped = np.asarray(adjusted["y_cuped"].data).ravel()

    np.testing.assert_allclose(y_cuped, y_orig)
