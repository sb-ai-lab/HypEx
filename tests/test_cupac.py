"""Regression tests for CUPAC theta-residualization.

CUPAC predicts the target from pre-experiment covariates and then removes the
predicted component via CUPED. The adjustment must use the optimal coefficient
``theta = Cov(pred, y) / Var(pred)`` rather than assuming ``theta = 1`` (i.e.
subtracting the prediction directly). With theta=1 the variance reduction is
suboptimal whenever the prediction is not on the same scale as the target.
"""

import warnings

import numpy as np
import pytest

from hypex import ABTest
from hypex.dataset import (
    Dataset,
    FeatureRole,
    InfoRole,
    PreTargetRole,
    TargetRole,
    TreatmentRole,
)
from hypex.extensions.cupac import CupacExtension
from hypex.utils.tutorial_data_creation import DataGenerator


def _vr(original: np.ndarray, adjusted: np.ndarray) -> float:
    return 1 - adjusted.var() / original.var()


def test_cuped_theta_equals_cov_over_var():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 5, 10_000)
    pred = 0.3 * y + rng.normal(0, 1, 10_000)  # correlated, different scale

    theta = CupacExtension._cuped_theta(y, pred)
    expected = np.cov(pred, y, bias=True)[0, 1] / pred.var()

    assert theta == pytest.approx(expected, rel=1e-9)


def test_theta_residualize_beats_theta_one():
    """theta-scaled residualization must never reduce variance less than theta=1,
    and strictly more when the prediction scale differs from the target."""
    rng = np.random.default_rng(1)
    y = rng.normal(0, 5, 10_000)
    pred = 0.3 * y + rng.normal(0, 1, 10_000)

    theta = CupacExtension._cuped_theta(y, pred)
    vr_theta = _vr(y, y - theta * (pred - pred.mean()))
    vr_one = _vr(y, y - (pred - pred.mean()))

    assert vr_theta >= vr_one - 1e-9
    assert vr_theta > vr_one  # strictly better here (scale mismatch)
    assert vr_theta >= -1e-9  # never increases variance


def test_constant_prediction_is_noop():
    """A zero-variance prediction yields theta=0 (no adjustment)."""
    y = np.random.default_rng(2).normal(0, 1, 1000)
    pred = np.full(1000, 4.2)

    theta = CupacExtension._cuped_theta(y, pred)

    assert theta == 0.0
    np.testing.assert_allclose(y - theta * (pred - pred.mean()), y)


def test_cupac_end_to_end_variance_reduction_non_negative():
    """End-to-end CUPAC through ABTest produces a finite, non-negative real
    variance reduction and a CV reduction consistent with it."""
    gen = DataGenerator(
        n_samples=1000,
        distributions={
            "X1": {"type": "normal", "mean": 1, "std": 1},
            "X2": {"type": "bernoulli", "p": 0.5},
            "y0": {"type": "normal", "mean": 1, "std": 5},
        },
        time_correlations={"X1": 0.2, "X2": 0.1, "y0": 0.8},
        effect_size=0.1,
        seed=42,
    )
    df = gen.generate().drop(columns=["y0", "z", "U", "D", "y1"])
    df = df.rename(columns={"y0_lag_1": "y_lag1", "y0_lag_2": "y_lag2"})

    data = Dataset(
        roles={
            "d": TreatmentRole(),
            "y": TargetRole(cofounders=["X1", "X2"]),
            "y_lag1": PreTargetRole(parent="y", lag=1),
            "X1_lag1": FeatureRole(parent="X1", lag=1),
            "X2_lag1": FeatureRole(parent="X2", lag=1),
            "y_lag2": PreTargetRole(parent="y", lag=2),
            "X1_lag2": FeatureRole(parent="X1", lag=2),
            "X2_lag2": FeatureRole(parent="X2", lag=2),
        },
        data=df,
        default_role=InfoRole(),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ABTest(
            enable_cupac=True, cupac_models=["linear", "ridge", "lasso"]
        ).execute(data)

    vr = result.cupac.variance_reductions.data
    real = float(vr["variance_reduction_real"].iloc[0])
    cv = float(vr["variance_reduction_cv"].iloc[0])

    assert np.isfinite(real) and real >= 0.0
    assert np.isfinite(cv) and cv >= 0.0
    # On this well-correlated data CUPAC should remove a substantial share.
    assert real > 40.0
