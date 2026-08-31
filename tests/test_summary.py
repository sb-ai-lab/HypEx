"""Tests for the ``summary`` view that shows several result tables at once."""

import warnings

import numpy as np
import pandas as pd
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
from hypex.ui import Summary
from hypex.ui.cupac import CupacOutput
from hypex.utils.tutorial_data_creation import DataGenerator


def _cuped_data(n: int = 500, seed: int = 5) -> Dataset:
    rng = np.random.default_rng(seed)
    pre = rng.normal(100, 10, n)
    df = pd.DataFrame(
        {
            "user_id": np.arange(n),
            "treat": rng.integers(0, 2, n),
            "pre_spends": pre,
            "post_spends": pre + rng.normal(0, 5, n),
        }
    )
    return Dataset(
        roles={
            "user_id": InfoRole(int),
            "treat": TreatmentRole(),
            "pre_spends": TargetRole(),
            "post_spends": TargetRole(),
        },
        data=df,
    )


def _cupac_data() -> Dataset:
    gen = DataGenerator(
        n_samples=500,
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
    return Dataset(
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


def test_summary_collects_main_and_additional_tables():
    """``summary`` shows the main tables plus the ones of every extra output."""
    result = ABTest(cuped_features={"post_spends": "pre_spends"}).execute(_cuped_data())

    assert isinstance(result.summary, Summary)
    assert list(result.summary) == [
        "resume",
        "multitest",
        "sizes",
        "cuped.resume",
        "cuped.variance_reductions",
    ]
    # a section name is the attribute path of the very same table
    assert result.summary["resume"] is result.resume
    assert result.summary["cuped.resume"] is result.cuped.resume


def test_summary_renders_every_table():
    """Both the console and the notebook rendering list all the sections."""
    result = ABTest(cuped_features={"post_spends": "pre_spends"}).execute(_cuped_data())

    text = str(result.summary)
    html = result.summary._repr_html_()
    for name in result.summary:
        assert f"{name}:" in text
        assert name in html
    assert "post_spends_cuped" in text


def test_cupac_output_is_not_shadowed_by_the_main_output():
    """``result.cupac`` must be the multi-table CUPAC output, not a stub."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ABTest(enable_cupac=True, cupac_models="linear").execute(_cupac_data())

    assert isinstance(result.cupac, CupacOutput)
    assert result.cupac.resume is not None
    # back-compatible access paths keep working
    assert result.cupac.variance_reductions is not None
    assert result.cupac.feature_importances is not None
    assert list(result.cupac.summary) == [
        "resume",
        "variance_reductions",
        "feature_importances",
    ]
    assert "cupac.feature_importances" in result.summary


def test_no_cupac_output_without_cupac():
    """Nothing CUPAC-related is exposed when CUPAC was not requested."""
    result = ABTest().execute(_cuped_data())

    assert "cupac" not in result.outputs
    assert not any(name.startswith("cupac.") for name in result.summary)
    with pytest.raises(AttributeError):
        result.cupac
