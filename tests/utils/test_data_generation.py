"""Tests for tutorial data generators in hypex.utils.tutorial_data_creation."""
from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd
import pytest

from hypex.utils.tutorial_data_creation import (
    DataGenerator,
    create_test_data,
    gen_control_variates_df,
    gen_oracle_df,
    gen_special_medicine_df,
    set_nans,
    sigmoid,
    sigmoid_division,
)


@contextlib.contextmanager
def _preserve_np_random():
    """Save and restore the global numpy RNG state around a block."""
    state = np.random.get_state()
    try:
        yield
    finally:
        np.random.set_state(state)


# ── create_test_data ──────────────────────────────────────────────────────
@pytest.mark.parametrize("num_users", [50, 200])
def test_create_test_data_shape(num_users: int) -> None:
    """create_test_data returns one row per user with expected columns."""
    df = create_test_data(num_users=num_users, rs=7)

    assert len(df) == num_users
    expected = {
        "user_id", "signup_month", "treat", "pre_spends", "post_spends",
        "age", "gender", "industry",
    }
    assert expected.issubset(df.columns)


def test_create_test_data_reproducible() -> None:
    """The same seed produces identical frames."""
    a = create_test_data(num_users=100, rs=42)
    b = create_test_data(num_users=100, rs=42)
    pd.testing.assert_frame_equal(a, b)


def test_create_test_data_treat_is_binary() -> None:
    """Treatment assignment is strictly binary."""
    df = create_test_data(num_users=200, rs=1)
    assert set(df["treat"].unique()).issubset({0, 1})


def test_create_test_data_exact_att_shifts_post_spends() -> None:
    """Larger exact_ATT increases post-signup spending of treated users."""
    low = create_test_data(num_users=2000, exact_ATT=10, rs=3)
    high = create_test_data(num_users=2000, exact_ATT=500, rs=3)

    low_mean = low.loc[low["treat"] == 1, "post_spends"].mean()
    high_mean = high.loc[high["treat"] == 1, "post_spends"].mean()
    assert high_mean > low_mean


# ── set_nans ──────────────────────────────────────────────────────────────
def test_set_nans_injects_missing_values() -> None:
    """set_nans inserts NaNs at the configured step interval."""
    df = pd.DataFrame({"a": range(10), "b": range(10)})
    result = set_nans(df, na_step=5, nan_cols="a")

    assert result["a"].isna().sum() == 2
    assert result["b"].isna().sum() == 0


def test_set_nans_does_not_mutate_input() -> None:
    """set_nans returns a copy and leaves the input untouched."""
    df = pd.DataFrame({"a": [1.0, 2.0]})
    _ = set_nans(df, na_step=1, nan_cols="a")
    assert df["a"].isna().sum() == 0


# ── oracle / domain generators ────────────────────────────────────────────
def test_gen_oracle_df_structure() -> None:
    """gen_oracle_df returns the documented columns and TE consistency."""
    df = gen_oracle_df(data_size=50, random_state=11)

    assert len(df) == 50
    assert set(df.columns) == {
        "X", "Target_untreated", "Target_treated", "Treatment", "Target", "TE",
    }
    assert np.allclose(df["TE"], df["Target_treated"] - df["Target_untreated"])


def test_gen_oracle_df_reproducible() -> None:
    """gen_oracle_df is deterministic under a fixed seed."""
    a = gen_oracle_df(data_size=30, random_state=5)
    b = gen_oracle_df(data_size=30, random_state=5)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("dependent", [True, False])
def test_gen_oracle_df_dependent_division(dependent: bool) -> None:
    """dependent_division flag is accepted for both modes."""
    df = gen_oracle_df(data_size=40, dependent_division=dependent, random_state=2)
    assert set(df["Treatment"].unique()).issubset({0, 1})


def test_gen_special_medicine_df_columns() -> None:
    """gen_special_medicine_df returns medical trial columns."""
    df = gen_special_medicine_df(data_size=100, random_state=9)

    assert set(df.columns) == {
        "age", "disease_degree", "experimental_treatment", "residual_lifetime",
    }
    assert df["disease_degree"].between(1, 5).all()
    assert (df["residual_lifetime"] >= 0).all()


def test_gen_control_variates_df_columns() -> None:
    """gen_control_variates_df returns the CUPED-friendly layout."""
    df = gen_control_variates_df(data_size=100, random_state=4)
    assert set(df.columns) == {
        "X_lag_1", "Target_lag_1", "X", "Treatment", "Target",
    }
    assert len(df) == 100


# ── DataGenerator ─────────────────────────────────────────────────────────
def test_data_generator_generate_shape() -> None:
    """DataGenerator produces the documented feature set."""
    gen = DataGenerator(n_samples=100, seed=21)
    df = gen.generate()

    assert len(df) == 100
    for col in ("z", "U", "D", "d", "X1_lag1", "X2_lag1", "X1_lag2", "X2_lag2",
                "y0", "y1", "y"):
        assert col in df.columns


def test_data_generator_reproducible() -> None:
    """The same seed yields identical generated frames."""
    a = DataGenerator(n_samples=50, seed=13).generate()
    b = DataGenerator(n_samples=50, seed=13).generate()
    pd.testing.assert_frame_equal(a, b)


# ── sigmoid helpers ───────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "x,expected", [(0.0, 0.5), (100.0, pytest.approx(1.0)), (-100.0, pytest.approx(0.0))]
)
def test_sigmoid_values(x: float, expected) -> None:
    """sigmoid is monotone and bounded in (0, 1)."""
    value = float(sigmoid(np.array([x]))[0])
    assert value == expected
    assert 0.0 <= value <= 1.0


def test_sigmoid_division_is_binary() -> None:
    """sigmoid_division always returns a binary vector."""
    with _preserve_np_random():
        result = sigmoid_division(np.arange(50), dependent_division=True)
    assert set(np.unique(result)).issubset({0, 1})