"""Tests for MinSampleSize (closed-form equal-variance math and validation)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hypex.dataset import Dataset, TargetRole, TreatmentRole
from hypex.executor.calculators import MinSampleSize
from hypex.utils import BackendsEnum, NotSuitableFieldError


def _closed_form(var: float, q1: float, q2: float, mde: float) -> int:
    return int(2 * var * ((q1 - q2) / mde) ** 2) + 1


@pytest.fixture
def two_groups() -> Dataset:
    rng = np.random.RandomState(0)
    df = pd.DataFrame(
        {
            "g": ["a"] * 100 + ["b"] * 100,
            "y": np.r_[rng.normal(0, 2, 100), rng.normal(0, 4, 100)],
        }
    )
    return Dataset(
        roles={"g": TreatmentRole(), "y": TargetRole()},
        data=df,
        backend=BackendsEnum.pandas,
    )


@pytest.mark.parametrize(
    "var,q1,q2,mde",
    [(4.0, 2.5, -0.84, 1.0), (1.0, 1.96, -0.84, 0.5), (9.0, 2.24, -0.52, 2.0)],
)
def test_equal_variance_closed_form(var, q1, q2, mde) -> None:
    n = MinSampleSize._inner_function(
        num_samples=2,
        mde=mde,
        variances=var,
        quantile_1=q1,
        quantile_2=q2,
        equal_variance=True,
    )
    assert n == _closed_form(var, q1, q2, mde)


def test_list_variances_use_first_entry_in_equal_variance_mode() -> None:
    n = MinSampleSize._inner_function(
        num_samples=2,
        mde=1.0,
        variances=[4.0, 100.0],
        quantile_1=2.0,
        quantile_2=-1.0,
        equal_variance=True,
    )
    assert n == _closed_form(4.0, 2.0, -1.0, 1.0)


def test_sample_size_scales_inverse_square_with_mde() -> None:
    kwargs = dict(
        num_samples=2,
        variances=4.0,
        quantile_1=2.0,
        quantile_2=-1.0,
        equal_variance=True,
    )
    n1 = MinSampleSize._inner_function(mde=1.0, **kwargs)
    n2 = MinSampleSize._inner_function(mde=0.5, **kwargs)
    assert n1 == _closed_form(4.0, 2.0, -1.0, 1.0)
    assert n2 == _closed_form(4.0, 2.0, -1.0, 0.5)
    assert n2 == pytest.approx(4 * n1, abs=4)


def test_sample_size_scales_linearly_with_variance() -> None:
    kwargs = dict(
        num_samples=2, mde=1.0, quantile_1=2.0, quantile_2=-1.0, equal_variance=True
    )
    n1 = MinSampleSize._inner_function(variances=1.0, **kwargs)
    n2 = MinSampleSize._inner_function(variances=3.0, **kwargs)
    assert n2 == pytest.approx(3 * n1, rel=0.05)


def test_unequal_variance_requires_list() -> None:
    with pytest.raises(TypeError, match="variances must be a list"):
        MinSampleSize._inner_function(
            num_samples=2,
            mde=1.0,
            variances=4.0,
            quantile_1=2.0,
            quantile_2=-1.0,
            equal_variance=False,
        )


def test_estimated_quantiles_are_reproducible() -> None:
    kwargs = dict(
        num_samples=2,
        mde=1.0,
        variances=4.0,
        equal_variance=True,
        iteration_size=500,
        random_state=1,
    )
    assert MinSampleSize._inner_function(**kwargs) == MinSampleSize._inner_function(
        **kwargs
    )


def test_estimated_quantiles_give_larger_n_for_smaller_mde() -> None:
    kwargs = dict(
        num_samples=2,
        variances=4.0,
        equal_variance=True,
        iteration_size=500,
        random_state=1,
    )
    assert MinSampleSize._inner_function(
        mde=0.5, **kwargs
    ) > MinSampleSize._inner_function(mde=2.0, **kwargs)


@pytest.mark.xfail(
    strict=True,
    raises=FutureWarning,
    reason="Issue: PandasDataset.count_groups does int(df[cols].nunique()) on a Series, emitting a FutureWarning (TypeError for several group cols)",
)
def test_calc_equal_variance_uses_mean_group_variance(two_groups) -> None:
    calc = MinSampleSize(mde=1.0, equal_variance=True, quantile_1=2.0, quantile_2=-1.0)
    result = calc.calc(two_groups)
    df = two_groups.backend_data.data
    mean_var = np.mean([df.y[df.g == g].var() for g in ("a", "b")])
    expected = _closed_form(mean_var, 2.0, -1.0, 1.0)
    assert result["y"]["min sample size"] == expected
    assert result["overall"]["min sample size"] == expected


@pytest.mark.xfail(
    strict=True,
    raises=FutureWarning,
    reason="Issue: PandasDataset.count_groups does int(df[cols].nunique()) on a Series, emitting a FutureWarning (TypeError for several group cols)",
)
def test_calc_explicit_variances_override_data(two_groups) -> None:
    calc = MinSampleSize(
        mde=1.0, equal_variance=True, variances=9.0, quantile_1=2.0, quantile_2=-1.0
    )
    assert calc.calc(two_groups)["y"]["min sample size"] == _closed_form(
        9.0, 2.0, -1.0, 1.0
    )


@pytest.mark.xfail(
    strict=True,
    raises=FutureWarning,
    reason="Issue: PandasDataset.count_groups does int(df[cols].nunique()) on a Series, emitting a FutureWarning (TypeError for several group cols)",
)
def test_calc_sets_key_to_single_target(two_groups) -> None:
    calc = MinSampleSize(mde=1.0, equal_variance=True, quantile_1=2.0, quantile_2=-1.0)
    calc.calc(two_groups)
    assert calc.key == "y"


@pytest.mark.xfail(
    strict=True,
    raises=FutureWarning,
    reason="Issue: PandasDataset.count_groups does int(df[cols].nunique()) on a Series, emitting a FutureWarning (TypeError for several group cols)",
)
def test_calc_overall_is_max_over_targets() -> None:
    rng = np.random.RandomState(1)
    df = pd.DataFrame(
        {
            "g": ["a"] * 50 + ["b"] * 50,
            "small": rng.normal(0, 1, 100),
            "big": rng.normal(0, 10, 100),
        }
    )
    ds = Dataset(
        roles={"g": TreatmentRole(), "small": TargetRole(), "big": TargetRole()},
        data=df,
        backend=BackendsEnum.pandas,
    )
    result = MinSampleSize(
        mde=1.0, equal_variance=True, quantile_1=2.0, quantile_2=-1.0
    ).calc(ds)
    assert result["big"]["min sample size"] > result["small"]["min sample size"]
    assert result["overall"]["min sample size"] == result["big"]["min sample size"]


@pytest.mark.xfail(
    strict=True,
    raises=FutureWarning,
    reason="Issue: PandasDataset.count_groups does int(df[cols].nunique()) on a Series, emitting a FutureWarning (TypeError for several group cols)",
)
def test_calc_single_group_raises() -> None:
    df = pd.DataFrame({"g": ["a"] * 10, "y": np.arange(10.0)})
    ds = Dataset(
        roles={"g": TreatmentRole(), "y": TargetRole()},
        data=df,
        backend=BackendsEnum.pandas,
    )
    with pytest.raises(NotSuitableFieldError):
        MinSampleSize(
            mde=1.0, equal_variance=True, quantile_1=2.0, quantile_2=-1.0
        ).calc(ds)


def test_mde_is_keyword_only_and_required() -> None:
    with pytest.raises(TypeError):
        MinSampleSize()  # type: ignore[call-arg]


@pytest.mark.slow
def test_unequal_variance_search_returns_multiple_of_step(two_groups) -> None:
    n = MinSampleSize._inner_function(
        num_samples=2,
        mde=2.0,
        variances=[4.0, 16.0],
        quantile_1=1.5,
        quantile_2=-0.8,
        equal_variance=False,
        power_iteration_size=200,
        random_state=0,
    )
    assert n % 100 == 0 and n >= 100
