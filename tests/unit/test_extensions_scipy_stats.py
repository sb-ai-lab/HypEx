"""Tests for hypex.extensions.scipy_stats test extensions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency, ks_2samp, mannwhitneyu, ttest_ind

from hypex.dataset import Dataset, FeatureRole
from hypex.extensions.scipy_stats import (
    GroupChi2TestExtension,
    GroupKSTestExtension,
    GroupStatTest,
    GroupTTestExtension,
    GroupUTestExtension,
    NormCDF,
    PandasChi2TestExtension,
    PandasKSTestExtension,
    UniformCheck,
)
from hypex.utils import BackendsEnum
from hypex.utils.registry import backend_factory


def _ds(values, name="x", backend=BackendsEnum.pandas, session=None) -> Dataset:
    return Dataset(
        roles={name: FeatureRole()},
        data=pd.DataFrame({name: values}),
        backend=backend,
        session=session,
    )


def _res(small) -> dict:
    return {
        "p-value": float(small.get_values(column="p-value")[0]),
        "statistic": float(small.get_values(column="statistic")[0]),
    }


@pytest.fixture
def samples():
    rng = np.random.RandomState(3)
    return rng.normal(0, 1, 80), rng.normal(0.6, 1, 90)


def test_check_other_none_raises() -> None:
    with pytest.raises(ValueError, match="No other dataset"):
        GroupStatTest.check_other(None)


def test_check_dataset_requires_single_column() -> None:
    ds = Dataset(
        roles={"a": FeatureRole(), "b": FeatureRole()},
        data=pd.DataFrame({"a": [1.0], "b": [2.0]}),
        backend=BackendsEnum.pandas,
    )
    with pytest.raises(ValueError, match="one-dimensional"):
        GroupStatTest.check_dataset(ds)


def test_calc_without_test_function_raises() -> None:
    with pytest.raises(ValueError, match="test_function"):
        GroupStatTest().calc(_ds([1.0, 2.0]), _ds([1.0, 2.0]))


def test_extract_arrays_is_backend_dependent() -> None:
    with pytest.raises(NotImplementedError):
        GroupStatTest()._extract_arrays(None, None)


def test_ttest_matches_scipy_welch(samples) -> None:
    a, b = samples
    res = _res(GroupTTestExtension().calc(_ds(a), _ds(b)))
    ref = ttest_ind(a, b, equal_var=False)
    assert res["statistic"] == pytest.approx(ref.statistic)
    assert res["p-value"] == pytest.approx(ref.pvalue)


def test_ttest_equal_variance_kwarg_maps_to_equal_var(samples) -> None:
    a, b = samples
    res = _res(GroupTTestExtension().calc(_ds(a), _ds(b), equal_variance=True))
    ref = ttest_ind(a, b, equal_var=True)
    assert res["p-value"] == pytest.approx(ref.pvalue)


def test_invalid_kwargs_warn_and_are_ignored(samples) -> None:
    a, b = samples
    with pytest.warns(UserWarning, match="not accepted"):
        res = _res(GroupTTestExtension().calc(_ds(a), _ds(b), bogus=1))
    assert res["p-value"] == pytest.approx(ttest_ind(a, b, equal_var=False).pvalue)


def test_pass_flag_follows_reliability(samples) -> None:
    a, b = samples
    small = GroupTTestExtension(reliability=1.0).calc(_ds(a), _ds(b))
    assert bool(small.get_values(column="pass")[0]) is True
    small = GroupTTestExtension(reliability=1e-30).calc(_ds(a), _ds(b))
    assert bool(small.get_values(column="pass")[0]) is False


def test_utest_matches_scipy(samples) -> None:
    a, b = samples
    res = _res(GroupUTestExtension().calc(_ds(a), _ds(b)))
    ref = mannwhitneyu(a, b)
    assert res["statistic"] == pytest.approx(ref.statistic)
    assert res["p-value"] == pytest.approx(ref.pvalue)


def test_pandas_kstest_matches_scipy(samples) -> None:
    a, b = samples
    ext = backend_factory.resolve_backend(GroupKSTestExtension, _ds(a))(0.05)
    assert isinstance(ext, PandasKSTestExtension)
    res = _res(ext.calc(_ds(a), _ds(b)))
    ref = ks_2samp(a, b)
    assert res["statistic"] == pytest.approx(ref.statistic)
    assert res["p-value"] == pytest.approx(ref.pvalue)


def test_chi2_matrix_preparation_merges_rare_categories() -> None:
    data = _ds(["a"] * 20 + ["b"] * 15 + ["c"] * 3 + ["d"] * 2, "c")
    other = _ds(["a"] * 10 + ["b"] * 25 + ["c"] * 4, "c")
    ext = PandasChi2TestExtension()
    table = ext.matrix_preparation(data, other)
    assert table is not None
    arr = table.raw_data
    # rare c, d (< 7) in data merge into "other"; "other" is again < 7 -> dropped
    # everything below 7 is removed, leaving categories a and b
    assert len(arr) == 2
    assert set(arr.columns) == {"count_x", "count_y"}
    prop = 40 / (40 + 39)
    assert sorted(arr["count_x"].round(6)) == sorted(
        [round(20 * (1 - prop), 6), round(15 * (1 - prop), 6)]
    )


def test_chi2_matrix_preparation_single_category_returns_none() -> None:
    data = _ds(["a"] * 20, "c")
    other = _ds(["a"] * 10 + ["b"] * 10, "c")
    assert PandasChi2TestExtension().matrix_preparation(data, other) is None


def test_chi2_calc_degenerate_warns_and_returns_none() -> None:
    data = _ds(["a"] * 20, "c")
    other = _ds(["a"] * 10 + ["b"] * 10, "c")
    with pytest.warns(UserWarning, match="Matrix Chi2 is empty"):
        small = PandasChi2TestExtension().calc(data, other)
    assert small.get_values(column="p-value")[0] is None


@pytest.mark.xfail(
    strict=True,
    reason="GroupChi2TestExtension.calc passes (statistic, p_value) to "
    "_form_results(p_value, statistic, ...), so the two are swapped",
)
def test_chi2_calc_p_value_and_statistic_are_not_swapped() -> None:
    data = _ds(["a"] * 40 + ["b"] * 20, "c")
    other = _ds(["a"] * 20 + ["b"] * 40, "c")
    ext = PandasChi2TestExtension()
    table = ext.matrix_preparation(data, other).raw_data.values
    stat, p, *_ = chi2_contingency(table)
    res = _res(ext.calc(data, other))
    assert res["p-value"] == pytest.approx(p)
    assert res["statistic"] == pytest.approx(stat)


def test_normcdf_two_sided_p_value() -> None:
    ds = _ds([1.96])
    small = NormCDF().calc(ds)
    assert float(small.get_values(column="p-value")[0]) == pytest.approx(0.05, abs=1e-3)


def test_uniform_check_detects_uniform_vs_normal() -> None:
    rng = np.random.RandomState(0)
    p_uniform = _res(UniformCheck().calc(_ds(rng.uniform(0, 1, 500))))["p-value"]
    p_normal = _res(UniformCheck().calc(_ds(rng.normal(0, 1, 500))))["p-value"]
    assert p_uniform > 0.05 > p_normal


# ---------------------------------------------------------------------------
# Spark implementations
# ---------------------------------------------------------------------------
@pytest.fixture
def spark_ds(spark_session):
    def _make(values, name="x"):
        return _ds(values, name, BackendsEnum.spark, spark_session)

    return _make


@pytest.mark.spark
def test_spark_kstest_close_to_scipy(spark_ds, samples) -> None:
    a, b = samples
    ext = backend_factory.resolve_backend(GroupKSTestExtension, spark_ds(a))(0.05)
    res = _res(ext.calc(spark_ds(a), spark_ds(b)))
    ref = ks_2samp(a, b)
    assert res["statistic"] == pytest.approx(ref.statistic, abs=0.03)
    assert 0.0 <= res["p-value"] <= 1.0


@pytest.mark.spark
def test_spark_kstest_identical_constant_samples(spark_ds) -> None:
    ext = backend_factory.resolve_backend(GroupKSTestExtension, spark_ds([1.0]))(0.05)
    small = ext.calc(spark_ds([2.0, 2.0, 2.0]), spark_ds([2.0, 2.0]))
    assert _res(small) == {"p-value": 1.0, "statistic": 0.0}


@pytest.mark.spark
def test_spark_kstest_nan_policy_raise(spark_ds) -> None:
    ext = backend_factory.resolve_backend(GroupKSTestExtension, spark_ds([1.0]))(0.05)
    with pytest.raises(ValueError, match="NaN values found"):
        ext.calc(spark_ds([1.0, np.nan, 3.0]), spark_ds([1.0, 2.0]), nan_policy="raise")


@pytest.mark.spark
def test_spark_kstest_nan_policy_propagate(spark_ds) -> None:
    ext = backend_factory.resolve_backend(GroupKSTestExtension, spark_ds([1.0]))(0.05)
    small = ext.calc(
        spark_ds([1.0, np.nan, 3.0]), spark_ds([1.0, 2.0]), nan_policy="propagate"
    )
    assert np.isnan(float(small.get_values(column="p-value")[0]))


@pytest.mark.spark
@pytest.mark.xfail(
    strict=True,
    reason="SparkKSTestExtension default nan_policy='omit' does not drop NaN rows "
    "(NaN poisons min/max/bucket), unlike the pandas ks_2samp(nan_policy='omit')",
)
def test_spark_kstest_nan_policy_omit_matches_clean_data(spark_ds) -> None:
    ext = backend_factory.resolve_backend(GroupKSTestExtension, spark_ds([1.0]))(0.05)
    b = [0.5, 1.5, 2.5, 3.5, 4.5]
    with_nan = _res(ext.calc(spark_ds([1.0, 2.0, 3.0, np.nan]), spark_ds(b)))
    clean = _res(ext.calc(spark_ds([1.0, 2.0, 3.0]), spark_ds(b)))
    assert with_nan == pytest.approx(clean)


@pytest.mark.spark
@pytest.mark.xfail(
    strict=True,
    reason="GroupChi2TestExtension.calc swaps p-value and statistic",
)
def test_spark_chi2_matches_scipy_on_counts(spark_ds) -> None:
    data = spark_ds(["a"] * 40 + ["b"] * 20, "c")
    other = spark_ds(["a"] * 20 + ["b"] * 40, "c")
    ext = backend_factory.resolve_backend(GroupChi2TestExtension, data)()
    table = ext.matrix_preparation(data, other)
    assert table.shape == (2, 2)
    assert sorted(table.sum(axis=1)) == [60.0, 60.0]
    stat, p, *_ = chi2_contingency(table)
    res = _res(ext.calc(data, other))
    assert res["p-value"] == pytest.approx(p)
