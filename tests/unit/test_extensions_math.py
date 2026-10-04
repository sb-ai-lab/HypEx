"""Tests for MultiTest, MultitestQuantile, CholeskyExtension and LstsqExtension."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from hypex.dataset import Dataset, FeatureRole, StatisticRole, TargetRole
from hypex.extensions import (
    CholeskyExtension,
    InverseExtension,
    LstsqExtension,
    MultiTest,
    MultitestQuantile,
    PandasLstsqExtension,
    SparkLstsqExtension,
    UniteCovExtension,
)
from hypex.extensions.scipy_stats import GroupTTestExtension
from hypex.utils import ABNTestMethodsEnum, BackendsEnum
from hypex.utils.constants import ID_SPLIT_SYMBOL as S

TOL = 1e-8

RAW_P = [0.001, 0.008, 0.02, 0.04, 0.3, 0.7]


def _pvalue_dataset(p_values, test="GroupTTest", fields=None, groups=None) -> Dataset:
    fields = fields or [f"y{i}" for i in range(len(p_values))]
    groups = groups or ["b"] * len(p_values)
    index = [f"{test}{S}hash{S}{f}{S}{g}" for f, g in zip(fields, groups)]
    return Dataset(
        roles={"p-value": StatisticRole()},
        data=pd.DataFrame({"p-value": p_values}, index=index),
        backend=BackendsEnum.pandas,
    )


def _frame(ds) -> pd.DataFrame:
    return ds.backend_data.data


def _as_bool(value) -> bool:
    """MultiTest stores booleans as strings ("True"/"False"), see the dtype test below."""
    return str(value) == "True"


# ---------------------------------------------------------------------------
# MultiTest
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "method,sm_name",
    [
        (ABNTestMethodsEnum.bonferroni, "bonferroni"),
        (ABNTestMethodsEnum.sidak, "sidak"),
        (ABNTestMethodsEnum.holm, "holm"),
        (ABNTestMethodsEnum.holm_sidak, "holm-sidak"),
        (ABNTestMethodsEnum.simes_hochberg, "simes-hochberg"),
        (ABNTestMethodsEnum.hommel, "hommel"),
        (ABNTestMethodsEnum.fdr_bh, "fdr_bh"),
        (ABNTestMethodsEnum.fdr_by, "fdr_by"),
        (ABNTestMethodsEnum.fdr_tsbh, "fdr_tsbh"),
    ],
)
def test_multitest_matches_statsmodels(method, sm_name) -> None:
    result = _frame(MultiTest(method, 0.05).calc(_pvalue_dataset(RAW_P)))
    expected = multipletests(RAW_P, alpha=0.05, method=sm_name)
    np.testing.assert_allclose(
        result["new p-value"].astype(float), expected[1], atol=TOL
    )
    assert [_as_bool(v) for v in result["rejected"]] == expected[0].tolist()
    np.testing.assert_allclose(result["old p-value"].astype(float), RAW_P, atol=TOL)


@pytest.mark.xfail(
    strict=True,
    reason="Issue: MultiTest result columns are stringified (p-values and 'rejected' come "
    "back as str), so bool(rejected) is True for every row",
)
def test_multitest_result_columns_are_typed() -> None:
    result = _frame(
        MultiTest(ABNTestMethodsEnum.bonferroni).calc(_pvalue_dataset(RAW_P))
    )
    assert result["rejected"].dtype == bool
    assert result["new p-value"].dtype.kind == "f"


def test_bonferroni_closed_form() -> None:
    result = _frame(
        MultiTest(ABNTestMethodsEnum.bonferroni).calc(_pvalue_dataset(RAW_P))
    )
    np.testing.assert_allclose(
        result["new p-value"].astype(float),
        np.minimum(np.array(RAW_P) * len(RAW_P), 1.0),
        atol=TOL,
    )


def test_sidak_closed_form() -> None:
    result = _frame(MultiTest(ABNTestMethodsEnum.sidak).calc(_pvalue_dataset(RAW_P)))
    expected = 1 - (1 - np.array(RAW_P)) ** len(RAW_P)
    np.testing.assert_allclose(result["new p-value"].astype(float), expected, atol=TOL)


def test_holm_closed_form_is_step_down_and_monotone() -> None:
    result = _frame(MultiTest(ABNTestMethodsEnum.holm).calc(_pvalue_dataset(RAW_P)))
    m = len(RAW_P)
    sorted_p = np.sort(RAW_P)
    manual = np.maximum.accumulate((m - np.arange(m)) * sorted_p).clip(max=1.0)
    np.testing.assert_allclose(
        np.sort(result["new p-value"].astype(float)), manual, atol=TOL
    )


def test_fdr_bh_closed_form() -> None:
    result = _frame(MultiTest(ABNTestMethodsEnum.fdr_bh).calc(_pvalue_dataset(RAW_P)))
    m = len(RAW_P)
    ranked = np.array(RAW_P) * m / (np.arange(m) + 1)
    manual = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    np.testing.assert_allclose(result["new p-value"].astype(float), manual, atol=TOL)


def test_corrected_pvalues_are_never_smaller_than_raw() -> None:
    for method in (
        ABNTestMethodsEnum.bonferroni,
        ABNTestMethodsEnum.holm,
        ABNTestMethodsEnum.fdr_bh,
    ):
        result = _frame(MultiTest(method).calc(_pvalue_dataset(RAW_P)))
        assert (
            result["new p-value"].astype(float).to_numpy() >= np.array(RAW_P) - TOL
        ).all()


def test_alpha_controls_rejection() -> None:
    strict = _frame(
        MultiTest(ABNTestMethodsEnum.bonferroni, alpha=0.001).calc(
            _pvalue_dataset(RAW_P)
        )
    )
    loose = _frame(
        MultiTest(ABNTestMethodsEnum.bonferroni, alpha=0.5).calc(_pvalue_dataset(RAW_P))
    )
    assert sum(_as_bool(v) for v in strict["rejected"]) < sum(
        _as_bool(v) for v in loose["rejected"]
    )


def test_correction_is_applied_per_test_family() -> None:
    p_values = [0.01, 0.02, 0.03, 0.04]
    index = [
        f"GroupTTest{S}h{S}y{S}b",
        f"GroupTTest{S}h{S}y2{S}b",
        f"GroupKSTest{S}h{S}y{S}b",
        f"GroupKSTest{S}h{S}y2{S}b",
    ]
    ds = Dataset(
        roles={"p-value": StatisticRole()},
        data=pd.DataFrame({"p-value": p_values}, index=index),
        backend=BackendsEnum.pandas,
    )
    result = _frame(MultiTest(ABNTestMethodsEnum.bonferroni).calc(ds))
    # two p-values per family -> multiplied by 2, not by 4
    np.testing.assert_allclose(
        result["new p-value"].astype(float), np.array(p_values) * 2, atol=TOL
    )
    assert list(result["test"]) == ["TTest", "TTest", "KSTest", "KSTest"]


def test_index_parts_split_test_field_group() -> None:
    tests, fields, groups = MultiTest._index_parts(
        [f"GroupTTest{S}h{S}rev{S}b", f"X{S}h{S}f"]
    )
    assert tests == ["GroupTTest", "X"]
    assert fields == ["rev", "f"]
    assert groups == ["b", ""]


def test_correction_ratio_and_zero_pvalue_guard() -> None:
    result = _frame(
        MultiTest(ABNTestMethodsEnum.bonferroni).calc(_pvalue_dataset([0.0, 0.2]))
    )
    assert float(result["correction"].iloc[0]) == 0.0
    assert float(result["correction"].iloc[1]) == pytest.approx(0.2 / 0.4)


# ---------------------------------------------------------------------------
# MultitestQuantile
# ---------------------------------------------------------------------------
def test_quantile_equal_variance_matches_analytic_value() -> None:
    """For 2 groups the statistic (Z_j - Z_i)/sqrt(2) is standard normal."""
    mtq = MultitestQuantile(
        alpha=0.05, iteration_size=20000, equal_variance=True, random_state=0
    )
    q = mtq.quantile_of_marginal_distribution(num_samples=2, quantile_level=0.975)
    assert q == [q[0]] * 2
    assert q[0] == pytest.approx(norm.ppf(0.975), abs=0.06)


def test_quantile_is_reproducible_with_seed() -> None:
    kwargs = dict(num_samples=3, quantile_level=0.9)
    a = MultitestQuantile(
        iteration_size=500, random_state=1
    ).quantile_of_marginal_distribution(**kwargs)
    b = MultitestQuantile(
        iteration_size=500, random_state=1
    ).quantile_of_marginal_distribution(**kwargs)
    assert a == b


def test_quantile_increases_with_level() -> None:
    mtq = MultitestQuantile(iteration_size=2000, random_state=0)
    low = mtq.quantile_of_marginal_distribution(3, 0.5)[0]
    high = mtq.quantile_of_marginal_distribution(3, 0.99)[0]
    assert high > low


def test_quantile_unequal_variance_needs_variances() -> None:
    mtq = MultitestQuantile(iteration_size=100, equal_variance=False, random_state=0)
    q = mtq.quantile_of_marginal_distribution(2, 0.9, variances=[1.0, 4.0])
    assert len(q) == 2


def test_quantile_without_variances_forces_equal_variance() -> None:
    mtq = MultitestQuantile(iteration_size=100, equal_variance=False, random_state=0)
    q = mtq.quantile_of_marginal_distribution(3, 0.9)
    assert mtq.equal_variance is True
    assert len(set(q)) == 1


def test_quantile_calc_accepts_best_hypothesis() -> None:
    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame(
        {
            "g": ["a"] * n + ["b"] * n + ["c"] * n,
            "y": np.r_[rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(3, 1, n)],
        }
    )
    ds = Dataset(
        roles={"g": FeatureRole(), "y": TargetRole()},
        data=df,
        backend=BackendsEnum.pandas,
    )
    mtq = MultitestQuantile(alpha=0.05, iteration_size=1000, random_state=0)
    result = mtq._calc_pandas(ds, group_field="g", target_field="y")
    assert int(_frame(result).iloc[0]["accepted hypothesis"]) == 3


def test_quantile_calc_rejects_when_groups_equal() -> None:
    rng = np.random.RandomState(1)
    n = 200
    df = pd.DataFrame({"g": ["a"] * n + ["b"] * n, "y": rng.normal(0, 1, 2 * n)})
    ds = Dataset(
        roles={"g": FeatureRole(), "y": TargetRole()},
        data=df,
        backend=BackendsEnum.pandas,
    )
    mtq = MultitestQuantile(alpha=0.05, iteration_size=1000, random_state=0)
    result = mtq._calc_pandas(ds, group_field="g", target_field="y", quantiles=5.0)
    assert int(_frame(result).iloc[0]["accepted hypothesis"]) == 0


# ---------------------------------------------------------------------------
# Linear algebra extensions
# ---------------------------------------------------------------------------
def _spd(dim: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    a = rng.normal(0, 1, (dim, dim))
    return a @ a.T + np.eye(dim)


def _matrix_ds(matrix: np.ndarray) -> Dataset:
    cols = [f"c{i}" for i in range(matrix.shape[1])]
    return Dataset(
        roles={c: FeatureRole() for c in cols},
        data=pd.DataFrame(matrix, columns=cols),
        backend=BackendsEnum.pandas,
    )


def test_cholesky_reconstructs_matrix_with_epsilon() -> None:
    cov = _spd()
    lower = _frame(CholeskyExtension().calc(_matrix_ds(cov), epsilon=1e-3)).to_numpy(
        dtype=float
    )
    np.testing.assert_allclose(lower @ lower.T, cov + 1e-3 * np.eye(3), atol=1e-9)
    assert np.allclose(lower, np.tril(lower))


def test_cholesky_zero_epsilon_is_exact() -> None:
    cov = _spd()
    lower = _frame(CholeskyExtension().calc(_matrix_ds(cov), epsilon=0.0)).to_numpy(
        dtype=float
    )
    np.testing.assert_allclose(lower, np.linalg.cholesky(cov), atol=1e-10)


def test_cholesky_epsilon_regularises_singular_matrix() -> None:
    singular = np.ones((3, 3))
    with pytest.raises(np.linalg.LinAlgError):
        CholeskyExtension().calc(_matrix_ds(singular), epsilon=0.0)
    lower = _frame(CholeskyExtension().calc(_matrix_ds(singular), epsilon=1e-3))
    assert np.isfinite(lower.to_numpy(dtype=float)).all()


def test_inverse_extension() -> None:
    cov = _spd()
    inv = _frame(InverseExtension().calc(_matrix_ds(cov))).to_numpy(dtype=float)
    np.testing.assert_allclose(inv @ cov, np.eye(3), atol=1e-9)


def test_unite_cov_averages_group_covariances() -> None:
    rng = np.random.RandomState(0)
    a = pd.DataFrame(rng.normal(0, 1, (50, 2)), columns=["x", "y"])
    b = pd.DataFrame(rng.normal(0, 3, (70, 2)), columns=["x", "y"])
    roles = {"x": FeatureRole(), "y": FeatureRole()}
    ds_a = Dataset(roles=dict(roles), data=a, backend=BackendsEnum.pandas)
    ds_b = Dataset(roles=dict(roles), data=b, backend=BackendsEnum.pandas)
    united = _frame(UniteCovExtension().calc(ds_a, ds_b)).to_numpy(dtype=float)
    np.testing.assert_allclose(
        united, (a.cov().to_numpy() + b.cov().to_numpy()) / 2, atol=1e-9
    )
    single = _frame(UniteCovExtension().calc(ds_a)).to_numpy(dtype=float)
    np.testing.assert_allclose(single, a.cov().to_numpy(), atol=1e-9)


def _regression_ds(backend=BackendsEnum.pandas, session=None, noise=0.0):
    rng = np.random.RandomState(0)
    n = 200
    X = rng.normal(0, 1, (n, 2))
    y = 4.0 + 2.0 * X[:, 0] - 3.0 * X[:, 1] + rng.normal(0, noise, n)
    df = pd.DataFrame({"y": y, "a": X[:, 0], "b": X[:, 1]})
    return Dataset(
        roles={"y": TargetRole(), "a": FeatureRole(), "b": FeatureRole()},
        data=df,
        backend=backend,
        session=session if backend == BackendsEnum.spark else None,
    ), df


def test_lstsq_get_columns_puts_target_first() -> None:
    ds, _ = _regression_ds()
    assert LstsqExtension.get_columns(ds) == ["y", "a", "b"]


def test_pandas_lstsq_recovers_coefficients_without_intercept() -> None:
    ds, _ = _regression_ds()
    coefs = np.ravel(PandasLstsqExtension().calc(ds))
    np.testing.assert_allclose(coefs, [2.0, -3.0], atol=1e-8)


def test_pandas_lstsq_matches_numpy_with_noise() -> None:
    ds, df = _regression_ds(noise=0.5)
    coefs = np.ravel(PandasLstsqExtension().calc(ds))
    design = np.c_[np.ones(len(df)), df[["a", "b"]].to_numpy()]
    expected = np.linalg.lstsq(design, df.y.to_numpy(), rcond=None)[0][1:]
    np.testing.assert_allclose(coefs, expected, atol=1e-8)


@pytest.mark.spark
def test_spark_lstsq_close_to_pandas(spark_session) -> None:
    pandas_ds, _ = _regression_ds(noise=0.1)
    spark_ds, _ = _regression_ds(BackendsEnum.spark, spark_session, noise=0.1)
    expected = np.ravel(PandasLstsqExtension().calc(pandas_ds))
    got = np.ravel(SparkLstsqExtension().calc(spark_ds))
    # Spark uses regParam=0.01 (ridge), so only approximate agreement is expected
    np.testing.assert_allclose(got, expected, atol=0.05)


# ---------------------------------------------------------------------------
# GroupTTestExtension: equal_variance -> equal_var mapping
# ---------------------------------------------------------------------------
def _ttest_samples() -> tuple[Dataset, Dataset, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(0)
    a, b = rng.normal(0, 1, 30), rng.normal(0, 4, 20)

    def make(values):
        return Dataset(
            roles={"y": TargetRole()},
            data=pd.DataFrame({"y": values}),
            backend=BackendsEnum.pandas,
        )

    return make(a), make(b), a, b


def test_group_ttest_calc_does_not_mutate_callers_kwargs() -> None:
    """Behavioural guard: ``**kwargs`` unpacking already copies, so this cannot
    fail on the old in-place ``kwargs.pop`` implementation; it pins the contract."""
    first, second, _, _ = _ttest_samples()
    caller_kwargs = {"equal_variance": True}
    GroupTTestExtension().calc(first, second, **caller_kwargs)
    assert caller_kwargs == {"equal_variance": True}


@pytest.mark.parametrize("flag", [True, False])
def test_group_ttest_equal_variance_maps_to_scipy_equal_var(flag) -> None:
    from scipy.stats import ttest_ind

    first, second, a, b = _ttest_samples()
    result = GroupTTestExtension().calc(first, second, equal_variance=flag)
    expected = ttest_ind(a, b, equal_var=flag)
    row = result.backend_data.data
    assert row["p-value"].iloc[0] == pytest.approx(expected.pvalue)
    assert row["statistic"].iloc[0] == pytest.approx(expected.statistic)


@pytest.mark.parametrize("flag", [True, False])
def test_group_ttest_equal_variance_beats_equal_var(flag) -> None:
    from scipy.stats import ttest_ind

    first, second, a, b = _ttest_samples()
    result = GroupTTestExtension().calc(
        first, second, equal_variance=flag, equal_var=not flag
    )
    expected = ttest_ind(a, b, equal_var=flag)
    assert result.backend_data.data["p-value"].iloc[0] == pytest.approx(expected.pvalue)
