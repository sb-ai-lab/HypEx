"""Tests for MahalanobisDistance, PSI and MDEBySize."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from hypex.comparators import PSI, MahalanobisDistance, MDEBySize
from hypex.dataset import FeatureRole, GroupingRole, TargetRole
from hypex.utils import NotSuitableFieldError

from ._utils import build_dataset, to_pandas

TOL = 1e-6


def _col(values, name="y", role=TargetRole):
    return build_dataset(pd.DataFrame({name: values}), {name: role()})


# ---------------------------------------------------------------------------
# MDEBySize
# ---------------------------------------------------------------------------
def test_mde_exact_formula() -> None:
    rng = np.random.RandomState(0)
    control, test = rng.normal(0, 1, 50), rng.normal(0, 2, 70)
    mde = MDEBySize._inner_function(_col(control), _col(test))
    m = norm.ppf(0.975) + norm.ppf(0.8)
    s = np.sqrt(np.var(test, ddof=1) / 70 + np.var(control, ddof=1) / 50)
    assert mde == pytest.approx(m * s, abs=TOL)


@pytest.mark.parametrize("significance,power", [(0.9, 0.8), (0.95, 0.9), (0.99, 0.8)])
def test_mde_uses_significance_and_power(significance, power) -> None:
    rng = np.random.RandomState(1)
    control, test = rng.normal(0, 1, 40), rng.normal(0, 1, 40)
    mde = MDEBySize._inner_function(_col(control), _col(test), significance=significance, power=power)
    m = norm.ppf((1 + significance) / 2) + norm.ppf(power)
    s = np.sqrt(np.var(test, ddof=1) / 40 + np.var(control, ddof=1) / 40)
    assert mde == pytest.approx(m * s, abs=TOL)


def test_mde_decreases_with_more_data() -> None:
    rng = np.random.RandomState(2)
    small = MDEBySize._inner_function(_col(rng.normal(0, 1, 30)), _col(rng.normal(0, 1, 30)))
    large = MDEBySize._inner_function(_col(rng.normal(0, 1, 3000)), _col(rng.normal(0, 1, 3000)))
    assert large < small


def test_mde_requires_test_data() -> None:
    with pytest.raises(ValueError, match="test_data is required"):
        MDEBySize._inner_function(_col([1.0, 2.0, 3.0]), None)


def test_mde_defaults_stored() -> None:
    ex = MDEBySize()
    assert ex.significance == 0.95 and ex.power == 0.8


# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="Issue: PSI._inner_function calls Dataset.cut which does not exist",
)
def test_psi_identical_samples_is_zero() -> None:
    x = np.linspace(0, 10, 200)
    res = PSI._inner_function(_col(x), _col(x.copy()))
    assert res["PSI"] == pytest.approx(0.0, abs=TOL)


def test_psi_requires_test_data() -> None:
    with pytest.raises(ValueError, match="test_data"):
        PSI._inner_function(_col([1.0, 2.0, 3.0]), None)


# ---------------------------------------------------------------------------
# MahalanobisDistance
# ---------------------------------------------------------------------------
@pytest.fixture
def two_group_features():
    rng = np.random.RandomState(1)
    n = 300
    cov = [[2.0, 0.8], [0.8, 1.0]]
    a = rng.multivariate_normal([0, 0], cov, n)
    b = rng.multivariate_normal([1, 1], cov, n)
    df = pd.DataFrame(np.r_[a, b], columns=["f1", "f2"])
    df["g"] = [0] * n + [1] * n
    roles = {"f1": FeatureRole(), "f2": FeatureRole(), "g": GroupingRole()}
    return build_dataset(df, roles), a, b


@pytest.mark.xfail(
    strict=True,
    raises=FutureWarning,
    reason="Issue: PandasDataset.count_groups does int(df[cols].nunique()) on a Series, emitting a FutureWarning (TypeError for several group cols)",
)
def test_mahalanobis_whitens_pooled_covariance(two_group_features) -> None:
    ds, a, b = two_group_features
    result = MahalanobisDistance.calc(ds, group_field="g", target_fields=["f1", "f2"])
    w = to_pandas(result).to_numpy(dtype=float)
    pooled = (np.cov(a.T) + np.cov(b.T)) / 2
    # CholeskyExtension adds epsilon=1e-3 to the diagonal, hence the loose tolerance.
    np.testing.assert_allclose(w.T @ pooled @ w, np.eye(2), atol=5e-3)


@pytest.mark.xfail(
    strict=True,
    raises=FutureWarning,
    reason="Issue: PandasDataset.count_groups does int(df[cols].nunique()) on a Series, emitting a FutureWarning (TypeError for several group cols)",
)
def test_mahalanobis_matches_numpy_cholesky(two_group_features) -> None:
    ds, a, b = two_group_features
    result = MahalanobisDistance.calc(ds, group_field="g", target_fields=["f1", "f2"])
    pooled = (np.cov(a.T) + np.cov(b.T)) / 2 + 1e-3 * np.eye(2)
    expected = np.linalg.inv(np.linalg.cholesky(pooled)).T
    np.testing.assert_allclose(to_pandas(result).to_numpy(dtype=float), expected, atol=TOL)


@pytest.mark.xfail(
    strict=True,
    reason="Issue: Dataset.dot(ndarray) returns an empty-role dataset, so weighted "
    "transform collapses to shape (0, 0)",
)
def test_mahalanobis_weights_change_transform(two_group_features) -> None:
    ds, _, _ = two_group_features
    plain = to_pandas(MahalanobisDistance.calc(ds, group_field="g", target_fields=["f1", "f2"]))
    weighted = to_pandas(
        MahalanobisDistance.calc(
            ds, group_field="g", target_fields=["f1", "f2"], weights={"f1": 3.0, "f2": 1.0}
        )
    )
    assert not np.allclose(plain.to_numpy(dtype=float), weighted.to_numpy(dtype=float))


@pytest.mark.xfail(
    strict=True,
    raises=FutureWarning,
    reason="Issue: PandasDataset.count_groups does int(df[cols].nunique()) on a Series, emitting a FutureWarning (TypeError for several group cols)",
)
def test_mahalanobis_single_group_raises() -> None:
    df = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [2.0, 1.0, 5.0], "g": [0, 0, 0]})
    ds = build_dataset(
        df, {"f1": FeatureRole(), "f2": FeatureRole(), "g": GroupingRole()}
    )
    with pytest.raises(NotSuitableFieldError):
        MahalanobisDistance.calc(ds, group_field="g", target_fields=["f1", "f2"])


def test_mahalanobis_search_types() -> None:
    assert MahalanobisDistance().search_types == [int, float]


# ---------------------------------------------------------------------------
# MahalanobisDistance with pre-grouped data (bypasses PandasDataset.count_groups)
# ---------------------------------------------------------------------------
def _pregrouped(ds):
    df = to_pandas(ds)
    roles = {"f1": FeatureRole(), "f2": FeatureRole(), "g": GroupingRole()}
    return [
        ((g,), build_dataset(part.reset_index(drop=True), roles))
        for g, part in df.groupby("g")
    ]


def test_mahalanobis_pregrouped_matches_numpy_cholesky(two_group_features) -> None:
    ds, a, b = two_group_features
    result = MahalanobisDistance.calc(
        ds,
        group_field="g",
        grouping_data=_pregrouped(ds),
        target_fields=["f1", "f2"],
    )
    pooled = (np.cov(a.T) + np.cov(b.T)) / 2 + 1e-3 * np.eye(2)
    expected = np.linalg.inv(np.linalg.cholesky(pooled)).T
    np.testing.assert_allclose(
        to_pandas(result).to_numpy(dtype=float), expected, atol=TOL
    )


def test_mahalanobis_pregrouped_single_group_raises(two_group_features) -> None:
    ds, _, _ = two_group_features
    with pytest.raises(NotSuitableFieldError):
        MahalanobisDistance.calc(
            ds,
            group_field="g",
            grouping_data=_pregrouped(ds)[:1],
            target_fields=["f1", "f2"],
        )


def test_mahalanobis_execute_inner_function_single_group_requires_test_data(
    two_group_features,
) -> None:
    ds, _, _ = two_group_features
    with pytest.raises(ValueError, match="test_data"):
        MahalanobisDistance._execute_inner_function(
            _pregrouped(ds)[:1], target_fields=["f1", "f2"]
        )


def test_mahalanobis_get_fields_and_set_value(two_group_features) -> None:
    from hypex.dataset import ExperimentData

    ds, _, _ = two_group_features
    data = ExperimentData(ds)
    op = MahalanobisDistance()
    group_field, target_fields = op._get_fields(data)
    assert group_field == ["g"]
    assert sorted(target_fields) == ["f1", "f2"]

    transform = MahalanobisDistance.calc(
        ds, group_field="g", grouping_data=_pregrouped(ds), target_fields=target_fields
    )
    op.key = "k"
    out = op._set_value(data, transform)
    assert out is data
    stored = out.variables[op.id]["k"]
    np.testing.assert_allclose(
        to_pandas(stored).to_numpy(dtype=float),
        to_pandas(transform).to_numpy(dtype=float),
    )


def test_mahalanobis_stores_weights_and_grouping_role() -> None:
    op = MahalanobisDistance(weights={"f1": 2.0})
    assert op.weights == {"f1": 2.0}
    assert isinstance(op.grouping_role, GroupingRole)


def test_mahalanobis_execute_uses_precomputed_groups(two_group_features) -> None:
    from hypex.dataset import ExperimentData
    from hypex.utils import ExperimentDataEnum

    ds, a, b = two_group_features
    data = ExperimentData(ds)
    for key, part in _pregrouped(ds):
        data.set_value(ExperimentDataEnum.groups, "g", part, key=key[0])
    op = MahalanobisDistance()
    out = op.execute(data)

    stored = out.variables[op.id]
    assert op.key == str(["f1", "f2"])
    pooled = (np.cov(a.T) + np.cov(b.T)) / 2 + 1e-3 * np.eye(2)
    expected = np.linalg.inv(np.linalg.cholesky(pooled)).T
    np.testing.assert_allclose(
        to_pandas(next(iter(stored.values()))).to_numpy(dtype=float),
        expected,
        atol=TOL,
    )


def test_mahalanobis_execute_without_targets_and_tmp_roles_is_noop() -> None:
    from hypex.dataset import ExperimentData

    df = pd.DataFrame({"t": [1.0, 2.0, 3.0, 4.0], "g": [0, 0, 1, 1]})
    ds = build_dataset(df, {"t": TargetRole(), "g": GroupingRole()})
    ds.tmp_roles = {"t": TargetRole()}
    data = ExperimentData(ds)
    op = MahalanobisDistance()
    assert op.execute(data) is data
    assert op.id not in data.variables
