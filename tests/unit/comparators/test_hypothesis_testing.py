"""Group hypothesis tests compared against ``scipy.stats`` reference values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from hypex.comparators import GroupChi2Test, GroupKSTest, GroupTTest, GroupUTest
from hypex.dataset import ExperimentData, TargetRole, TreatmentRole

from ._utils import build_dataset, result_frame, three_groups_df, to_pandas

TOL = 1e-6


def _col(values, name="y"):
    return build_dataset(pd.DataFrame({name: values}), {name: TargetRole()})


@pytest.fixture
def samples():
    rng = np.random.RandomState(42)
    return rng.normal(0.0, 1.0, 60), rng.normal(0.4, 1.7, 80)


def _row(ds):
    return to_pandas(ds).iloc[0]


# ---------------------------------------------------------------------------
# GroupTTest (Welch)
# ---------------------------------------------------------------------------
def test_ttest_matches_scipy_welch(samples) -> None:
    a, b = samples
    row = _row(GroupTTest._inner_function(_col(a), _col(b)))
    ref = stats.ttest_ind(a, b, equal_var=False)
    assert row["statistic"] == pytest.approx(ref.statistic, abs=TOL)
    assert row["p-value"] == pytest.approx(ref.pvalue, abs=TOL)
    assert bool(row["pass"]) == (ref.pvalue < 0.05)


def test_ttest_reliability_controls_pass(samples) -> None:
    a, b = samples
    p = stats.ttest_ind(a, b, equal_var=False).pvalue
    loose = _row(
        GroupTTest._inner_function(_col(a), _col(b), reliability=min(1.0, p * 2))
    )
    strict = _row(GroupTTest._inner_function(_col(a), _col(b), reliability=p / 2))
    assert bool(loose["pass"]) is True
    assert bool(strict["pass"]) is False


def test_ttest_omits_nan(samples) -> None:
    a, b = samples
    a_nan = np.r_[a, np.nan, np.nan]
    row = _row(GroupTTest._inner_function(_col(a_nan), _col(b)))
    ref = stats.ttest_ind(a, b, equal_var=False)
    assert row["p-value"] == pytest.approx(ref.pvalue, abs=TOL)


def test_ttest_identical_samples_not_significant() -> None:
    x = np.arange(20, dtype=float)
    row = _row(GroupTTest._inner_function(_col(x), _col(x.copy())))
    assert row["p-value"] == pytest.approx(1.0, abs=TOL)
    assert bool(row["pass"]) is False


def test_ttest_requires_test_data(samples) -> None:
    with pytest.raises(ValueError):
        GroupTTest._inner_function(_col(samples[0]), None)


def test_ttest_rejects_multi_column_data(samples) -> None:
    a, b = samples
    wide = build_dataset(
        pd.DataFrame({"y": a[:10], "z": a[:10]}), {"y": TargetRole(), "z": TargetRole()}
    )
    with pytest.raises(ValueError, match="one-dimensional"):
        GroupTTest._inner_function(wide, _col(b))


# ---------------------------------------------------------------------------
# GroupUTest
# ---------------------------------------------------------------------------
def test_utest_matches_scipy(samples) -> None:
    a, b = samples
    row = _row(GroupUTest._inner_function(_col(a), _col(b)))
    ref = stats.mannwhitneyu(a, b)
    assert row["statistic"] == pytest.approx(ref.statistic, abs=TOL)
    assert row["p-value"] == pytest.approx(ref.pvalue, abs=TOL)


# ---------------------------------------------------------------------------
# GroupKSTest
# ---------------------------------------------------------------------------
def test_kstest_matches_scipy(samples) -> None:
    a, b = samples
    row = _row(GroupKSTest._inner_function(_col(a), _col(b)))
    ref = stats.ks_2samp(a, b)
    assert row["statistic"] == pytest.approx(ref.statistic, abs=TOL)
    assert row["p-value"] == pytest.approx(ref.pvalue, abs=TOL)


def test_kstest_detects_shifted_distribution() -> None:
    rng = np.random.RandomState(1)
    row = _row(
        GroupKSTest._inner_function(
            _col(rng.normal(0, 1, 300)), _col(rng.normal(3, 1, 300))
        )
    )
    assert bool(row["pass"]) is True
    assert row["statistic"] > 0.8


# ---------------------------------------------------------------------------
# GroupChi2Test
# ---------------------------------------------------------------------------
def _cat(values):
    return build_dataset(pd.DataFrame({"c": values}), {"c": TargetRole()})


def test_chi2_single_category_returns_none_row() -> None:
    with pytest.warns(UserWarning, match="Matrix Chi2 is empty"):
        ds = GroupChi2Test._inner_function(_cat(["u"] * 20), _cat(["u"] * 25))
    row = _row(ds)
    assert pd.isna(row["p-value"]) and pd.isna(row["statistic"])


@pytest.mark.xfail(
    strict=True,
    reason="Issue: matrix_preparation scales counts by group proportion instead of "
    "building a real contingency table, so results differ from chi2_contingency",
)
def test_chi2_matches_scipy_contingency() -> None:
    a = ["u"] * 30 + ["v"] * 10
    b = ["u"] * 15 + ["v"] * 25
    row = _row(GroupChi2Test._inner_function(_cat(a), _cat(b)))
    ref = stats.chi2_contingency(np.array([[30, 10], [15, 25]]))
    assert row["statistic"] == pytest.approx(ref[0], abs=TOL)
    assert row["p-value"] == pytest.approx(ref[1], abs=TOL)


# ---------------------------------------------------------------------------
# End-to-end execute
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cls,scipy_fn",
    [
        (GroupTTest, lambda x, y: stats.ttest_ind(x, y, equal_var=False)),
        (GroupUTest, lambda x, y: stats.mannwhitneyu(x, y)),
        (GroupKSTest, lambda x, y: stats.ks_2samp(x, y)),
    ],
)
def test_execute_compares_each_group_with_baseline(cls, scipy_fn) -> None:
    df = three_groups_df()
    ds = build_dataset(df, {"g": TreatmentRole(), "y": TargetRole()})
    ex = cls(compare_by="groups", grouping_role=TreatmentRole())
    table = result_frame(ex.execute(ExperimentData(ds)), ex)
    assert sorted(table.index) == ["b", "c"]
    base = df.y[df.g == "a"]
    for group in ("b", "c"):
        ref = scipy_fn(base, df.y[df.g == group])
        assert table.loc[group, "p-value"] == pytest.approx(ref.pvalue, abs=TOL)
        assert table.loc[group, "statistic"] == pytest.approx(ref.statistic, abs=TOL)


@pytest.mark.xfail(
    strict=True,
    reason="Issue: GroupHypothesisTesting.reliability is stored but never forwarded "
    "to _inner_function (calc_kwargs stays empty), so 0.05 is always used",
)
def test_execute_reliability_is_configurable() -> None:
    df = three_groups_df()
    ds = build_dataset(df, {"g": TreatmentRole(), "y": TargetRole()})
    ex = GroupTTest(
        compare_by="groups", grouping_role=TreatmentRole(), reliability=1e-12
    )
    table = result_frame(ex.execute(ExperimentData(ds)), ex)
    assert not table["pass"].astype(bool).any()


def test_search_types() -> None:
    assert int in GroupTTest(compare_by="groups").search_types
    assert GroupChi2Test(compare_by="groups").search_types == [str]
