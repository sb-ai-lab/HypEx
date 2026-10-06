"""Degenerate inputs for the public tests (I6).

Where the library currently misbehaves the test is an ``xfail(strict=True)`` that
states the desired behaviour (a descriptive error or a correct result).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from hypex import AATest, ABTest, HomogeneityTest
from hypex.dataset import FeatureRole, TargetRole, TreatmentRole
from hypex.utils import ID_SPLIT_SYMBOL as S
from hypex.utils import NoColumnsError, NotSuitableFieldError

from ._utils import ab_frame, make_dataset, to_pandas

pytestmark = pytest.mark.slow

ROLES = {"treat": TreatmentRole(), "y": TargetRole(), "x": FeatureRole()}


def _frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    return ab_frame(n=n, effect=0.0, seed=seed)[["treat", "y", "x"]]


def _run_all(df, roles=None):
    ds = lambda: make_dataset(df, roles or ROLES)  # noqa: E731
    return {
        "ab": lambda: ABTest().execute(ds()),
        "homo": lambda: HomogeneityTest().execute(ds()),
        "aa": lambda: AATest(n_iterations=2, random_states=[1, 2]).execute(ds()),
    }


# ---------------------------------------------------------------------------
# Empty dataset
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["ab", "homo", "aa"])
def test_empty_dataset_raises(name) -> None:
    with pytest.raises(Exception):
        _run_all(_frame().iloc[0:0])[name]()


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            "ab",
            marks=pytest.mark.xfail(
                strict=True,
                reason="Issue: an empty dataset fails with a bare IndexError('pop from empty list') "
                "instead of a descriptive error",
            ),
        ),
        pytest.param(
            "homo",
            marks=pytest.mark.xfail(
                strict=True,
                reason="Issue: HomogeneityTest() construction emits the DeprecationWarning of "
                "HomoDatasetReporter before any data validation (and an empty dataset gives a "
                "bare IndexError)",
            ),
        ),
        "aa",
    ],
)
def test_empty_dataset_error_is_descriptive(name) -> None:
    with pytest.raises((NotSuitableFieldError, NoColumnsError, ValueError)):
        _run_all(_frame().iloc[0:0])[name]()


# ---------------------------------------------------------------------------
# Single row / single group
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name",
    [
        "ab",
        pytest.param(
            "homo",
            marks=pytest.mark.xfail(
                strict=True,
                raises=DeprecationWarning,
                reason="Issue: HomogeneityTest() construction emits the DeprecationWarning of "
                "HomoDatasetReporter",
            ),
        ),
        "aa",
    ],
)
def test_single_row_raises_not_suitable_field(name) -> None:
    with pytest.raises(NotSuitableFieldError):
        _run_all(_frame().iloc[:1])[name]()


@pytest.mark.parametrize(
    "name",
    [
        "ab",
        pytest.param(
            "homo",
            marks=pytest.mark.xfail(
                strict=True,
                raises=DeprecationWarning,
                reason="Issue: HomogeneityTest() construction emits the DeprecationWarning of "
                "HomoDatasetReporter",
            ),
        ),
    ],
)
def test_single_group_raises_not_suitable_field(name) -> None:
    df = _frame().assign(treat=0)
    with pytest.raises(NotSuitableFieldError):
        _run_all(df)[name]()


# ---------------------------------------------------------------------------
# All NaN target
# ---------------------------------------------------------------------------
def test_all_nan_target_ab_does_not_crash_and_has_no_estimates() -> None:
    out = ABTest().execute(make_dataset(_frame().assign(y=np.nan), ROLES))
    row = to_pandas(out.resume).iloc[0]
    assert row["TTest p-value"] is None or pd.isna(row["TTest p-value"])
    assert row["difference"] is None or pd.isna(row["difference"])


def test_all_nan_target_is_not_reported_as_passed() -> None:
    out = ABTest().execute(make_dataset(_frame().assign(y=np.nan), ROLES))
    assert to_pandas(out.resume).iloc[0]["TTest pass"] != "OK"


def test_all_nan_target_aa_raises() -> None:
    with pytest.raises(Exception):
        AATest(n_iterations=2).execute(make_dataset(_frame().assign(y=np.nan), ROLES))


def test_partial_nan_target_is_ignored_in_means() -> None:
    df = _frame(400, seed=5)
    df.loc[df.sample(frac=0.2, random_state=0).index, "y"] = np.nan
    out = ABTest().execute(make_dataset(df, ROLES))
    row = to_pandas(out.resume).iloc[0]
    expected = df.y[df.treat == 1].mean() - df.y[df.treat == 0].mean()
    assert float(row["difference"]) == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# Missing roles
# ---------------------------------------------------------------------------
def test_missing_treatment_role_raises() -> None:
    df = _frame()[["y", "x"]]
    with pytest.raises(Exception):
        ABTest().execute(make_dataset(df, {"y": TargetRole(), "x": FeatureRole()}))


@pytest.mark.xfail(
    strict=True,
    reason="Issue: a dataset without any TargetRole column silently yields an empty report "
    "instead of raising NoColumnsError",
)
def test_missing_target_role_raises() -> None:
    df = _frame()[["treat", "x"]]
    with pytest.raises(NoColumnsError):
        ABTest().execute(
            make_dataset(df, {"treat": TreatmentRole(), "x": FeatureRole()})
        )


@pytest.mark.xfail(
    strict=True,
    reason="Issue: AATest without a TreatmentRole works, but ABTest/Homogeneity raise a "
    "NoColumnsError that names the *Target* role even though Treatment is missing",
)
def test_missing_treatment_error_names_the_treatment_role() -> None:
    df = _frame()[["y", "x"]]
    with pytest.raises(NoColumnsError, match="Treatment"):
        ABTest().execute(make_dataset(df, {"y": TargetRole(), "x": FeatureRole()}))


def test_aa_does_not_need_a_treatment_role() -> None:
    df = _frame()[["y", "x"]]
    out = AATest(n_iterations=2, random_states=[1, 2]).execute(
        make_dataset(df, {"y": TargetRole(), "x": FeatureRole()})
    )
    assert len(to_pandas(out.experiments)) == 2


# ---------------------------------------------------------------------------
# Duplicate index
# ---------------------------------------------------------------------------
def test_duplicate_index_ab_matches_manual_difference() -> None:
    df = _frame(300, seed=6)
    expected = df.y[df.treat == 1].mean() - df.y[df.treat == 0].mean()
    dup = df.copy()
    dup.index = [0] * len(dup)
    out = ABTest().execute(make_dataset(dup, ROLES))
    assert float(to_pandas(out.resume).iloc[0]["difference"]) == pytest.approx(
        expected, abs=1e-9
    )


@pytest.mark.xfail(
    strict=True,
    raises=NotSuitableFieldError,
    reason="Issue: AATest fails on a duplicated index because the split is keyed by the index",
)
def test_duplicate_index_aa_runs() -> None:
    dup = _frame(300).copy()
    dup.index = [0] * len(dup)
    AATest(n_iterations=2, random_states=[1, 2]).execute(make_dataset(dup, ROLES))


# ---------------------------------------------------------------------------
# ID_SPLIT_SYMBOL in column names
# ---------------------------------------------------------------------------
def _splitter_frame() -> tuple[pd.DataFrame, dict]:
    df = _frame(400, seed=7).rename(columns={"y": f"y{S}z"})
    roles = {"treat": TreatmentRole(), f"y{S}z": TargetRole(), "x": FeatureRole()}
    return df, roles


def test_splitter_symbol_in_target_name_still_computes_the_test() -> None:
    df, roles = _splitter_frame()
    out = ABTest().execute(make_dataset(df, roles))
    row = to_pandas(out.resume).iloc[0]
    ref = stats.ttest_ind(
        df[f"y{S}z"][df.treat == 1], df[f"y{S}z"][df.treat == 0], equal_var=False
    )
    assert float(row["TTest p-value"]) == pytest.approx(ref.pvalue, rel=0.05)


@pytest.mark.xfail(
    strict=True,
    reason="Issue: the feature name is rewritten ('y┴z' -> 'y|z') because ID_SPLIT_SYMBOL is "
    "escaped inside executor ids and never restored in the report",
)
def test_splitter_symbol_in_target_name_is_preserved_in_report() -> None:
    df, roles = _splitter_frame()
    out = ABTest().execute(make_dataset(df, roles))
    assert to_pandas(out.resume).iloc[0]["feature"] == f"y{S}z"


def test_splitter_symbol_in_target_name_aa_runs() -> None:
    df, roles = _splitter_frame()
    out = AATest(n_iterations=2, random_states=[1, 2]).execute(make_dataset(df, roles))
    assert len(to_pandas(out.experiments)) == 2
