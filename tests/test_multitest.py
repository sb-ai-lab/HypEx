"""Tests for the multiple-testing correction table of an A/B/n test."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests

from hypex import ABTest
from hypex.analyzers.ab import ABAnalyzer
from hypex.dataset import Dataset, InfoRole, TargetRole, TreatmentRole
from hypex.utils import ExperimentDataEnum

METRICS = ("m1", "m2", "m3")
TEST_GROUPS = ("1", "2")


@pytest.fixture
def abn_data() -> pd.DataFrame:
    """Three metrics over three groups: m1 moves in group 1, m2 in group 2."""
    rng = np.random.default_rng(1)
    n = 4000
    treat = rng.integers(0, 3, n)
    return pd.DataFrame(
        {
            "user_id": np.arange(n),
            "treat": treat,
            "m1": rng.normal(100, 10, n) + (treat == 1) * 1.6,
            "m2": rng.normal(100, 10, n) + (treat == 2) * 1.6,
            "m3": rng.normal(100, 10, n),
        }
    )


def _dataset(df: pd.DataFrame) -> Dataset:
    return Dataset(
        roles={
            "user_id": InfoRole(int),
            "treat": TreatmentRole(),
            **{metric: TargetRole() for metric in METRICS},
        },
        data=df,
    )


def _row(table: pd.DataFrame, field: str, group: str, test: str = "TTest") -> pd.Series:
    rows = table[
        (table["field"] == field)
        & (table["group"].astype(str) == group)
        & (table["test"] == test)
    ]
    assert len(rows) == 1, f"expected one row for {test} {field} {group}, got {len(rows)}"
    return rows.iloc[0]


def _p_values(df: pd.DataFrame, test: str) -> dict[tuple[str, str], float]:
    """Raw p-values per (metric, group) straight from scipy."""
    statistic = {
        "TTest": lambda a, b: ttest_ind(a, b, equal_var=False).pvalue,
        "UTest": lambda a, b: mannwhitneyu(a, b).pvalue,
    }[test]
    return {
        (metric, group): statistic(
            df[df["treat"] == 0][metric], df[df["treat"] == int(group)][metric]
        )
        for metric in METRICS
        for group in TEST_GROUPS
    }


def test_every_row_is_labelled_with_its_own_metric_and_group(abn_data):
    """The labels used to be transposed: rows come metric-major while field and
    group were filled group-major, so the flags landed on the wrong metric."""
    result = ABTest().execute(_dataset(abn_data))
    table = result.multitest.data
    expected = _p_values(abn_data, "TTest")

    assert len(table) == len(expected)
    for (metric, group), p_value in expected.items():
        assert _row(table, metric, group)["old p-value"] == pytest.approx(p_value)


def test_multitest_agrees_with_the_resume(abn_data):
    """The same comparison must carry the same p-value in both tables."""
    result = ABTest().execute(_dataset(abn_data))
    resume = result.resume.data
    table = result.multitest.data

    for _, resume_row in resume.iterrows():
        row = _row(table, resume_row["feature"], str(resume_row["group"]))
        assert row["old p-value"] == pytest.approx(resume_row["TTest p-value"])


def test_correction_is_applied_within_each_test(abn_data):
    """A metric checked by a t-test and by a u-test must not inflate the
    correction of the other: each test is a family of its own."""
    result = ABTest(additional_tests=["t-test", "u-test"]).execute(_dataset(abn_data))
    table = result.multitest.data

    pooled = []
    for test in ("TTest", "UTest"):
        expected = _p_values(abn_data, test)
        keys = list(expected)
        rejected, corrected, _, _ = multipletests(
            [expected[key] for key in keys], method="holm", alpha=0.05
        )
        pooled += [expected[key] for key in keys]
        for key, corrected_value, rejected_value in zip(keys, corrected, rejected):
            row = _row(table, key[0], key[1], test)
            assert row["new p-value"] == pytest.approx(corrected_value)
            assert bool(row["H0 rejected"]) == bool(rejected_value)

    # and the result is really not the correction over both tests at once
    pooled_corrected = multipletests(pooled, method="holm", alpha=0.05)[1]
    assert not np.allclose(sorted(table["new p-value"]), sorted(pooled_corrected))


def test_alpha_reaches_the_correction(abn_data):
    """``ABAnalyzer.alpha`` used to be dropped on the way to ``multipletests``,
    so everything was always rejected at 0.05."""
    corrected, rejections = None, {}
    for alpha in (0.05, 0.2):
        test = ABTest()
        test.experiment.set_params({ABAnalyzer: {"alpha": alpha}})
        row = _row(test.execute(_dataset(abn_data)).multitest.data, "m2", "1")
        corrected = row["new p-value"]
        rejections[alpha] = bool(row["H0 rejected"])

    # the corrected p-value itself does not depend on alpha, only the verdict does
    assert 0.05 < corrected < 0.2
    assert rejections == {0.05: False, 0.2: True}


def test_analyzer_aggregates_over_the_group_it_names(abn_data):
    """``TTest p-value 1`` used to be the mean over the first metric instead of
    the mean over the first group - the same transposition as in the table."""
    result = ABTest().execute(_dataset(abn_data))
    expected = _p_values(abn_data, "TTest")

    experiment_data = result.main_output._experiment_data
    analyzer_id = next(
        table_id
        for table_id in experiment_data.get_ids(
            ABAnalyzer, ExperimentDataEnum.analysis_tables
        )[ABAnalyzer.__name__]["analysis_tables"]
        if "MultiTest" not in table_id
    )
    aggregates = experiment_data.analysis_tables[analyzer_id].data

    for group in TEST_GROUPS:
        of_the_group = [
            p_value for (_, key), p_value in expected.items() if key == group
        ]
        assert aggregates[f"TTest p-value {group}"].iloc[0] == pytest.approx(
            np.mean(of_the_group)
        )
