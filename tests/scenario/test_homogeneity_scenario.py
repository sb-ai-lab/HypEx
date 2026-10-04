"""Homogeneity test scenarios: balanced vs imbalanced groups (I3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hypex import HomogeneityTest
from hypex.dataset import InfoRole, TargetRole, TreatmentRole

from ._utils import make_dataset, resume_row, to_pandas

pytestmark = pytest.mark.slow

ROLES = {
    "id": InfoRole(),
    "treat": TreatmentRole(),
    "age": TargetRole(),
    "spend": TargetRole(),
}


def _frame(n=1000, seed=0, imbalance=0.0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    treat = rng.randint(0, 2, n)
    age = rng.normal(40, 10, n) + imbalance * treat
    spend = rng.normal(100, 20, n)
    return pd.DataFrame(
        {"id": np.arange(n), "treat": treat, "age": age, "spend": spend}
    )


def _run(df):
    return HomogeneityTest().execute(make_dataset(df, ROLES))


def test_balanced_groups_pass_every_check() -> None:
    resume = to_pandas(_run(_frame(seed=1)).resume)
    assert set(resume["feature"]) == {"age", "spend"}
    for column in [c for c in resume.columns if c.endswith("pass")]:
        assert (resume[column] == "OK").all(), column


def test_imbalanced_feature_is_flagged_only_for_that_feature() -> None:
    output = _run(_frame(seed=2, imbalance=5.0))
    assert resume_row(output, "age")["TTest pass"] == "NOT OK"
    assert resume_row(output, "age")["KSTest pass"] == "NOT OK"
    assert resume_row(output, "spend")["TTest pass"] == "OK"


def test_imbalance_is_reflected_in_group_means() -> None:
    df = _frame(seed=3, imbalance=5.0)
    row = resume_row(_run(df), "age")
    control, test = df.age[df.treat == 0], df.age[df.treat == 1]
    assert float(row["control mean"]) == pytest.approx(control.mean(), abs=1e-9)
    assert float(row["test mean"]) == pytest.approx(test.mean(), abs=1e-9)
    assert float(row["difference"]) == pytest.approx(
        test.mean() - control.mean(), abs=1e-9
    )


def test_pvalue_decreases_as_imbalance_grows() -> None:
    p = [
        float(resume_row(_run(_frame(seed=4, imbalance=d)), "age")["TTest p-value"])
        for d in (0.0, 1.0, 3.0, 6.0)
    ]
    assert p[0] > p[1] > p[2] > p[3]
    assert p[-1] < 1e-10


def test_resume_has_expected_columns() -> None:
    resume = to_pandas(_run(_frame()).resume)
    assert {
        "feature",
        "group",
        "control mean",
        "test mean",
        "difference",
        "difference %",
        "TTest pass",
        "TTest p-value",
        "KSTest pass",
        "KSTest p-value",
    } <= set(resume.columns)


def test_three_groups_produce_one_row_per_comparison() -> None:
    rng = np.random.RandomState(5)
    df = _frame(900)
    df["treat"] = rng.randint(0, 3, len(df))
    resume = to_pandas(_run(df).resume)
    assert len(resume) == 2 * 2  # two features x two non-baseline groups
    assert sorted(set(resume["group"].astype(str))) == ["1", "2"]


def test_result_is_deterministic() -> None:
    first, second = _run(_frame(seed=6)), _run(_frame(seed=6))
    pd.testing.assert_frame_equal(to_pandas(first.resume), to_pandas(second.resume))
