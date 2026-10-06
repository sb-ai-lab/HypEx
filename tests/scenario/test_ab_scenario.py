"""A/B test scenarios with a known effect (I2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from hypex import ABTest
from hypex.dataset import (
    InfoRole,
    PreTargetRole,
    TargetRole,
    TreatmentRole,
)

from ._utils import ab_frame, make_dataset, resume_row, to_pandas

pytestmark = pytest.mark.slow


def _run(df, roles=None, **kwargs):
    return ABTest(**kwargs).execute(make_dataset(df, roles))


def test_known_effect_is_recovered_exactly_from_group_means() -> None:
    df = ab_frame(n=2000, effect=3.0, seed=1)
    row = resume_row(_run(df), "y", 1)
    control, test = df.y[df.treat == 0], df.y[df.treat == 1]
    assert float(row["difference"]) == pytest.approx(
        test.mean() - control.mean(), abs=1e-9
    )
    assert float(row["control mean"]) == pytest.approx(control.mean(), abs=1e-9)
    assert float(row["test mean"]) == pytest.approx(test.mean(), abs=1e-9)


def test_estimated_effect_is_close_to_truth() -> None:
    df = ab_frame(n=4000, effect=3.0, seed=2)
    row = resume_row(_run(df), "y", 1)
    assert float(row["difference"]) == pytest.approx(3.0, abs=0.5)


def test_percentage_difference_is_relative_to_control_mean() -> None:
    df = ab_frame(n=1000, effect=1.0, seed=3)
    df["y"] = df["y"] + 50  # keep the control mean far from zero
    row = resume_row(_run(df), "y", 1)
    assert float(row["difference %"]) == pytest.approx(
        float(row["difference"]) / float(row["control mean"]) * 100, rel=1e-6
    )


def test_real_effect_is_flagged_and_null_is_not() -> None:
    # AB convention: a significant effect is reported as "OK".
    assert resume_row(_run(ab_frame(n=1000, effect=1.0, seed=4)))["TTest pass"] == "OK"
    assert (
        resume_row(_run(ab_frame(n=1000, effect=0.0, seed=4)))["TTest pass"] == "NOT OK"
    )


def test_pvalue_matches_welch_ttest() -> None:
    df = ab_frame(n=800, effect=0.3, seed=5)
    row = resume_row(_run(df))
    ref = stats.ttest_ind(df.y[df.treat == 1], df.y[df.treat == 0], equal_var=False)
    # StatsTTest picks pooled/Welch from the variance ratio; with equal sds both agree closely
    assert float(row["TTest p-value"]) == pytest.approx(ref.pvalue, rel=0.05, abs=1e-4)


def test_t_test_equal_var_option_changes_nothing_when_variances_match() -> None:
    df = ab_frame(n=800, effect=0.3, seed=6)
    welch = float(resume_row(_run(df, t_test_equal_var=False))["TTest p-value"])
    pooled = float(resume_row(_run(df, t_test_equal_var=True))["TTest p-value"])
    assert welch == pytest.approx(pooled, rel=0.05)


@pytest.mark.parametrize("extra", ["u-test", "t-test"])
def test_additional_tests_are_reported(extra) -> None:
    out = _run(ab_frame(n=600, effect=1.0, seed=7), additional_tests=extra)
    columns = set(to_pandas(out.resume).columns)
    name = {"u-test": "UTest", "t-test": "TTest"}[extra]
    assert f"{name} p-value" in columns


def test_sizes_table_reports_group_counts() -> None:
    df = ab_frame(n=900, effect=1.0, seed=8)
    sizes = to_pandas(_run(df).sizes)
    assert int(sizes["control size"].iloc[0]) == (df.treat == 0).sum()
    assert int(sizes["test size"].iloc[0]) == (df.treat == 1).sum()


# ---------------------------------------------------------------------------
# Three groups and multiple testing
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def three_groups() -> pd.DataFrame:
    return ab_frame(n=1500, effect=1.0, groups=3, seed=9)  # group g gets +1*g


def test_three_groups_each_compared_to_control(three_groups) -> None:
    out = _run(three_groups)
    resume = to_pandas(out.resume)
    assert sorted(resume["group"].astype(str)) == ["1", "2"]
    for g in (1, 2):
        row = resume_row(out, "y", g)
        expected = (
            three_groups.y[three_groups.treat == g].mean()
            - three_groups.y[three_groups.treat == 0].mean()
        )
        assert float(row["difference"]) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    "method",
    [
        "bonferroni",
        "sidak",
        "holm",
        "holm-sidak",
        "simes-hochberg",
        "hommel",
        "fdr_bh",
        "fdr_by",
        "fdr_tsbh",
    ],
)
def test_multitest_methods_never_decrease_pvalues(three_groups, method) -> None:
    out = _run(three_groups, multitest_method=method)
    table = to_pandas(out.multitest)
    old = table["old p-value"].astype(float).to_numpy()
    new = table["new p-value"].astype(float).to_numpy()
    assert len(table) == 2
    assert (new >= old - 1e-12).all() and (new <= 1.0 + 1e-12).all()


def test_bonferroni_is_at_least_as_conservative_as_holm(three_groups) -> None:
    bonf = to_pandas(_run(three_groups, multitest_method="bonferroni").multitest)
    holm = to_pandas(_run(three_groups, multitest_method="holm").multitest)
    assert (
        bonf["new p-value"].astype(float).to_numpy()
        >= holm["new p-value"].astype(float).to_numpy() - 1e-12
    ).all()


def test_bonferroni_equals_pvalue_times_comparisons(three_groups) -> None:
    table = to_pandas(_run(three_groups, multitest_method="bonferroni").multitest)
    old = table["old p-value"].astype(float).to_numpy()
    np.testing.assert_allclose(
        table["new p-value"].astype(float), np.minimum(old * 2, 1.0), atol=1e-12
    )


@pytest.mark.xfail(
    strict=True,
    reason="Issue: ABTest._make_experiment replaces None (and any unknown name) with 'holm', "
    "so multitest_method=None cannot disable the correction",
)
def test_no_multitest_method_gives_message(three_groups) -> None:
    out = _run(three_groups, multitest_method=None)
    assert isinstance(out.multitest, str)


def test_two_groups_do_not_need_multitest() -> None:
    out = _run(ab_frame(n=500, effect=1.0, seed=10), multitest_method="holm")
    assert isinstance(out.multitest, str)


@pytest.mark.xfail(
    strict=True,
    raises=Exception,
    reason="Issue: ABTest(multitest_method='quantile') always fails because "
    "MultitestQuantile has no calc()",
)
def test_quantile_multitest_runs(three_groups) -> None:
    _run(three_groups, multitest_method="quantile")


@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason="Issue: unknown multitest method names (including the documented 'fdr_tsbhy') are "
    "silently replaced by 'holm' instead of raising",
)
@pytest.mark.parametrize("name", ["fdr_tsbhy", "nonsense"])
def test_unknown_multitest_method_is_rejected(name) -> None:
    with pytest.raises(ValueError):
        ABTest(multitest_method=name)


# ---------------------------------------------------------------------------
# Variance reduction
# ---------------------------------------------------------------------------
def _with_pre(n=2000, effect=1.0, seed=11) -> tuple[pd.DataFrame, dict]:
    rng = np.random.RandomState(seed)
    base = rng.normal(0, 3, n)
    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "treat": rng.randint(0, 2, n),
            "y_pre": base + rng.normal(0, 0.5, n),
        }
    )
    df["y"] = base + effect * df.treat + rng.normal(0, 0.5, n)
    roles = {
        "id": InfoRole(),
        "treat": TreatmentRole(),
        "y": TargetRole(),
        "y_pre": PreTargetRole(),
    }
    return df, roles


def test_cuped_reduces_pvalue_noise_end_to_end() -> None:
    df, roles = _with_pre()
    out = _run(df, roles, cuped_features={"y": "y_pre"})
    plain = float(resume_row(_run(df, roles), "y", 1)["TTest p-value"])
    cuped = float(resume_row(out, "y_cuped", 1)["TTest p-value"])
    assert cuped < plain


@pytest.mark.xfail(
    strict=True,
    reason="Issue: CUPAC cannot run end to end (CupacExtension.calc dispatch and "
    "_agg_data_from_cupac_data add_column errors)",
)
def test_cupac_runs_end_to_end() -> None:
    df, roles = _with_pre()
    roles = dict(roles, y=TargetRole(cofounders=["f"]))
    _run(df, roles, enable_cupac=True, cupac_models="linear")


def test_exact_att_reference_dataset() -> None:
    """create_test_data(exact_ATT=100) embeds a +100 effect; the post-minus-pre ATT is ~63.5."""
    from hypex.utils.tutorial_data_creation import create_test_data

    try:
        df = create_test_data(num_users=3000, rs=0, exact_ATT=100)
    except (
        TypeError
    ) as exc:  # pandas < 2.2 does not know groupby.apply(include_groups=...)
        pytest.xfail(f"create_test_data is incompatible with this pandas: {exc}")
    roles = {
        "user_id": InfoRole(),
        "treat": TreatmentRole(),
        "post_spends": TargetRole(),
        "pre_spends": PreTargetRole(),
    }
    roles = {c: r for c, r in roles.items() if c in df.columns}
    out = ABTest().execute(make_dataset(df, roles))
    diff = float(resume_row(out, "post_spends", 1)["difference"])
    assert diff == pytest.approx(63.5, abs=15.0)
