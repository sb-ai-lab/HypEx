"""Tests for AdaptiveHypothesisTest backend dispatch."""

from __future__ import annotations

import pytest

from hypex.comparators import GroupTTest, StatsTTest
from hypex.comparators.abstract import AdaptiveHypothesisTest
from hypex.dataset import ExperimentData, TargetRole, TreatmentRole
from hypex.utils import BackendsEnum

from ._utils import build_dataset, result_frame, three_groups_df, to_pandas


class _AdaptiveT(AdaptiveHypothesisTest):
    BACKEND_MAP = {  # noqa: RUF012
        BackendsEnum.pandas: GroupTTest,
        BackendsEnum.spark: StatsTTest,
    }


class _PandasOnly(AdaptiveHypothesisTest):
    BACKEND_MAP = {BackendsEnum.pandas: GroupTTest}  # noqa: RUF012


ROLES = {"g": TreatmentRole(), "y": TargetRole()}


def test_build_delegate_group_branch_copies_config_and_id() -> None:
    adaptive = _AdaptiveT(
        grouping_role=TreatmentRole(), reliability=0.01, compare_by="groups", key="k"
    )
    delegate = adaptive._build_delegate(GroupTTest)
    assert isinstance(delegate, GroupTTest)
    assert delegate.reliability == 0.01
    assert delegate.compare_by == "groups"
    assert delegate.id == adaptive.id


def test_build_delegate_stats_branch_copies_config_and_id() -> None:
    adaptive = _AdaptiveT(reliability=0.2, key="k")
    delegate = adaptive._build_delegate(StatsTTest)
    assert isinstance(delegate, StatsTTest)
    assert delegate.reliability == 0.2
    assert delegate.id == adaptive.id


def _only_table(out):
    assert len(out.analysis_tables) == 1
    return to_pandas(next(iter(out.analysis_tables.values())))


def test_pandas_dispatches_to_group_test() -> None:
    ds = build_dataset(three_groups_df(), ROLES)
    out = _AdaptiveT(grouping_role=TreatmentRole()).execute(ExperimentData(ds))
    assert sorted(_only_table(out).index) == ["b", "c"]


@pytest.mark.xfail(
    strict=True,
    reason="Issue: execute() assigns self.key on the delegate, which regenerates its id, "
    "so results are NOT stored under the adaptive instance's id",
)
def test_results_stored_under_adaptive_id() -> None:
    ds = build_dataset(three_groups_df(), ROLES)
    adaptive = _AdaptiveT(grouping_role=TreatmentRole())
    out = adaptive.execute(ExperimentData(ds))
    assert adaptive.id in out.analysis_tables


@pytest.mark.spark
def test_spark_dispatches_to_stats_test(spark_session) -> None:
    ds = build_dataset(three_groups_df(), ROLES, BackendsEnum.spark, spark_session)
    adaptive = _AdaptiveT(grouping_role=TreatmentRole())
    out = adaptive.execute(ExperimentData(ds))
    stats_free = [k for k in out.analysis_tables if not k.endswith("stats")]
    assert len(stats_free) == 1
    assert sorted(to_pandas(out.analysis_tables[stats_free[0]]).index) == ["b┆y", "c┆y"]


def test_unregistered_backend_raises(spark_session) -> None:
    ds = build_dataset(three_groups_df(), ROLES, BackendsEnum.spark, spark_session)
    with pytest.raises(ValueError, match="no implementation for backend"):
        _PandasOnly(grouping_role=TreatmentRole()).execute(ExperimentData(ds))


def test_empty_backend_map_raises() -> None:
    ds = build_dataset(three_groups_df(), ROLES)
    with pytest.raises(ValueError, match="no implementation"):
        AdaptiveHypothesisTest().execute(ExperimentData(ds))


def test_inner_function_is_not_callable() -> None:
    with pytest.raises(NotImplementedError):
        AdaptiveHypothesisTest._inner_function(None)


def test_results_match_direct_delegate() -> None:
    df = three_groups_df()
    ds = build_dataset(df, ROLES)
    adaptive = _AdaptiveT(grouping_role=TreatmentRole())
    direct = GroupTTest(compare_by="groups", grouping_role=TreatmentRole())
    via_adaptive = _only_table(adaptive.execute(ExperimentData(ds)))
    via_direct = result_frame(direct.execute(ExperimentData(ds)), direct)
    assert via_adaptive["p-value"].tolist() == pytest.approx(
        via_direct["p-value"].tolist()
    )
